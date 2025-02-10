# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions and classes for simulating document editing."""

import copy
import dataclasses
import functools
import glob
import io
import itertools
import json
import math
import os
from typing import Collection, Mapping, Sequence

from PIL import Image

_BACKGROUND_IMAGE_ID = -1
_BBOX_CHARACTER_HEIGHT = 11
_BBOX_CHARACTER_WIDTH = 7.5
_BBOX_MARGIN_WORD = 5
_ELEMENT_CHANGED_COLOR = 'fuchsia'


@dataclasses.dataclass
class Point:
  """Represents a point in 2D space.

  Attributes:
    x: The x-coordinate of the point.
    y: The y-coordinate of the point.
  """

  x: float
  y: float


@dataclasses.dataclass
class BoundingBox:
  """A class representing a bounding box.

  It assumes that the horizontal axis goes from left to right and the vertical
  axis goes from top to bottom.

  Attributes:
      top: The y-coordinate of the top edge.
      left: The x-coordinate of the left edge.
      bottom: The y-coordinate of the bottom edge.
      right: The x-coordinate of the right edge.
  """

  top: float
  left: float
  bottom: float
  right: float

  def __init__(self, top: float, left: float, bottom: float, right: float):
    if left > right:
      raise ValueError(
          f'Left edge must be to the left of the right edge but got left={left}'
          f' and right={right}'
      )
    if top > bottom:
      raise ValueError(
          f'Top edge must be above the bottom edge but got top={top} and'
          f' bottom={bottom}'
      )
    self.top = top
    self.left = left
    self.bottom = bottom
    self.right = right

  @property
  def width(self) -> float:
    """Returns the width of the bounding box."""
    return self.right - self.left

  @property
  def height(self) -> float:
    """Returns the height of the bounding box."""
    return self.bottom - self.top

  @property
  def area(self) -> float:
    """Returns the area of the bounding box."""
    return self.width * self.height

  @property
  def center(self) -> Point:
    """Calculates the center of the bounding box."""
    return Point(x=(self.left + self.right) / 2, y=(self.top + self.bottom) / 2)

  def union(self, other: 'BoundingBox') -> 'BoundingBox':
    """Returns the union of this box with another."""
    return BoundingBox(
        left=min(self.left, other.left),
        top=min(self.top, other.top),
        right=max(self.right, other.right),
        bottom=max(self.bottom, other.bottom),
    )

  def intersection(self, other: 'BoundingBox') -> 'BoundingBox':
    """Returns the intersection of this box with another."""
    top = max(self.top, other.top)
    left = max(self.left, other.left)
    bottom = min(self.bottom, other.bottom)
    right = min(self.right, other.right)
    if top > bottom or left > right:  # Check for disjoint boxes.
      return BoundingBox(top=0, left=0, bottom=0, right=0)
    return BoundingBox(
        top=top,
        left=left,
        bottom=bottom,
        right=right,
    )

  def get_vertical_overlap(self, other: 'BoundingBox') -> float:
    """Computes the vertical overlap between this bounding box and another."""
    return max(0, min(self.bottom, other.bottom) - max(self.top, other.top))

  def get_horizontal_overlap(self, other: 'BoundingBox') -> float:
    """Computes the horizontal overlap between this bounding box and another."""
    return max(0, min(self.right, other.right) - max(self.left, other.left))

  def normalize(self, normalize: float | tuple[float, float]) -> 'BoundingBox':
    """Normalizes the bounding box by the given factor.

    Args:
      normalize: The factor to normalize by. If a tuple is provided, the first
        value is used for the horizontal normalization and the second value is
        used for the vertical normalization.

    Returns:
      The normalized bounding box.
    """
    if not isinstance(normalize, tuple):
      normalize = (normalize, normalize)
    return BoundingBox(
        top=self.top / normalize[1],
        left=self.left / normalize[0],
        bottom=self.bottom / normalize[1],
        right=self.right / normalize[0],
    )


@dataclasses.dataclass
class Element:
  """Represents an element within a structured document.

  Attributes:
    id: Unique identifier for the element.
    class_name: The class name of the element.
    bbox: The bounding box of the element, represented by a BoundingBox object.
    children_ids: The IDs of child elements contained within this element.
    parent_id: The ID of the parent element.
    text: The text content of the element, if applicable.
    next_id: The ID of the next sibling element.
    prev_id: The ID of the previous sibling element.
    color: The color of the element, if applicable.
  """

  id: int
  class_name: str
  bbox: BoundingBox
  children_ids: list[int] = dataclasses.field(default_factory=list)
  parent_id: int | None = None
  text: str | None = None
  next_id: int | None = None
  prev_id: int | None = None
  color: str | None = None

  @classmethod
  def from_mapping(cls, bbox: Mapping[str, float], **kwargs) -> 'Element':
    """Returns an Element from a mapping of data (e.g. from parsed JSON)."""
    return Element(bbox=BoundingBox(**bbox), **kwargs)


@dataclasses.dataclass
class Stroke:
  """Represents a single stroke of an ink.

  Attributes:
    xs: The x-coordinates of the stroke.
    ys: The y-coordinates of the stroke.
  """

  xs: list[float]
  ys: list[float]


@dataclasses.dataclass
class Ink:
  """Represents an ink (handwritten input).

  Attributes:
    strokes: The strokes of the ink.
  """

  strokes: list[Stroke]

  def get_bbox(self) -> BoundingBox:
    """Returns the bounding box of the ink."""
    return BoundingBox(
        top=min(min(stroke.ys) for stroke in self.strokes),
        left=min(min(stroke.xs) for stroke in self.strokes),
        bottom=max(max(stroke.ys) for stroke in self.strokes),
        right=max(max(stroke.xs) for stroke in self.strokes),
    )

  def get_stroke_lengths(self) -> Sequence[float]:
    """Returns the length of each stroke in the ink."""
    return list(map(len, [s.xs for s in self.strokes]))


def _estimate_character_width(character: str) -> float:
  """Estimates the width of a character when rendered on the canvas."""
  if character in "1iIlt:.,;'()":
    return _BBOX_CHARACTER_WIDTH * 0.8
  return _BBOX_CHARACTER_WIDTH


def _estimate_text_width(text: str) -> float:
  """Estimates the width of a text when rendered on the canvas."""
  return sum(_estimate_character_width(character) for character in text)


@dataclasses.dataclass
class Page:
  """Represents a document with its elements and images.

  Attributes:
    id: The ID of the page in the dataset.
    element_from_id: A mapping from element ID to page element.
    image_from_id: A mapping from image ID to image..
  """

  id: str
  element_from_id: dict[int, Element]
  image_from_id: dict[int, Image.Image]

  def delete_element(self, element_id: int):
    """Deletes an element and its parents if they become empty."""
    element = self.element_from_id.pop(element_id)

    if element.prev_id:
      self.element_from_id[element.prev_id].next_id = element.next_id
    if element.next_id:
      self.element_from_id[element.next_id].prev_id = element.prev_id

    parent = self.element_from_id.get(element.parent_id)
    if parent and parent.children_ids:
      parent.children_ids.remove(element.id)
      if not parent.children_ids:
        self.delete_element(parent.id)

  def insert_line(
      self,
      bbox: BoundingBox,
      parent_id: int,
      children_ids: list[int],
  ) -> Element:
    """Inserts a line element in the document."""
    new_line = Element(
        id=len(self.element_from_id),
        class_name='textline',
        bbox=bbox,
        children_ids=children_ids,
        parent_id=parent_id,
    )

    self.element_from_id[parent_id].children_ids.append(new_line.id)
    self.element_from_id[new_line.id] = new_line

    for children_id in children_ids:
      self.element_from_id[children_id].parent_id = new_line.id

    return new_line

  def insert_word(
      self,
      bbox: BoundingBox,
      parent_id: int,
      text: str,
  ) -> Element:
    """Inserts a word element in the document."""
    new_word = Element(
        id=len(self.element_from_id),
        class_name='word',
        text=text,
        bbox=bbox,
        parent_id=parent_id,
        color=_ELEMENT_CHANGED_COLOR,
    )

    self.element_from_id[parent_id].children_ids.append(new_word.id)
    self.element_from_id[new_word.id] = new_word

    return new_word

  def _vertically_shift_element(self, element: Element, shift_amount: float):
    """Vertically shifts an element and its children by the given amount.

    This function recursively shifts an element and its children vertically by
    the
    specified amount, updating their bounding boxes accordingly.

    Args:
      element: The element to shift.
      shift_amount: The amount to shift the element vertically.
    """
    element.bbox.top += shift_amount
    element.bbox.bottom += shift_amount
    if element.children_ids:
      for child_id in element.children_ids:
        self._vertically_shift_element(
            self.element_from_id[child_id], shift_amount
        )

  def reflow_paragraphs_and_images(
      self,
      source: Element,
      source_previous_bbox: BoundingBox,
  ):
    """Reflows paragraphs and images below a given source element.

    This function iterates through elements in the document and shifts
    paragraphs
    and images vertically if their overlap with the source element has increased
    due to changes in the source's bounding box.

    Args:
      source: The source element that has been modified.
      source_previous_bbox: The previous bounding box of the source element.
        This is used to determine if the overlap with other elements has
        increased or decreased due to the modification of the source element.
    """
    for element in self.element_from_id.values():
      if element == source or element.class_name not in ('paragraph', 'image'):
        continue

      overlap_before = source_previous_bbox.intersection(element.bbox).area
      overlap_after = source.bbox.intersection(element.bbox).area

      if overlap_after - overlap_before > 0.01:
        overlap_y = source.bbox.get_vertical_overlap(element.bbox)
        element_previous_bbox = copy.copy(element.bbox)
        self._vertically_shift_element(element, overlap_y + 10)
        self.reflow_paragraphs_and_images(element, element_previous_bbox)

  def recompute_bbox(self, element_id: int):
    """Recomputes the bounding box of an element based on its children."""
    if element_id not in self.element_from_id:
      print(
          f'Could not recompute bbox for element {element_id} since it could'
          ' not be found in the page.'
      )
      return

    element = self.element_from_id[element_id]
    if not element.children_ids:
      return

    children_bboxes = []
    for child_id in element.children_ids:
      self.recompute_bbox(child_id)
      children_bboxes.append(self.element_from_id[child_id].bbox)

    element.bbox = functools.reduce(
        lambda bbox1, bbox2: bbox1.union(bbox2), children_bboxes
    )

  def shift_words(self, word: Element | None, shift_amount: float):
    """Shifts a word horizontally and reflows the surrounding text as needed.

    This function shifts a word horizontally within its paragraph, handling
    overflows to the next line or underflows to the previous line. It also
    updates the bounding boxes of affected elements and reflows the surrounding
    paragraphs to maintain document structure.

    Args:
      word: The word element to shift.
      shift_amount: The amount to shift the word horizontally.
    """
    if word is None:
      return

    # Mark this element as being altered so that we can visualize it later.
    word.color = _ELEMENT_CHANGED_COLOR
    line = self.element_from_id[word.parent_id]
    paragraph = self.element_from_id[line.parent_id]
    next_word = self.element_from_id.get(word.next_id)
    prev_word = self.element_from_id.get(word.prev_id)
    word_width = word.bbox.width
    word_height = word.bbox.height

    if word.bbox.right + shift_amount > paragraph.bbox.right:
      # This word is overflowing, we push it to the next line instead.
      line.children_ids.remove(word.id)

      if next_word:
        # If the next word is on the same line, it will overflow too so we push
        # it first.
        if next_word.parent_id == word.parent_id:
          self.shift_words(next_word, shift_amount)

        word.parent_id = next_word.parent_id
        self.element_from_id[word.parent_id].children_ids.append(word.id)

        # Move the current word to the next line.
        word.bbox = BoundingBox(
            left=next_word.bbox.left,
            top=next_word.bbox.top,
            right=next_word.bbox.left + word_width,
            bottom=next_word.bbox.top + word_height,
        )
        self.shift_words(next_word, word_width + _BBOX_MARGIN_WORD)
      else:
        # We reached the end of a line so we create a new one.
        word.bbox = BoundingBox(
            left=line.bbox.left,
            top=line.bbox.bottom + _BBOX_MARGIN_WORD,
            right=line.bbox.left + word_width,
            bottom=line.bbox.bottom + _BBOX_MARGIN_WORD + word_height,
        )
        self.insert_line(
            word.bbox,
            parent_id=paragraph.id,
            children_ids=[word.id],
        )

    elif word.bbox.left + shift_amount < paragraph.bbox.left:
      # This word is underflowing, we move it to the previous line if there is
      # room. Otherwise, we clamp it to the side of the paragraph.
      if not prev_word:
        # This word is the first word of the paragraph, we clamp it to the side
        # of the paragraph.
        self.shift_words(word, -max(0, word.bbox.left - paragraph.bbox.left))
        return

      if (
          prev_word.bbox.right + word_width + _BBOX_MARGIN_WORD
          > paragraph.bbox.right
      ):
        # There is no space in the line above, we clamp to the side of the
        # paragraph.
        new_shift_amount = -(word.bbox.left - paragraph.bbox.left)
        word.bbox.left = paragraph.bbox.left
        word.bbox.right = word.bbox.left + word_width
        self.shift_words(next_word, new_shift_amount)
      else:
        # Move the word to the line above it.
        line.children_ids.remove(word.id)
        word.parent_id = prev_word.parent_id
        self.element_from_id[word.parent_id].children_ids.append(word.id)

        word.bbox = BoundingBox(
            left=prev_word.bbox.right + _BBOX_MARGIN_WORD,
            top=prev_word.bbox.top,
            right=prev_word.bbox.right + _BBOX_MARGIN_WORD + word_width,
            bottom=prev_word.bbox.top + word_height,
        )

        if next_word:
          new_shift_amount = -(next_word.bbox.left - paragraph.bbox.left + 1)
          self.shift_words(next_word, new_shift_amount)
        else:
          # The parent is now empty, we delete it.
          self.delete_element(line.id)

    else:
      # Shift the word normally, and propagate the change to the next element.
      word.bbox.left += shift_amount
      word.bbox.right += shift_amount
      if (
          next_word
          and next_word.parent_id == word.parent_id
          or shift_amount < 0
      ):
        self.shift_words(next_word, shift_amount)

  def _elements_under_bbox(
      self,
      bbox: BoundingBox,
      class_names: Collection[str],
      threshold: float = 0.5,
  ) -> Sequence[Element]:
    """Returns the elements that overlap the bbox by more than the threshold."""
    return [
        element
        for element in self.element_from_id.values()
        if element.class_name in class_names
        and element.bbox.intersection(bbox).area / element.bbox.area > threshold
    ]

  def _get_root_ancestor_id(self, element: Element):
    """Returns the ID of the topmost parent of the given element."""
    topmost_element = element
    while topmost_element.parent_id is not None:
      topmost_element = self.element_from_id[topmost_element.parent_id]
    return topmost_element.id

  def delete_elements_under_bbox(self, bbox: BoundingBox):
    """Deletes elements under a bounding box and reflows text as necessary.

    This function identifies elements that overlap with the given bounding box,
    deletes them, and adjusts the positions of the remaining elements to
    maintain the document structure.

    Args:
      bbox: The bounding box to delete elements from.
    """
    overlapping_words = []
    for overlapping_element in self._elements_under_bbox(
        bbox, class_names={'word', 'image'}
    ):
      # Delete images directly as they are not affected by reflow.
      if overlapping_element.class_name == 'image':
        self.delete_element(overlapping_element.id)
      else:
        overlapping_words.append(overlapping_element)

    # Determines the amount to shift words for each line separately.
    root_ancestor_ids = set()
    sorted_overlapping_words = sorted(
        overlapping_words,
        key=lambda element: (
            # Sort first by line, bottom to top, then by word left to right.
            -self.element_from_id[element.parent_id].bbox.top,
            element.bbox.right,
        ),
    )
    for _, elements in itertools.groupby(
        sorted_overlapping_words, lambda element: element.parent_id
    ):
      elements = list(elements)
      next_element = self.element_from_id.get(elements[-1].next_id)
      bboxes = [element.bbox for element in elements]
      union_bbox = functools.reduce(
          lambda bbox1, bbox2: bbox1.union(bbox2), bboxes
      )
      for element in elements:
        root_ancestor_ids.add(self._get_root_ancestor_id(element))
        self.delete_element(element.id)
      if next_element and next_element not in overlapping_words:
        self.shift_words(next_element, -union_bbox.width)

    # Adjust the bounding boxes of parents that have been affected by deletion.
    for root_ancestor_id in root_ancestor_ids:
      if root_ancestor_id in self.element_from_id:
        self.recompute_bbox(root_ancestor_id)

  def _find_closest_word(self, insertion_bbox: BoundingBox) -> Element | None:
    """Returns the closest, vertically overlapping word.

    For computing the distance, the horizontal center of the insertion bounding
    boxes is considered.

    Args:
      insertion_bbox: The bounding box to find the closest word to.
    """
    closest_element = None
    min_distance = math.inf
    insertion_x = (insertion_bbox.left + insertion_bbox.right) / 2

    for element in self.element_from_id.values():
      if (
          element.class_name != 'word'
          or not element.bbox.get_vertical_overlap(insertion_bbox) > 0.001
      ):
        continue

      distance = min(
          abs(insertion_x - element.bbox.left),
          abs(insertion_x - element.bbox.right),
      )
      if distance < min_distance:
        min_distance = distance
        closest_element = element

    return closest_element

  def _insert_word_at_location(
      self,
      insertion_bbox: BoundingBox,
      word_text: str,
  ) -> Element | None:
    """Inserts a word into the document.

    Args:
      insertion_bbox: The bounding box of the word to insert.
      word_text: The text of the word to insert.

    Returns:
      The inserted word element, or None if the word could not be inserted.
    """
    # Find the closest element.
    closest_element = self._find_closest_word(insertion_bbox)
    if not closest_element:
      return

    paragraph = self.element_from_id[
        self._get_root_ancestor_id(closest_element)
    ]
    paragraph_bbox_before = copy.copy(paragraph.bbox)

    # Find whether to insert before or after the closest element.
    insertion_x = (insertion_bbox.left + insertion_bbox.right) / 2
    insert_before = abs(insertion_x - closest_element.bbox.left) < abs(
        insertion_x - closest_element.bbox.right
    )
    word = self.insert_word(
        bbox=BoundingBox(top=0, left=0, bottom=0, right=0),
        parent_id=closest_element.parent_id,
        text=word_text,
    )
    word_width = _estimate_text_width(word_text)

    # Determine the target element for linking and shifting
    if insert_before:
      left = self.element_from_id.get(closest_element.prev_id)
      right = closest_element
      word_bbox_anchor = closest_element.bbox.left - _BBOX_MARGIN_WORD
    else:
      left = closest_element
      right = self.element_from_id.get(closest_element.next_id)
      word_bbox_anchor = closest_element.bbox.right

    if left:
      left.next_id = word.id
      word.prev_id = left.id
    if right:
      word.next_id = right.id
      right.prev_id = word.id

    word.bbox = BoundingBox(
        top=closest_element.bbox.top,
        # We shift the word itself by word_width to ensure no overflow.
        left=word_bbox_anchor - word_width,
        bottom=closest_element.bbox.bottom,
        right=word_bbox_anchor,
    )

    self.shift_words(word, word_width + _BBOX_MARGIN_WORD)

    self.recompute_bbox(paragraph.id)
    self.reflow_paragraphs_and_images(paragraph, paragraph_bbox_before)

    return word

  def insert_words(
      self,
      insertion_bbox: BoundingBox,
      words: Sequence[str],
  ):
    """Inserts words into the document at the given location.

    Each word in the sequence is inserted to the right of the previous word.

    Args:
      insertion_bbox: The bounding box indicating the location of the first word
        to be inserted.
      words: The list of words to be inserted.
    """
    for word in words:
      inserted_word = self._insert_word_at_location(insertion_bbox, word)
      if inserted_word:
        insertion_bbox = copy.copy(inserted_word.bbox)
        insertion_bbox.left = insertion_bbox.right

  def highlight_words_under_bbox(self, bbox: BoundingBox):
    """Highlights the elements that overlap with the given bounding box."""
    for overlapping_element in self._elements_under_bbox(
        bbox, class_names={'word'}
    ):
      overlapping_element.color = _ELEMENT_CHANGED_COLOR

  def crop_image(self, bbox: BoundingBox):
    """Crops overlapping images to the given bounding box."""
    overlapping_images = self._elements_under_bbox(
        bbox,
        threshold=0.05,
        class_names={'image'},
    )
    for image_element in overlapping_images:
      cropped_bbox = bbox.intersection(image_element.bbox)
      self.image_from_id[image_element.id] = self.image_from_id[
          image_element.id
      ].crop((
          cropped_bbox.left - image_element.bbox.left,
          cropped_bbox.top - image_element.bbox.top,
          cropped_bbox.right - image_element.bbox.left,
          cropped_bbox.bottom - image_element.bbox.top,
      ))
      image_element.bbox = cropped_bbox

  def edit(self, edit_name: str, edit_bbox: BoundingBox, text: str = ''):
    """Edits the page according to the given class name and bbox.

    Args:
      edit_name: The type of edit to perform.
      edit_bbox: The bounding box of the edit.
      text: Optional text to use for the edit.
    """
    if edit_name == 'insert':
      words = text.split()
      self.insert_words(edit_bbox, words)
    elif edit_name in (
        'question',
        'underline',
        'question',
        'instruct_text',
        'select',
    ):
      self.highlight_words_under_bbox(edit_bbox)
    elif edit_name == 'delete':
      self.delete_elements_under_bbox(edit_bbox)
    elif edit_name == 'crop':
      self.crop_image(edit_bbox)

  def tighten_bboxes_for_colab_canvas(self):
    """Tightens the bounding boxes of the elements.

    Use this function to tighten the bounding boxes of the elements to better
    reflect their size when they are rendered with `rendering.render_document`.
    By default, the bounding boxes are set to the size of the text in the
    original image from which OCR results were extracted. However, this is not
    suitable for the interactive canvas in the notebook, which displays the
    elements at a smaller size than the original image.
    """
    for element in self.element_from_id.values():
      if element.class_name == 'word':
        element.bbox = BoundingBox(
            top=element.bbox.top,
            left=element.bbox.left,
            bottom=element.bbox.top + _BBOX_CHARACTER_HEIGHT,
            right=min(
                element.bbox.right,
                element.bbox.left + _estimate_text_width(element.text),
            ),
        )

    for element in self.element_from_id.values():
      if element.class_name == 'paragraph':
        self.recompute_bbox(element.id)


def load_page(base_dir: str, page_id: str) -> Page:
  """Reads a page from the pages directory."""
  with open(os.path.join(base_dir, page_id, 'page.json'), 'r') as f:
    json_page = json.load(f)
    element_from_id = {
        int(element['id']): Element.from_mapping(**element)
        for element in json_page['elements']
    }

  image_from_id = {}
  for image_path in glob.glob(os.path.join(base_dir, page_id, '*.png')):
    if 'background' in image_path:
      image_id = _BACKGROUND_IMAGE_ID
    else:
      image_id = int(os.path.basename(image_path).removesuffix('.png'))
    with open(image_path, 'rb') as f:
      image_from_id[image_id] = Image.open(io.BytesIO(f.read()))

  return Page(
      id=json_page['page_id'],
      element_from_id=element_from_id,
      image_from_id=image_from_id,
  )
