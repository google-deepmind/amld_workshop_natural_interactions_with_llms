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

"""Tools for arrows from MathWriting dataset.

Some code is based on
https://github.com/google-research/google-research/blob/master/mathwriting/mathwriting_code_examples.ipynb
"""

import copy
import dataclasses
import itertools
import json
import os
from typing import Literal
from xml.etree import ElementTree

import document_editing
import numpy as np
import paligemma_tools


def read_inkml_file(file_name: str) -> document_editing.Ink:
  """Simple reader for MathWriting's InkML files.

  Copied from:
  https://github.com/google-research/google-research/blob/master/mathwriting/mathwriting_code_examples.ipynb

  Args:
    file_name: The name of the InkML file to read.

  Returns:
    An Ink object representing the ink in the file.
  """
  if not os.path.exists(file_name):
    raise ValueError(f'File {file_name} does not exist.')

  with open(file_name, 'r') as f:
    root = ElementTree.fromstring(f.read())

  strokes = []

  for element in root:
    tag_name = element.tag.removeprefix('{http://www.w3.org/2003/InkML}')

    if tag_name == 'trace':
      if not element.text:
        continue
      points = element.text.split(',')
      stroke_x, stroke_y, stroke_t = [], [], []
      for point in points:
        x, y, t = point.split(' ')
        stroke_x.append(float(x))
        stroke_y.append(float(y))
        stroke_t.append(float(t))
      strokes.append(document_editing.Stroke(xs=stroke_x, ys=stroke_y))

  return document_editing.Ink(strokes=strokes)


@dataclasses.dataclass
class InkPart:
  r"""Holds information about an ink part, corresponding to a single symbol.

  Copied from:
  https://github.com/google-research/google-research/blob/master/mathwriting/mathwriting_code_examples.ipynb

  Attributes:
    source_sample_id: Which sample the symbol is from. Ex: '00016221aae38d32'.
    label: Which symbol it is. Ex: '\sum'.
    stroke_indices: Indices of strokes in the source ink that cover the symbol.
  """

  source_sample_id: str
  label: str
  stroke_indices: list[int]


def read_symbols_file(file_name: str) -> list[InkPart]:
  """Returns the InkPart objects corresponding to the symbols in the file.

  Copied from:
  https://github.com/google-research/google-research/blob/master/mathwriting/mathwriting_code_examples.ipynb

  Args:
    file_name: The name of the symbols file to read.
  """
  symbols = []
  if not os.path.exists(file_name):
    raise ValueError(f'File {file_name} does not exist.')

  with open(file_name, 'r') as f:
    for line in f:
      symbol_json = json.loads(line)
      symbols.append(
          InkPart(
              source_sample_id=symbol_json['sourceSampleId'],
              label=symbol_json['label'],
              stroke_indices=symbol_json['strokeIndices'],
          )
      )
  return symbols


def find_rotation_angle(
    p1: document_editing.Point, p2: document_editing.Point
) -> float:
  """Calculates the angle of rotation between two points, representing vectors.

  This function determines the angle required to rotate vector p1 to align with
  vector p2, considering the rotation to be around the origin (0,0).

  Args:
    p1: The starting point (representing the initial vector).
    p2: The ending point (representing the target vector).

  Returns:
    The angle of rotation (in radians) required to rotate p1 to p2. The angle
    is always positive and represents a counter-clockwise rotation.
  """
  ang1 = np.arctan2(p1.y, p1.x)
  ang2 = np.arctan2(p2.y, p2.x)
  a = (p2.x**2 + p2.y**2) ** 0.5
  b = (p1.x**2 + p1.y**2) ** 0.5
  c = (a**2 + b**2 - 2 * (a * b * np.cos(ang1 - ang2))) ** 0.5
  return np.arccos((b**2 + c**2 - a**2) / (2 * b * c + 1e-8)) + ang1


def rotate_stroke_points(
    p: list[tuple[float, float]],
    origin: tuple[float, float] = (0, 0),
    degrees: float = 0,
) -> np.ndarray:
  """Rotates the given points around the origin by the given degrees."""
  angle = np.deg2rad(degrees)
  r = np.array(
      [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
  )
  o = np.atleast_2d(origin)
  p = np.atleast_2d(p)
  return np.squeeze((r @ (p.T - o.T) + o.T).T)


def _get_critical_stroke_point(
    ink: document_editing.Ink, side: Literal['left', 'right']
) -> tuple[int, int]:
  """Returns the critical stroke point of an arrow ink."""
  cumulative_lengths = np.cumsum(ink.get_stroke_lengths())
  xs_flat = list(itertools.chain.from_iterable([s.xs for s in ink.strokes]))
  critical_point_index = int(
      np.argmin(xs_flat) if side == 'left' else np.argmax(xs_flat)
  )
  critical_point_stroke_index = np.searchsorted(
      cumulative_lengths, critical_point_index, side='right'
  )

  if critical_point_stroke_index > 0:
    critical_point_local_index = (
        critical_point_index
        - cumulative_lengths[critical_point_stroke_index - 1]
    )
  else:
    critical_point_local_index = critical_point_index
  return (critical_point_stroke_index, critical_point_local_index)


class ArrowInk:
  """Holds information about an ink representing a horizontal arrow.

  Attributes:
    ink: The underlying ink object.
    left_critical_point: The left head of the arrow.
    right_critical_point: The right head of the arrow.
    left_critical_point_id: The id of the left head of the arrow.
    right_critical_point_id: The id of the right head of the arrow.
  """

  ink: document_editing.Ink

  @property
  def left_critical_point(self) -> document_editing.Point:
    stroke_id, point_id = self.left_critical_point_id
    return document_editing.Point(
        x=self.ink.strokes[stroke_id].xs[point_id],
        y=self.ink.strokes[stroke_id].ys[point_id],
    )

  @property
  def right_critical_point(self) -> document_editing.Point:
    stroke_id, point_id = self.right_critical_point_id
    return document_editing.Point(
        x=self.ink.strokes[stroke_id].xs[point_id],
        y=self.ink.strokes[stroke_id].ys[point_id],
    )

  def __init__(self, ink: document_editing.Ink):
    self.ink = ink
    self.left_critical_point_id = _get_critical_stroke_point(ink, 'left')
    self.right_critical_point_id = _get_critical_stroke_point(ink, 'right')


class ArrowPageFitter:
  """A class to draw an arrow between two words on a page.

  We move the left head of the arrow
  to the center of the first word, rotate it to the correct orientation and
  resize it to fit right head of the arrow.

  This code is designed for horizontal arrows only.

  Attributes:
    page: The page on which the arrow is drawn.
    word_id1: The id of the first word.
    word_id2: The id of the second word.
  """

  page: document_editing.Page
  word_id1: int
  word_id2: int

  def __init__(self, page: document_editing.Page, word_id1: int, word_id2: int):
    self.page = page
    self.word_id1 = word_id1
    self.word_id2 = word_id2

  @property
  def center1(self) -> document_editing.Point:
    """Returns the center of the first word."""
    return self.page.element_from_id[self.word_id1].bbox.center

  @property
  def center2(self) -> document_editing.Point:
    """Returns the center of the second word."""
    return self.page.element_from_id[self.word_id2].bbox.center

  @property
  def word_id1_bbox(self) -> document_editing.BoundingBox:
    """Returns the bounding box of the first word."""
    return self.page.element_from_id[self.word_id1].bbox

  @property
  def word_id2_bbox(self) -> document_editing.BoundingBox:
    """Returns the bounding box of the second word."""
    return self.page.element_from_id[self.word_id2].bbox

  def move_coord_values_to_center(
      self,
      arrow: ArrowInk,
      center: document_editing.Point,
      coord_values: document_editing.Point,
      resize: tuple[float, float] = (1, 1),
  ) -> ArrowInk:
    """Moves the arrow ink to the center of the given coordinates."""
    arrow_moved = copy.deepcopy(arrow)
    for s in arrow_moved.ink.strokes:
      for i in range(len(s.xs)):
        s.xs[i] = s.xs[i] * resize[0] + (center.x - coord_values.x * resize[0])
        s.ys[i] = s.ys[i] * resize[1] + (center.y - coord_values.y * resize[1])
    return arrow_moved

  def rotate_ink(
      self,
      arrow: ArrowInk,
      center1: document_editing.Point,
      center2: document_editing.Point,
      verbose: bool = False,
  ) -> ArrowInk:
    """Rotates the arrow ink from center1 to center2."""
    angle = find_rotation_angle(center1, center2)
    if verbose:
      print('angle', np.rad2deg(angle))
    new_strokes = []
    for stroke in arrow.ink.strokes:
      rotated_stroke = rotate_stroke_points(
          list(zip(stroke.xs, stroke.ys)),
          origin=(center1.x, center1.y),
          degrees=np.rad2deg(angle) % 180,
      )
      new_strokes.append(
          document_editing.Stroke(
              xs=rotated_stroke[:, 0], ys=rotated_stroke[:, 1]
          )
      )
    arrow_ink = ArrowInk(ink=document_editing.Ink(strokes=new_strokes))
    arrow_ink.left_critical_point_id = arrow.left_critical_point_id
    arrow_ink.right_critical_point_id = arrow.right_critical_point_id
    return arrow_ink

  def resize_ink(
      self,
      arrow: ArrowInk,
      center1: document_editing.Point,
      center2: document_editing.Point,
      verbose: bool = False,
  ) -> ArrowInk:
    """Resizes the arrow ink in order to fit center2."""
    share_y = abs(center2.y - center1.y) / (
        abs(arrow.right_critical_point.y - arrow.left_critical_point.y) + 1e-6
    )
    share_x = abs(center2.x - center1.x) / (
        abs(arrow.right_critical_point.x - arrow.left_critical_point.x) + 1e-6
    )

    if verbose:
      print('share_x', share_x)
      print('share_y', share_y)

    return self.move_coord_values_to_center(
        arrow, center2, arrow.right_critical_point, resize=(share_x, share_y)
    )

  def fit_to_page(self, arrow: ArrowInk, verbose: bool = False) -> ArrowInk:
    """Fits the arrow ink to two words on the page."""
    moved_arrow = self.move_coord_values_to_center(
        arrow, self.center1, arrow.left_critical_point
    )
    rotated_arrow = self.rotate_ink(
        moved_arrow, self.center1, self.center2, verbose=verbose
    )
    resized_arrow = self.resize_ink(
        rotated_arrow, self.center1, self.center2, verbose=verbose
    )

    return resized_arrow


def get_arrow_target(
    arrow_fitter: ArrowPageFitter,
    area: document_editing.BoundingBox,
) -> str:
  """Returns an arrow target corresponding to the given arrow fitter."""
  x1 = paligemma_tools.to_location_coordinate(
      arrow_fitter.center1.x, area.left, area.right
  )
  x2 = paligemma_tools.to_location_coordinate(
      arrow_fitter.center2.x, area.left, area.right
  )
  y1 = paligemma_tools.to_location_coordinate(
      arrow_fitter.center1.y, area.top, area.bottom
  )
  y2 = paligemma_tools.to_location_coordinate(
      arrow_fitter.center2.y, area.top, area.bottom
  )
  coordinates = ''.join([f'<loc{i:04d}>' for i in [y1, x1, y2, x2]])
  return f'point {coordinates}'


def check_that_arrow_is_located_correctly(
    arrow_fitter: ArrowPageFitter, arrow: ArrowInk
) -> bool:
  """Checks that the arrow's critical points are in the correct location."""
  if not (
      arrow_fitter.word_id1_bbox.left
      <= arrow.left_critical_point.x
      <= arrow_fitter.word_id1_bbox.right
  ):
    return False
  if not (
      arrow_fitter.word_id1_bbox.top
      <= arrow.left_critical_point.y
      <= arrow_fitter.word_id1_bbox.bottom
  ):
    return False
  if not (
      arrow_fitter.word_id2_bbox.left
      <= arrow.right_critical_point.x
      <= arrow_fitter.word_id2_bbox.right
  ):
    return False
  if not (
      arrow_fitter.word_id2_bbox.top
      <= arrow.right_critical_point.y
      <= arrow_fitter.word_id2_bbox.bottom
  ):
    return False
  return True
