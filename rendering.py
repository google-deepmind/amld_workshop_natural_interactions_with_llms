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

"""Utilities for rendering pages."""

import base64
import dataclasses
import functools
import io
from typing import Mapping, Sequence

import document_editing
import matplotlib
import matplotlib.pyplot as plt
import paligemma_tools
from PIL import Image

_FIGURE_MAX_HEIGHT_INCHES = 16
_FIGURE_MAX_WIDTH_INCHES = 16
_FIGURE_DPI = 80


@dataclasses.dataclass
class DocumentRender:
  """Represents the rendering of a document.

  Attributes:
    image: The rendering of the document itself.
    area_of_interest: The area of interest in the rendering. Typically contains
      the ink overlay, the elements modified in previous edits and any
      additional overlaid bounding box.
    extent: The vertical and horizontal extent covered by all the page elements
      shown in the rendering, in the form of a bounding box.
  """

  image: Image.Image
  area_of_interest: document_editing.BoundingBox
  extent: document_editing.BoundingBox


def _get_visually_relevant_square_bbox(
    bboxes_to_show: Sequence[document_editing.BoundingBox], margin: int = 100
):
  """Computes a visually relevant square area from a set of bounding boxes."""
  visible_area = functools.reduce(
      lambda bbox1, bbox2: bbox1.union(bbox2), bboxes_to_show
  )
  largest_side = max(visible_area.width, visible_area.height)
  width = visible_area.width
  height = visible_area.height

  return document_editing.BoundingBox(
      top=visible_area.top - margin + (largest_side - width) / 2,
      left=visible_area.left - margin + (largest_side - width) / 2,
      bottom=visible_area.bottom + margin + (largest_side - height) / 2,
      right=visible_area.right + margin + (largest_side - height) / 2,
  )


def _plot_one_element(
    axes: matplotlib.axes.Axes,
    element: document_editing.Element,
    image_from_id: Mapping[int, Image.Image],
):
  """Plots a single element on the given axes.

  Args:
      axes: The matplotlib axes to plot on.
      element: The element to plot.
      image_from_id: A mapping from page element ID to images.
  """
  rect_kwargs = {"fill": False, "zorder": 1, "linewidth": 1, "alpha": 0.3}
  if element.class_name == "paragraph":
    rect_kwargs.update({"color": "orange", "zorder": 3, "alpha": 1})
  elif element.class_name == "textline":
    rect_kwargs.update({"color": "dodgerblue", "zorder": 2, "alpha": 1})
  elif element.class_name == "word":
    if element.text:
      axes.text(
          element.bbox.left,
          element.bbox.top,
          s=element.text,
          fontsize=10,
          va="top",
          font="sans",
      )
    if element.color:
      rect_kwargs.update({"color": element.color, "fill": True})
    else:
      rect_kwargs.update({"color": "dodgerblue"})
  elif element.class_name == "image":
    image = image_from_id[element.id]
    axes.imshow(
        image,
        extent=(
            element.bbox.left,
            element.bbox.right,
            element.bbox.bottom,
            element.bbox.top,
        ),
        zorder=4,
    )
    rect_kwargs.update({"color": "red"})

  rect = matplotlib.patches.Rectangle(
      (element.bbox.left, element.bbox.top),
      element.bbox.width,
      element.bbox.height,
      **rect_kwargs,
  )
  axes.add_patch(rect)


def _plot_one_bbox(
    axes: plt.Axes, bbox: document_editing.BoundingBox, color: str
):
  """Plots a single bounding box on the given axes."""
  rect = matplotlib.patches.Rectangle(
      (bbox.left, bbox.top),
      bbox.width,
      bbox.height,
      zorder=1,
      linewidth=2,
      alpha=1,
      color=color,
      fill=False,
  )
  axes.add_patch(rect)


def _plot_one_ink(axes: plt.Axes, ink: document_editing.Ink):
  """Plots the ink data on the given axes."""
  for stroke in ink.strokes:
    axes.plot(stroke.xs, stroke.ys, color="red", linewidth=2, zorder=100)


def _figure_to_image(
    fig: plt.Figure,
    ax: plt.Axes,
    extent: document_editing.BoundingBox,
    crop_area: document_editing.BoundingBox | None = None,
) -> Image.Image:
  """Renders the document figure to an image."""
  ratio = extent.width / extent.height
  fig_width = min(_FIGURE_MAX_WIDTH_INCHES, _FIGURE_MAX_HEIGHT_INCHES * ratio)
  fig_height = fig_width / ratio

  fig.set_size_inches(fig_width, fig_height)
  fig.set_dpi(_FIGURE_DPI)
  ax.set_xlim(extent.left, extent.right)
  ax.set_ylim(extent.bottom, extent.top)

  fig.canvas.draw()
  image_bytes = fig.canvas.buffer_rgba()
  canvas_width, canvas_height = fig.canvas.get_width_height()
  image = Image.frombuffer(
      "RGBA", (canvas_width, canvas_height), image_bytes, "raw", "RGBA", 0, 1
  )

  if crop_area:
    scale_x = canvas_width / extent.width
    scale_y = canvas_height / extent.height
    image_crop = (
        (crop_area.left - extent.left) * scale_x,
        (crop_area.top - extent.top) * scale_y,
        (crop_area.right - extent.left) * scale_x,
        (crop_area.bottom - extent.top) * scale_y,
    )
    image = image.crop(image_crop)

  return image


def render_document(
    page: document_editing.Page,
    overlay_bboxes: Mapping[str, document_editing.BoundingBox] | None = None,
    ink: document_editing.Ink | None = None,
    crop_area: document_editing.BoundingBox | bool = False,
) -> DocumentRender:
  """Returns a rendering of a page with its elements, images, and ink gesture.

  Args:
      page: Page to visualize.
      overlay_bboxes: Colors and bounding boxes to draw for visualization
        purposes. Color can be set to the empty string to skip drawing but still
        ensure the area covered by the bounding box will be visible in the final
        image. This is useful to ensure that the area around a gesture is
        visible or to show the image that was fed as input to the model.
      ink: Optional ink data to overlay on the plot.
      crop_area: If set to True, crops the final image around a relevant area
        consisting of the ink, any element with a color attribute and any
        provided overlay bounding box. If set to a BoundingBox, crops the final
        image around this bounding box.
  """
  plt.ioff()
  fig = plt.figure(frameon=False)

  axes = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
  axes.set_axis_off()
  fig.add_axes(axes)

  # Determine the visible area of the document. Any element that is not in that
  # area will be skipped to speed up the rendering.
  visible_area = None
  if isinstance(crop_area, document_editing.BoundingBox):
    visible_area = crop_area
  elif crop_area:
    boxes_to_show = []
    for element in page.element_from_id.values():
      if element.color:
        boxes_to_show.append(element.bbox)
    if overlay_bboxes:
      boxes_to_show.extend(overlay_bboxes.values())
    if ink:
      boxes_to_show.append(ink.get_bbox())
    if boxes_to_show:
      visible_area = _get_visually_relevant_square_bbox(boxes_to_show)

  # Keeps track of the overall extent of the canvas, as elements get added.
  extent = next(iter(page.element_from_id.values())).bbox
  if ink:
    extent = extent.union(ink.get_bbox())
    _plot_one_ink(axes, ink)

  for element in page.element_from_id.values():
    extent = extent.union(element.bbox)
    if crop_area and element.bbox.intersection(visible_area).area < 0.001:
      continue
    _plot_one_element(axes, element, page.image_from_id)

  if overlay_bboxes:
    for color, bbox in overlay_bboxes.items():
      extent = extent.union(bbox)
      if color:
        _plot_one_bbox(axes, bbox, color)

  image = _figure_to_image(fig, axes, extent, visible_area)
  plt.close(fig)
  return DocumentRender(
      image=image, area_of_interest=visible_area, extent=extent
  )


def to_data_url(image: Image.Image) -> str:
  """Returns the data URL value for the provided image."""
  buffer = io.BytesIO()
  image.save(buffer, format="PNG")
  base64_image = base64.b64encode(buffer.getvalue()).decode()
  return f"data:image/png;base64,{base64_image}"


def to_html_image(image: Image.Image, width: int = 400) -> str:
  """Returns an HTML element for the provided image."""
  return f'<img src="{to_data_url(image)}" width="{width}">'


def bbox_to_image_space(
    bbox: document_editing.BoundingBox,
    composition_bbox: document_editing.BoundingBox,
) -> document_editing.BoundingBox:
  """Shifts and scales a bounding box from composition space to image space.

  The composition space is defined as a 1024x1024 unit rectangle, with the
  origin at the top left corner. These are the coordinates used by the model
  when it makes a prediction. This rectangle covers only a portion of the
  original document image.

  The image space on the other hand is defined as the original image coordinates
  in number of pixels, also with the origin at the top left corner. For example:

        Document image
        +----------------------------------------------------+
        |                                                    |
        |                                                    |
        |                                                    |
        |                                                    |
        |                                                    |
        |                                                    |
        |                                                    |
        |           Composition (1024x1024 unit rectangle)   |
        |           +-------------------+                    |
        |           |                   |                    |
        |           |                   |                    |
        |           |                   |                    |
        |           | Bbox (prediction) |                    |
        |           | [  ]              |                    |
        |           |                   |                    |
        |           +-------------------+                    |
        |                                                    |
        |                                                    |
        |                                                    |
        |                                                    |
        |                                                    |
        +----------------------------------------------------+

  Args:
    bbox: The bounding box in composition space.
    composition_bbox: The bounding box of the composition, relative to the
      document image.

  Returns:
    The bounding box in image space.
  """

  # Convert prediction composition bbox into image bbox
  converted_bbox = document_editing.BoundingBox(
      left=paligemma_tools.from_location_coordinate(
          bbox.left, composition_bbox.left, composition_bbox.right
      ),
      top=paligemma_tools.from_location_coordinate(
          bbox.top, composition_bbox.top, composition_bbox.bottom
      ),
      right=paligemma_tools.from_location_coordinate(
          bbox.right, composition_bbox.left, composition_bbox.right
      ),
      bottom=paligemma_tools.from_location_coordinate(
          bbox.bottom, composition_bbox.top, composition_bbox.bottom
      ),
  )

  return converted_bbox


def ink_to_image_space(
    ink: document_editing.Ink,
    writing_guide: document_editing.BoundingBox,
    document_image: Image.Image,
) -> document_editing.Ink:
  """Returns the ink with coordinates converted to image space.

  When composing the ink gesture and the document image, the image scaled and
  shifted to fit within the writing guide while preserving the aspect ratio of
  the document.

  Args:
    ink: The ink in composition space.
    writing_guide: The bounding box of the writing guide in image space.
    document_image: The image object.
  """
  # The image is rescaled so that it fits in the writing guide while
  # preserving the aspect ratio. Dimension is < 1 if the image size had to be
  # increased to fit the screen.
  scale = max(
      document_image.width / writing_guide.width,
      document_image.height / writing_guide.height,
  )

  # The image is centered within the writing guide.
  scaled_guide_width = writing_guide.width * scale
  scaled_guide_height = writing_guide.height * scale
  offset_x = abs(document_image.width - scaled_guide_width) / 2
  offset_y = abs(document_image.height - scaled_guide_height) / 2

  # The composition in image coordinates
  scaled_strokes = []
  for stroke in ink.strokes:
    xs = []
    ys = []
    for x, y in zip(stroke.xs, stroke.ys):
      xs.append(x * scale - offset_x)
      ys.append(y * scale - offset_y)
    scaled_strokes.append(document_editing.Stroke(xs=xs, ys=ys))

  return document_editing.Ink(strokes=scaled_strokes)
