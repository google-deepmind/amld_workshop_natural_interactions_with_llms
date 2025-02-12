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

"""Interactive canvas for experimenting with gesture recognition."""

import base64
import copy
import io
from typing import Callable

import data_processing
import document_editing
from google.colab import output
from IPython import display
from PIL import Image
import rendering

_CANVAS_HTML = """
<style>
  #drawCanvas {{
    border: 1px solid lightgrey;
    cursor: crosshair;
  }}
  #modelOutput {{
    border: 1px solid lightgrey;
    margin-left: 10px;
  }}
  #container {{
    display: flex;
    flex-direction: row;
    align-items: flex-start;
  }}
</style>
<div id="container">
    <canvas id="drawCanvas"></canvas>
    <div id="modelOutput">
      <b>Controls:</b>
      <button id="resetButton">Reset document</button>
      <button id="clearButton">Clear ink</button>
      <button id="interpretButton">Interpret</button><br>
      <b>Status:</b> <span id="status"></span><br>
      <b>Model output:</b> <span id="modelOutputText"></span><br><img id="modelOutputImage" src="" width="300">
    </div>
</div>
<script>
    const canvas = document.getElementById('drawCanvas');
    let ctx = canvas.getContext('2d');
    let drawing = false;
    let image = new Image();
    let drawingCoordinates = [];

    function setUpContext(context) {{
      context.strokeStyle = '{color}';
      context.lineWidth = {line_width};
      context.lineCap = 'round';
      context.lineJoin = 'round';
    }}

    function drawBackgroundImage() {{
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    }}

    function setBackgroundImage(imageData) {{
      image.src = imageData;
      image.onload = () => {{
        // Redraw the background image and any existing drawing
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        canvas.height = image.height;
        canvas.width = image.width;
        setUpContext();

        drawBackgroundImage(image);
      }};
    }}

    function setModelOutput(text, imageData) {{
      const modelOutputText = document.getElementById('modelOutputText');
      modelOutputText.textContent = text;

      const modelOutputImage = document.getElementById('modelOutputImage');
      modelOutputImage.src = imageData;
    }}

    canvas.addEventListener('mousedown', (e) => {{
      let x = e.offsetX;
      let y = e.offsetY;
      drawingCoordinates.push([[x, y]]);
      drawing = true;
      ctx.beginPath();
      ctx.moveTo(x, y);
    }});

    canvas.addEventListener('mousemove', (e) => {{
      if (drawing) {{
        let x = e.offsetX;
        let y = e.offsetY;
        drawingCoordinates[drawingCoordinates.length - 1].push([x, y]);

        ctx.lineTo(x, y);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(x, y);
      }}
    }});

    function stopDrawing() {{
        drawing = false;
    }}

    function setStatus(text) {{
      const status = document.getElementById('status');
      status.textContent = text;
    }}

    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', stopDrawing);

    document.getElementById('clearButton').addEventListener('click', function() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawingCoordinates = [];
      drawBackgroundImage();
    }});

    document.getElementById('interpretButton').addEventListener('click', function() {{
      // Create a temporary canvas to combine background and drawing.
      const tempCanvas = document.createElement('canvas');
      const tempCtx = tempCanvas.getContext('2d');
      tempCanvas.width = canvas.width;
      tempCanvas.height = canvas.height;

      // Background image.
      tempCtx.drawImage(image, 0, 0);

      // White interlayer.
      tempCtx.globalAlpha = 0.5;
      tempCtx.fillStyle = 'white';
      tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
      tempCtx.globalAlpha = 1;

      // Gesture on top.
      for (const stroke of drawingCoordinates) {{
        tempCtx.beginPath();
        tempCtx.lineTo(stroke[0][0], stroke[0][1]);
        for (let i = 1; i < stroke.length; i++) {{
          const [x, y] = stroke[i];
          tempCtx.lineTo(x, y); // Draw a line to the current point
        }}
        tempCtx.stroke();
      }}

      const imageData = tempCanvas.toDataURL();
      google.colab.kernel.invokeFunction('notebook.InterpretDrawing', [imageData, drawingCoordinates], {{}});
      drawingCoordinates = [];
    }});

    document.getElementById('resetButton').addEventListener('click', function() {{
      drawingCoordinates = [];
      google.colab.kernel.invokeFunction('notebook.ResetDocument', [], {{}});
    }});

    setUpContext();
    google.colab.output.setIframeHeight(0, true, {{maxHeight: 5000}});
</script>
"""


def _display_javascript(javascript: str):
  """Displays the provided JavaScript code in the Colab output."""
  display.display(display.Javascript(javascript))


def _display_status(status: str):
  """Displays the provided status message in the Colab output."""
  _display_javascript(f"window.setStatus('{status}');")


class Canvas:
  """Interactive canvas for experimenting with gesture recognition."""

  def __init__(
      self,
      document: document_editing.Page,
      predict_fn: Callable[
          [document_editing.Ink, Image.Image],
          data_processing.DocumentEditingLabel,
      ],
      canvas_max_width: int = 600,
      canvas_max_height: int = 600,
  ):
    self._canvas_max_width = canvas_max_width
    self._canvas_max_height = canvas_max_height
    self._original_document = document
    self._document = copy.deepcopy(document)
    self._document_extent = document_editing.BoundingBox(
        top=0, left=0, bottom=0, right=0
    )
    self._predict_fn = predict_fn

  def _render_document_on_canvas(self):
    """Renders the current document and update the canvas background."""
    rendered_document = rendering.render_document(
        self._document, show_bboxes=False
    )
    rendered_document.image.thumbnail(
        (self._canvas_max_width, self._canvas_max_height)
    )
    self._document_extent = rendered_document.extent
    _display_javascript(
        "window.setBackgroundImage("
        f"'{rendering.to_data_url(rendered_document.image)}');"
    )

  def _interpret_gesture(self, image_data, coordinates):
    """Interprets the user's gesture and update the underlying document."""
    _display_status("Interpreting... ⏳")

    image_data = image_data.split(",")[1]
    decoded_data = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(decoded_data))

    # Interpret the gesture
    strokes = []
    for stroke in coordinates:
      xs = []
      ys = []
      for x, y in stroke:
        xs.append(x)
        ys.append(y)
      strokes.append(document_editing.Stroke(xs=xs, ys=ys))
    ink = document_editing.Ink(strokes=strokes)

    if not ink.strokes:
      _display_status("Empty ink, nothing to interpret.")
      return

    prediction = self._predict_fn(ink, image)

    # Convert the bounding box to document coordinates.
    scale = self._document_extent.width / image.width
    prediction.bbox = document_editing.BoundingBox(
        self._document_extent.top + prediction.bbox.top * scale,
        self._document_extent.left + prediction.bbox.left * scale,
        self._document_extent.top + prediction.bbox.bottom * scale,
        self._document_extent.left + prediction.bbox.right * scale,
    )

    self._document.edit(
        edit_name=prediction.gesture,
        edit_bbox=prediction.bbox,
        text=prediction.text,
    )
    self._render_document_on_canvas()
    _display_status("")

  def _reset_document(self):
    """Render a fresh copy of the original document and update the canvas."""
    _display_status("Resetting document... ⏳")
    self._document = copy.deepcopy(self._original_document)
    self._render_document_on_canvas()
    _display_status("")

  def display_interaction_widget(self):
    """Displays the interaction widget."""
    output.register_callback(
        "notebook.InterpretDrawing", self._interpret_gesture
    )
    output.register_callback("notebook.ResetDocument", self._reset_document)

    display.display(
        display.HTML(_CANVAS_HTML.format(color="#FF0000", line_width=3))
    )
    self._render_document_on_canvas()
