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

"""Utilities for working with document editing data."""

import dataclasses
import re
from typing import Any, Optional

import document_editing

Example = dict[str, Any]

"""The pattern/format to match/generate gestures defined using standard bounding
box coordinates.

Example:
  select 10 20 30 40 text
"""
BBOX_PATTERN = re.compile(
    r"(?P<gesture>.*) (?P<top>\d{1,4}) (?P<left>\d{1,4}) (?P<bottom>\d{1,4})"
    r" (?P<right>\d{1,4})( (?P<text>.*))?"
)
BBOX_FORMAT = "{gesture} {top:.0f} {left:.0f} {bottom:.0f} {right:.0f} {text}"

"""The pattern/format to match/generate gestures defined using location tokens.

Example:
  select <loc0010><loc020><loc0030><040> text
"""
LOC_PATTERN = re.compile(
    r"(?P<gesture>.*)"
    r" <loc(?P<top>\d{4})><loc(?P<left>\d{4})><loc(?P<bottom>\d{4})><loc(?P<right>\d{4})>("
    r" (?P<text>.*))?"
)

LOC_FORMAT = (
    "{gesture}"
    " <loc{top:04.0f}><loc{left:04.0f}><loc{bottom:04.0f}><loc{right:04.0f}>"
    " {text}"
)


@dataclasses.dataclass
class DocumentEditingLabel:
  """Represents a document editing label.

  Attributes:
    gesture: The gesture of the label.
    bbox: The bounding box of the label.
    text: The text of the label.
  """

  gesture: str
  bbox: document_editing.BoundingBox
  text: str = ""

  @classmethod
  def from_string(
      cls, label: str, pattern: re.Pattern[str] = LOC_PATTERN
  ) -> Optional["DocumentEditingLabel"]:
    """Parse the document editing label using the provided pattern.

    Args:
      label: The label to parse.
      pattern: The pattern to match the label against. By default, use the
        PaLI-Gemma location token pattern.

    Returns:
      The parsed label, or None if it could not be parsed.
    """
    try:
      if regex_match := pattern.match(label):
        groups = regex_match.groupdict()
        gesture = groups.pop("gesture")
        text = groups.pop("text", None)
        if text is None:
          text = ""
        bbox = groups
        bbox = document_editing.BoundingBox(
            **{k: float(v) for k, v in bbox.items()}
        )
        return cls(gesture=gesture, bbox=bbox, text=text)
    except ValueError:
      pass

  def to_string(self, format_str: str) -> str:
    """Returns the document editing label as a string.

    Args:
      format_str: The format string to use. The string must contain the
        following placeholders: gesture, top, left, bottom, right, and text.

    Returns:
      The document editing label as a string.
    """
    return format_str.format(**{
        "gesture": self.gesture,
        "top": self.bbox.top,
        "left": self.bbox.left,
        "bottom": self.bbox.bottom,
        "right": self.bbox.right,
        "text": self.text,
    })

  @classmethod
  def from_output(
      cls,
      label: str,
      *,
      loc_tokens: bool = False,
      normalize: float | None = None,
  ) -> Optional["DocumentEditingLabel"]:
    """Parses output of a model into a DocumentEditingLabel.

    Args:
      label: The label to parse.
      loc_tokens: Whether the output uses location tokens.
      normalize: If provided, the bounding box is normalized by this value.

    Returns:
      A tuple of the class, bounding box, and text.
    """
    pattern = LOC_PATTERN if loc_tokens else BBOX_PATTERN
    document_editing_label = cls.from_string(label, pattern)
    if document_editing_label is not None:
      if normalize is not None:
        document_editing_label.bbox = document_editing_label.bbox.normalize(
            normalize
        )
    return document_editing_label

  @classmethod
  def transform(
      cls,
      label: str,
      pattern_from: re.Pattern[str],
      format_to: str,
      normalize: float | None = None,
  ) -> str | None:
    """Transforms the document editing label into a different format.

    Args:
      label: The label to transform.
      pattern_from: The pattern to match the output against.
      format_to: The format to use for the output.
      normalize: If provided, the bounding box is normalized by this value.

    Returns:
      The transformed label, or None if the label could not be parsed.
    """
    document_editing_label = cls.from_string(label, pattern_from)
    if document_editing_label is None:
      return None
    if normalize is not None:
      document_editing_label.bbox = document_editing_label.bbox.normalize(
          normalize
      )
    return document_editing_label.to_string(format_to)


def transform_example_label(
    example: Example,
    pattern_from: re.Pattern[str],
    format_to: str,
    normalize: float | None = None,
) -> Example:
  """Transforms a document editing label from one format to another.

  Args:
    example: The example to transform.
    pattern_from: The pattern to parse the example's label against.
    format_to: The format to use for the output.
    normalize: If provided, the bounding box is normalized by this value.

  Returns:
    The transformed example.
  """
  new_example = {**example}
  label = new_example.pop("label").decode()
  new_label = DocumentEditingLabel.transform(
      label, pattern_from, format_to, normalize
  )
  if new_label is None:
    new_label = ""
  new_example["label"] = new_label.encode()
  return new_example
