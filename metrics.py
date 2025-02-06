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

"""Utilities for computing metrics for document editing tasks."""

import collections
from typing import Any

import data_processing
import document_editing
import jiwer
from matplotlib import pyplot as plt
import numpy as np

Example = dict[str, Any]


def cer(string_1: str, string_2: str) -> float:
  """Computes the character error rate between two strings."""
  if not string_1:
    return 0
  if not string_2:
    string_2 = ''
  result: float = jiwer.cer(string_1, string_2)  # pytype: disable=annotation-type-mismatch
  return result


def accuracy(string_1: str, string_2: str) -> bool:
  """Computes the accuracy between two strings."""
  return string_1 == string_2


def intersection_over_union(
    bbox_1: document_editing.BoundingBox, bbox_2: document_editing.BoundingBox
) -> float:
  """Computes the intersection over union between two bounding boxes."""
  if bbox_1 is None or bbox_2 is None:
    return 0
  inter = bbox_1.intersection(bbox_2)
  return inter.area / (bbox_1.area + bbox_2.area - inter.area)


def compute_document_editing_metrics(
    targets: list[data_processing.DocumentEditingLabel | None],
    predictions: list[data_processing.DocumentEditingLabel | None],
) -> tuple[list[bool], list[float], list[float]]:
  """Computes the accuracies, ious, and cers for a list of predictions and targets.

  Args:
    targets: The list of targets.
    predictions: The list of predictions.

  Returns:
    A tuple of the accuracies, ious, and cers.
  """
  accuracies = []
  ious = []
  cers = []

  for prediction, target in zip(predictions, targets):
    if target is None:
      continue
    if prediction is None:
      accuracies.append(False)
      ious.append(0)
      continue
    accuracies.append(accuracy(target.gesture, prediction.gesture))
    ious.append(intersection_over_union(target.bbox, prediction.bbox))
    if target.text is not None:
      cers.append(cer(target.text, prediction.text))

  return accuracies, ious, cers


def confusion_matrix(
    predictions: list[data_processing.DocumentEditingLabel | None],
    targets: list[data_processing.DocumentEditingLabel | None],
    *,
    none_key: str = '<none>',
) -> dict[str, dict[str, int]]:
  """Computes the confusion matrix for a list of predictions and targets.

  Args:
    predictions: The list of predictions.
    targets: The list of targets.
    none_key: The key to use for the 'none' class (corresponding to an invalid
      label).

  Returns:
    The confusion matrix.
  """
  cm = collections.defaultdict(lambda: collections.defaultdict(int))
  for prediction, target in zip(predictions, targets):
    if target is None:
      if prediction is None:
        continue
      cm[none_key][prediction.gesture] += 1
    elif prediction is None:
      cm[target.gesture][none_key] += 1
    else:
      cm[target.gesture][prediction.gesture] += 1
  return cm


def plot_confusion_matrix(cm: dict[str, dict[str, int]], **kwargs: Any) -> None:
  """Plots the given confusion matrix.

  Args:
    cm: The confusion matrix.
    **kwargs: Keyword arguments to pass to matplotlib.pyplot.imshow.
  """
  figure, ax = plt.subplots()
  classes = sorted(cm.keys())
  cm_array = np.array([[cm[c1][c2] for c2 in classes] for c1 in classes])
  art = ax.imshow(cm_array, **kwargs)
  ax.set_xticks(range(len(classes)), classes, rotation=90)
  ax.set_yticks(range(len(classes)), classes)
  ax.set_xlabel('Predicted')
  ax.set_ylabel('Actual')
  figure.colorbar(art)


def plot_fewshot_results(
    results: dict[int, dict[str, Any]],
    ax: plt.Axes | None = None,
):
  """Plots the few-shot results."""
  if ax is None:
    _, ax = plt.subplots()
  shots = sorted(results.keys())
  metrics = sorted(results[shots[0]].keys())
  for metric in metrics:
    ax.plot(
        shots,
        [results[shot][metric] for shot in shots],
        label=metric,
        marker='o',
        markersize=10,
    )
  ax.set_xlabel('Number of shots')
  ax.set_ylabel('Metric')
  ax.legend()
