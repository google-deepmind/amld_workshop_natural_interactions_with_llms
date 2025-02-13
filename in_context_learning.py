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

"""Utilities for few-shot prompting."""

import abc
from collections.abc import Callable
import io
import time
from typing import Any, Iterator

import data_processing
from IPython import display
import numpy as np
from PIL import Image
import tensorflow as tf
import tqdm

Example = dict[str, Any]
PromptChunk = str | Image.Image


class FewShotPrompter(abc.ABC):
  """A class for generating few-shot prompts."""

  __slots__ = ["_prefix", "_shot_prefix", "_shots"]

  def __init__(
      self, prefix: str, shot_prefix: str, shots: list[Example]
  ) -> None:
    """Initializes the few-shot prompter.

    Args:
      prefix: The prefix to add to the prompt.
      shot_prefix: The prefix to add to the shot examples.
      shots: The shot examples to add to the prompt.
    """
    self._prefix = prefix
    self._shot_prefix = shot_prefix
    self._shots = [self.prepare_example(shot, True) for shot in shots]

  def load_image(self, array: np.ndarray) -> Image.Image:
    """Loads an image from a numpy array."""

    image = Image.fromarray(array)
    f = io.BytesIO()
    image.save(f, format="PNG")
    image_png = Image.open(f)
    return image_png

  @abc.abstractmethod
  def prepare_example(
      self, example: Example, is_shot: bool
  ) -> list[PromptChunk]:
    """Returns prepared examples for few-shot prompting.

    Args:
      example: The example to prepare.
      is_shot: Whether the example is a shot example.
    """
    raise NotImplementedError()

  def __call__(self, example: Example) -> list[PromptChunk]:
    inputs = []
    inputs.append(self._prefix)
    if self._shots:
      inputs.append(self._shot_prefix)
      for shot in self._shots:
        inputs.extend(shot)
    inputs.extend(self.prepare_example(example, False))
    return inputs

  def iterate(
      self, dataset: tf.data.Dataset
  ) -> Iterator[tuple[Example, list[PromptChunk]]]:
    for example in dataset.as_numpy_iterator():
      yield example, self(example)

  def display_prompt(self, example: Example) -> None:
    """Displays the prompt for a given example in IPython.

    Args:
      example: The example to display the prompt for.
    """
    prompt = self(example)
    display_prompt(prompt)


def infer_fewshot(
    inference_fn: Callable[[list[PromptChunk]], str],
    prompter: FewShotPrompter,
    dataset: tf.data.Dataset,
    *,
    progress: bool = True,
    normalize_fn: (
        Callable[[dict[str, Any]], float | tuple[float, float]] | None
    ) = None,
    rpm: float = 15,
) -> tuple[
    list[data_processing.DocumentEditingLabel | None],
    list[data_processing.DocumentEditingLabel | None],
]:
  """Runs `inference_fn` on a dataset with a few-shot prompter.

  Args:
    inference_fn: The function to use for inference.
    prompter: The prompter to use for generating the prompts.
    dataset: The dataset to run inference on.
    progress: Whether to show progress bars.
    normalize_fn: A function that returns a normalization factor for locations
      in a given example. Either a single float for both x and y or a tuple of
      two floats for x and y.
    rpm: The number of requests per minute to limit the inference rate to.

  Returns:
    The predictions and targets.
  """
  predictions = []
  targets = []

  for example, prompt in tqdm.tqdm(
      prompter.iterate(dataset), total=len(dataset), disable=not progress
  ):
    try:
      start_time = time.time()
      output = inference_fn(prompt)
      target = example["label"].decode()
      predicted_label = data_processing.DocumentEditingLabel.from_output(
          output, loc_tokens=False
      )
      expected_label = data_processing.DocumentEditingLabel.from_output(
          target,
          loc_tokens=False,
          normalize=None if normalize_fn is None else normalize_fn(example),
      )
      predictions.append(predicted_label)
      targets.append(expected_label)
      end_time = time.time()
      sleep_time = (60 / rpm) - (end_time - start_time)
      if sleep_time > 0:
        time.sleep(sleep_time)
    except KeyboardInterrupt:
      break
  return targets, predictions


def display_prompt(prompt: list[PromptChunk]) -> None:
  """Displays a given prompt in IPython.

  Args:
    prompt: The prompt to display.
  """
  for chunk in prompt:
    if isinstance(chunk, str):
      print(chunk)
      pass
    elif isinstance(chunk, Image.Image):
      display.display(chunk)
    else:
      raise ValueError(f"Unknown chunk type: {type(chunk)}")
