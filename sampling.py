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

"""Sampling utilities for few-shot prompting."""

import functools
import random
from typing import Any, Callable, Iterable, Iterator

import tensorflow as tf

Example = dict[str, Any]


def _get_strata_from_example(
    example: Example,
    strata_value: Any,
    example_to_stratum: Callable[[Example], Any],
) -> bool:
  """Returns whether the example's stratum mapping matches the given stratum.

  Args:
    example: The example to check.
    strata_value: The stratum value to check against.
    example_to_stratum: The function to use to map an example to a stratum.
  """

  example_strata_value = example_to_stratum(example)
  return tf.math.equal(example_strata_value, strata_value)


class StratifiedSampler:
  """Samples examples from a dataset in a stratified manner."""

  __slots__ = ("_mapping_fn", "_dataset", "_values", "_datasets")

  def __init__(
      self,
      dataset: tf.data.Dataset,
      example_to_stratum: Callable[[Example], Any],
      *,
      stratum_values: Iterable[Any] | None = None,
  ):
    """Initializes a stratified sampler.

    Args:
      dataset: The dataset to sample from.
      example_to_stratum: A function that maps an example to a stratum.
      stratum_values: The values to stratify by. If not provided, all unique
        values obtained using the example_to_stratum function on the entire
        dataset are used.
    """
    if stratum_values is None:
      stratum_values = {
          example_to_stratum(example).numpy().decode() for example in dataset
      }

    self._mapping_fn = example_to_stratum
    self._dataset = dataset
    self._values = stratum_values
    self._datasets = {
        value: dataset.filter(
            functools.partial(
                _get_strata_from_example,
                strata_value=value,
                example_to_stratum=example_to_stratum,
            )
        )
        for value in stratum_values
    }

  def sample(
      self,
      *,
      num_examples: int | None = None,
      num_examples_by_class: int | None = None,
  ) -> Iterator[Example]:
    """Yields examples from the dataset in a stratified manner.

    Args:
      num_examples: The number of examples to sample (mutually exclusive with
        num_examples_by_class).
      num_examples_by_class: The number of examples to sample for each class
        (mutually exclusive with num_examples).
    """
    if num_examples is None and num_examples_by_class is None:
      raise ValueError("Either n or n_class must be specified")
    if num_examples is not None and num_examples_by_class is not None:
      raise ValueError("Only one of n or n_class can be specified")
    if num_examples_by_class is not None:
      num_examples = num_examples_by_class * len(self._datasets)
    if num_examples == 0:
      return
    count = 0
    remaining_class = len(self._datasets)
    datasets = list(self._datasets.values())
    random.shuffle(datasets)
    for dataset in datasets:
      class_num_example = (num_examples - count) // remaining_class
      dataset = dataset.shuffle(len(self._dataset))
      subset = dataset.take(class_num_example)
      for example in subset.as_numpy_iterator():
        count += 1
        yield example
      remaining_class -= 1
