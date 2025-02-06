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

"""Tools for the PaLI-Gemma model.

Code is based on
https://colab.sandbox.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/finetune_paligemma.ipynb
"""

import jax
import numpy as np
from PIL import Image
import tensorflow as tf


LOCATION_TOKENS_RANGE_MAX = 1024
_SEPARATOR = '\n'


def to_location_coordinate(
    coordinate: float, min_value: float, max_value: float
) -> int:
  """Converts a coordinate from document to model coordinates."""
  return int(
      (coordinate - min_value)
      / (max_value - min_value)
      * LOCATION_TOKENS_RANGE_MAX
  )


def from_location_coordinate(
    coordinate: float, min_value: float, max_value: float
) -> float:
  """Converts a coordinate from model to document coordinates."""
  return (
      min_value
      + (max_value - min_value) * coordinate / LOCATION_TOKENS_RANGE_MAX
  )


def preprocess_image(
    image: np.ndarray | Image.Image, size: int = 448
) -> np.ndarray:
  """Resizes and normalizes image for PaLI-Gemma model.

  Args:
    image: Image to preprocess.
    size: Size of the output image.

  Returns:
    Preprocessed image.
  """
  # The model has been trained to handle images of different aspects ratios
  # resized to 224x224 in the range [-1, 1]. Bilinear and antialias resizing
  # options are helpful to improve the quality in some tasks.
  image = np.asarray(image)
  if image.ndim == 2:  # Convert image without last channel into greyscale.
    image = np.stack((image,) * 3, axis=-1)
  image = image[..., :3]  # Remove alpha layer.
  assert image.shape[-1] == 3

  image = tf.constant(image)
  image = tf.image.resize(
      image, (size, size), method='bilinear', antialias=True
  )
  return image.numpy() / 127.5 - 1.0  # [0, 255]->[-1,1]


class PaliGemmaTokenizer:
  """Tokenizes text into tokens."""

  def __init__(self, tokenizer) -> None:
    self._tokenizer = tokenizer

  def preprocess_tokens(
      self, prefix: str, suffix: str | None = None, seqlen: int | None = None
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Preprocess the tokens for PaLI-Gemma model.

    Args:
      prefix: Prefix of the text.
      suffix: Suffix of the text.
      seqlen: Max sequence length of the text.

    Returns:
      Preprocessed tokens.
    """
    # Model has been trained to handle tokenized text composed of a prefix with
    # full attention and a suffix with causal attention.

    tokens = self._tokenizer.encode(
        prefix, add_bos=True
    ) + self._tokenizer.encode(_SEPARATOR)
    mask_ar = [0] * len(tokens)  # 0 to use full attention for prefix.
    mask_loss = [0] * len(tokens)  # 0 to not use prefix tokens in the loss.

    if suffix:
      suffix = self._tokenizer.encode(suffix, add_eos=True)
      tokens += suffix
      mask_ar += [1] * len(suffix)  # 1 to use causal attention for suffix.
      mask_loss += [1] * len(suffix)  # 1 to use suffix tokens in the loss.

    mask_input = [1] * len(tokens)  # 1 if it's a token, 0 if padding.
    if seqlen:
      padding = [0] * max(0, seqlen - len(tokens))
      tokens = tokens[:seqlen] + padding
      mask_ar = mask_ar[:seqlen] + padding
      mask_loss = mask_loss[:seqlen] + padding
      mask_input = mask_input[:seqlen] + padding

    return jax.tree.map(np.array, (tokens, mask_ar, mask_loss, mask_input))

  def postprocess_tokens(self, tokens: np.ndarray) -> str:
    tokens = tokens.tolist()  # np.array to list[int].
    try:  # Remove tokens at and after EOS if any.
      eos_pos = tokens.index(self._tokenizer.eos_id())
      tokens = tokens[:eos_pos]
    except ValueError:
      pass
    return self._tokenizer.decode(tokens)
