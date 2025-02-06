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

"""Tools for dataset iterators for PaLI-Gemma model.

Code is based on
https://colab.sandbox.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/finetune_paligemma.ipynb
"""

from typing import Mapping

import numpy as np
import paligemma_tools
from PIL import Image


_SEQLEN = 128
_PROMPT = "Recognize gesture"


def prepare_inference_input(
    tokenizer: paligemma_tools.PaliGemmaTokenizer,
    image: np.ndarray | Image.Image,
) -> Mapping[str, np.ndarray]:
  """Prepares input data for PaLI-Gemma model.

  Args:
    tokenizer: PaLI-Gemma tokenizer.
    image: Input image.

  Returns:
    image: np.ndarray[H, W, 3] image with values in [-1, 1].
    text: int32[N] tokens.
    input_mask: bool[B, N] true if its part of the input, false if padding.
    mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
      it and 0 where it shares the same attention mask as the previous token.
  """
  if not isinstance(image, Image.Image):
    image = Image.fromarray(image, "RGB")
  image = paligemma_tools.preprocess_image(image)

  tokens, mask_ar, _, mask_input = tokenizer.preprocess_tokens(
      _PROMPT, seqlen=_SEQLEN
  )

  return {
      "image": np.asarray(image),
      "text": np.asarray(tokens),
      "mask_ar": np.asarray(mask_ar),
      "mask_input": np.asarray(mask_input),
  }


def prepare_train_input(
    tokenizer: paligemma_tools.PaliGemmaTokenizer,
    image: np.ndarray | Image.Image,
    suffix: str,
) -> Mapping[str, np.ndarray]:
  """Prepares input data for PaLI-Gemma model for training.

  Args:
    tokenizer: PaLI-Gemma tokenizer.
    image: Input image.
    suffix: target text.

  Returns:
    Dictionary containing the prepared input data.
  """
  if not isinstance(image, Image.Image):
    image = Image.fromarray(image, "RGB")
  image = paligemma_tools.preprocess_image(image)

  suffix = suffix.lower()
  tokens, mask_ar, mask_loss, _ = tokenizer.preprocess_tokens(
      _PROMPT, suffix, seqlen=_SEQLEN
  )

  return {
      "image": np.asarray(image),
      "text": np.asarray(tokens),
      "mask_ar": np.asarray(mask_ar),
      "mask_loss": np.asarray(mask_loss),
  }
