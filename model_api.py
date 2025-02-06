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

"""Utilities for interacting with Gemini models."""

from typing import Callable
from google import genai
from PIL import Image


def get_client(
    api_key: str,
) -> genai.Client:
  """Returns a client for the given model URL."""
  client = genai.Client(api_key=api_key)
  return client


def generate(
    client: genai.Client, model_name: str, prompt: list[str | Image.Image]
) -> str:
  """Generates text from the given prompt using the given client."""
  response = client.models.generate_content(model=model_name, contents=prompt)
  output = response.text
  return output


def client_to_inference_fn(
    client: genai.Client,
    model_name: str
) -> Callable[[list[str | Image.Image]], str]:
  """Returns an inference function for the given client."""

  def inference_fn(prompt: list[str | Image.Image]) -> str:
    return generate(client, model_name, prompt)

  return inference_fn
