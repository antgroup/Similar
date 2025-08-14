# Copyright 2024 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Some LLM inference interface."""

import abc
import base64
import io
import os
import time
from typing import Any, Optional
import google.generativeai as genai
from google.generativeai import types
from google.generativeai.types import answer_types
from google.generativeai.types import content_types
from google.generativeai.types import generation_types
from google.generativeai.types import safety_types
import numpy as np
from PIL import Image
import requests

ERROR_CALLING_LLM = 'Error calling LLM'


def ask_chatgpt(model, payload):
    param = get_default_config(model)
    # param["queryConditions"]["messages"][0]["content"] = [{}, {}]
    # param["queryConditions"]["messages"][0]["content"][1]["type"] = "image_url"
    # param["queryConditions"]["messages"][0]["content"][1]["image_url"] = {"url": "https://up.enterdesk.com/edpic/14/e7/33/14e733e0a7dc301f0d2d02c2a9fda007.jpg"}
    # param["queryConditions"]["messages"][0]["content"][0]["type"] = "text"
    # param["queryConditions"]["messages"][0]["content"][0]["text"] = msg
    # param["queryConditions"]["temperature"] = temp

    param["queryConditions"]["messages"] = payload["messages"]
    param["queryConditions"]["temperature"] = str(payload["temperature"])

    try:
        ask_chatgpt_async_send(param)
    except Exception as e:
        print("send error")
        print(e)
        return False

    try:
        return ask_chatgpt_async_fetch(param)
    except Exception as e:
        print("fetch error")
        print(e)
        return None


def array_to_jpeg_bytes(image: np.ndarray) -> bytes:
    """Converts a numpy array into a byte string for a JPEG image."""
    image = Image.fromarray(image)
    return image_to_jpeg_bytes(image)


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
    in_mem_file = io.BytesIO()
    image.save(in_mem_file, format='JPEG')
    # Reset file pointer to start
    in_mem_file.seek(0)
    img_bytes = in_mem_file.read()
    return img_bytes


class LlmWrapper(abc.ABC):
  """Abstract interface for (text only) LLM."""

  @abc.abstractmethod
  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    """Calling multimodal LLM with a prompt and a list of images.

    Args:
      text_prompt: Text prompt.

    Returns:
      Text output, is_safe, and raw output.
    """


class MultimodalLlmWrapper(abc.ABC):
  """Abstract interface for Multimodal LLM."""

  @abc.abstractmethod
  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    """Calling multimodal LLM with a prompt and a list of images.

    Args:
      text_prompt: Text prompt.
      images: List of images as numpy ndarray.

    Returns:
      Text output and raw output.
    """


SAFETY_SETTINGS_BLOCK_NONE = {
    types.HarmCategory.HARM_CATEGORY_HARASSMENT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
}


class Gpt4Wrapper(LlmWrapper, MultimodalLlmWrapper):
    """OpenAI GPT4 wrapper.

    Attributes:
      openai_api_key: The class gets the OpenAI api key either explicitly, or
        through env variable in which case just leave this empty.
      max_retry: Max number of retries when some error happens.
      temperature: The temperature parameter in LLM to control result stability.
      model: GPT model to use based on if it is multimodal.
    """

    RETRY_WAITING_SECONDS = 20

    def __init__(
            self,
            model_name: str,
            max_retry: int = 3,
            temperature: float = 0.0,
    ):
        # if 'OPENAI_API_KEY' not in os.environ:
        #   raise RuntimeError('OpenAI API key not set.')
        # self.openai_api_key = os.environ['OPENAI_API_KEY']
        if max_retry <= 0:
            max_retry = 3
            print('Max_retry must be positive. Reset it to 3')
        self.max_retry = min(max_retry, 5)
        self.temperature = temperature
        self.model = model_name

    @classmethod
    def encode_image(cls, image: np.ndarray) -> str:
        return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

    def predict(
            self,
            text_prompt: str,
    ) -> tuple[str, Optional[bool], Any]:
        return self.predict_mm(text_prompt, [])

    def predict_mm(
            self, text_prompt: str, images: list[np.ndarray]
    ) -> tuple[str, Optional[bool], Any]:
        # headers = {
        #     'Content-Type': 'application/json',
        #     'Authorization': f'Bearer {self.openai_api_key}',
        # }

        payload = {
            'model': self.model,
            'temperature': self.temperature,
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': text_prompt},
                ],
            }],
            'max_tokens': 1000,
        }

        # print("\nmsg = \n", payload['messages'][0]['content'])
        # print("\n")

        # Gpt-4v supports multiple images, just need to insert them in the content
        # list.
        for image in images:
            payload['messages'][0]['content'].append({
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/jpeg;base64,{self.encode_image(image)}'
                },
            })

        model = 'gpt-4o'

        start_time = time.time()

        # 请求openai，得到agent回复
        while True:
            response = ask_chatgpt(model, payload)
            if (response != None) and (response != False):
                print("\nyes\n")
                print("\nresponse = -----------------------------------------------------------\n", response)
                print("-----------------------------------------------------------\n")
                break
            else:
                print("\nno\n")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("spend time = ", elapsed_time)
        print("\n")

        return (
            response,
            None
        )
