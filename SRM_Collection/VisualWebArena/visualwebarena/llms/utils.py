import argparse
import time
from typing import Any

try:
    from vertexai.preview.generative_models import Image
    from llms import generate_from_gemini_completion
except:
    print('Google Cloud not set up, skipping import of vertexai.preview.generative_models.Image and llms.generate_from_gemini_completion')

from llms import (
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
)

from .ant_utils.utils import (
    get_default_config,
    ask_chatgpt_async_send,
    ask_chatgpt_async_fetch,
)


APIInput = str | list[Any] | dict[str, Any]


def ask_chatgpt(model, msg, temp='0.2'):
    param = get_default_config(model)
    # param["queryConditions"]["messages"][0]["content"] = [{}]
    # param["queryConditions"]["messages"][0]["content"][0]["type"] = "text"
    # param["queryConditions"]["messages"][0]["content"][0]["text"] = msg

    param["queryConditions"]["messages"] = msg
    param["queryConditions"]["temperature"] = temp

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


def call_llm(
    lm_config: lm_config.LMConfig,
    prompt: APIInput,
) -> str:
    response: str
    if lm_config.provider == "openai": # 如果是gpt模型
        if lm_config.mode == "chat":
            print("\ncall_llm chat\n")

            model = "gpt-4o"

            # print("msg = \n", prompt)
            print("\n")

            start_time = time.time()

            while True:
                response = ask_chatgpt(model, prompt, str(lm_config.gen_config["temperature"]))
                if (response != None) and (response != False):
                    print("\nyes\n")
                    print("\nresponse = -------------------------------------------------------\n", response)
                    print("\n-------------------------------------------------------\n")
                    break
                else:
                    print("\nno\n")

            end_time = time.time()
            elapsed_time = end_time - start_time
            print("spend time = ", elapsed_time)
            print("\n")


            # assert isinstance(prompt, list)
            # response = generate_from_openai_chat_completion(
            #     messages=prompt,
            #     model=lm_config.model,
            #     temperature=lm_config.gen_config["temperature"],
            #     top_p=lm_config.gen_config["top_p"],
            #     context_length=lm_config.gen_config["context_length"],
            #     max_tokens=lm_config.gen_config["max_tokens"],
            #     stop_token=None,
            # )
        elif lm_config.mode == "completion":
            print("\ncall_llm completion\n")

            model = "gpt-4o"

            start_time = time.time()

            while True:
                response = ask_chatgpt(model, prompt, str(lm_config.gen_config["temperature"]))
                if (response != None) and (response != False):
                    print("\nyes\n")
                    print("\nresponse = -------------------------------------------------------\n", response)
                    print("\n-------------------------------------------------------\n")
                    break
                else:
                    print("\nno\n")

            end_time = time.time()
            elapsed_time = end_time - start_time
            print("spend time = ", elapsed_time)
            print("\n")

            # assert isinstance(prompt, str)
            # response = generate_from_openai_completion(
            #     prompt=prompt,
            #     engine=lm_config.model,
            #     temperature=lm_config.gen_config["temperature"],
            #     max_tokens=lm_config.gen_config["max_tokens"],
            #     top_p=lm_config.gen_config["top_p"],
            #     stop_token=lm_config.gen_config["stop_token"],
            # )
        else:
            raise ValueError(
                f"OpenAI models do not support mode {lm_config.mode}"
            )
    elif lm_config.provider == "huggingface":
        assert isinstance(prompt, str)
        response = generate_from_huggingface_completion(
            prompt=prompt,
            model_endpoint=lm_config.gen_config["model_endpoint"],
            temperature=lm_config.gen_config["temperature"],
            top_p=lm_config.gen_config["top_p"],
            stop_sequences=lm_config.gen_config["stop_sequences"],
            max_new_tokens=lm_config.gen_config["max_new_tokens"],
        )
    elif lm_config.provider == "google": # 如果是gemini模型
        assert isinstance(prompt, list)
        assert all(
            [isinstance(p, str) or isinstance(p, Image) for p in prompt]
        )
        response = generate_from_gemini_completion(
            prompt=prompt,
            engine=lm_config.model,
            temperature=lm_config.gen_config["temperature"],
            max_tokens=lm_config.gen_config["max_tokens"],
            top_p=lm_config.gen_config["top_p"],
        )
    else:
        raise NotImplementedError(
            f"Provider {lm_config.provider} not implemented"
        )

    return response
