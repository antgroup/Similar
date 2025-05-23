import asyncio
import json
import os
import threading
import socket
import sys
import subprocess
import time
import re
import toml
import traceback
import transformers
import uuid
import torch
import numpy as np
from enum import Enum
from typing import Dict, List, Union
from pathlib import Path
from packaging import version

from vllm import __version__ as vllm_version
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_tokenization import OpenAIServingTokenization
from vllm.entrypoints.openai.api_server import build_async_engine_client
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.logger import init_logger
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              ErrorResponse)
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

# token监控 maya sdk依赖
from maya_tools.context import get_request_context, _InferenceMetric
# 统一继承MayaBaseHandler，可以自动获取组件面板的模型文件
from aistudio_serving.hanlder.pymps_handler import MayaBaseHandler
from maya_config import config_python_backend_log, config_vllm_log

import logging
import logging.config

logger = logging.getLogger()

ENTRY_POINT_KEY = "__entry_point__"
API_VERSION = "api_version"
OPEN_AI_CHAT_COMPLETION = "openai.chat.completion"
VLLM_USE_V1 = 1


def start_cleaner(pid):
    """
    保证Tritonserver被杀掉的时候，vllm的进程退出。
    """
    dir_name = os.path.dirname(__file__)
    cleaner = os.path.join(dir_name, "vllm_cleaner.py")
    cmd = f"python {cleaner} --pid {pid} 2>&1 &"
    logger.info(f"#########Start Cleaner: {cmd}")
    proc = subprocess.run(cmd, shell=True)
    return_code = proc.returncode
    if return_code != 0:
        raise Exception(f"return_value: {return_code}, failed executing command '{cmd}'")


class InputSchema(Enum):
    Legacy = 0
    LegacyV2 = 1
    OPEN_AI_CHAT_COMPLETION = 2


class GenerationMeta:
    def __init__(self, schema, trace_id, user, think_output):
        self.schema: InputSchema = schema
        self.trace_id: str = trace_id
        self.user: str = user
        self.think_output: bool = think_output


class OpenaiAppState:
    def __init__(self,
                 openai_serving_chat: OpenAIServingChat,
                 openai_serving_completion: OpenAIServingCompletion,
                 openai_serving_embedding: OpenAIServingEmbedding,
                 openai_serving_tokenization: OpenAIServingTokenization):
        self.openai_serving_chat: OpenAIServingChat = openai_serving_chat
        self.openai_serving_completion: OpenAIServingCompletion = openai_serving_completion
        self.openai_serving_embedding: OpenAIServingEmbedding = openai_serving_embedding
        self.openai_serving_tokenization: OpenAIServingTokenization = openai_serving_tokenization


class PayloadHandler:
    def __init__(self, for_github, default_values):
        self.for_github = for_github
        self.default_do_sample = False
        self.default_system = default_values["system"]
        self.default_max_tokens = default_values["max_tokens"]
        self.default_temperature = default_values["temperature"]
        self.default_top_p = default_values["top_p"]
        self.default_top_k = default_values["top_k"]
        self.default_stream = default_values["stream"]
        self.default_repetition_penalty = default_values["repetition_penalty"]
        self.limit_max_out_tokens = int(os.getenv("LIMIT_MAX_OUT_TOKENS", -1))
        self.think_output = False

        self.deepseek_think_template = '<think>\n'
        logger.info(f"PayloadHandler's default_values: {default_values}")

    def clean_process(self, inputs):
        inputs = re.sub('[\n]+', '', inputs)
        return inputs

    def handle_payload(self, payload):
        payload = self.clean_process(payload)
        payload = json.loads(payload, strict=False)

        # 白名单-筛选推理参数
        standard_input = {}

        # trace id
        trace_id = payload.get('trace_id', "")
        request_context = get_request_context()
        randid = str(uuid.uuid4()).replace('-', '')[:8]
        if request_context is None:
            sub_model = None
            trace_id = f"{trace_id} randid {randid}"
        else:
            sub_model = request_context.sub_model_name
            trace_id = f"{trace_id} {request_context.request_id} {randid}"
            if len(sub_model) == 0:
                sub_model = None

        # mutli lora配置，插入调用的模型，没有传默认 auto 请求base模型
        if "model" not in payload and sub_model is None:
            standard_input["model"] = "auto"
        elif "model" not in payload and sub_model is not None:
            standard_input["model"] = sub_model
        elif "model" in payload and sub_model and payload["model"] != sub_model:
            if payload["model"] == "auto":
                standard_input["model"] = sub_model
            else:
                raise Exception(f"data's model: {payload['model']} and sub_model are both exits and not equal!")

        top_k = payload.get('top_k', self.default_top_k)
        top_p = payload.get('top_p', self.default_top_p)
        standard_input['top_p'] = self.default_top_p if top_p >= 1.0 else top_p
        standard_input['top_k'] = self.default_top_k if top_k < 1 else top_k
        standard_input['temperature'] = payload.get('temperature', self.default_temperature)
        standard_input['stop'] = payload.get("stop", None)
        standard_input['repetition_penalty'] = payload.get("repetition_penalty", self.default_repetition_penalty)

        # 是否输出<think>
        think_output = payload.get('think_output', self.think_output)

        # 最大生成token参数
        max_tokens = payload.get("max_tokens", None)
        if self.limit_max_out_tokens > 0 and (max_tokens is None or max_tokens >= self.limit_max_out_tokens):
            max_tokens = self.limit_max_out_tokens
        standard_input['max_tokens'] = max_tokens

        # 流式参数
        stream = payload.get("stream", self.default_stream)
        standard_input['stream'] = stream
        if stream and "stream_options" not in payload:
            standard_input["stream_options"] = {
                "include_usage": True,
                "continuous_usage_stats": True
            }

        # 用户请求处理，适配字符串和标准message格式
        user_query = payload.pop("query", None)
        think = payload.pop("think", False)
        chat_history = payload.pop("history", [])
        system_prompt = payload.pop("system_prompt", self.default_system)
        if isinstance(user_query, list):
            standard_input["messages"] = user_query
        else:
            # 是否开启深度思考
            if think:
                user_query += self.deepseek_think_template

            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            if chat_history is not None and isinstance(chat_history, list):
                for dialog in chat_history:
                    user = dialog["user"].strip()
                    bot = dialog["bot"].strip()
                    messages.append({"role": "user", "content": user})
                    messages.append({"role": "assistant", "content": bot})
            messages.append({"role": "user", "content": user_query})
            standard_input["messages"] = messages

        # 受控参数配置
        guided_decoding_params_inputs = payload.pop("guided_decoding_params", None)
        self.convert_guided_decoding_params(standard_input, guided_decoding_params_inputs)

        # mata数据
        user = payload.get('user', "")
        meta = GenerationMeta(None, trace_id, user, think_output)
        return standard_input, meta

    def convert_guided_decoding_params(self, payload, guided_decoding_params_inputs):
        if not guided_decoding_params_inputs:
            return None
        try:
            guide_type = guided_decoding_params_inputs.get('guide_type')
            guide_content = guided_decoding_params_inputs.get('guide_content')
            if 'regex' == guide_type:
                payload['guided_regex'] = guide_content
            if 'choice' == guide_type:
                payload['guided_choice'] = guide_content
            if 'json' == guide_type:
                payload['guided_json'] = guide_content
            if 'grammar' == guide_type:
                payload['guided_grammar'] = guide_content
        except Exception as e:
            logger.error(f"convert guided decoding params failed: {e}")
        return None



# 用户自定义代码处理类
class UserHandler(MayaBaseHandler):
    """
     model_dir: model.py 文件所在目录
    """

    def __init__(self, model_dir):
        # 父类初始化
        super(UserHandler, self).__init__(model_dir)

        # 可以认为 self.resource_path 就是上游组件输入的模型或者python组件面板设置的 '自定义资源地址'在本地磁盘的路径，如果都没有返回 None
        # model_path = os.path.join(self.resource_path, "xxx")

    """
     测试demo
     1 输入配置
        query:TYPE_STRING:[1]        对应bytes 类型
     2 输出配置
        out_float:TYPE_FP64:[1]        对应float64 类型
        out_string:TYPE_STRING:[1]     对应bytes_ 类型
        out_int:TYPE_INT32:[1].        对应int32 类型
     其他
     1. TYPE_STRING对应是python bytes类型(或者np.bytes_)，目标是方便传递二进制内容，比如图片的binary内容，减少base64转换开销;
        bytes类型可以通过decode函数明确转换成python str类型
     2. 参数维度见使用文档
    """

    def predict_np(self, features, trace_id):
        # 解析请求, 字符串类型默认是bytes类型
        text = features.get("text").tostring().decode()

        # 添加预测逻辑的代码
        resultMap = {
            "res": res
        }

        # 处理结果返回
        resultCode = 0  # 0表示成功，其它为失败
        errorMessage = "ok"  # errorMessage为predict函数对外透出的信息
        logger.info("predict result")  # 可以在平台查看相关日志
        return (resultCode, errorMessage, resultMap)


# 用于调试UserHandler类的功能
if __name__ == "__main__":
    # 示例
    import os
    # 使用绝对路径初始化
    # user_handler = UserHandler(os.getcwd())

    # request = {}
    # str 类型使用 .encode() 编码模拟引擎调用的真实输入, 代码中进行decode()
    # request["query"] = "12345".encode()
    # print(user_handler.predict_np(request, "test_trace_id"))

    from alps.pytorch.api.exporter.raw_python_exporter import raw_python_export_and_deploy

    # 设置用户工号
    USER = "456420"
    # 进行导出
    raw_python_export_and_deploy(user=USER)