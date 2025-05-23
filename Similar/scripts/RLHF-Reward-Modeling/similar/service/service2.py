import json
from http.client import HTTPException
from requests import Request
import requests
import time

# 设置日志记录器

# 设置常量
URL_PREFIX = "https://paiplusinferencepre.alipay.com/inference" # 设置URL前缀
SOCKET_TIME_OUT = 8 # 设置套接字超时时间
CONNECT_TIME_OUT = 8 # 设置连接超时时间
CONNECT_REQUEST_TIME_OUT = 8 # 设置连接请求超时时间
CONTENT_TYPE = "application/json" # 设置内容类型
APP_NAME = "9f9ac67d9907e17e_qwen2_72B_instruct_novllm_inference" # 设置应用名称
APP_VERSION = "v1" # 设置应用版本
PROTOCOL_VERSION = "1.0" # 设置协议版本
MPS_FLOW_TYPE = "online|test" # 设置MPS流类型
CHARSET = "utf-8" # 设置字符集

# 构造URLs
url = f"{URL_PREFIX}/{APP_NAME}/{APP_VERSION}"

try:
    # 构造POST请求
    post = Request("POST", url)
    post.headers["Content-type"] = CONTENT_TYPE
    post.headers["Accept"] = CONTENT_TYPE
    post.headers["MPS-app-name"] = APP_NAME
    post.headers["MPS-http-version"] = PROTOCOL_VERSION
    post.headers["MPS-trace-id"] = 'xxxxx'
    post.headers["MPS-flow-type"] = MPS_FLOW_TYPE

# data = {
# "temperature": 0.9,