"""Implements helper functions to assist evaluation cases where other evaluators are not suitable."""
import json
from typing import Any
from urllib.parse import urlparse

import requests
from playwright.sync_api import CDPSession, Page

from browser_env.env_config import (
    ACCOUNTS,
    SHOPPING,
    SHOPPING_ADMIN,
)
from llms.providers.openai_utils import (
    generate_from_openai_chat_completion,
)
from evaluation_harness.utils import (
    get_asyn_config,
    get_fetch_config,
    send_request,
    parse_response,
    parse_fetch_result,
    get_default_config,
    ask_chatgpt_async_send,
    ask_chatgpt_async_fetch,
)

import hashlib
import time
from time import sleep


def send_asyn_request(model, msg, msg_key, temp=0.2):
    # print("send--model = ", model)
    param = get_asyn_config(model=model)

    param["queryConditions"]["messages"][0]["content"] = [{}]
    param["queryConditions"]["messages"][0]["content"][0]["type"] = "text"
    param["queryConditions"]["messages"][0]["content"][0]["text"] = msg
    param["queryConditions"]["messageKey"] = msg_key
    param["queryConditions"]["temperature"] = temp
    # print(param['queryConditions'])
    response = send_request(param)
    # print("response = ", response)
    try:
        result = parse_response(response)
        return result["data"]["success"]
    except Exception as e:
        print(e)
        return False

def fetch_asyn_result(model, msg_key):
    param = get_fetch_config(model=model)
    param["queryConditions"]["messageKey"] = msg_key
    response = send_request(param)
    try:
        return parse_fetch_result(response)
    except Exception as e:
        print(e)
        return None

def ask_chatgpt(model, msg, temp='0.2'):
    param = get_default_config(model)
    param["queryConditions"]["messages"][0]["content"] = [{}]
    param["queryConditions"]["messages"][0]["content"][0]["type"] = "text"
    param["queryConditions"]["messages"][0]["content"][0]["text"] = msg
    param["queryConditions"]["temperature"] = temp

    # print("ask_chatgpt")
    try:
        ask_chatgpt_async_send(param)
    except Exception as e:
        # print("send error")
        # print(e)
        return False

    try:
        return ask_chatgpt_async_fetch(param)
    except Exception as e:
        # print("fetch error")
        # print(e)
        return None



def shopping_get_auth_token() -> str:
    response = requests.post(
        url=f"{SHOPPING}/rest/default/V1/integration/admin/token",
        headers={"content-type": "application/json"},
        data=json.dumps(
            {
                "username": ACCOUNTS["shopping_site_admin"]["username"],
                "password": ACCOUNTS["shopping_site_admin"]["password"],
            }
        ),
    )
    token: str = response.json()
    return token


def shopping_get_latest_order_url() -> str:
    """Get the latest order url from the shopping website."""

    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }

    params = {
        "searchCriteria[sortOrders][0][field]": "created_at",
        "searchCriteria[sortOrders][0][direction]": "DESC",
        "searchCriteria[pageSize]": "1",
    }

    response = requests.get(
        f"{SHOPPING}/rest/V1/orders", params=params, headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()["items"][0]
    order_id = int(response_obj["increment_id"])
    order_url = f"{SHOPPING}/sales/order/view/order_id/{order_id}/"
    return order_url


def shopping_get_sku_latest_review_author(sku: str) -> str:
    """Get the latest review for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{SHOPPING}/rest/V1/products/{sku}/reviews", headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    author: str = response_obj[-1]["nickname"]
    return author


def shopping_get_sku_latest_review_rating(sku: str) -> str:
    """Get the latest review for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{SHOPPING}/rest/V1/products/{sku}/reviews", headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    assert response_obj[0]["ratings"][0]["rating_name"] == "Rating"
    rating: str = str(response_obj[-1]["ratings"][0]["percent"])
    return rating



def llm_fuzzy_match(pred: str, reference: str, question: str) -> float:
    """Check whether the prediction matches the reference with GPT4-turbo"""
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = "Help a teacher to grade the answer of a student given a question. Keep in mind that the student may use different phrasing or wording to answer the question. The goal is to evaluate whether the answer is semantically equivalent to the reference answer.\n"
    message += f"question: {question}\n"
    message += f"reference answer: {reference}\n"
    message += "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
    message += f"student answer: {pred}\n"
    message += "Conclude the judgement by correct/incorrect/partially correct. Please not that all output letters are lowercase."
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant"},
    #     {"role": "user", "content": message},
    # ]

    # response = generate_from_openai_chat_completion(
    #     # model="gpt-4-1106-preview",
    #     model="gpt-4o",
    #     messages=messages,
    #     temperature=0,
    #     max_tokens=768,
    #     top_p=1.0,
    #     context_length=0,
    # ).lower()

    model = "gpt-4-turbo"
    # temperature = 0.3
    # temperature_str = str(temperature)

    msg = message
    # msg_key = hashlib.sha256(str(msg + temperature_str).encode('utf-8')).hexdigest()

    flag = 1

    print("llm_fuzzy_match")

    start_time = time.time()
    while flag:

        # result = send_asyn_request(model, msg, msg_key, temperature)

        # response = fetch_asyn_result(model, msg_key)
        # print("\n\nresponse = ----------------------------------------------------------------------------------------\n\n", response)
        # print("\n---------------------------------------------------------------------------------------------------\n\n")
        response = ask_chatgpt(model, msg)
        # sleep(0.5)
        if (response != None) and (response != False):
            flag = 0
            print("\nyes_fuzzy_match")
            print("%s\n" % response)
        else:
            flag = 1
            # print("no")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("spend time = ", elapsed_time)

    response = response.lower()

    if "partially correct" in response or "incorrect" in response:
        return 0.0
    else:
        assert "correct" in response
        return 1.0


def llm_ua_match(pred: str, reference: str, question: str) -> float:
    """Check whether the prediction matches the reference with GPT-turbo"""
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = ""
    message += f"task: {question}\n"
    message += f"actual unachievable reason: {reference}\n"
    message += f"reported unachievable reason: {pred}\n"
    message += (
        "The task described above is inherently unachievable due to the reason specified under 'actual unachievable reason'. "
        "An individual previously attempted this task and was unable to complete it. They provided a reason for their failure, "
        "which is listed under 'reported unachievable reason'. Your role is to review both the actual and reported reasons. "
        "Determine if the reported reason aligns with the actual reason, even if implicitly. "
        "If the stated reason is in line with the actual reason, respond with 'same'. Otherwise, respond with 'different'."
    )
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant"},
    #     {"role": "user", "content": message},
    # ]

    # response = generate_from_openai_chat_completion(
    #     # model="gpt-4-1106-preview",
    #     model="gpt-4o",
    #     messages=messages,
    #     temperature=0,
    #     max_tokens=768,
    #     top_p=1.0,
    #     context_length=0,
    # ).lower()

    model = "gpt-4-turbo"
    # temperature = 0.3
    # temperature_str = str(temperature)

    msg = message
    # msg_key = hashlib.sha256(str(msg + temperature_str).encode('utf-8')).hexdigest()

    flag = 1

    start_time = time.time()
    while flag:

        # result = send_asyn_request(model, msg, msg_key, temperature)

        # response = fetch_asyn_result(model, msg_key)
        # print("\n\nresponse = ----------------------------------------------------------------------------------------\n\n", response)
        # print("\n---------------------------------------------------------------------------------------------------\n\n")
        response = ask_chatgpt(model, msg)
        # sleep(0.5)
        if (response != None) and (response != False):
            flag = 0
            print("\nyes_ua_match")
            print("%s\n" % response)
        else:
            flag = 1
            # print("no")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("spend time = ", elapsed_time)

    response = response.lower()

    if "different" in response:
        return 0.0
    else:
        assert "same" in response
        return 1.0


class PseudoPage:
    def __init__(self, original_page: Page, url: str):
        self.url = url
        self.original_page = original_page

    def __getattr__(self, attr: str) -> Any:
        # Delegate attribute access to the original page object
        if attr not in ["url"]:
            return getattr(self.original_page, attr)
        else:
            return getattr(self, attr)
