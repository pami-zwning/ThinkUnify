import json
import requests
import os
import base64
import random
import math
import time
import argparse
import threading
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from openai import OpenAI
from openai import AzureOpenAI

from desc import TOOLS, SYSTEM_ROLE_PROMPT, SYSTEM_PROMPT
from utils import run_toolcall, toolcall_message_convert

# ==========================================
# 配置区域
# ==========================================

# 路径配置
DATA_PATH = '/mnt/afs/share_data/shangziyi/datasets/unimmmu/vqa/unimmmu_tool_call.jsonl'

def read_jsonl_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [json.loads(line) for line in lines]

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def encode_image_with_format(image_path):
    try:
        if not os.path.exists(image_path):
            print(f"Image path does not exist: {image_path}")
            return None

        with Image.open(image_path) as img:
            # 判断格式
            image_format = img.format.lower()
            if image_format == "jpg":
                image_format = "jpeg"
            
            # 保存到内存 buffer
            buffered = BytesIO()
            img.save(buffered, format=image_format.upper())
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        data_url = f"data:image/{image_format};base64,{image_base64}"
        return data_url

    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def clean_message_history(messages):
    """
    清洗消息历史，移除图片的 Base64 编码，减小保存体积。
    """
    import copy
    cleaned = []
    try:
        for msg in messages:
            new_msg = copy.deepcopy(msg)
            # 检查 content 是否包含 image_url
            if "content" in new_msg and isinstance(new_msg["content"], list):
                for item in new_msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        if "url" in item["image_url"] and item["image_url"]["url"].startswith("data:image"):
                            item["image_url"]["url"] = "<IMAGE_BASE64_OMITTED>"
            cleaned.append(new_msg)
    except Exception as e:
        return [{"role": "system", "content": f"Error cleaning history: {str(e)}"}]
    return cleaned


def convert_message(messages, use_tools):
    cur_messages = []
    if use_tools:
        cur_messages.append({"role": "system","content": SYSTEM_ROLE_PROMPT})

    user_content = []
    fn_image_list = []
    all_images_list = []
    
    message_contents = messages[0]['content']
    for message_content in message_contents:
        if message_content['type'] == 'text':
            user_content.append(message_content)
        elif message_content['type'] == 'image_url':
            user_content.append(
                {
                    "type": "image_url", 
                    "image_url": {'url':encode_image_with_format(message_content['image_url']['url'])}
                })
            fn_image_list.append(Image.open(message_content['image_url']['url']))
            all_images_list.append(message_content['image_url']['url'])

    cur_messages.append({"role": "user","content": user_content})

    return cur_messages, fn_image_list, all_images_list

def model_response_func(client, model_name, messages, hash_uid, args):
    cur_messages, fn_image_list, all_images_list = convert_message(messages, args.use_tools)
    # ==========================================
    # 循环逻辑
    # ==========================================

    img_name = hash_uid
    img_format = os.path.basename(all_images_list[0]).split('.')[-1]
    
    for turn_idx in range(args.max_turns):
        try:
            request_kwargs = {
                "model": model_name,
                "messages": cur_messages,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "max_tokens":args.max_tokens,
                "extra_body": {
                    "repetition_penalty": 1.05,
                    "chat_template_kwargs": {"enable_thinking": False}
                },
            }

            # 2. 根据条件动态调整参数
            if args.use_tools:
                request_kwargs["tools"] = TOOLS

            # 3. 统一调用
            response = client.chat.completions.create(**request_kwargs)
            #print(response.choices[0].message.model_dump())
            msg = response.choices[0].message
            cur_messages.append(msg.model_dump())
            
            # 判断是否调用工具
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)

                    function_return = run_toolcall(func_name, fn_args, fn_image_list)
                    function_return_content = toolcall_message_convert(function_return)

                    cur_messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": func_name,
                        "content": function_return_content,
                    })

                    if args.save_interim:
                        if 'image' in function_return:
                            for image_idx, image_item in enumerate(function_return['image']):
                                save_path = os.path.abspath(os.path.join(args.image_save_dir, f"{img_name}_{turn_idx}_{func_name}_{image_idx}.{img_format}"))
                                if img_format == "jpeg" or img_format == "jpg":
                                    image_item_new = image_item.convert("RGB")
                                    image_item_new.save(save_path)
                                else:
                                    image_item.save(save_path)
                                all_images_list.append(save_path)
                                fn_image_list.append(image_item)

            else:
                # 没有调用工具，返回文本
                return msg.content, cur_messages, all_images_list

        except Exception as e:
            import traceback;traceback.print_exc()
            return f"[ERROR in Loop] {str(e)}", cur_messages, all_images_list

    # 超过最大轮数，让模型直接进行总结
    if args.need_summary:
        try:
            cur_messages.append({
                "role": "user",
                "content": [
                    { "type": "text", "text": "Please output the answer to <answer> </answer>."},
                ]
            })
            final_res = client.chat.completions.create(
                model=model_name,
                messages=cur_messages,
                tool_choice="none" 
            )
            final_msg = final_res.choices[0].message
            cur_messages.append(final_msg.model_dump())
            return final_msg.content, cur_messages, all_images_list
        except Exception as e:
            return f"[ERROR Final Summary] {str(e)}", cur_messages, all_images_list
    else:
        return f"I give up answering.", cur_messages, all_images_list

def worker_process(input_args):
    item, client_urls, args = input_args

    if args.close_api:
        if 'gpt' in args.model_name:
            azure_client = get_thread_azure()
            result_text, full_history, all_images_list = azure_chat(azure_client, item['messages'], args)
        elif 'gemini' in args.model_name:
            result_text, full_history, all_images_list = gemini_api(item, args)

    
    else:
        client_url = random.choice(client_urls)
        client = OpenAI(api_key="EMPTY", base_url=client_url)
        try:
            model_name = client.models.list().data[0].id
        except:
            model_name = "default-model"

        result_text, full_history, all_images_list = model_response_func(client, model_name, item['messages'], item['hash_uid'], args)
    
    if result_text:
        item["model_response"] = result_text
    else:
        item["model_response"] = "[ERROR] Empty Response"
    
    if full_history:
        item["mid_process"] = clean_message_history(full_history)
    else:
        item["mid_process"] = []

    if args.save_interim:
        item["all_images_list"] = all_images_list

    return item

def get_azure_client():
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://duomotai.openai.azure.com/")
    AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "YOUR_AZURE_OPENAI_API_KEY")
    AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

    client = AzureOpenAI(
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
    )
    return client


_tls = threading.local()
def get_thread_azure():
    if getattr(_tls, "azure", None) is None:
        _tls.azure = get_azure_client()
    return _tls.azure

def azure_chat(azure_client: AzureOpenAI, messages, args):
    cur_messages, _, all_images_list = convert_message(messages, args.use_tools)
    last_err = None
    for _ in range(max(1, args.max_turns)):
        try:
            resp = azure_client.chat.completions.create(
                model=args.model_name,
                messages=cur_messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            txt = resp.choices[0].message.content or ""
            return txt, cur_messages, all_images_list
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    return f"[AZURE_ERROR] {repr(last_err)}", {}, all_images_list

def image_base64(image_path):
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
        return image_b64

def convert_message_gemini(messages, use_tools):
    data = {}
    if use_tools:
        data["system_instruction"] = {
                    "parts": [
                        {"text": SYSTEM_PROMPT}
                    ]
                }

    user_content = []
    fn_image_list = []
    all_images_list = []
    
    message_contents = messages[0]['content']
    for message_content in message_contents:
        if message_content['type'] == 'text':
            user_content.append({"text": message_content['text']})
            
        elif message_content['type'] == 'image_url':
            img_url = image_base64(message_content['image_url']['url'])
            user_content.append(
                {
                    "inline_data": {
                            "mime_type": "image/png",
                            "data": img_url
                    }
                })
            fn_image_list.append(Image.open(message_content['image_url']['url']))
            all_images_list.append(message_content['image_url']['url'])

    cur_messages = {
        "role": "user",
        "parts": user_content
    }

    data['contents'] = [cur_messages]

    return data, fn_image_list, all_images_list


def gemini_api(item, args):
    data, _, all_images_list = convert_message_gemini(item['messages'], args.use_tools)

    key = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY")
    url = f"https://api.ppchat.vip/v1beta/models/{args.model_name}-preview:generateContent"

    headers = {
    "x-goog-api-key": key,
    "Content-Type": "application/json",
    }
    response = requests.post(url, headers=headers, json=data)

    if response.status_code==200:
        return response.json()['candidates'][0]['content']['parts'][0]['text'], data, all_images_list
    else:
        return "ERROR", data, all_images_list
