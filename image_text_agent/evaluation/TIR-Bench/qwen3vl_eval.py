import json
import requests
import os
import base64
import random
import math
import time
import argparse
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from openai import OpenAI

from desc import TOOLS, SYSTEM_ROLE_PROMPT
from utils import run_toolcall, toolcall_message_convert

# ==========================================
# 配置区域
# ==========================================

# 路径配置
DATA_PATH = '/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/TIR-Bench/TIR-Bench.json'
IMAGE_ROOT = "/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/TIR-Bench/"

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def encode_image_with_format(image_path, max_size=0):
    try:
        if not os.path.exists(image_path):
            print(f"Image path does not exist: {image_path}")
            return None

        with Image.open(image_path) as img:
            # 判断格式
            image_format = img.format.lower()
            if image_format == "jpg":
                image_format = "jpeg"
            
            if max_size > 0:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
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

def model_response_func(client, model_name, prompt, image_path_list, args):
    max_size = 0

    img_urls = [{ "type": "image_url", "image_url":  {'url':encode_image_with_format(item, max_size)}} for item in image_path_list]
    user_content = [{ "type": "text", "text": prompt}]
    user_content = user_content+img_urls

    cur_messages = []
    if args.use_tools:
        cur_messages.append({"role": "system","content": SYSTEM_ROLE_PROMPT})
    cur_messages.append({"role": "user","content": user_content})

    # ==========================================
    # 循环逻辑
    # ==========================================

    fn_image_list = [Image.open(image_path) for image_path in image_path_list]
    all_images_list = image_path_list

    img_name = os.path.basename(image_path_list[0]).split('.')[0]
    img_format = os.path.basename(image_path_list[0]).split('.')[-1]
    dirname = os.path.basename(os.path.dirname(image_path_list[0]))
    save_sub_dir = os.path.join(args.image_save_dir, dirname)
    os.makedirs(save_sub_dir, exist_ok=True)
    
    for turn_idx in range(args.max_turns):
        try:
            if args.use_tools:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=cur_messages,
                    tools=TOOLS,
                    top_p=0.5,
                    temperature=0.5,
                    max_tokens=1024*4,
                    extra_body={
                        "repetition_penalty": 1.05,
                        "chat_template_kwargs": {"enable_thinking": False}  # default to True
                        },
                    )
            else:
                response = client.chat.completions.create(
                model=model_name,
                messages=cur_messages,
                top_p=0.5,
                temperature=0.5,
                max_tokens=1024*4,
                extra_body={
                    "repetition_penalty": 1.05,
                    "chat_template_kwargs": {"enable_thinking": False}  # default to True
                    },
                )
            #print(response.choices[0].message.model_dump())
            msg = response.choices[0].message
            cur_messages.append(msg.model_dump())
            
            # 判断是否调用工具
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)

                    function_return = run_toolcall(func_name, fn_args, fn_image_list)
                    function_return_content = toolcall_message_convert(function_return, max_size=max_size)

                    cur_messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": func_name,
                        "content": function_return_content,
                    })

                    if args.save_interim:
                        if 'image' in function_return:
                            for image_idx, image_item in enumerate(function_return['image']):
                                save_path = os.path.abspath(os.path.join(save_sub_dir, f"{img_name}_{turn_idx}_{func_name}_{image_idx}.{img_format}"))
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
    client_url = random.choice(client_urls)
    
    client = OpenAI(api_key="EMPTY", base_url=client_url)
    try:
        model_name = client.models.list().data[0].id
    except:
        model_name = "default-model"
    
    image_path_list = [os.path.join(IMAGE_ROOT, item["image_1"])]
    if item["image_2"]:
        image_path_list.append(os.path.join(IMAGE_ROOT, item["image_2"]))
    prompt = item["prompt"]

    result_text, full_history, all_images_list = model_response_func(client, model_name, prompt, image_path_list, args)
    
    
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

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='测评的模型名字')
    parser.add_argument('--ip_list', type=str, nargs='+',help='vllm api 列表')
    parser.add_argument('--save_interim', action='store_true', default=False, help='是否需要保存过程中的中间结果文件。')
    parser.add_argument('--result_path', type=str, help='保存测评结果路径')
    parser.add_argument('--workers', type=int, help='并发数')
    parser.add_argument('--max_turns', type=int, help='最大toolcall轮数')
    parser.add_argument('--need_summary', action='store_true', default=False, help='是否需要强制给出一个选项。')
    parser.add_argument('--use_tools', action='store_true', default=False, help='是否需要调用工具。')

    args = parser.parse_args()
    #import pdb;pdb.set_trace()

    # 1. 准备数据
    client_urls = [f"http://{host_ip}:8000/v1" for host_ip in args.ip_list]
    save_path = os.path.join(args.result_path, "Jsons", f'{args.model_name}.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


    if args.save_interim:
        image_save_dir = os.path.join(args.result_path, "Images", args.model_name)
        args.image_save_dir = image_save_dir
        os.makedirs(image_save_dir, exist_ok=True)

    with open(DATA_PATH, 'r') as f:
        datas = json.load(f)

    #datas = [item for item in datas if item['task'] not in ['jigsaw', 'maze']]

    for item in datas:
        if item['task'] == 'ocr':
            item['prompt'] = item['prompt'] + "Please answer in a case-sensitive manner."
        if item['task'] == 'math':
            if item['id'] in [
                35, 217, 240, 245, 256, 330, 377, 413, 416, 443,
                493, 551, 566, 589, 604, 608, 671, 716, 767, 809,
                887, 923, 941, 977, 1012, 1013, 1180, 1187, 1188,
                1000, 1097
            ]:
                w,h = Image.open(os.path.join(IMAGE_ROOT, item["image_1"])).size
                item['prompt'] = "Image resolution: {}x{}\n".format(w,h) +item['prompt'] 

    # 断点续传逻辑
    processed_ids = set()
    aa = []
    if os.path.exists(save_path):
        try:
            tested_data = read_json_file(save_path)
            filtered_data = [item for item in tested_data if 'ERROR' not in item['model_response']]
            with open(save_path, 'w') as f:
                json.dump(filtered_data, f, ensure_ascii=False, indent=2)

            with open(save_path, 'r') as f:
                aa = json.load(f)
                processed_ids = {x["image_1"] for x in aa}
        except json.JSONDecodeError:
            print("JSON file corrupted, starting over or handle manually.")
            aa = []

    print(f"Loaded {len(aa)} processed samples.")
    print(f"Total samples: {len(datas)}")

    remaining = [x for x in datas if x["image_1"] not in processed_ids]
    print(f"Remaining to process: {len(remaining)}")

    if not remaining:
        print("All done.")
        return

    # 2. 准备 Worker 参数
    worker_args = [(item, client_urls, args) for item in remaining]
    
    with Pool(args.workers) as pool:
        processed_count = 0
        for result in tqdm(pool.imap_unordered(worker_process, worker_args), total=len(remaining)):
            aa.append(result)
            processed_count += 1

            if processed_count % 10 == 0:
                with open(save_path, 'w') as f:
                    json.dump(aa, f, ensure_ascii=False, indent=2)
    
    with open(save_path, 'w') as f:
        json.dump(aa, f, ensure_ascii=False, indent=2)

    print(f"Finished. Total saved: {len(aa)}")

if __name__ == "__main__":
    main()