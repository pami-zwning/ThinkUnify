import os
import random
import uuid

import numpy as np
import torch
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from PIL import Image
import sys
import json

from .base import BaseModel
from openai import OpenAI
def setup_environment():
    """设置PYTHONPATH环境变量"""
    tir_path = "/mnt/afs/nijingcheng/workspace/codes/benchmarks/image_text_agent/evaluation/TIR-Bench"
    if tir_path not in sys.path:
        sys.path.insert(0, tir_path)
    
    # 也设置环境变量
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if tir_path not in current_pythonpath:
        if current_pythonpath:
            os.environ['PYTHONPATH'] = f"{tir_path}:{current_pythonpath}"
        else:
            os.environ['PYTHONPATH'] = tir_path
    
    print(f"✅ 已设置PYTHONPATH: {os.environ.get('PYTHONPATH')}")

class Qwen3Internal(BaseModel):
    def __init__(self, **kwargs):
        # self.check_install()
        from easydict import EasyDict as edict
        args = edict()
        args.ip_list = os.environ["IP_LIST"].split(' ')
        args.model_name = os.environ["MODEL_NAME"]
        args.save_interim = (os.environ["SAVE_INTERIM"].lower() == "true")
        args.result_path = os.environ["RESULT_PATH"]
        args.max_turns = int(os.environ["MAX_TURNS"])
        args.need_summary = (os.environ["NEED_SUMMARY"].lower() == "true")
        args.use_tools = (os.environ["USE_TOOLS"].lower() == "true")
        self.client_urls = [f"http://{host_ip}:8000/v1" for host_ip in args.ip_list]
        save_path = os.path.join(args.result_path, "Jsons", f'{args.model_name}.jsonl')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if args.save_interim:
            image_save_dir = os.path.join(args.result_path, "Images", args.model_name)
            args.image_save_dir = image_save_dir
            os.makedirs(image_save_dir, exist_ok=True)
        self.args = args
        self.save_path = save_path
        with open(self.save_path, "w", encoding="utf-8") as f:
            # ensure_ascii=False 保证中文正常显示
            pass

        setup_environment()
        from qwen3vl_eval import model_response_func, clean_message_history
        self.model_response_func = model_response_func
        self.clean_message_history = clean_message_history

    def build_input(self, message):
        # according to https://github.com/ByteDance-Seed/Bagel/issues/83

        images = []
        text_parts = []

        for m in message:
            if m['type'] == 'image':                
                images.append(m['value'])

            elif m['type'] == 'text':
                text_parts.append(m['value'])
            else:
                raise ValueError(f"Unsupported message type {m['type']}")

        if not images:
            raise ValueError("Bagel requires at least one image input")

        prompt = " ".join(text_parts)

        return prompt, images
    
    def generate_inner(self, message, dataset=None):
        prompt, image_path_list = self.build_input(message)
        client_url = random.choice(self.client_urls)
        client = OpenAI(api_key="EMPTY", base_url=client_url)
        try:
            model_name = client.models.list().data[0].id
        except:
            model_name = "default-model"
        result_text, full_history, all_images_list = self.model_response_func(
            client, model_name, prompt, image_path_list, self.args)

        item = dict(prompt=prompt)
        if result_text:
            item["model_response"] = result_text
        else:
            item["model_response"] = "[ERROR] Empty Response"

        print(item["model_response"])
        
        if full_history:
            item["mid_process"] = self.clean_message_history(full_history)
        else:
            item["mid_process"] = []

        if self.args.save_interim:
            item["all_images_list"] = all_images_list

        with open(self.save_path, "a", encoding="utf-8") as f:
            # ensure_ascii=False 保证中文正常显示
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return result_text