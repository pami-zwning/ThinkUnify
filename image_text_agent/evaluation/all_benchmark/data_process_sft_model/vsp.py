import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import PIL.Image
import time
import os
import json
import random
import numpy as np
import hashlib  # 1. 引入 hashlib

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(1)

prompt_input_3 = '''You are a maze solver. Your goal is to guide a player from the start to the goal on a grid map while avoiding holes.

Generate the complete, optimized sequence of instructions to guide the robot.
Commands are: R (Right), L (Left), U (Up), D (Down). Do not include pauses or diagonal movements.
Provide your solution as a sequence of moves wrapped in <answer> </answer>, such as <answer>LRUD</answer>.
Please generate action plan for the following maze:
'''

levels = [3,4,5,6,7,8]
in_context_example_num = 0 # 0, 1, 2, 4, 8

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


GOOGLE_API_KEY='YOUR-API-KEY'
# genai.configure(api_key=GOOGLE_API_KEY) # 注释掉以防止没有API Key报错
# model = genai.GenerativeModel('gemini-pro-vision')

if in_context_example_num > 0:
    output_path = "output/output_img_%d/"%(in_context_example_num)
    input_backup_path = "input/input_backup_img_%d/"%(in_context_example_num)
else:
    output_path = "output/output_img/"
    input_backup_path = "input/input_backup_img/"

os.makedirs(output_path, exist_ok=True)
os.makedirs(input_backup_path, exist_ok=True)

loop_time = 0

EXAMPLE_DICT = {
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
}

jsonl_path = './maze_task_main.jsonl'
with open(jsonl_path, "w", encoding="utf-8") as jf:
    for level in levels:
        start_idx = 0
        end_idx = 100
        runned_term = 0
        map_path = "/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/VSP/maps/level%d/img/"%(level)
        while True:
            try:
                curr_id = start_idx + runned_term
                if curr_id >= end_idx:
                    break
                img_input_path = map_path + "%d.png"%(curr_id)
                
                # 获取绝对路径
                abs_img_path = os.path.abspath(img_input_path)

                message = {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": prompt_input_3},
                        { "type": "image_url", "image_url": { "url": abs_img_path }}]
                }

                # ==========================================
                # 生成 hash_uid
                # ==========================================
                # 使用 maze + level + curr_id 确保唯一性
                # 只要是同一个等级的同一张图，生成的 hash 就是固定的
                unique_str = f"maze_level{level}_{curr_id}"
                hash_uid = hashlib.md5(unique_str.encode("utf-8")).hexdigest()

                record = {
                    "hash_uid": hash_uid,  # 添加 hash_uid
                    "level": level,
                    "curr_id": curr_id,
                    "prompt": prompt_input_3,
                    "image_path": abs_img_path,
                    "save_path": output_path + "level%d/%d.txt"%(level, curr_id),
                    "messages": [message] # 这里我顺手把你原本的 "message" 改成了标准的列表包裹 "messages"，如果不适应你的评测代码，可以改回 "message": message
                }
                
                # 如果你的评测代码严格要求字段名是 "message" 而不是 "messages"，请使用下面这一行：
                # record["message"] = message 

                jf.write(json.dumps(record, ensure_ascii=False) + "\n")

                runned_term += 1
            except Exception as e:
                # 打印错误以便调试
                print(f"Error processing level {level}, id {curr_id}: {e}")
                time.sleep(2)
                # 可以在这里决定是否要 continue 或者 pass
                pass