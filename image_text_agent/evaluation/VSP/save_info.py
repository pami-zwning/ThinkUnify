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
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # env.seed(seed)
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
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro-vision')
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
        map_path = "./maps/level%d/img/"%(level)
        while True:
            try:
                curr_id = start_idx + runned_term
                if curr_id >= end_idx:
                    break
                img_input_path = map_path + "%d.png"%(curr_id)
                
                record = {
                    "level": level,
                    "curr_id": curr_id,
                    "prompt": prompt_input_3,
                    "image_path":os.path.abspath(img_input_path),
                    "save_path":output_path + "level%d/%d.txt"%(level, curr_id)
                }
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")

                runned_term += 1
            except:
                time.sleep(2)
                pass






# pass



# cookie_picture = [{
#     'mime_type': 'image/png',
#     'data': Path('cookie.png').read_bytes()
# }]
# prompt = "Do these look store-bought or homemade?"

# response = model.generate_content(
#     model="gemini-pro-vision",
#     content=[prompt, cookie_picture]
# )
# print(response.text)