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

prompt_input_1 = '''
As a professional maze solver, your task is to analyze a grid-based map and devise an action plan that enables a player to reach the goal from the starting point without falling into any holes, using the fewest possible moves. Since coding is not within your skill set, your approach relies on logical reasoning of the map.

## Game Setup
- The game presents a fully observable grid-based map.
- The player starts at a specified grid square, with the goal located elsewhere on the map.
- Each grid square is either safe or contains a hole.
- Your goal is to guide the player to the goal while avoiding holes.
The following figure shows how the player, the holes (non-safe grid), the lands (safe grids), and the goals look like.
'''
prompt_input_2 = '''
## Moving Rules
- The action plan involves a series of moves: 'L' (left), 'R' (right), 'U' (up), or 'D' (down).
- Each move transfers the player to the adjacent square in that direction, provided it is a safe square. The player cannot move more than one square at a time.
- Moving off the edge of the map has no effect. The player will remain at the same square.
- DO NOT MOVE INTO A HOLE! Falling into a hole results in defeat.
- Locating at the grid containing the goal results in victory.
We provide an example to further illustrate the rules.
'''
prompt_input_3 = '''
In this provided example:
- The player is at Row 1, Column 1;
- The goal is at Row 4, Column 4;
- There are two holes: one at Row 1, Column 2, and another at Row 4, Column 1.
- The player can move DOWN. This is because moving down brings them to Row 2, Column 1, and this cell is safe (without holes).
- Moving UP has no effects. This is because the player is already in the topmost row.
- Similarly, moving LEFT has no effects because the player is already in the left-most column.
- Moving RIGHT places the player at Row 1, Column 2. Since there is a hole at this grid, this move results in a loss.

## Procedure and Output
Now you will solve the given maze. To solve it, please generate text EXACTLY FOLLOW THE FOLLOWING STEPS:
1. First, interpret map. List where the player is at now, where is the goal, and where are the holes.
2. Then, generate an action plan to navigate to the goal step by step. At each step, you should check:
    (a) Where the current move leads the player to (the row and column);
    (b) What is in that grid. Is it a hole? Is it the goal? Is it an empty space?
    (c) Determine if that is a safe action. If not, correct it and re-generate the action plan.
3. Next, verify if the steps successfully navigate the player to the goal without falling into the hole. If not, restart from step 2 and re-generate this step.
4. If succeed, output an aggregated plan using "Action plan: <PLAN>", where <PLAN> is a string concatenated action in each step. For example, "Action plan: L,L,R,U,D" meaning an action plan of left, left, right, up, and down. Double check the final action plan is consistent with the previous analysis.
Do not output any extra content after the above aggregated output.
'''
prompt_input_4 = "\nPlease generate action plan for the following maze:"

levels = [3,4,5,6,7,8]
in_context_example_num = 0 # 0, 1, 2, 4, 8

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


GOOGLE_API_KEY='YOUR-API-KEY'
# genai.configure(api_key=GOOGLE_API_KEY)
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

jsonl_path = './maze_task_main_other_models.jsonl'
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

                message = {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": prompt_input_1},
                        { "type": "image_url", "image_url": { "url": "/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/VSP/prompt-visual-images/system-figure-1.png" }},
                        { "type": "text", "text": prompt_input_2},
                        { "type": "image_url", "image_url": { "url": "/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/VSP/prompt-visual-images/system-figure-2.png" }},
                        { "type": "text", "text": prompt_input_3},
                        { "type": "text", "text": prompt_input_4},
                        { "type": "image_url", "image_url": { "url": os.path.abspath(img_input_path)}}
                    ]
                }
                
                # ==========================================
                # 生成 hash_uid
                # ==========================================
                # 使用 task_type + level + curr_id 组合
                unique_str = f"maze_vsp_level{level}_{curr_id}"
                hash_uid = hashlib.md5(unique_str.encode("utf-8")).hexdigest()

                record = {
                    "hash_uid": hash_uid,  # 新增 hash_uid
                    "level": level,
                    "curr_id": curr_id,
                    "prompt": prompt_input_3,
                    "image_path": os.path.abspath(img_input_path),
                    "save_path": output_path + "level%d/%d.txt"%(level, curr_id),
                    "messages": [message]  # 修改为 list 格式
                }
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")

                runned_term += 1
            except Exception as e:
                print(f"Error at level {level}, id {curr_id}: {e}")
                time.sleep(2)
                pass