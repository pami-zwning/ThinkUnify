# read a map, receive a solution, determine if the solution reaches the goal

import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import random
import os
import numpy as np
# from PIL import Image

def generate_map_from_table(table_list):
    """
    根据输入的表格字符串列表生成 nxn 的二维地图列表。

    Args:
        table_list (list of str): 表示地图的字符串列表。
                                    示例: ['| | Col 1 | ... |', '| Row 1 | _ | # | ... |', ...]

    Returns:
        list of str: 表示 nxn 地图的字符串列表，例如 ['FFFFF', 'FHFFF', ...]。
                     'S' for Start, 'G' for Goal, 'F' for Frozen, 'H' for Hole.
    """
    # 检查输入是否为空
    if not table_list or len(table_list) < 2:
        return []

    # 地图的实际行数据从第二个元素开始 (索引 1)
    # n 是地图的维度 (行数/列数)
    n = len(table_list) - 1

    # 初始化 nxn 的地图列表
    nxn_map = []

    # 遍历地图的每一行 (对应 table_list 中的索引 1 到 n)
    for i in range(1, n + 1):
        row_str = table_list[i]
        
        # 移除首尾的 '|' 字符，并按 '|' 分割，得到一个包含单元格内容的列表
        # 示例: '| Row 1 | _ | # | _ | _ | _ |' -> [' Row 1 ', ' _ ', ' # ', ' _ ', ' _ ', ' _ ']
        cells = [cell.strip() for cell in row_str.strip().strip('|').split('|')]
        
        # 实际的地图数据从 cells 列表的索引 1 开始，对应 Col 1 到 Col n
        # 排除第一个元素 ('Row i')
        map_row = ""
        
        # 遍历单元格内容，从索引 1 开始 (对应第一列数据)
        # 单元格内容: '@' -> 'S', '#' -> 'H', '*' -> 'G', '_' -> 'F'
        for j in range(1, n + 1):
            cell_content = cells[j]
            
            if cell_content == '@':
                # S for Start tile
                map_row += 'S'
            elif cell_content == '#':
                # H for a tile with a hole
                map_row += 'H'
            elif cell_content == '*':
                # G for Goal tile
                map_row += 'G'
            elif cell_content == '_':
                # F for frozen tile (默认状态)
                map_row += 'F'
            else:
                # 遇到未知字符时，默认为 Frozen (F) 或抛出错误，这里我们选择默认 F
                map_row += 'F'
        
        nxn_map.append(map_row)

    return nxn_map


# import ipdb; ipdb.set_trace()
import re
def extract_answer(response):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.S)
    if not match:
        return ''

    result = match.group(1)
    # 使用正则表达式替换掉所有空白字符（\s 包括空格、回车、换行、制表符）
    # 同时去掉逗号
    result = re.sub(r'[\s,]+', '', result) 
    
    return result

levels = [3,4,5,6,7,8]

for level in levels:
    check_map_dir = "/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/VSP/maps/level%d/table/"%(level)
    check_answer_dir = "./output/output_img/"
    count = 0
    correct = 0

    for test_id in range(100):
        # print(test_id)
        # map
        with open(check_map_dir + '%d.txt'%(test_id), 'r') as f:
            contents = f.read()
            rows = contents.split('\n')

        # action
        
        with open(check_answer_dir + 'level%d/%d.txt'%(level, test_id), 'r') as f:
            answer = f.read()
            answer = list(extract_answer(answer))
            answer = [item.upper() for item in answer]
        #import ipdb; ipdb.set_trace()
        desc = generate_map_from_table(rows[:level+1])
        env = gym.make('FrozenLake-v1', desc=desc, is_slippery=False)
        env.reset(seed=42)
        # temp = env.render()
        # 

        action_number_map = {'L':0, 'D':1, 'R':2, 'U':3}
        for action_index in range(len(answer)):
            observation, reward, terminated, truncated, info = env.step(action_number_map[answer[action_index]])
            # temp = env.render()

            if terminated or truncated:
                if reward > 0:
                    correct += 1
                observation, info = env.reset()
                break
        count += 1

        env.close()

    print("Level %d Total tested: %d"%(level, count))
    print("Level %d Total correct: %d"%(level, correct))
    print('='*20)
