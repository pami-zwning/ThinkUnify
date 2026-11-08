import os
import base64
from openai import OpenAI
from typing import Optional, Dict, Any, Union, List
import logging
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
# import cairosvg
import tempfile
from tqdm import tqdm
import re
import json
import os
from openai import AzureOpenAI
import re

from tir_tools import *
import ast
import argparse


tasks = 'maze, color, word_search, jigsaw, rotation_game, ocr, refcoco, spot_difference, visual_search, symbolic, math, contrast, instrument'
tasks = tasks.split(', ')
tasks = [task.strip() for task in tasks]
assert len(tasks)==13

safe_tasks = 'maze, color, rotation_game, refcoco, visual_search, symbolic, math, contrast, instrument'
safe_tasks = safe_tasks.split(',')
safe_tasks = [task.strip() for task in safe_tasks]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--input_path', type=str)

    args = parser.parse_args()

    path = os.path.join(args.input_path, "Extract_Answer", f"{args.model_name}.json")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    task_correct = {}
    for task in tasks:
        task_correct[task] = []
    correct = 0

    for item in tqdm(data):
        correctness = 0
        if 'task' in item.keys():
            task = item['task']
        else:
            task = item['category']
        if task!='ocr':
            assert 'extracted_answer' in item.keys()
            if item['extracted_answer'] == None:
                item['extracted_answer'] = ''

        if task in safe_tasks:
            extracted_answer = item['extracted_answer'].replace('*', '').strip()
            answer = str(item['answer'])
            string_type = classify_string(answer)
            if string_type == 1:
                correctness = judge_choice(extracted_answer, answer, item)
                correct = correct + correctness
            elif string_type == 2:
                #int
                correctness = judge_int(extracted_answer, answer)
                correct = correct + correctness
            elif string_type == 3:
                # float
                correctness = judge_float(extracted_answer, answer)
                correct = correct + correctness
            elif string_type == 4:
                print(f'error {task}, answer is wrong')
                print(item['question_id'])

        elif task=='ocr':
            answer = str(item['answer'])
            response = item['model_response']
            if type(response) == list:
                response = response[0]
            if '60.jpg' in item['image_1']:
                answer = 'mobi'
            if '62.jpg' in item['image_1']:
                answer = 'aires'
            if answer in response:
                correctness = 1
                correct += correctness

        elif task=='word_search':
            answer = str(item['answer'])
            extracted_answer = item['extracted_answer'].replace('*', '').strip()
            extraction = re.sub(r"[A-Za-z*:\s]+", "", extracted_answer).strip()
            if classify_string(answer)==2:
                correctness = judge_int(extracted_answer, answer)
                correct = correct + correctness
            else:
                try:
                    a1, a2 =extract_two_numbers(answer)
                    r1, r2 = extract_two_numbers(extracted_answer)
                    if a1==r1 and a2==r2:
                        correctness = 1
                        correct += 1
                except:
                    pass

        elif task=='spot_difference':
            answer = str(item['answer'])
            extracted_answer = item['extracted_answer'].replace('*', '').strip()
            if classify_string(answer)==2:
                correctness = judge_int(extracted_answer, answer)
                correct = correct + correctness
            else:
                try:
                    list_answer =extract_consecutive_integers(answer)
                    list_response = extract_consecutive_integers(extracted_answer)
                    correctness = list_iou(list_response, list_answer)
                    correct += correctness
                except:
                    pass

        elif task=='jigsaw':
            answer = str(item['answer'])
            extracted_answer = item['extracted_answer'].replace('*', '').strip()
            try:
                if 'metadata' in item.keys():
                    meta_key = 'metadata'
                else:
                    meta_key = 'meta_data'
                m_re = extract_consecutive_n_squared(extracted_answer, item[meta_key]['difficulty'])
                a_re = extract_consecutive_n_squared(answer, item[meta_key]['difficulty'])
                correctness = compare(a_re, m_re)
                correct += correctness
            except Exception as e:
                #print(e)
                pass

        item['true_false'] = correctness
        task_correct[task].append(correctness)

    path = os.path.join(args.input_path, "Extract_Answer", f"{args.model_name}_final.json")
    with open(path , 'w') as f:
        json.dump(data, f,ensure_ascii=False, indent=2)

    for task in tasks:
        if len(task_correct[task])==0:
            continue
        print(f'{task}(number: {len(task_correct[task])}): {sum(task_correct[task])/len(task_correct[task])}')

    print('\n\n')

if __name__ == "__main__":
    main()