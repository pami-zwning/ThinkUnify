import os
import json
import argparse
import concurrent.futures
from tqdm import tqdm
from openai import AzureOpenAI

from .demo_prompts import *
from .tir_tools import *


# ================= Configuration =================
# 建议将 Key 放入环境变量，这里保留你提供的默认配置
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "https://duomotai.openai.azure.com/")
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "YOUR_OPENAI_API_KEY")
AZURE_API_VERSION = "2025-01-01-preview"

# Initialize Client
client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION
)

# Task definitions
TASKS = [
    'maze', 'color', 'word_search', 'jigsaw', 'rotation_game', 'ocr', 
    'refcoco', 'spot_difference', 'visual_search', 'symbolic', 
    'math', 'contrast', 'instrument'
]
SAFE_TASKS = [
    'maze', 'color', 'rotation_game', 'refcoco', 'visual_search', 
    'symbolic', 'math', 'contrast', 'instrument'
]

# ================= Helper Functions =================

def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"Question: {query}\n\nModel response: {response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt

def get_response_text(model, text):
    patience = 10
    while patience > 0:
        patience -= 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": " Assistant is a large language model trained by OpenAI."}]},
                    {"role": "user", "content": [{"type": "text", "text": text}]}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error: {e}, Retrying... ({patience} left)")
    return ""

def api_call_worker(api_name, full_prompt):
    return get_response_text(api_name, full_prompt)

def get_demo_prompt(task):
    if task == 'maze': return demo_prompt_maze
    elif task == 'color': return demo_prompt_color
    elif task == 'math': return demo_prompt_math
    elif task == 'word_search': return demo_prompt_word_search
    elif task == 'jigsaw': return demo_prompt_jigsaw
    elif task == 'rotation_game': return demo_prompt_rotation_game
    elif task == 'refcoco': return demo_prompt_refcoco
    elif task == 'spot_difference': return demo_prompt_spot_difference
    elif task == 'visual_search': return demo_prompt_visual_search
    elif task == 'symbolic': return demo_prompt_symbolic
    elif task == 'contrast': return demo_prompt_contrast
    elif task == 'instrument': return demo_prompt_instrument
    else:
        raise ValueError(f"Incorrect task name: {task}")

# ================= Core Evaluation Logic =================

def evaluate_tir_json(
    input_json_path, 
    api_model="gpt-4o-2024-11-20", 
    workers=8,
    save_intermediate=True
):
    """
    核心入口函数：读取JSON -> 提取答案 -> 计算分数 -> 保存结果
    """
    if not os.path.exists(input_json_path):
        raise FileNotFoundError(f"Input file not found: {input_json_path}")

    print(f"Loading data from {input_json_path}...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # ---------------- Step 1: Extract Answers ----------------
    tasks_to_run = []
    ocr_items = []
    
    # Pre-process items
    for item in data:
        # Normalize task key
        task = item.get('task') or item.get('category')
        item['task'] = task # Ensure 'task' key exists for later

        if task == 'ocr':
            ocr_items.append(item)
            continue
        
        # Skip if already extracted (optional, currently re-extracting based on logic)
        # if 'extracted_answer' in item and item['extracted_answer']:
        #    ocr_items.append(item) # Treat as finished
        #    continue

        try:
            demo_prompt = get_demo_prompt(task)
            text = item['prompt']
            response = item['model_response']
            if response is None:
                print(f"Warning: None response for item {item.get('index', 'unknown')}")
                response = ""
            
            full_prompt = create_test_prompt(demo_prompt, text, response)
            tasks_to_run.append((item, full_prompt))
        except Exception as e:
            print(f"Skipping item due to error: {e}")

    extracted_items = []
    if tasks_to_run:
        print(f"Extracting answers for {len(tasks_to_run)} items using {api_model} with {workers} workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {
                executor.submit(api_call_worker, api_model, full_prompt): original_item
                for original_item, full_prompt in tasks_to_run
            }
            
            results_iterator = tqdm(
                concurrent.futures.as_completed(future_to_item), 
                total=len(future_to_item), 
                desc="LLM Extraction"
            )
            
            for future in results_iterator:
                original_item = future_to_item[future]
                try:
                    extracted_answer = future.result()
                    original_item['extracted_answer'] = extracted_answer
                except Exception as exc:
                    print(f"\n[Error] Task failed: {exc}")
                    original_item['extracted_answer'] = ""
                extracted_items.append(original_item)
    
    # Merge data
    all_processed_data = extracted_items + ocr_items
    
    # Save Intermediate Result (input_extract_answer.json)
    base_name, ext = os.path.splitext(input_json_path)
    extract_output_path = f"{base_name}_extract_answer{ext}"
    
    if save_intermediate:
        with open(extract_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_processed_data, f, ensure_ascii=False, indent=2)
        print(f"Intermediate extracted answers saved to: {extract_output_path}")

    # ---------------- Step 2: Calculate Scores ----------------
    print("Calculating scores...")
    
    task_correct = {t: [] for t in TASKS}
    total_correct = 0
    valid_count = 0

    for item in tqdm(all_processed_data, desc="Scoring"):
        task = item.get('task')
        correctness = 0
        
        # Handle Non-OCR Items
        if task != 'ocr':
            # Ensure extracted_answer exists
            if 'extracted_answer' not in item or item['extracted_answer'] is None:
                item['extracted_answer'] = ''
        
        # --- Logic from calculate_score.py ---
        try:
            if task in SAFE_TASKS:
                extracted_answer = item['extracted_answer'].replace('*', '').strip()
                answer = str(item['answer'])
                string_type = classify_string(answer)
                
                if string_type == 1:
                    correctness = judge_choice(extracted_answer, answer, item)
                elif string_type == 2: # int
                    correctness = judge_int(extracted_answer, answer)
                elif string_type == 3: # float
                    correctness = judge_float(extracted_answer, answer)
                elif string_type == 4:
                    print(f"Error: Unknown answer type for question_id: {item.get('question_id')}")

            elif task == 'ocr':
                answer = str(item['answer'])
                response = item.get('model_response', '')
                if isinstance(response, list):
                    response = response[0] if response else ""
                
                # Hardcoded fixes from original script
                if 'image_1' in item:
                    if '60.jpg' in item['image_1']: answer = 'mobi'
                    if '62.jpg' in item['image_1']: answer = 'aires'
                
                if answer in str(response):
                    correctness = 1

            elif task == 'word_search':
                answer = str(item['answer'])
                extracted_answer = item['extracted_answer'].replace('*', '').strip()
                if classify_string(answer) == 2:
                    correctness = judge_int(extracted_answer, answer)
                else:
                    try:
                        a1, a2 = extract_two_numbers(answer)
                        r1, r2 = extract_two_numbers(extracted_answer)
                        if a1 == r1 and a2 == r2:
                            correctness = 1
                    except:
                        pass

            elif task == 'spot_difference':
                answer = str(item['answer'])
                extracted_answer = item['extracted_answer'].replace('*', '').strip()
                if classify_string(answer) == 2:
                    correctness = judge_int(extracted_answer, answer)
                else:
                    try:
                        list_answer = extract_consecutive_integers(answer)
                        list_response = extract_consecutive_integers(extracted_answer)
                        correctness = list_iou(list_response, list_answer)
                    except:
                        pass

            elif task == 'jigsaw':
                answer = str(item['answer'])
                extracted_answer = item['extracted_answer'].replace('*', '').strip()
                try:
                    meta_key = 'metadata' if 'metadata' in item else 'meta_data'
                    difficulty = item[meta_key]['difficulty']
                    m_re = extract_consecutive_n_squared(extracted_answer, difficulty)
                    a_re = extract_consecutive_n_squared(answer, difficulty)
                    correctness = compare(a_re, m_re)
                except Exception:
                    pass
        except Exception as e:
            # print(f"Scoring error for item {item.get('index')}: {e}")
            pass

        # Update item and stats
        item['true_false'] = correctness
        if task in task_correct:
            task_correct[task].append(correctness)
        total_correct += correctness
        valid_count += 1

    # ---------------- Step 3: Save Final Results ----------------
    final_output_path = f"{base_name}_final{ext}"
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nFinal results saved to: {final_output_path}")

    # ---------------- Step 4: Print Report ----------------
    print("\n" + "="*40)
    print("Evaluation Report")
    print("="*40)
    
    aggregated_score = 0
    tasks_reported = 0

    for task in TASKS:
        scores = task_correct[task]
        if len(scores) == 0:
            continue
        avg_score = sum(scores) / len(scores)
        print(f"{task:<20} (n={len(scores):<3}): {avg_score:.4f}")
        aggregated_score += avg_score
        tasks_reported += 1
    
    if tasks_reported > 0:
        print("-" * 40)
        print(f"{'Average Score':<20} : {aggregated_score / tasks_reported:.4f}")
    print("="*40 + "\n")
    
    return all_processed_data

# ================= CLI Entry Point =================

def calculate_score(json_path, workers):
    evaluate_tir_json(
    input_json_path=json_path,
    workers=workers
)

def main():
    parser = argparse.ArgumentParser(description="TIR-Bench End-to-End Evaluation")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the model response JSON file')
    parser.add_argument('--api_model', type=str, default='gpt-4o-2024-11-20', help='LLM used for answer extraction')
    parser.add_argument('--workers', type=int, default=8, help='Number of threads for extraction')
    parser.add_argument('--no_intermediate', action='store_true', help='Do not save the intermediate extraction JSON')

    args = parser.parse_args()

    evaluate_tir_json(
        input_json_path=args.input_path,
        api_model=args.api_model,
        workers=args.workers,
        save_intermediate=(not args.no_intermediate)
    )

if __name__ == "__main__":
    main()