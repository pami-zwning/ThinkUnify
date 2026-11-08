import json
import os
import re
import hashlib  # 1. 引入 hashlib
from tqdm import tqdm

# ==========================================
# 配置区域
# ==========================================

# 原始数据路径
DATA_PATH = "/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/RealUnify/GEU_step.json"
# 处理后的保存路径
SAVE_PATH = "/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/RealUnify/GEU_step_processed.jsonl"

# 路径替换配置
DATASET_ROOT = "/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/RealUnify"
OLD_PATH_PREFIX = "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Data/RealUnify"

def refine_prompt(prompt_text):
    """
    清洗 Prompt：
    1. 如果存在 "Question:"，则去掉其之前的所有内容。
    2. 去掉结尾处的 "The best answer is:"。
    """
    if not prompt_text:
        return ""

    # 步骤 1: 寻找 Question: 并截取其后的内容
    if "Question:" in prompt_text:
        prompt_text = prompt_text.split("Question:", 1)[-1].strip()
    
    # 步骤 2: 去掉结尾的 "The best answer is:"
    prompt_text = re.sub(r"The best answer is:\s*$", "", prompt_text, flags=re.IGNORECASE).strip()
    
    return prompt_text

def process_data():
    # 1. 读取原始数据
    if not os.path.exists(DATA_PATH):
        print(f"Error: Input file not found: {DATA_PATH}")
        return

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    print(f"Total samples to process: {len(data_list)}")
    
    processed_data = []

    # 2. 遍历处理 (使用 enumerate 获取 idx)
    for idx, item in enumerate(tqdm(data_list, desc="Processing Data")):
        # 复制对象，避免修改原始引用
        new_item = item.copy()

        # ------------------------------------------
        # Step 1: 路径修正 (Image Manipulation)
        # ------------------------------------------
        raw_img_path = new_item.get('input_image', '')
        
        # 执行路径替换逻辑
        if raw_img_path:
            img_path = raw_img_path.replace(OLD_PATH_PREFIX, DATASET_ROOT)
            # 转为绝对路径
            img_abs_path = os.path.abspath(img_path)
            # 更新 item 中的路径
            new_item['input_image'] = img_abs_path
        else:
            img_abs_path = None

        # ------------------------------------------
        # Step 2: Prompt 清洗
        # ------------------------------------------
        raw_prompt = new_item.get("evaluation_prompt", "")
        clean_prompt = refine_prompt(raw_prompt)
        
        new_item['processed_prompt'] = clean_prompt

        # ------------------------------------------
        # Step 3: 构建 Messages (VQA 格式)
        # ------------------------------------------
        content_list = [
            { "type": "text", "text": clean_prompt }
        ]

        # 添加图片 (如果存在)
        if img_abs_path and os.path.exists(img_abs_path):
            content_list.append({ 
                "type": "image_url", 
                "image_url": { "url": img_abs_path }
            })
        elif img_abs_path:
            print(f"Warning: Image path does not exist: {img_abs_path}")

        vqa_structure = {
            "role": "user",
            "content": content_list
        }

        new_item["messages"] = [vqa_structure]

        # ------------------------------------------
        # Step 4: 生成 hash_uid
        # ------------------------------------------
        # 使用 id + idx 组合，确保唯一性。如果数据中没有id，就用 unknown
        unique_str = f"{item.get('id', 'unknown')}_{idx}"
        hash_uid = hashlib.md5(unique_str.encode("utf-8")).hexdigest()
        new_item["hash_uid"] = hash_uid

        processed_data.append(new_item)

    # 3. 保存结果到 JSONL
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        for p_item in processed_data:
            f.write(json.dumps(p_item, ensure_ascii=False) + '\n')

    print(f"Processing complete. Saved to {SAVE_PATH}")

if __name__ == "__main__":
    process_data()