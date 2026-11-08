import json
import os
import hashlib  # 1. 引入 hashlib 库
from PIL import Image

# ==========================================
# 配置区域
# ==========================================

# 输入和输出路径
DATA_PATH = '/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/TIR-Bench/TIR-Bench.json'
SAVE_PATH = '/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/TIR-Bench/TIR-Bench_data.jsonl'
IMAGE_ROOT = "/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/TIR-Bench/"

def process_data():
    # 1. 读取原始数据
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    processed_data = []

    print(f"Total samples to process: {len(datas)}")

    # 2. 使用 enumerate 获取索引，用于确保 hash 的绝对唯一性
    for idx, item in enumerate(datas):
        # ==========================================
        # 原始逻辑：Prompt 预处理
        # ==========================================
        current_prompt = item['prompt']
        
        # 处理 OCR 任务
        if item.get('task') == 'ocr':
            current_prompt = current_prompt + "Please answer in a case-sensitive manner."
        
        # 处理 Math 任务的特定 ID (添加分辨率信息)
        if item.get('task') == 'math':
            target_ids = [
                35, 217, 240, 245, 256, 330, 377, 413, 416, 443,
                493, 551, 566, 589, 604, 608, 671, 716, 767, 809,
                887, 923, 941, 977, 1012, 1013, 1180, 1187, 1188,
                1000, 1097
            ]
            if item.get('id') in target_ids:
                img_path = os.path.join(IMAGE_ROOT, item["image_1"])
                if os.path.exists(img_path):
                    try:
                        with Image.open(img_path) as img:
                            w, h = img.size
                        current_prompt = "Image resolution: {}x{}\n".format(w, h) + current_prompt
                    except Exception as e:
                        print(f"Warning: Could not read image size for {img_path}: {e}")

        # ==========================================
        # 构建新的 VQA 格式
        # ==========================================
        
        img_input_rel = item["image_1"]
        img_input_abs = os.path.abspath(os.path.join(IMAGE_ROOT, img_input_rel))
        
        content_list = [
            { "type": "text", "text": current_prompt },
        ]
        if os.path.exists(img_input_abs):
            content_list.append({ "type": "image_url", "image_url": { "url": img_input_abs }})

        # 处理 image_2
        if item.get("image_2"):
            img2_abs = os.path.abspath(os.path.join(IMAGE_ROOT, item["image_2"]))
            if os.path.exists(img2_abs):
                content_list.append({ "type": "image_url", "image_url": { "url": img2_abs }})

        vqa_structure = {
            "role": "user",
            "content": content_list
        }

        # ==========================================
        # 保存结果 & 生成 hash_uid
        # ==========================================
        new_item = item.copy()
        
        # 3. 生成 hash_uid
        # 使用 task + id + index 组合，确保唯一
        unique_str = f"{item.get('task', 'unknown')}_{item.get('id', 'unknown')}_{idx}"
        hash_uid = hashlib.md5(unique_str.encode("utf-8")).hexdigest()
        
        new_item["hash_uid"] = hash_uid
        new_item["messages"] = [vqa_structure]

        processed_data.append(new_item)

    # 写入文件
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False)+'\n')

    print(f"Processing complete. Saved to {SAVE_PATH}")

if __name__ == "__main__":
    process_data()