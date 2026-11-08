import json
import os
import hashlib  # 1. 引入 hashlib
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 配置区域
# ==========================================

# 原始 Benchmark 根目录
# 请确保这是包含各个任务子文件夹的那个目录
BENCHMARK_DIR = "/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/MIRA/data" 
# 处理结果保存路径
SAVE_PATH = "/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/MIRA/processed_data.jsonl"

def get_target_image(task_dir: Path, data_item: dict):
    """
    根据 text_cot 场景查找主图路径
    """
    base_name = Path(data_item.get('image_path', '')).stem
    if not base_name:
        return None

    # 在 task_dir/image 下查找支持的图片格式
    for ext in ['.png', '.jpg', '.jpeg', '.webp']:
        p = task_dir / "image" / (base_name + ext)
        if p.exists():
            return str(p.resolve())  # 返回绝对路径
            
    return None

def process_data():
    bench_root = Path(BENCHMARK_DIR)
    
    if not bench_root.exists():
        print(f"❌ Error: Benchmark directory not found: {BENCHMARK_DIR}")
        return

    # 1. 扫描所有 jsonl 文件
    source_files = list(bench_root.rglob('*.jsonl'))
    print(f"📂 Found {len(source_files)} source files in {BENCHMARK_DIR}")

    processed_data = []
    total_processed = 0

    # 2. 遍历文件处理
    for file_path in tqdm(source_files, desc="Processing Files"):
        # 获取任务文件夹路径
        task_dir = file_path.parent
        # 获取任务名称 (通常是文件夹名)
        task_name = task_dir.name 
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # ==========================================
                    # 核心逻辑：路径提取与 Prompt 构建
                    # ==========================================
                    
                    # 1. 获取图片绝对路径
                    img_abs_path = get_target_image(task_dir, item)
                    
                    # 2. 构建 Prompt
                    question = item.get('question', '')
                    prompt_text = (
                        f"Question: {question}\n\n"
                        "Please first conduct step-by-step reasoning, and then provide the final answer. "
                        "The final answer is placed in <answer></answer>."
                    )

                    # 3. 构建 Messages 结构
                    content_list = [
                        { "type": "text", "text": prompt_text }
                    ]
                    
                    if img_abs_path:
                        content_list.append({
                            "type": "image_url",
                            "image_url": { "url": img_abs_path }
                        })

                    vqa_structure = {
                        "role": "user",
                        "content": content_list
                    }

                    # ==========================================
                    # 保存数据 & 生成 Hash UID
                    # ==========================================
                    new_item = item.copy()
                    
                    # --- 记录 Task 名称 ---
                    new_item["task"] = os.path.basename(os.path.dirname(file_path))
                    
                    # 显式记录处理类型
                    new_item["scenario"] = "text_cot"
                    
                    # 记录处理后的 prompt 以便核对
                    new_item["processed_prompt"] = prompt_text
                    
                    # 记录绝对路径
                    if img_abs_path:
                        new_item["image_abs_path"] = img_abs_path

                    new_item["messages"] = [vqa_structure]
                    
                    # --- 新增：生成唯一的 hash_uid ---
                    # 组合策略：Task名 + 原始ID(uid或id) + 全局计数索引
                    # 这样即使原始数据ID有重复，加上索引也能保证绝对唯一
                    raw_id = item.get('uid', item.get('id', 'unknown'))
                    unique_str = f"{new_item['task']}_{raw_id}_{total_processed}"
                    hash_uid = hashlib.md5(unique_str.encode("utf-8")).hexdigest()
                    
                    new_item["hash_uid"] = hash_uid
                    
                    processed_data.append(new_item)
                    total_processed += 1
                    
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    # 3. 写入结果
    print(f"💾 Saving {len(processed_data)} samples to {SAVE_PATH} ...")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        for p_item in processed_data:
            f.write(json.dumps(p_item, ensure_ascii=False) + '\n')

    print("✅ Processing complete.")

if __name__ == "__main__":
    process_data()