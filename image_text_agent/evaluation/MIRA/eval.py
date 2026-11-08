import os
import sys
import json
import random
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# --- 1. 环境与依赖设置 ---
# 将 TIR-Bench 加入路径以便 import
TIR_PATH = "/mnt/afs/mashengnan/o3like/Test_Code/image_text_agent/evaluation/TIR-Bench"
if TIR_PATH not in sys.path: sys.path.insert(0, TIR_PATH)

try:
    from qwen3vl_eval import clean_message_history, model_response_func
except ImportError:
    print("❌ Error: 无法导入 qwen3vl_eval，请检查路径。")
    sys.exit(1)

# --- 2. 辅助工具 ---
FILE_LOCKS = {} # 文件写入锁

def get_target_images(task_dir: Path, data_item: dict, scenario: str):
    """查找任务相关的图片路径"""
    imgs = []
    base_name = Path(data_item.get('image_path', '')).stem
    if not base_name: return imgs

    # 1. 找主图 (支持常见格式)
    for ext in ['.png', '.jpg', '.jpeg', '.webp']:
        p = task_dir / "image" / (base_name + ext)
        if p.exists():
            imgs.append(str(p))
            break
    
    # 2. 如果是 Visual CoT，找辅助图
    if scenario == "visual_cot":
        cot_dir = task_dir / "cot"
        # 找 base_name.xxx 和 base_name_xxx.xxx
        candidates = list(cot_dir.glob(f"{base_name}.*")) + list(cot_dir.glob(f"{base_name}_*"))
        imgs.extend(sorted([str(p) for p in candidates if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]))
    
    return imgs

def run_single_task(client_urls, model_label, data_item, task_dir, output_dir, args):
    """单个任务执行逻辑"""
    scenario = "text_cot" # 固定只跑这个场景，如需扩展可传参
    
    # 1. 准备 Client
    base_url = f"http://{random.choice(client_urls)}:8000/v1"
    client = OpenAI(api_key="EMPTY", base_url=base_url)

    try:
        model_name = client.models.list().data[0].id
    except:
        model_name = "default-model"

    # 2. 准备 Prompt 和 图片
    prompt = (f"Question: {data_item.get('question', '')}\n\n"
              "Please first conduct step-by-step reasoning, and then provide the final answer. "
              "The final answer is placed in <answer></answer>.")
    
    img_paths = get_target_images(task_dir, data_item, scenario)

    # 3. 执行推理 (API Model Name 设为 default 即可，反正 vLLM 既然加载了模型通常不挑这个)
    try:
        resp, history, all_imgs = model_response_func(client, model_name, prompt, img_paths, args)
    except Exception as e:
        print(f"[Err] UID {data_item.get('uid')} @ {base_url}: {e}")
        resp, history, all_imgs = f"[ERROR] {str(e)}", [], []

    # 4. 构造结果
    res = {
        "uid": data_item.get('uid'),
        "scenario": scenario,
        "model_name": model_label, # 记录用户指定的 tag
        "response": resp,
        "original_data": data_item
    }
    if args.save_interim:
        res["mid_process"] = clean_message_history(history) if history else []
        res["all_images_list"] = all_imgs

    # 5. 写入文件 (加锁)
    out_file = output_dir / f"{model_label}_{task_dir.name}.jsonl"
    lock = FILE_LOCKS.setdefault(str(out_file), threading.Lock())
    with lock:
        with open(out_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
            f.flush()

# --- 3. 主流程 ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="用于记录结果的文件名标签")
    parser.add_argument("--benchmark_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ip_list", type=str, nargs='+', required=True)
    parser.add_argument("--workers", type=int, default=8)
    # 传递给 evaluation 内部使用的参数
    parser.add_argument('--save_interim', action='store_true')
    parser.add_argument('--result_path', type=str, default="", help="用于存放生成图片的根目录")
    parser.add_argument('--max_turns', type=int, default=10)
    parser.add_argument('--use_tools', action='store_true', default=False)
    parser.add_argument('--need_summary', action='store_true', default=False, help='是否需要强制给出一个选项。')
    args = parser.parse_args()

    # 初始化目录
    bench_root = Path(args.benchmark_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    
    if args.save_interim and args.result_path:
        args.image_save_dir = os.path.join(args.result_path, "Images", args.model_name)
        os.makedirs(args.image_save_dir, exist_ok=True)

    # A. 扫描已完成任务 (断点续传)
    finished_uids = set()
    for f in out_root.glob(f"{args.model_name}_*.jsonl"):
        try:
            with open(f, 'r', encoding='utf-8') as fr:
                for line in fr:
                    if line.strip(): finished_uids.add(json.loads(line).get('uid'))
        except: pass
    print(f"⏩ 跳过已完成任务数: {len(finished_uids)}")

    # B. 生成待办任务列表
    tasks = []
    print("📂 扫描任务文件...")
    for f in bench_root.rglob('*.jsonl'):
        with open(f, 'r', encoding='utf-8') as fr:
            for line in fr:
                try:
                    item = json.loads(line)
                    if item.get('uid') not in finished_uids:
                        tasks.append((item, f.parent))
                except: continue
    
    if not tasks:
        print("🎉 所有任务已完成。")
        return

    # C. 多线程执行
    print(f"🚀 开始执行 {len(tasks)} 个任务 (并发: {args.workers})...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                run_single_task, 
                args.ip_list, args.model_name, item, task_dir, out_root, args
            ) 
            for item, task_dir in tasks
        ]
        
        # 进度条
        for _ in tqdm(as_completed(futures), total=len(tasks), desc=args.model_name):
            pass

    print("✅ 评测结束。")

if __name__ == "__main__":
    main()