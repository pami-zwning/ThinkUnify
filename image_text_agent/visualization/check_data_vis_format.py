import os
import json
import warnings
import re
from multiprocessing import Pool
from aoss_client.client import Client

warnings.filterwarnings('ignore')

# AOSS 配置
CONF_PATH = '/mnt/afs1/tianhao2/aoss.conf'
try:
    client = Client(CONF_PATH)
except:
    client = None

def check_format(args):
    """
    args: (data_item, verbose, is_sft_mode, root_path)
    """
    data, verbose, is_sft_mode, root_path = args
    try:
        # 1. 字段存在性检查 (新格式: conversations 和 image)
        if 'conversations' not in data or 'image' not in data:
            if verbose: print("[Error] 缺失 conversations 或 image 字段")
            return False

        conversations = data['conversations']
        image_files = data['image']
        
        # 2. 对话轮数与角色检查 (忽略 system)
        valid_msgs = [m for m in conversations if m['from'] != 'system']
        if len(valid_msgs) < 2 or len(valid_msgs) % 2 != 0:
            if verbose: print(f"[Error] 对话轮数错误: {len(valid_msgs)}")
            return False

        for i, msg in enumerate(valid_msgs):
            role = msg['from']
            content = msg.get('value', '').strip()
            # 角色映射: human -> user, gpt -> assistant
            expected_role = 'human' if i % 2 == 0 else 'gpt'
            
            if role != expected_role:
                if verbose: print(f"[Error] 角色顺序错误: 第 {i} 轮应该是 {expected_role}，实际是 {role}")
                return False

            if is_sft_mode:
                # --- Human 角色检查 ---
                if role == 'human':
                    if i == 0:
                        if '<image>' not in content:
                            if verbose: print("[Error] 首轮 human 必须包含 <image>")
                            return False
                    else:
                        # 非首轮 human 必须符合 <tool_response> 格式
                        pattern = r"^<tool_response>\n([\s\S]*?)\n</tool_response>$"
                        if not re.match(pattern, content):
                            if verbose: print(f"[Error] 第 {i} 轮 human 格式不符合 <tool_response> 模板")
                            return False

                # --- GPT 角色检查 ---
                elif role == 'gpt':
                    think_pattern = r"^<thinking>([\s\S]*?)</thinking>\s*"
                    tool_call_pattern = r"<tool_call>\n([\s\S]*?)\n</tool_call>$"
                    answer_pattern = r"<answer>([\s\S]*?)</answer>$"

                    if not re.match(think_pattern, content):
                        if verbose: print(f"[Error] GPT 第 {i} 轮缺失或格式错误的 <thinking> 模块")
                        return False
                    
                    is_tool = re.search(tool_call_pattern, content)
                    is_answer = re.search(answer_pattern, content)

                    if is_tool:
                        try:
                            tool_json_str = is_tool.group(1).strip()
                            tool_dict = json.loads(tool_json_str)
                            
                            # 【新增】检查 image_index
                            arguments = tool_dict.get('arguments', {})
                            if 'image_index' not in arguments:
                                if verbose: print(f"[Error] GPT 第 {i} 轮 tool_call 缺失 image_index 参数")
                                return False
                        except json.JSONDecodeError:
                            if verbose: print(f"[Error] GPT 第 {i} 轮 <tool_call> 内部 JSON 解析失败")
                            return False
                    elif not is_answer:
                        if verbose: print(f"[Error] GPT 第 {i} 轮结尾必须是 <tool_call> 或 <answer>")
                        return False

        # 3. 路径校验 (结合 root_path)
        for img_path in image_files:
            full_path = os.path.join(root_path, img_path) if root_path else img_path
            
            if full_path.startswith('s3://'):
                if client and not client.contains(full_path):
                    if verbose: print(f"[Error] S3 图片不存在: {full_path}")
                    return False
            elif not os.path.exists(full_path):
                if verbose: print(f"[Error] 本地图片不存在: {full_path}")
                return False

        # 4. 图片标签数量校验
        total_image_tags = sum(msg.get('value', '').count('<image>') for msg in conversations)
        if total_image_tags != len(image_files):
            if verbose: print(f"[Error] 图片标签数 ({total_image_tags}) 与实际图片数 ({len(image_files)}) 不符")
            return False

        return True

    except Exception as e:
        if verbose: print(f"[Error] 运行异常: {e}")
        return False

def process_config(config_path):
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    print(f"开始检查，共 {len(config_data)} 个数据集任务\n")
    
    for task_name, info in config_data.items():
        file_path = info.get('annotation')
        root_path = info.get('root', "")
        
        if not os.path.exists(file_path):
            print(f"[{task_name}] 跳过不存在的标注文件: {file_path}")
            continue
            
        is_sft_mode = "sft" in file_path.lower() or "sft" in task_name.lower()
        
        dataset = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
        
        # 多进程处理 (传入 root_path)
        with Pool(processes=4) as pool:
            results = pool.map(check_format, [(d, False, is_sft_mode, root_path) for d in dataset])
            
        valid_count = sum(results)
        invalid_count = len(results) - valid_count
        
        print(f"任务名称: {task_name}")
        print(f"标注路径: {file_path}")
        print(f"验证模式: {'[严格 SFT]' if is_sft_mode else '[基础校验]'}")
        print(f"统计结果: 总计 {len(results)}, 通过 {valid_count}, 失败 {invalid_count}")

        if invalid_count > 0:
            first_fail_idx = results.index(False)
            print("-" * 20 + " 错误数据样例排查 " + "-" * 20)
            check_format((dataset[first_fail_idx], True, is_sft_mode, root_path))
            print(f"样例索引: {first_fail_idx}")
            print(f"样例内容:\n{json.dumps(dataset[first_fail_idx], indent=2, ensure_ascii=False)}")
            print("-" * 50)
        print("\n")

if __name__ == "__main__":
    # 假设你的配置 JSON 保存为 dataset_config.json
    CONFIG_JSON = '/mnt/afs/shangziyi/Data/WorkSpace/training_data/SFT_TIR_math_map_ruler_clock_reflection_251231_meta.json' 
    process_config(CONFIG_JSON)