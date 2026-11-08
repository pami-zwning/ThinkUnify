import gym
import os
import re
import json

# 默认地图根目录配置
DEFAULT_MAP_ROOT = "/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/VSP/maps"

def generate_map_from_table(table_list):
    """
    根据输入的表格字符串列表生成 nxn 的二维地图列表。
    """
    if not table_list or len(table_list) < 2:
        return []

    n = len(table_list) - 1
    nxn_map = []

    for i in range(1, n + 1):
        row_str = table_list[i]
        cells = [cell.strip() for cell in row_str.strip().strip('|').split('|')]
        map_row = ""
        
        for j in range(1, n + 1):
            if j < len(cells):
                cell_content = cells[j]
            else:
                cell_content = '_' # 防止越界，默认填充

            if cell_content == '@':
                map_row += 'S'
            elif cell_content == '#':
                map_row += 'H'
            elif cell_content == '*':
                map_row += 'G'
            elif cell_content == '_':
                map_row += 'F'
            else:
                map_row += 'F'
        
        nxn_map.append(map_row)

    return nxn_map

def extract_answer(response):
    """
    从模型回复中提取 <answer> 标签内的内容
    """
    if not response:
        return ''
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.S)
    if not match:
        return ''

    result = match.group(1)
    result = re.sub(r'[\s,]+', '', result) 
    return result

def evaluate_json_data(json_data, map_root_dir=DEFAULT_MAP_ROOT):
    """
    核心评测函数。
    
    Args:
        json_data (list): 包含评测数据的列表 (通常由 json.load 得到)
        map_root_dir (str): 地图文件的根目录路径
        
    Returns:
        dict: 包含各等级统计信息的字典
    """
    # 结构: { level_id: {'total': 0, 'correct': 0} }
    stats = {}

    for entry in json_data:
        # 1. 提取元数据
        level = entry.get('level')
        test_id = entry.get('curr_id')
        model_response = entry.get('model_response', '')

        if level is None or test_id is None:
            continue

        # 初始化该等级的统计
        if level not in stats:
            stats[level] = {'total': 0, 'correct': 0}

        # 2. 读取地图文件
        # 构建路径: .../maps/levelX/table/Y.txt
        check_map_path = os.path.join(map_root_dir, f"level{level}", "table", f"{test_id}.txt")
        
        try:
            with open(check_map_path, 'r') as f:
                contents = f.read()
                rows = contents.strip().split('\n')
        except FileNotFoundError:
            print(f"[Warn] Map file not found: Level {level}, ID {test_id} path: {check_map_path}")
            continue

        # 3. 提取并处理模型的答案
        raw_answer = extract_answer(model_response)
        answer_moves = list(raw_answer)
        answer_moves = [item.upper() for item in answer_moves]

        # 4. 生成环境
        map_rows = rows[:level+1]
        desc = generate_map_from_table(map_rows)
        
        if not desc:
            print(f"[Warn] Empty map generated: Level {level}, ID {test_id}")
            stats[level]['total'] += 1
            continue

        try:
            # 这里的 render_mode 设为 None 以加快速度，如需调试可设为 'human'
            env = gym.make('FrozenLake-v1', desc=desc, is_slippery=False)
            env.reset(seed=42)
        except Exception as e:
            print(f"[Error] Gym Init Error Level {level} ID {test_id}: {e}")
            stats[level]['total'] += 1
            continue

        # 5. 执行动作并评测
        action_number_map = {'L':0, 'D':1, 'R':2, 'U':3}
        is_success = False

        for move in answer_moves:
            if move not in action_number_map:
                continue 
                
            observation, reward, terminated, truncated, info = env.step(action_number_map[move])

            if terminated or truncated:
                if reward > 0:
                    is_success = True
                break
        
        env.close()

        # 6. 更新统计
        stats[level]['total'] += 1
        if is_success:
            stats[level]['correct'] += 1
            
    return stats

def print_report(stats):
    """
    格式化打印评测结果，并返回全局准确率。
    """
    print("\n" + "="*30)
    print("Evaluation Results")
    print("="*30)

    sorted_levels = sorted(stats.keys())
    total_all_levels = 0
    correct_all_levels = 0

    for level in sorted_levels:
        total = stats[level]['total']
        correct = stats[level]['correct']
        acc = (correct / total * 100) if total > 0 else 0.0
        
        print(f"Level {level}: Total: {total}, Correct: {correct}, Accuracy: {acc:.2f}%")
        
        total_all_levels += total
        correct_all_levels += correct

    print("-" * 30)
    global_acc = (correct_all_levels / total_all_levels * 100) if total_all_levels > 0 else 0.0
    print(f"Overall: Total: {total_all_levels}, Correct: {correct_all_levels}, Accuracy: {global_acc:.2f}%")
    print("="*30)
    
    return global_acc


def calculate_score(json_path):
    if os.path.exists(json_path):
        print(f"Loading data from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # 调用封装好的函数
        result_stats = evaluate_json_data(test_data)
        
        # 打印结果
        print_report(result_stats)
    else:
        print(f"File not found: {json_path}")

# ==========================================
# 使用示例 (当直接运行此脚本时执行)
# ==========================================
if __name__ == "__main__":
    # 你的 JSON 文件路径
    JSON_FILE_PATH = '/mnt/afs/mashengnan/o3like/Test_Code/image_text_agent/evaluation/all_benchmark/Results/VSP/test_api.json'
    calculate_score(JSON_FILE_PATH)
    