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

def read_txt(txt_path):
    lines = open(txt_path).read().splitlines()
    return lines

def check_format(args):
    """
    args: (data_item, verbose, is_sft_mode)
    """
    data, verbose, is_sft_mode = args
    try:
        if 'messages' not in data or 'images' not in data:
            if verbose: print("[Error] 缺失 messages 或 images 字段")
            return False

        messages = data['messages']
        image_files = data['images']
        
        valid_msgs = [m for m in messages if m['role'] != 'system']
        if len(valid_msgs) < 2 or len(valid_msgs) % 2 != 0:
            if verbose: print(f"[Error] 对话轮数错误: {len(valid_msgs)}")
            return False

        for i, msg in enumerate(valid_msgs):
            role = msg['role']
            content = msg.get('content', '').strip()
            expected_role = 'user' if i % 2 == 0 else 'assistant'
            
            if role != expected_role:
                if verbose: print(f"[Error] 角色顺序错误: 第 {i} 轮应该是 {expected_role}，实际是 {role}")
                return False

            if is_sft_mode:
                if role == 'user':
                    if i == 0:
                        if '<image>' not in content:
                            if verbose: print("[Error] 首轮 User 必须包含 <image>")
                            return False
                    else:
                        # 非首轮 User 必须符合 tool_response 格式
                        pattern = r"^<tool_response>\n([\s\S]*?)\n</tool_response>$"
                        if not re.match(pattern, content):
                            if verbose: print(f"[Error] 第 {i} 轮 User 格式不符合 <tool_response> 模板")
                            return False

                elif role == 'assistant':
                    # 允许标签之间有或没有换行符 \s*
                    think_pattern = r"^<thinking>([\s\S]*?)</thinking>\s*"
                    tool_call_pattern = r"<tool_call>\n([\s\S]*?)\n</tool_call>$"
                    answer_pattern = r"<answer>([\s\S]*?)</answer>$"

                    if not re.match(think_pattern, content):
                        if verbose: print(f"[Error] Assistant 第 {i} 轮缺失或格式错误的 <thinking> 模块")
                        return False
                    
                    is_tool = re.search(tool_call_pattern, content)
                    is_answer = re.search(answer_pattern, content)

                    if is_tool:
                        try:
                            tool_json_str = is_tool.group(1).strip()
                            tool_dict = json.loads(tool_json_str)
                            if not isinstance(tool_dict, dict): raise ValueError
                        except:
                            if verbose: print(f"[Error] Assistant 第 {i} 轮 <tool_call> 内部 JSON 解析失败")
                            return False
                    elif not is_answer:
                        if verbose: print(f"[Error] Assistant 第 {i} 轮结尾必须是 <tool_call> 或 <answer>")
                        return False

        # 路径和标签校验
        for img_path in image_files:
            if img_path.startswith('s3://'):
                if client and not client.contains(img_path):
                    if verbose: print(f"[Error] S3 图片不存在: {img_path}")
                    return False
            elif not os.path.exists(img_path):
                if verbose: print(f"[Error] 本地图片不存在: {img_path}")
                return False

        total_image_tags = sum(msg.get('content', '').count('<image>') for msg in messages)
        if total_image_tags != len(image_files):
            if verbose: print(f"[Error] 图片标签数 ({total_image_tags}) 与实际图片数 ({len(image_files)}) 不符")
            return False

        return True

    except Exception as e:
        if verbose: print(f"[Error] 运行异常: {e}")
        return False

def process_all_files(file_list):
    print(f"开始检查，共 {len(file_list)} 个文件\n")
    
    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"跳过不存在的文件: {file_path}")
            continue
            
        file_name = os.path.basename(file_path)
        is_sft_mode = "sft" in file_name.lower()
        
        dataset = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): dataset.append(json.loads(line))
        
        # 多进程处理
        with Pool(processes=4) as pool:
            results = pool.map(check_format, [(d, False, is_sft_mode) for d in dataset])
            
        valid_count = sum(results)
        invalid_count = len(results) - valid_count
        
        print(f"文件: {file_name}")
        print(f"模式: {'[严格 SFT]' if is_sft_mode else '[基础校验]'}")
        print(f"结果: 总计 {len(results)}, 通过 {valid_count}, 失败 {invalid_count}")

        if invalid_count > 0:
            # 找到第一个失败的索引
            first_fail_idx = results.index(False)
            print("-" * 20 + " 错误数据样例排查 " + "-" * 20)
            # 重新调用 check_format 开启 verbose 输出具体原因
            check_format((dataset[first_fail_idx], True, is_sft_mode))
            print(f"样例数据内容:\n{json.dumps(dataset[first_fail_idx], indent=2, ensure_ascii=False)}")
            print("-" * 50)
        print("\n")

if __name__ == "__main__":
    file_paths = read_txt('./check_files.txt')
    process_all_files(file_paths)