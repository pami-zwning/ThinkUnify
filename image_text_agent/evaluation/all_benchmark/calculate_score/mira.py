import os
import re
import time
import json
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI, APIError

# 任务到领域的映射表
TASK2DOMAIN = {
    'convex_hull':'EG', 'mirror_pattern':'EG', 'localizer':'EG','overlap':'EG','cubes_count':'EG','cubes_missing':'EG',
    'billiards':'PBR','electric_charge':'PBR','mirror_clock':'PBR',
    'trailer_cubes_missing':'ASLP','trailer_cubes_count':'ASLP','puzzle':'ASLP','multi_piece_puzzle':'ASLP','defuse_a_bomb':'ASLP','unfolded_cube':'ASLP',
    'rolling_dice_sum':'CT','rolling_dice_top':'CT','gear_rotation':'CT','paper_airplane':'CT','rolling_dice_two':'CT'
}

class BenchmarkEvaluator:
    def __init__(self, api_key=None, model_name="gpt-4o", use_llm_judge=False):
        """
        初始化评测器
        :param api_key: OpenAI API Key (如果为None则尝试读取环境变量)
        :param model_name: 裁判模型的名称
        :param use_llm_judge: 是否启用 LLM 进行模糊判题
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.use_llm_judge = use_llm_judge
        self.client = None
        
        # 只有在确实需要 LLM 裁判时才初始化客户端
        if self.use_llm_judge:
            if not self.api_key:
                print("[Warning] use_llm_judge is True but no API Key provided. Fallback to Exact Match only.")
                self.use_llm_judge = False
            else:
                try:
                    self.client = OpenAI(api_key=self.api_key)
                except Exception as e:
                    print(f"[Error] Failed to init OpenAI client: {e}")
                    self.use_llm_judge = False

        # 缓存与锁
        self.judge_cache = {}
        self.cache_lock = threading.Lock()

    def _extract_answer(self, response_text: str):
        if not response_text: return None
        match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _normalize_answer(self, text) -> str:
        return str(text).strip().lower()

    def _judge_with_llm(self, question, ground_truth, model_response):
        """调用 LLM 进行裁判"""
        cache_key = (question, str(ground_truth), model_response)
        with self.cache_lock:
            if cache_key in self.judge_cache:
                return self.judge_cache[cache_key]

        judge_prompt = f"""
        You are a strict and precise evaluator. Your task is to determine if the model's final answer is correct based on the ground truth.
        Your evaluation MUST focus EXCLUSIVELY on the final answer provided within the `<answer>` and `</answer>` tags.
        IGNORE all reasoning.

        - Question: "{question}"
        - Ground Truth Answer: "{ground_truth}"
        - Model's Full Response:
        ---
        {model_response}
        ---

        Based on the ground truth, is the answer inside the `<answer>` tag correct?
        Please respond with only one word: 'Correct' or 'Incorrect'.
        """
        
        retries = 3
        for i in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": judge_prompt}],
                    max_tokens=10,
                    temperature=0,
                )
                result = response.choices[0].message.content.strip().lower()
                is_correct = (result == 'correct')
                with self.cache_lock:
                    self.judge_cache[cache_key] = is_correct
                return is_correct
            except Exception as e:
                # print(f"LLM Judge retry {i+1}: {e}")
                time.sleep(1)
        
        return False

    def _process_single_item(self, item):
        """处理单条数据"""
        try:
            # 兼容两种数据格式：
            # 1. 扁平格式 (你刚才提供的 json 样例)
            # 2. 嵌套格式 (原始脚本读取的 jsonl 格式)
            
            # 尝试提取关键字段
            if 'data' in item and isinstance(item['data'], dict):
                # 嵌套格式
                data_body = item['data']
                original_data = data_body.get('original_data', {})
                
                model_alias = item.get('model_alias', 'default_model')
                task_name = item.get('task_name', original_data.get('task', 'unknown_task'))
                
                uid = original_data.get('uid')
                question = original_data.get('question', '')
                ground_truth = original_data.get('answer')
                model_response = data_body.get('response', '')
                scenario = data_body.get('scenario', original_data.get('scenario', 'unknown'))
            else:
                # 扁平格式 (推荐)
                model_alias = item.get('model_alias', 'default_model') # 如果没有，给个默认名
                task_name = item.get('task', 'unknown_task')
                
                uid = item.get('uid')
                question = item.get('question', '')
                ground_truth = item.get('answer')
                model_response = item.get('model_response', '')
                scenario = item.get('scenario', 'unknown')

            if ground_truth is None:
                return None

            # 提取与对比
            predicted_answer = self._extract_answer(model_response)
            is_correct = False
            
            if predicted_answer is not None:
                # 1. 精确匹配
                if self._normalize_answer(predicted_answer) == self._normalize_answer(ground_truth):
                    is_correct = True
                # 2. LLM 裁判 (如果精确匹配失败且开启了LLM)
                elif self.use_llm_judge and self.client:
                    is_correct = self._judge_with_llm(question, ground_truth, model_response)

            key = (model_alias, task_name, scenario)
            return (key, is_correct, task_name, uid)

        except Exception as e:
            # print(f"Error processing item: {e}")
            return None

    def evaluate(self, data_list, workers=10):
        """
        主入口函数
        :param data_list: 包含所有测试数据的列表 (list of dicts)
        :param workers: 并发线程数
        :return: 包含统计结果的字典
        """
        print(f"Starting evaluation on {len(data_list)} items with {workers} workers...")
        
        intermediate_results = defaultdict(list)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self._process_single_item, item) for item in data_list]
            
            for future in tqdm(as_completed(futures), total=len(data_list), desc="Judging"):
                res = future.result()
                if res:
                    key, is_correct, task_name, uid = res
                    intermediate_results[key].append({
                        "correct": is_correct,
                        "uid": uid
                    })

        # 聚合结果 (包含 Gear 任务的特殊逻辑)
        stats = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for key, results_list in intermediate_results.items():
            model, task, scenario = key
            
            # 特殊逻辑：Gear Rotation (齿轮) 任务需要成对正确才算分
            if 'gear_rotation' in task:
                paired_results = defaultdict(dict)
                for res in results_list:
                    uid_str = str(res['uid'])
                    is_c = res['correct']
                    
                    # 尝试解析 1.1, 1.2 这种格式
                    if '.' in uid_str:
                        base, sub = uid_str.split('.', 1)
                        paired_results[base][sub] = is_c
                    else:
                        paired_results[uid_str]['single'] = is_c
                
                # 计算成对正确数
                pair_correct = 0
                for base_res in paired_results.values():
                    # 必须 .1 和 .2 都正确 (或者单个非拆分任务正确)
                    if (base_res.get('1') and base_res.get('2')) or base_res.get('single'):
                        pair_correct += 1
                
                stats[key]['total'] = len(paired_results)
                stats[key]['correct'] = pair_correct
            else:
                # 普通逻辑
                stats[key]['total'] = len(results_list)
                stats[key]['correct'] = sum(1 for r in results_list if r['correct'])

        return dict(stats)

    def print_report(self, stats):
        """
        打印类似于原始脚本的报告
        """
        if not stats:
            print("No results.")
            return

        sorted_keys = sorted(stats.keys())
        
        # 1. 详细任务列表
        print("\n" + "="*90)
        print(f"{'Model':<20} | {'Task':<30} | {'Scenario':<15} | {'Acc':<8} | {'Count'}")
        print("-" * 90)
        
        aggregated_stats = {} # By Domain
        scenario_stats = {}   # By Scenario

        for key in sorted_keys:
            model, task, scenario = key
            correct = stats[key]['correct']
            total = stats[key]['total']
            acc = (correct / total * 100) if total > 0 else 0.0
            
            print(f"{model:<20} | {task:<30} | {scenario:<15} | {acc:>7.2f}% | ({correct}/{total})")

            # 统计聚合数据
            domain = TASK2DOMAIN.get(task, "Unknown")
            
            # Domain 聚合
            if domain not in aggregated_stats: aggregated_stats[domain] = {}
            if scenario not in aggregated_stats[domain]: 
                aggregated_stats[domain][scenario] = {'acc_sum': 0, 'count': 0}
            
            aggregated_stats[domain][scenario]['acc_sum'] += acc
            aggregated_stats[domain][scenario]['count'] += 1

            # Scenario 全局聚合
            if scenario not in scenario_stats: scenario_stats[scenario] = {'acc_sum': 0, 'count': 0}
            scenario_stats[scenario]['acc_sum'] += acc
            scenario_stats[scenario]['count'] += 1

        # 2. Domain 维度报告
        print("\n" + "="*30 + " Domain-wise Accuracy " + "="*30)
        for domain, scenarios in aggregated_stats.items():
            print(f"\n[ Domain: {domain} ]")
            domain_acc_sum = 0
            domain_count = 0
            
            for scen, s_data in scenarios.items():
                avg = s_data['acc_sum'] / s_data['count'] if s_data['count'] else 0
                print(f"  Scenario: {scen:<20} | Avg Acc: {avg:.2f}%")
                domain_acc_sum += s_data['acc_sum']
                domain_count += s_data['count']
            
            overall = domain_acc_sum / domain_count if domain_count else 0
            print(f"  >> Overall {domain}: {overall:.2f}%")

        # 3. Scenario 维度报告
        print("\n" + "="*30 + " Overall by Scenario " + "="*30)
        for scen, s_data in scenario_stats.items():
            overall = s_data['acc_sum'] / s_data['count'] if s_data['count'] else 0
            print(f"  Scenario: {scen:<20} | Overall Acc: {overall:.2f}%")
        print("="*90)


def calculate_score(json_path):
    with open(json_path, 'r') as f:
        data_list = json.load(f)
    # 2. 初始化评测器
    # 如果不想用 LLM 判题，use_llm_judge=False 即可 (速度快，只要格式标准准确率也高)
    evaluator = BenchmarkEvaluator(
        api_key="sk-xxxx",    # 如果环境变量设置了，这里可以不传
        use_llm_judge=False   # 如果你需要跑精确匹配，设为False；如果需要GPT-4o裁判，设为True
    )

    # 3. 运行评测
    stats = evaluator.evaluate(data_list, workers=10)

    # 4. 打印漂亮的结果
    evaluator.print_report(stats)

    # 5. 如果你想拿到具体数值做后续处理:
    # stats 的 key 是 (model_alias, task, scenario)
    # value 是 {'correct': int, 'total': int}
    # 例如获取 billiards 的结果:
    for key, val in stats.items():
        if key[1] == 'billiards':
            print(f"Billiards Acc: {val['correct']/val['total']}")