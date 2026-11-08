import os
import subprocess
import sys

# ================= 1. 全局配置区域 (修改这里即可) =================

# --- A. 基础路径与环境 ---
PROJECT_ROOT = "/mnt/afs/mashengnan/o3like/Test_Code/image_text_agent"
CONDA_ACTIVATE_SCRIPT = "/mnt/afs/mashengnan/miniconda3/bin/activate"
VSP_ENV_PATH = "/mnt/afs/mashengnan/.conda/envs/vsp"
TOOLS_DIR = os.path.join(PROJECT_ROOT, "tools")

# --- B. 评测任务选择 (新增) ---
# 可选值: "tir", "real", "mira", "mmmu", "vsp"
#在此列表中保留你想跑的任务即可
BENCHMARKS_TO_RUN = [
    "tir","vsp"
]

# --- C. 统一评测参数 ---
# 1. 统一的模型名称
MODEL_NAME = "RL_step190_2"

# 2. 统一的并发数 (控制显存压力)
WORKERS = 10

# 3. 统一的结果文件夹名称
RESULT_DIR_NAME = "Results"
MAX_TURNS = 8

# 4. 全局工具调用开关 (True: 开启, False: 关闭)
ENABLE_TOOLS = True 

# --- D. IP 资源池 ---
IP_LIST_MAIN = [
    "10.119.18.207", "10.119.18.201", "10.119.18.177","10.119.18.167"
]
#"10.119.16.114", "10.119.16.118", "10.119.16.104"
#"10.119.16.155", "10.119.16.143", "10.119.16.133", "10.119.16.125"
# ================= 2. 环境变量设置 =================
env = os.environ.copy()
env["PYTHONPATH"] = f"{TOOLS_DIR}:{env.get('PYTHONPATH', '')}"

# ================= 3. 任务具体配置 =================

# 任务 1: TIR-Bench
TIR_CONFIG = {
    "cwd": os.path.join(PROJECT_ROOT, "evaluation/TIR-Bench"),
    "result_path": f"./{RESULT_DIR_NAME}/", 
    "api_name": "gpt-4o-2024-11-20"
}

# 任务 2: RealUnify
REAL_CONFIG = {
    "cwd": os.path.join(PROJECT_ROOT, "evaluation/RealUnify"),
    "script_path": "eval.py",
    "result_path": f"./{RESULT_DIR_NAME}" 
}

# 任务 3: MIRA
MIRA_CONFIG = {
    "cwd": os.path.join(PROJECT_ROOT, "evaluation/MIRA"),
    "benchmark_dir": "/mnt/afs/share_data/mashengnan/o3like/visual_reasoning/MIRA",
    "result_path": f"./{RESULT_DIR_NAME}" 
}

# 任务 4: Uni-MMMU
UNIMMMU_CONFIG = {
    "cwd": os.path.join(PROJECT_ROOT, "evaluation/Uni-MMMU"),
    "dataset_root": "/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/Uni-MMMU",
    "result_path": f"./{RESULT_DIR_NAME}"
}

# 任务 5: VSP
VSP_CONFIG = {
    "cwd": os.path.join(PROJECT_ROOT, "evaluation/VSP"),
    "result_path": f"./{RESULT_DIR_NAME}/"
}

# ================= 4. 工具函数 =================

def run_cmd(cmd_list, cwd, description):
    """通用命令执行函数"""
    print(f"\n" + "-"*60)
    print(f"[Step] {description}")
    print(f"[Path] {cwd}")
    print(f"[Cmd]  {' '.join(cmd_list)}")
    print("-" * 60)
    try:
        subprocess.run(cmd_list, cwd=cwd, env=env, check=True)
        print(f"✅ [Success] {description}")
    except subprocess.CalledProcessError as e:
        print(f"❌ [Error] {description} failed (Exit Code: {e.returncode})")
        sys.exit(1)

# ================= 5. 主流程 =================

def main():
    print(f"🚀 Starting Pipeline for Model: {MODEL_NAME}")
    print(f"🔧 Workers: {WORKERS} | Result Dir: {RESULT_DIR_NAME}")
    print(f"🔧 Use Tools: {ENABLE_TOOLS}")
    
    # 转换为小写并去重，防止大小写输入错误
    tasks_to_run = set(t.lower() for t in BENCHMARKS_TO_RUN)
    print(f"📋 Scheduled Tasks: {list(tasks_to_run)}\n")

    # 根据全局开关生成参数列表
    tool_args = ["--use_tools"] if ENABLE_TOOLS else []

    # ------------------------------------------------------
    # Stage 1: TIR-Bench
    # ------------------------------------------------------
    if "tir" in tasks_to_run:
        print("\n" + "="*20 + " Stage 1: TIR-Bench Eval" + "="*20)
        
        run_cmd([
            "python", "qwen3vl_eval.py",
            "--model_name", MODEL_NAME,
            "--ip_list", *IP_LIST_MAIN,
            "--result_path", TIR_CONFIG["result_path"],
            "--workers", str(WORKERS),
            "--max_turns", str(MAX_TURNS),
            "--save_interim",
            *tool_args
        ], cwd=TIR_CONFIG["cwd"], description="TIR-Bench: Evaluation")

        run_cmd([
            "python", "extract_answer.py",
            "--model_name", MODEL_NAME,
            "--api_name", TIR_CONFIG["api_name"],
            "--input_path", TIR_CONFIG["result_path"],
            "--workers", str(WORKERS)
        ], cwd=TIR_CONFIG["cwd"], description="TIR-Bench: Extract Answer")

        print("\n" + "="*20 + " Stage 1: TIR-Bench Score" + "="*20)
        run_cmd([
            "python", "calculate_score.py",
            "--model_name", MODEL_NAME,
            "--input_path", TIR_CONFIG["result_path"]
        ], cwd=TIR_CONFIG["cwd"], description="TIR-Bench: Calculate Score")
    else:
        print("⏭️  Skipping TIR-Bench")

    # ------------------------------------------------------
    # Stage 2: RealUnify
    # ------------------------------------------------------
    if "real" in tasks_to_run:
        print("\n" + "="*20 + " Stage 2: RealUnify Eval" + "="*20)
        
        run_cmd([
            "python", REAL_CONFIG["script_path"],
            "--model_name", MODEL_NAME,
            "--ip_list", *IP_LIST_MAIN,
            "--result_path", REAL_CONFIG["result_path"],
            "--workers", str(WORKERS),
            "--max_turns", str(MAX_TURNS),
            "--save_interim",
            *tool_args
        ], cwd=REAL_CONFIG["cwd"], description="RealUnify: Evaluation")

        real_output_file = os.path.join(REAL_CONFIG["result_path"], "Jsons", f"{MODEL_NAME}.json")

        print("\n" + "="*20 + " Stage 2: RealUnify Score " + "="*20)
        run_cmd([
            "python", "score.py",
            "--input_file", real_output_file
        ], cwd=REAL_CONFIG["cwd"], description="RealUnify: Understanding Score")
    else:
        print("⏭️  Skipping RealUnify")


    # ------------------------------------------------------
    # Stage 3: MIRA
    # ------------------------------------------------------
    if "mira" in tasks_to_run:
        print("\n" + "="*20 + " Stage 3: MIRA Eval" + "="*20)

        mira_full_output_dir = os.path.join(MIRA_CONFIG["result_path"], MODEL_NAME)

        run_cmd([
            "python", "eval.py",
            "--benchmark_dir", MIRA_CONFIG["benchmark_dir"],
            "--model_name", MODEL_NAME,
            "--output_dir", mira_full_output_dir,
            "--ip_list", *IP_LIST_MAIN,
            "--result_path", MIRA_CONFIG["result_path"],
            "--workers", str(WORKERS),
            "--max_turns", str(MAX_TURNS),
            "--save_interim",
            *tool_args
        ], cwd=MIRA_CONFIG["cwd"], description="MIRA: Evaluation")

        print("\n" + "="*20 + " Stage 3: MIRA Score" + "="*20)
        run_cmd([
            "python", "score.py",
            "-r", mira_full_output_dir
        ], cwd=MIRA_CONFIG["cwd"], description="MIRA: Accuracy")
    else:
        print("⏭️  Skipping MIRA")

    # ------------------------------------------------------
    # Stage 4: Uni-MMMU
    # ------------------------------------------------------
    if "mmmu" in tasks_to_run:
        print("\n" + "="*20 + " Stage 4: Uni-MMMU Eval" + "="*20)

        unimmmu_result_dir = os.path.join(UNIMMMU_CONFIG["result_path"], MODEL_NAME)
        
        if ENABLE_TOOLS:
            print("👉 Running in Tool-Call Mode")
            unimmmu_script = "qwen3vl_tool_call.py"
            unimmmu_mode = "tool_call"
            unimmmu_extra_args = ["--save_images"]
        else:
            print("👉 Running in Direct Mode")
            unimmmu_script = "qwen3vl_direct.py"
            unimmmu_mode = "direct"
            unimmmu_extra_args = []

        run_cmd([
            "python", unimmmu_script,
            "--result_dir", unimmmu_result_dir,
            "--ip_list", *IP_LIST_MAIN,
            "--workers", str(WORKERS),
            "--max_turns", str(MAX_TURNS),
            *unimmmu_extra_args
        ], cwd=UNIMMMU_CONFIG["cwd"], description=f"Uni-MMMU: Inference ({unimmmu_mode})")

        print("\n" + "="*20 + " Stage 4: Uni-MMMU Score" + "="*20)
        run_cmd([
            "python", "eval_unimmmu.py",
            "--run_root", unimmmu_result_dir,
            "--dataset_root", UNIMMMU_CONFIG["dataset_root"],
            "--modes", unimmmu_mode,
            "--workers", str(WORKERS),
            "--max_retries", "2"
        ], cwd=UNIMMMU_CONFIG["cwd"], description="Uni-MMMU: Calculate Score")
    else:
        print("⏭️  Skipping Uni-MMMU")
    
    # ------------------------------------------------------
    # Stage 5: VSP
    # ------------------------------------------------------
    if "vsp" in tasks_to_run:
        print("\n" + "="*20 + " Stage 5: VSP Eval" + "="*20)

        # 5.1 Inference
        run_cmd([
            "python", "qwen3vl_eval.py",
            "--model_name", MODEL_NAME,
            "--ip_list", *IP_LIST_MAIN,
            "--result_path", VSP_CONFIG["result_path"],
            "--workers", str(WORKERS),
            "--max_turns", str(MAX_TURNS),
            "--save_interim",
            *tool_args
        ], cwd=VSP_CONFIG["cwd"], description="VSP: Inference")

        # 5.2 Score (Using Conda Activate)
        print("\n" + "="*20 + " Stage 5: VSP Score" + "="*20)
        score_cmd_str = f"source {CONDA_ACTIVATE_SCRIPT} {VSP_ENV_PATH} && python eval.py"
        
        run_cmd([
            "/bin/bash", "-c", score_cmd_str
        ], cwd=VSP_CONFIG["cwd"], description="VSP: Calculate Score (Conda Activated)")
    else:
        print("⏭️  Skipping VSP")


    print("\n🎉 Selected Evaluations Finished Successfully!")

if __name__ == "__main__":
    main()