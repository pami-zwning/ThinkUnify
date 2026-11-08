import json
import requests
import os
import copy
import time
import argparse
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from openai import OpenAI
from datetime import timedelta


import torch
import torch.distributed as dist
from evaluation.all_benchmark.calculate_score import mira, realunify, unimmmu, ai2d, chartqa
from evaluation.all_benchmark.calculate_score import hrbench, mathvista, mmmu, mmstar, realworldqa
from evaluation.all_benchmark.calculate_score import vstar, mmvp
from evaluation.all_benchmark.calculate_score import geneval as geneval_score, t2icompbench as t2icomp_score
from evaluation.all_benchmark.infer import read_jsonl_file, read_json_file
from evaluation.VLMEvalKit_Thinkmorph.vlmeval.vlm.ThinkMorph import ThinkMorph

# ==========================================
# 配置区域
# ==========================================

# Azure OpenAI (optional, used only by GPT-as-judge scoring for some benchmarks).
# Provide your own credentials via environment variables; no secrets are shipped.
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "YOUR_AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "YOUR_AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-2024-11-20")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

# Root directory that holds all evaluation data (interleaved-reasoning JSONLs and
# the general-benchmark cache). Override with the EVAL_DATA_ROOT environment
# variable; see README "Data Download" for the expected layout.
EVAL_DATA_ROOT = os.getenv("EVAL_DATA_ROOT", "YOUR_EVAL_DATA_ROOT")

# General understanding benchmark data cache root (JSONL + images produced
# from HuggingFace datasets on first run).
GENERAL_BENCH_CACHE_ROOT = os.getenv(
    "GENERAL_BENCH_CACHE_ROOT",
    os.path.join(EVAL_DATA_ROOT, "general_benchmark_cache"),
)

# 路径配置
ThinkMorph_data = {
    'MIRA': os.path.join(EVAL_DATA_ROOT, 'MIRA', 'processed_data.jsonl'),
    'Unimmmu': os.path.join(EVAL_DATA_ROOT, 'UniMMMU', 'vqa', 'unimmmu_tool_call.jsonl'),
    'Unimmmu_direct': os.path.join(EVAL_DATA_ROOT, 'UniMMMU', 'vqa', 'unimmmu_tool_call.jsonl'),
    'Realunify': os.path.join(EVAL_DATA_ROOT, 'RealUnify', 'GEU_step_processed.jsonl'),
    # General understanding benchmarks (prepared on-demand from HF datasets)
    'AI2D': os.path.join(GENERAL_BENCH_CACHE_ROOT, 'AI2D', 'test.jsonl'),
    'ChartQA': os.path.join(GENERAL_BENCH_CACHE_ROOT, 'ChartQA', 'test.jsonl'),
    'HR-Bench-4K': os.path.join(GENERAL_BENCH_CACHE_ROOT, 'HR-Bench-4K', 'test.jsonl'),
    'HR-Bench-8K': os.path.join(GENERAL_BENCH_CACHE_ROOT, 'HR-Bench-8K', 'test.jsonl'),
    'MathVista': os.path.join(GENERAL_BENCH_CACHE_ROOT, 'MathVista', 'test.jsonl'),
    'MMMU': os.path.join(GENERAL_BENCH_CACHE_ROOT, 'MMMU', 'test.jsonl'),
    'MMStar': os.path.join(GENERAL_BENCH_CACHE_ROOT, 'MMStar', 'test.jsonl'),
    'RealWorldQA': os.path.join(GENERAL_BENCH_CACHE_ROOT, 'RealWorldQA', 'test.jsonl'),
    'VStarBench': os.path.join(GENERAL_BENCH_CACHE_ROOT, 'VStarBench', 'test.jsonl'),
    'MMVP': os.path.join(GENERAL_BENCH_CACHE_ROOT, 'MMVP', 'test.jsonl'),
    # T2I generation benchmarks
    'GenEval': os.path.join(GENERAL_BENCH_CACHE_ROOT, 'GenEval', 'test.jsonl'),
    'T2I-CompBench': os.path.join(GENERAL_BENCH_CACHE_ROOT, 'T2I-CompBench', 'test.jsonl'),
}

# Tasks that do not generate images (skip gen_image, save no output images).
GENERAL_TASKS = {"AI2D", "ChartQA", "HR-Bench-4K", "HR-Bench-8K",
                            "MathVista", "MMMU", "MMStar", "RealWorldQA",
                            "VStarBench", "MMVP", "GenEval", "T2I-CompBench"}


def ensure_general_benchmark_data(task):
    """Download + convert HF datasets to JSONL format on rank 0 (first run only)."""
    data_path = ThinkMorph_data[task]
    if os.path.exists(data_path):
        print(f'===== exist data_path:{data_path} =====')
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        image_cache_dir = os.path.join(os.path.dirname(data_path), 'images')
        if task == 'AI2D':
            from evaluation.all_benchmark.inference.ai2d import prepare_ai2d_jsonl
            prepare_ai2d_jsonl(data_path, image_cache_dir)
        elif task == 'ChartQA':
            from evaluation.all_benchmark.inference.chartqa import prepare_chartqa_jsonl
            prepare_chartqa_jsonl(data_path, image_cache_dir)
        elif task == 'HR-Bench-4K':
            from evaluation.all_benchmark.inference.hrbench import prepare_hrbench_jsonl
            prepare_hrbench_jsonl(data_path, image_cache_dir, split='hrbench_4k')
        elif task == 'HR-Bench-8K':
            from evaluation.all_benchmark.inference.hrbench import prepare_hrbench_jsonl
            prepare_hrbench_jsonl(data_path, image_cache_dir, split='hrbench_8k')
        elif task == 'MathVista':
            from evaluation.all_benchmark.inference.mathvista import prepare_mathvista_jsonl
            prepare_mathvista_jsonl(data_path, image_cache_dir)
        elif task == 'MMMU':
            from evaluation.all_benchmark.inference.mmmu import prepare_mmmu_jsonl
            prepare_mmmu_jsonl(data_path, image_cache_dir)
        elif task == 'MMStar':
            from evaluation.all_benchmark.inference.mmstar import prepare_mmstar_jsonl
            prepare_mmstar_jsonl(data_path, image_cache_dir)
        elif task == 'RealWorldQA':
            from evaluation.all_benchmark.inference.realworldqa import prepare_realworldqa_jsonl
            prepare_realworldqa_jsonl(data_path, image_cache_dir)
        elif task == 'VStarBench':
            from evaluation.all_benchmark.inference.vstar import prepare_vstar_jsonl
            prepare_vstar_jsonl(data_path, image_cache_dir)
        elif task == 'MMVP':
            from evaluation.all_benchmark.inference.mmvp import prepare_mmvp_jsonl
            prepare_mmvp_jsonl(data_path, image_cache_dir)
        elif task == 'GenEval':
            from evaluation.all_benchmark.inference.geneval import prepare_geneval_jsonl
            prepare_geneval_jsonl(data_path, image_cache_dir)
        elif task == 'T2I-CompBench':
            from evaluation.all_benchmark.inference.t2icompbench import prepare_t2icompbench_jsonl
            prepare_t2icompbench_jsonl(data_path, image_cache_dir)
    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dist.barrier(device_ids=[local_rank])

def setup():
    # 初始化进程组，指定 device_id 避免 NCCL barrier 警告
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=5),
                            device_id=torch.device(f"cuda:{local_rank}"))
    # 获取当前进程的 rank (哪张卡) 和 world_size (总卡数)
    return local_rank, dist.get_world_size()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='测评的模型名字')
    parser.add_argument('--model_path', type=str, help='测评的模型路径')
    parser.add_argument('--save_interim', action='store_true', default=False, help='是否需要保存过程中的中间结果文件。')
    parser.add_argument('--result_path', type=str, help='保存测评结果路径')
    parser.add_argument('--tasks',  type=str, nargs='+',help='task 列表')
    parser.add_argument('--anchor_gen', action='store_true', default=False, help='Use anchor_gen for style conditioning')
    # CFG-Zero* + APG
    parser.add_argument('--cfg_zero_init_steps', type=int, default=0, help='CFG-Zero*: zero velocity for first N ODE steps')
    parser.add_argument('--cfg_zero_optimal_rescale', type=float, default=0.0, help='CFG-Zero*: optimal rescale alpha (0=off)')
    parser.add_argument('--cfg_guidance_mode', type=str, default='renorm', choices=['renorm', 'apg'], help='CFG guidance mode')
    parser.add_argument('--apg_text_eta', type=float, default=0.5, help='APG: text parallel suppression eta')
    parser.add_argument('--apg_img_eta', type=float, default=0.5, help='APG: image parallel suppression eta')
    # Q-Former distillation
    parser.add_argument('--enable_qformer_distill', action='store_true', default=False, help='Enable Q-Former VAE-to-understanding distillation')
    parser.add_argument('--qformer_num_layers', type=int, default=4, help='Number of Q-Former cross-attention layers')
    parser.add_argument('--qformer_num_heads', type=int, default=16, help='Number of Q-Former attention heads')
    parser.add_argument('--qformer_alpha', type=float, default=0.0, help='Q-Former alpha (1.0 = fully replace ViT)')
    parser.add_argument('--qformer_version', type=str, default='v2', help='Q-Former version: v2 or v3')
    parser.add_argument('--qformer_num_ca_blocks', type=int, default=4, help='v3: number of CA+SA blocks')
    parser.add_argument('--qformer_num_sa_blocks', type=int, default=9, help='v3: number of SA+SA blocks')
    # Flow matching steps
    parser.add_argument('--num_timesteps', type=int, default=50, help='Number of flow matching timesteps for image generation')
    # Verbosity control
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output (show image generation progress and think process)')

    args = parser.parse_args()

    setup()
    # 1. 准备数据
    vlm = ThinkMorph(args.model_path, think=True, understanding_output=False, save_dir=args.result_path,
                      anchor_gen=args.anchor_gen,
                      cfg_zero_init_steps=args.cfg_zero_init_steps,
                      cfg_zero_optimal_rescale=args.cfg_zero_optimal_rescale,
                      cfg_guidance_mode=args.cfg_guidance_mode,
                      apg_text_eta=args.apg_text_eta,
                      apg_img_eta=args.apg_img_eta,
                      enable_qformer_distill=args.enable_qformer_distill,
                      qformer_num_layers=args.qformer_num_layers,
                      qformer_num_heads=args.qformer_num_heads,
                      qformer_alpha=args.qformer_alpha,
                      qformer_version=args.qformer_version,
                      qformer_num_ca_blocks=args.qformer_num_ca_blocks,
                      qformer_num_sa_blocks=args.qformer_num_sa_blocks,
                      num_timesteps=args.num_timesteps,
                      verbose=args.verbose)
                      
    for task in args.tasks:
        # Prepare data for general understanding benchmarks (downloads from HF on first run)
        if task in GENERAL_TASKS:
            ensure_general_benchmark_data(task)

        data_path = ThinkMorph_data[task]

        rank, world_size = dist.get_rank(), dist.get_world_size()
        save_path = os.path.join(args.result_path, task, f'{args.model_name}.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_path_rank = os.path.join(args.result_path, task, f'{args.model_name}_{rank}_{world_size}.json')
        os.makedirs(os.path.dirname(save_path_rank), exist_ok=True)

        # Per-benchmark image directory under result_path/{task}/
        task_image_dir = os.path.join(args.result_path, task)
        os.makedirs(task_image_dir, exist_ok=True)
        vlm.save_dir = task_image_dir
        vlm.current_task = task

        datas = read_jsonl_file(data_path)
        datas = [datas[i] for i in range(len(datas)) if i%world_size == rank]

        # 断点续传逻辑
        processed_ids = set()
        aa = []
        if os.path.exists(save_path_rank):
            try:
                tested_data = read_json_file(save_path_rank)
                filtered_data = [item for item in tested_data if (item['model_response'] is None) or ('ERROR' not in item['model_response'])]
                with open(save_path_rank, 'w') as f:
                    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

                with open(save_path_rank, 'r') as f:
                    aa = json.load(f)
                    processed_ids = {x["hash_uid"] for x in aa}
            except json.JSONDecodeError:
                print("JSON file corrupted, starting over or handle manually.")
                aa = []

        print(f"Loaded {len(aa)} processed samples.")
        print(f"Total samples: {len(datas)}")

        remaining = [x for x in datas if x["hash_uid"] not in processed_ids]
        print(f"Remaining to process: {len(remaining)}")

        if remaining:
            # 2. 准备 Worker 参数
            # worker_args = [(item, task_args) for item in remaining]

            
            if task.split("_")[0] == 'Realunify':
                from evaluation.all_benchmark.inference.realunify import bagel_pipeline_step
                process_function = bagel_pipeline_step
            elif task.split("_")[0] == 'MIRA':
                from evaluation.all_benchmark.inference.mira import bagel_pipeline
                process_function = bagel_pipeline
            elif task.split("_")[0] == 'Unimmmu':
                if task == 'Unimmmu_direct':
                    from evaluation.all_benchmark.inference.unimmmu import bagel_pipeline_direct
                    process_function = bagel_pipeline_direct
                elif task == 'Unimmmu':
                    from evaluation.all_benchmark.inference.unimmmu import bagel_pipeline_step
                    process_function = bagel_pipeline_step
            elif task == 'AI2D':
                from evaluation.all_benchmark.inference.ai2d import bagel_pipeline
                process_function = bagel_pipeline
            elif task == 'ChartQA':
                from evaluation.all_benchmark.inference.chartqa import bagel_pipeline
                process_function = bagel_pipeline
            elif task in ('HR-Bench-4K', 'HR-Bench-8K'):
                from evaluation.all_benchmark.inference.hrbench import bagel_pipeline
                process_function = bagel_pipeline
            elif task == 'MathVista':
                from evaluation.all_benchmark.inference.mathvista import bagel_pipeline
                process_function = bagel_pipeline
            elif task == 'MMMU':
                from evaluation.all_benchmark.inference.mmmu import bagel_pipeline
                process_function = bagel_pipeline
            elif task == 'MMStar':
                from evaluation.all_benchmark.inference.mmstar import bagel_pipeline
                process_function = bagel_pipeline
            elif task == 'RealWorldQA':
                from evaluation.all_benchmark.inference.realworldqa import bagel_pipeline
                process_function = bagel_pipeline
            elif task == 'VStarBench':
                from evaluation.all_benchmark.inference.vstar import bagel_pipeline
                process_function = bagel_pipeline
            elif task == 'MMVP':
                from evaluation.all_benchmark.inference.mmvp import bagel_pipeline
                process_function = bagel_pipeline
            elif task == 'GenEval':
                from evaluation.all_benchmark.inference.geneval import bagel_pipeline
                process_function = bagel_pipeline
            elif task == 'T2I-CompBench':
                from evaluation.all_benchmark.inference.t2icompbench import bagel_pipeline
                process_function = bagel_pipeline
            else:
                raise ValueError(f"Unknown task: {task}")

            total_samples = len(remaining)
            # ---- Optional efficiency profiling (latency / throughput / memory) ----
            # Enabled by env var EVAL_PROFILE=1 (set via the eval shell script).
            _profile = os.environ.get("EVAL_PROFILE", "0") == "1"
            _per_sample_times = []
            if _profile:
                _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(_local_rank)
                _t_loop_start = time.perf_counter()
            for i in range(len(remaining)):
                if _profile:
                    _t_s = time.perf_counter()
                vlm.current_sample_idx = len(aa) + i  # global sample index including already processed
                result, image_list, mid_output = process_function(vlm, remaining[i])
                if _profile:
                    _per_sample_times.append(time.perf_counter() - _t_s)

                remaining[i]["model_response"] = result
                remaining[i]["generated_image"] = image_list
                remaining[i]["mid_output"] = mid_output
                aa.append(remaining[i])

                # Print progress for benchmark validation
                current_progress = len(aa)
                total_for_rank = len(datas)
                print(f"[{task}] Progress: {current_progress}/{total_for_rank}", flush=True)

                if i % 10 == 0:
                    with open(save_path_rank, 'w') as f:
                        json.dump(aa, f, ensure_ascii=False, indent=2)
            with open(save_path_rank, 'w') as f:
                json.dump(aa, f, ensure_ascii=False, indent=2)

            print(f"Finished. Total saved: {len(aa)}")

            # ---- Report efficiency metrics (per-rank + global) ----
            if _profile and len(_per_sample_times) > 0:
                _elapsed = time.perf_counter() - _t_loop_start
                _n = len(_per_sample_times)
                # Steady-state stats drop the first (warm-up) sample.
                _steady = _per_sample_times[1:] if _n > 1 else _per_sample_times
                _lat = sum(_steady) / len(_steady)          # s / sample (steady)
                _thr = 1.0 / _lat if _lat > 0 else 0.0       # samples / s (per GPU)
                _peak_gb = (torch.cuda.max_memory_allocated(_local_rank) / 1024**3
                            if torch.cuda.is_available() else 0.0)
                print(f"[{task}][PROFILE][rank{rank}] "
                      f"latency={_lat:.3f} s/sample | throughput={_thr:.4f} samples/s | "
                      f"peak_mem={_peak_gb:.2f} GB | n={_n} | elapsed={_elapsed:.1f}s", flush=True)
                # Global aggregation: slowest rank sets wall-clock; sum the samples.
                if dist.is_initialized() and world_size > 1:
                    _stat = torch.tensor([_elapsed, float(_n), _peak_gb],
                                         device=f'cuda:{_local_rank}')
                    _stat_max = _stat.clone(); dist.all_reduce(_stat_max, op=dist.ReduceOp.MAX)
                    _stat_sum = _stat.clone(); dist.all_reduce(_stat_sum, op=dist.ReduceOp.SUM)
                    if rank == 0:
                        _tot_n = _stat_sum[1].item()
                        _wall = _stat_max[0].item()          # slowest rank's wall-clock
                        _g_thr = _tot_n / _wall if _wall > 0 else 0.0
                        _g_peak = _stat_max[2].item()        # max peak mem across ranks
                        print(f"[{task}][PROFILE][GLOBAL] "
                              f"throughput={_g_thr:.4f} samples/s over {world_size} GPUs | "
                              f"total_samples={int(_tot_n)} | wall={_wall:.1f}s | "
                              f"peak_mem(max)={_g_peak:.2f} GB", flush=True)

        else:
            print("All done.")
        dist.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", "0"))])
        all_rank_data = []

        if rank == 0:
            for r in range(world_size):
                local_path = os.path.join(args.result_path, task, f'{args.model_name}_{r}_{world_size}.json')
                local_data = read_json_file(local_path)
                all_rank_data += local_data
            with open(save_path, 'w') as f:
                json.dump(all_rank_data, f, ensure_ascii=False, indent=2)

            print('*'*30)
            print(f'{task} score:')
            
            if task == 'VSP':
                pass
                # vsp.calculate_score(save_path)
            elif task == 'TIR':
                pass
                # tir.calculate_score(save_path, world_size)
            elif task.split("_")[0] == 'MIRA':
                mira.calculate_score(save_path)
            elif task.split("_")[0] == 'Realunify':
                realunify.calculate_score(save_path)
            elif task.split("_")[0] == 'Unimmmu':
                unimmmu.calculate_score(save_path, False, os.path.join(args.result_path, task))
            elif task == 'AI2D':
                ai2d.calculate_score(save_path)
            elif task == 'ChartQA':
                chartqa.calculate_score(save_path)
            elif task == 'HR-Bench-4K':
                hrbench.calculate_score_4k(save_path)
            elif task == 'HR-Bench-8K':
                hrbench.calculate_score_8k(save_path)
            elif task == 'MathVista':
                mathvista.calculate_score(save_path)
            elif task == 'MMMU':
                mmmu.calculate_score(save_path)
            elif task == 'MMStar':
                mmstar.calculate_score(save_path)
            elif task == 'RealWorldQA':
                realworldqa.calculate_score(save_path)
            elif task == 'VStarBench':
                vstar.calculate_score(save_path)
            elif task == 'MMVP':
                mmvp.calculate_score(save_path)
            elif task == 'GenEval':
                geneval_score.calculate_score(save_path)
            elif task == 'T2I-CompBench':
                t2icomp_score.calculate_score(save_path)


if __name__ == "__main__":
    main()