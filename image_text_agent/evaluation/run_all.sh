#!/bin/bash

# 1. 激活 Conda 环境
source /mnt/afs/ningzhiwei/miniconda3/bin/activate /mnt/afs/ningzhiwei/miniconda3/envs/bagel

# 2. 运行 Python 流程脚本
# -u 参数让日志实时输出，方便在 nohup 或 slurm 中查看进度
python -u run_pipeline.py