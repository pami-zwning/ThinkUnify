#!/usr/bin/env bash
set -x
MODEL_ID=interleaved_reasoning         # or pretrained model

export PATH=/mnt/afs1/luojiapeng/miniconda3/bin:$PATH
source activate
conda activate /mnt/afs/nijingcheng/miniconda3/envs/thinkmorph_eval

export MODEL_PATH=/mnt/afs/nijingcheng/workspace/work_dirs/thinkmorph/${MODEL_ID}/ckpt/0002000
PROJECT_ROOT=/mnt/afs/nijingcheng/workspace/codes/benchmarks/image_text_agent #改成自己的路径
export SAVE_DIR=$PROJECT_ROOT/evaluation/Results/VLMEvalKit_Thinkmorph/${MODEL_ID}
export LMUData=/mnt/afs/omni_data/evaluation/interleave_data/LMUData

cd ${PROJECT_ROOT}/evaluation/VLMEvalKit_Thinkmorph
python run.py \
--data VSP_maze_task_main_original VisPuzzle ChartQA_h_bar ChartQA_v_bar VStarBench BLINK_Jigsaw MMVP BLINK SAT_circular CV-Bench-2D CV-Bench-3D \
--model thinkmorph \
--judge gpt-4.1 \
--judge-args '{"use_azure": true}' \
--work-dir $SAVE_DIR