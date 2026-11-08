set -ex

T=$(date +%Y%m%d_%H%M%S)

PROJECT_ROOT=/mnt/afs/tongwenwen1/workspace/interleave_thinking/image_text_agent # 改成自己的路径
TOOLS_DIR="$PROJECT_ROOT/tools"
export PYTHONPATH="$TOOLS_DIR:$PYTHONPATH"

source /mnt/afs/mashengnan/miniconda3/bin/activate /mnt/afs/mashengnan/.conda/envs/vsp

# support tasks: TIR VSP MIRA Unimmmu Realunify

# qwen3vl series⬇
model_name='Qwen3-vl-8B-coldstart-v0.6.3-wo-jigsaw' # 改成测试的模型名称，按照模型来保存结果
result_path=$PROJECT_ROOT/evaluation/Results/${model_name}  # 按照模型来保存评测目录，运行结果在Json文件夹下
mkdir -p ${result_path}
logpath=${result_path}/${model_name}_${T}.log
workers=16

workdir=${PROJECT_ROOT}/evaluation/all_benchmark
cd ${workdir}
python run.py \
    --model_name ${model_name} \
    --result_path ${result_path} \
    --ip_list 10.119.27.130 10.119.27.118 10.119.27.127 10.119.27.123 10.119.27.115 10.119.27.113 10.119.27.110	10.119.27.106 \
    --workers ${workers} \
    --max_turns 10 \
    --save_interim \
    --use_tools \
    --tasks TIR VSP MIRA Unimmmu Realunify \
    2>&1 | tee ${logpath}