set -ex

T=$(date +%Y%m%d_%H%M%S)

PROJECT_ROOT=/mnt/afs/tongwenwen1/workspace/interleave_thinking/image_text_agent # 改成自己的路径
TOOLS_DIR="$PROJECT_ROOT/tools"
export PYTHONPATH="$TOOLS_DIR:$PYTHONPATH"

source /mnt/afs/mashengnan/miniconda3/bin/activate /mnt/afs/mashengnan/.conda/envs/vsp

# support tasks: TIR VSP MIRA Unimmmu Realunify

# qwen3vl series⬇
model_name='Qwen3-VL-8B-Instruct' # 改成测试的模型名称，按照模型来保存结果
result_path=$PROJECT_ROOT/evaluation/Results/${model_name}  #按照模型来保存评测目录，运行结果在Json文件夹下
mkdir -p ${result_path}
logpath=${result_path}/${model_name}_${T}.log

workers=32

workdir=${PROJECT_ROOT}/evaluation/all_benchmark
cd ${workdir}

python run.py \
    --model_name ${model_name} \
    --result_path ${result_path} \
    --ip_list 10.119.25.84 10.119.25.71 10.119.25.64 10.119.25.51 \
    --workers ${workers} \
    --max_turns 10 \
    --save_interim \
    --tasks TIR VSP MIRA Unimmmu Realunify \
    2>&1 | tee ${logpath}