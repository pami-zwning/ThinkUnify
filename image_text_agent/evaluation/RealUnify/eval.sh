PROJECT_ROOT=/mnt/afs/mashengnan/o3like/Test_Code/image_text_agent #改成自己的路径
TOOLS_DIR="$PROJECT_ROOT/tools"
export PYTHONPATH="$TOOLS_DIR:$PYTHONPATH"

model_name=qwen3vl-base # 定义测试模型名称，测试完保存同名json
result_path=/mnt/afs/nijingcheng/workspace/codes/benchmarks/outputs/RealUnify/ #保存目录，运行结果在Json文件夹下
workers=8

python /mnt/afs/nijingcheng/workspace/codes/benchmarks/image_text_agent/evaluation/RealUnify/evaluate_realunify.py \
    --model_name ${model_name} \
    --ip_list 10.119.16.18 10.119.16.26 10.119.16.32 10.119.16.10 \
    --result_path ${result_path} \
    --workers ${workers} \
    --max_turns 20 \
    --save_interim