PROJECT_ROOT=/mnt/afs/mashengnan/o3like/Test_Code/image_text_agent #改成自己的路径
TOOLS_DIR="$PROJECT_ROOT/tools"
export PYTHONPATH="$TOOLS_DIR:$PYTHONPATH"

model_name=qwen3vl-sft-xiangli-4k-1217 # 定义测试模型名称，测试完保存同名json
result_path=./Results/ #保存目录，运行结果在Json文件夹下
workers=8

python qwen3vl_eval.py \
    --model_name ${model_name} \
    --ip_list 10.119.26.34 10.119.26.30 10.119.26.16 \
    --result_path ${result_path} \
    --workers ${workers} \
    --max_turns 20 \
    --save_interim

python extract_answer.py \
    --model_name ${model_name} \
    --api_name gpt-4o-2024-11-20 \
    --input_path ${result_path} \
    --workers ${workers}

python calculate_score.py \
    --model_name ${model_name} \
    --input_path ${result_path}
