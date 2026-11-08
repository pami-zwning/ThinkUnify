PROJECT_ROOT=/mnt/afs/mashengnan/o3like/Test_Code/image_text_agent #改成自己的路径
TOOLS_DIR="$PROJECT_ROOT/tools"
export PYTHONPATH="$TOOLS_DIR:$PYTHONPATH"

result_path=./Result
model_name=qwen_123

python eval.py \
    --benchmark_dir /mnt/afs/share_data/mashengnan/o3like/visual_reasoning/MIRA \
    --model_name ${model_name} \
    --output_dir ${result_path}/${model_name} \
    --ip_list 10.119.17.220 \
    --result_path ${result_path} \
    --workers 2 \
    --save_interim \
    --max_turns 10 \
    --use_tools 

python acc.py -r ${result_path}/${model_name}