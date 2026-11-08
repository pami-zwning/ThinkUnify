PROJECT_ROOT=/mnt/afs/mashengnan/o3like/Test_Code/image_text_agent #改成自己的路径
TOOLS_DIR="$PROJECT_ROOT/tools"
export PYTHONPATH="$TOOLS_DIR:$PYTHONPATH"

model_name=qwen3vl_sft_20260109 # 定义测试模型名称，测试完保存同名json
result_path=./Results/ #保存目录，运行结果在Json文件夹下
workers=8

#conda activate /mnt/afs/mashengnan/.conda/envs/vsp

python qwen3vl_eval.py \
    --model_name ${model_name} \
    --ip_list 10.119.31.199 10.119.31.195 10.119.31.189 10.119.31.183 \
    --result_path ./Results/ \
    --workers ${workers} \
    --max_turns 10 \
    --save_interim
