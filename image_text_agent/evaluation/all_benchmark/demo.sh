PROJECT_ROOT=/mnt/afs/mashengnan/o3like/Test_Code/image_text_agent #改成自己的路径
TOOLS_DIR="$PROJECT_ROOT/tools"
export PYTHONPATH="$TOOLS_DIR:$PYTHONPATH"

source /mnt/afs/mashengnan/miniconda3/bin/activate /mnt/afs/mashengnan/.conda/envs/vsp

# support tasks: TIR VSP MIRA Unimmmu Realunify

# qwen3vl series⬇
model_name=RL_d260116_lr_1e-5_add_vsp_step150 # 定义测试模型名称，测试完保存同名json
result_path=./Results/ #保存目录，运行结果在Json文件夹下
workers=16

python run.py \
    --model_name ${model_name} \
    --result_path ${result_path} \
    --ip_list 10.119.21.198 10.119.21.158 10.119.21.161 10.119.21.154 10.119.21.142 10.119.21.135 10.119.21.128 10.119.21.121 \
    --workers ${workers} \
    --max_turns 10 \
    --save_interim \
    --use_tools \
    --tasks TIR VSP MIRA Unimmmu Realunify

#####################
# close api 
# ** need use --close_api **
# using api version name such as gpt-4o-2024-11-20/gemini-3-pro/gemini-3-flash
# Tool call is currently not supported by the closed-source model !!

model_name=gemini-3-flash # 定义测试模型名称，测试完保存同名json
result_path=./Results/ #保存目录，运行结果在Json文件夹下
workers=16

python run.py \
    --model_name ${model_name} \
    --result_path ${result_path} \
    --workers ${workers} \
    --save_interim \
    --tasks TIR VSP MIRA Unimmmu Realunify \
    --close_api
