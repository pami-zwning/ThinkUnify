import os
import shutil
import glob

# 定义源目录和目标目录
model_path = "/mnt/afs/omni_ckpt/BAGEL-7B-MoT"
training_result_path = "/mnt/afs/nijingcheng/workspace/work_dirs/thinkmorph/interleaved_reasoning/ckpt/0002000"

def sync_model_files(src, dst):
    # 1. 确保目标目录存在
    os.makedirs(dst, exist_ok=True)

    # 2. 定义需要匹配的文件模式
    # 包括所有 .json 文件和特定的 ae.safetensors
    patterns = ["*.json", "ae.safetensors"]
    
    files_to_copy = []
    for pattern in patterns:
        # 查找匹配的文件并加入列表
        matched = glob.glob(os.path.join(src, pattern))
        files_to_copy.extend(matched)

    if not files_to_copy:
        print(f"在 {src} 中未找到匹配的文件。")
        return

    # 3. 执行复制操作
    print(f"开始复制文件到: {dst}\n" + "-"*30)
    count = 0
    for file_path in files_to_copy:
        file_name = os.path.basename(file_path)
        dest_file = os.path.join(dst, file_name)
        
        try:
            shutil.copy(file_path, dest_file)
            print(f"成功复制: {file_name}")
            count += 1
        except Exception as e:
            print(f"复制 {file_name} 失败: {e}")

    print("-"*30)
    print(f"任务完成！共计成功复制 {count} 个文件。")


if __name__ == "__main__":
    sync_model_files(model_path, training_result_path)