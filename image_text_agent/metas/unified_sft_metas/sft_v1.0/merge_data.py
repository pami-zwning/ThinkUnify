import json
import glob
import re
import os
import io
import time
import string
import tarfile
import hashlib
import argparse
import random
from tqdm import tqdm
import multiprocessing 
import numpy as np
import pandas as pd


root_dir = '/mnt/afs/tongwenwen1/workspace/interleave_thinking/image_text_agent/metas/unified_sft_metas/sft_v1.0'
meta_folder =  os.path.join(root_dir, 'splits')
output_json = os.path.join(root_dir, "unified_interleave_data_v1.0.json")

# 获取meta_folder下所有json文件
json_files = glob.glob(os.path.join(meta_folder, "*.json"))
print(f"\n找到{len(json_files)}个json文件:")
for f in json_files:
    print(f"  - {os.path.basename(f)}")

# 合并数据，统计样本数目
ds_type_stats = {}

# 合并所有json文件
final_data = {}
total_samples = 0
for json_file in json_files:
    json_name = os.path.basename(json_file).split('.json')[0]
    actual_samples = 0
    with open(json_file, 'r') as f:
        data = json.load(f)
        for data_name, info in data.items():
            actual_samples = actual_samples + int(info['length']*info['repeat_time'])
            assert data_name not in final_data
            final_data[data_name] = info 
    
    ds_type_stats[json_name] = actual_samples
    total_samples = total_samples + actual_samples


print(f"\n合并后总共有{len(final_data)}个数据集")

# 保存合并后的json
output_path = output_json
with open(output_path, 'w') as f:
    json.dump(final_data, f, indent=4, ensure_ascii=False)
print(f"\n已保存合并结果到: {output_path}")

print(f"\n===总样本数: {total_samples:,.0f}")
print("\n各类数据集实际样本数统计:")
for ds_type, samples in sorted(ds_type_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = samples / total_samples * 100
    print(f"{ds_type}: {samples:,.0f} 占比{percentage:.1f}%")

