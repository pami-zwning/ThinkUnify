# 评测代码

## 说明
- 需要多任务并行测试提高测评速度
- 测试结果保存在： image_text_agent/evaluation/Results, 按照模型版本和benchmark来保存


```shell
evaluation/Results
├── Qwen3-VL-8B-Instruct
│   ├── MIRA
│   ├── Realunify
│   ├── TIR
│   ├── Unimmmu
│   └── VSP
├── Qwen3-vl-8B-coldstart-v0.6.3-wo-jigsaw
│   ├── MIRA
│   ├── Realunify
│   ├── TIR
│   ├── Unimmmu
│   └── VSP

```

## Qwen3-vl系列
支持的task: 集成 TIR, VSP, MIRA, Unimmmu, Realunify

### 测评流程
step1: 部署vllm API

step2: 基于API进行推理

1. **Direct Inference** 
```shell
cd evaluation/all_benchmark
bash eval_qwen3_vl_8b_instruct_wo_tools.sh
```

2. **agentict infer with tools**
```shell
cd evaluation/all_benchmark
bash eval_qwen3_vl_8b_instruct_with_tools.sh
```

### 参数名称,描述说明
| 参数 | 描述 |
| :--- | :--- |
| `--model_name` | 模型唯一标识，用于命名输出结果。 |
| `--result_path` | 结果保存目录，运行完成后 Json 结果会存放在此。 |
| `--ip_list` | 分布式评测的节点 IP 地址。 |
| `--workers` | 并行进程/线程数，默认推荐 16 |
| `--max_turns` | 工具调用的最大轮数，默认10 |
| `--save_interim` | 开启后将保存中间过程数据，方便调试 |
| `--use_tools` | 是否允许模型调用定义的外部工具|
| `--tasks` | 指定测评的benchmark，可一次性传入多个 |

### VLMEvalKit_Thinkmorph集成测评
1. **Direct Inference** 
```shell
cd evaluation
bash ThinkMorph/eval_qwen3_instruct_notool.sh
```

2. **agentict infer with tools**
```shell
cd evaluation
bash ThinkMorph/eval_qwen3_instruct.sh
```

## Bagel系列模型测评
目前不支持起API然后测试，直接在线推理测试

### VLMEvalKit_Thinkmorph 测评参考
评测目录: evaluation/VLMEvalKit_Thinkmorph
```shell
cd  evaluation/VLMEvalKit_Thinkmorph/eval_shells
# 多任务多卡并行，每个task单独起任务进行测试 VSP_maze_task_main_original VisPuzzle ChartQA_h_bar ChartQA_v_bar VStarBench BLINK_Jigsaw MMVP BLINK SAT_circular CV-Bench-2D CV-Bench-3D
bash srun_bagel.sh  
```
```shell
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    run.py \
    --data  ${task_name} \
    --model thinkmorph \
    --judge gpt-4.1 \
    --judge-args '{"use_azure": true}' \
    --work-dir $SAVE_DIR \
    2>&1 | tee ${logpath}
```

### TIR, MIRA, Unimmmu, Realunify测评参考
```shell
cd evaluation/all_benchmark
bash eval_bagel.sh {task} {GPUS}
```

- task, 测试的数据集目前包括: MIRA/Unimmmu/Unimmmu_direct/Realunify/Realunify_mt
    - 可以通过引入下划线构建自己的数据集, 比如Realunify_mt仅包含mental_tracking任务
    - Unimmmu表示基于官方的流程来测试, Unimmmu_direct则是直接测试, 如果使用官方测试, 请注意数据的多样性, 来应对多任务/多指令的情况
- GPUS, 显卡数, 默认基于torchrun测试, 存储不同rank的json来resume
