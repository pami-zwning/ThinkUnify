# NAME field of `sco aec2 clusters list`
PARTITION=amplarge   # AMP(DISPLAY NAME)
# PARTITION=cc4cdf6c-944a-4c46-a9de-3d508a06c4dd  # AMP
# PARTITION=b7c081ea-ab5a-4278-ab4a-c51bc222de13 # H100_Share
PARTITION=vqalarge
# PARTITION=amplarge2-large
PARTITION=amplarge2
PARTITION=interleave

# PARTITION=r1-m1
# SPEC=N6lS.Iq.I10.8

# NAME field of `sco ws instances list`
WORKSPACE=a58d023b-de76-475f-89c2-7e50f7aa3c7a  # 多模态(DISPLAY NAME)
# 获取用户下的所有私有镜像信息 `sco ccr images list`
# 获取大装置平台的公开镜像信息 `sco ccr images list -p`
CONTAINTER=registry.ms-sc-01.maoshanwangtech.com/studio-aicl/ubuntu20.04-py3.10-cuda11.8-cudnn8-transformer4.28.0:master-20230626-172512-32302
# 与开发机挂载路径保持一致
MOUNT=1f29056c-c3f2-11ee-967e-2aea81fd34ba:/mnt/afs2,047443d2-c3f2-11ee-a5f9-9e29792dec2f:/mnt/afs1,ce3b1174-f6eb-11ee-a372-82d352e10aed:/mnt/afs

############### 评测setting  ###############
# 评测benchmark
benchmarks=(
            # thinkmorph
            VSP_maze_task_main_original
            VisPuzzle 
            ChartQA_h_bar
            ChartQA_v_bar
            VStarBench
            BLINK_Jigsaw
            MMVP
            BLINK
            SAT_circular
            CV-Bench-2D
            CV-Bench-3D
            )

THISDIR=$(dirname $(realpath $0))
SCRIPT=$THISDIR/eval_bagel.sh
GPUS=4  # TODO 用几卡来推理
SPEC=N6lS.Iu.I80.${GPUS}

for DATASET in "${benchmarks[@]}"; do
    echo $DATASET
    JOBNAME=eval_bagel_$DATASET
    sco acp jobs create --workspace-name  $WORKSPACE --aec2-name $PARTITION --container-image-url $CONTAINTER --storage-mount $MOUNT --training-framework pytorch --worker-spec $SPEC --worker-nodes 1 -j $JOBNAME --command="bash $SCRIPT $DATASET $GPUS"
done

# --priority high \