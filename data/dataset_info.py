# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset,InterleavedBaseIterableDataset, InterleaveReasoningIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'inter_leave': InterleavedBaseIterableDataset,
    'inter_reason': InterleaveReasoningIterableDataset
}


DATASET_INFO = {
    'inter_reason':{
        "UniMMMu_Jigsaw_260204": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/UniMMMu_Jigsaw_260204",
            "num_files": 21,
            "num_total_samples": 2022,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/UniMMMu_Jigsaw_260204/UniMMMu_Jigsaw_260204_parquet_summary.json"
        },
        "Pretrain_UniMMMu_Geometry_260202": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/Pretrain_UniMMMu_Geometry_260202",
            "num_files": 119,
            "num_total_samples": 11811,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/Pretrain_UniMMMu_Geometry_260202/Pretrain_UniMMMu_Geometry_260202_parquet_summary.json"
        }
    },
    'unified_edit':{
        # v1.4
        "thinkmorph-Chart_Refocus": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/thinkmorph-Chart_Refocus",
            "num_files": 60,
            "num_total_samples": 6000,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/thinkmorph-Chart_Refocus/thinkmorph-Chart_Refocus_parquet_summary.json"
        },
        "thinkmorph-Jigsaw_Assembly": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/thinkmorph-Jigsaw_Assembly",
            "num_files": 60,
            "num_total_samples": 6000,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/thinkmorph-Jigsaw_Assembly/thinkmorph-Jigsaw_Assembly_parquet_summary.json"
        },
        "thinkmorph-Visual_Search": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/thinkmorph-Visual_Search",
            "num_files": 70,
            "num_total_samples": 6990,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/thinkmorph-Visual_Search/thinkmorph-Visual_Search_parquet_summary.json"
        },
        "thinkmorph-Spatial_Navigation": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/thinkmorph-Spatial_Navigation",
            "num_files": 60,
            "num_total_samples": 6000,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/thinkmorph-Spatial_Navigation/thinkmorph-Spatial_Navigation_parquet_summary.json"
        },
        "SFT_TIR_Maze_Bagel_Single_Step_260121": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/SFT_TIR_Maze_Bagel_Single_Step_260121",
            "num_files": 45,
            "num_total_samples": 4421,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/SFT_TIR_Maze_Bagel_Single_Step_260121/SFT_TIR_Maze_Bagel_Single_Step_260121_parquet_summary.json"
        },
        "SFT_Unimmmu_Maze_Single_Step_260121": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/SFT_Unimmmu_Maze_Single_Step_260121",
            "num_files": 11,
            "num_total_samples": 1025,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/SFT_Unimmmu_Maze_Single_Step_260121/SFT_Unimmmu_Maze_Single_Step_260121_parquet_summary.json"
        },
        "SFT_VSP_FrozenLakeMaze_Bagel_Single_step_260121": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/SFT_VSP_FrozenLakeMaze_Bagel_Single_step_260121",
            "num_files": 50,
            "num_total_samples": 4923,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/SFT_VSP_FrozenLakeMaze_Bagel_Single_step_260121/SFT_VSP_FrozenLakeMaze_Bagel_Single_step_260121_parquet_summary.json"
        },
        "SFT_TIR_Maze_Bagel_Multi_Step_260121": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/SFT_TIR_Maze_Bagel_Multi_Step_260121",
            "num_files": 45,
            "num_total_samples": 4421,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/SFT_TIR_Maze_Bagel_Multi_Step_260121/SFT_TIR_Maze_Bagel_Multi_Step_260121_parquet_summary.json"
        },
        "SFT_Unimmmu_Maze_Multi_Step_260121": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/SFT_Unimmmu_Maze_Multi_Step_260121",
            "num_files": 12,
            "num_total_samples": 1119,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/SFT_Unimmmu_Maze_Multi_Step_260121/SFT_Unimmmu_Maze_Multi_Step_260121_parquet_summary.json"
        },
        "SFT_VSP_FrozenLakeMaze_Bagel_Multi_step_260121": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/SFT_VSP_FrozenLakeMaze_Bagel_Multi_step_260121",
            "num_files": 50,
            "num_total_samples": 4923,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/SFT_VSP_FrozenLakeMaze_Bagel_Multi_step_260121/SFT_VSP_FrozenLakeMaze_Bagel_Multi_step_260121_parquet_summary.json"
        },
        "SFT_MIRA_convex_hull_counting_1_step": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/SFT_MIRA_convex_hull_counting_1_step",
            "num_files": 300,
            "num_total_samples": 30000,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/SFT_MIRA_convex_hull_counting_1_step/SFT_MIRA_convex_hull_counting_1_step_parquet_summary.json"
        },
        "jigsaw_edit_2x2_w_coord_2step_260121": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/jigsaw_edit_2x2_w_coord_2step_260121",
            "num_files": 200,
            "num_total_samples": 20000,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/jigsaw_edit_2x2_w_coord_2step_260121/jigsaw_edit_2x2_w_coord_2step_260121_parquet_summary.json"
        },
        "jigsaw_edit_1x2_1x3": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/jigsaw_edit_1x2_1x3",
            "num_files": 40,
            "num_total_samples": 4000,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/jigsaw_edit_1x2_1x3/jigsaw_edit_1x2_1x3_parquet_summary.json"
        },
        "TIR_jigsaw_3x3_4x4_260115": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/TIR_jigsaw_3x3_4x4_260115",
            "num_files": 303,
            "num_total_samples": 30220,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/TIR_jigsaw_3x3_4x4_260115/TIR_jigsaw_3x3_4x4_260115_parquet_summary.json"
        },
        "MathCanvas_Instruct": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/MathCanvas_Instruct",
            "num_files": 217,
            "num_total_samples": 21691,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/MathCanvas_Instruct/MathCanvas_Instruct_parquet_summary.json"
        },
        "localizer": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/localizer",
            "num_files": 50,
            "num_total_samples": 5000,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/localizer/localizer_parquet_summary.json"
        },
        "cubes_count_x": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/cubes_count_x",
            "num_files": 50,
            "num_total_samples": 5000,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/cubes_count_x/cubes_count_x_parquet_summary.json"
        },
        "cubes_missing_x": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/cubes_missing_x",
            "num_files": 50,
            "num_total_samples": 5000,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/cubes_missing_x/cubes_missing_x_parquet_summary.json"
        },
        "mirror_clock": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/mirror_clock",
            "num_files": 50,
            "num_total_samples": 5000,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/mirror_clock/mirror_clock_parquet_summary.json"
        },
        "rolling_dice_top": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/rolling_dice_top",
            "num_files": 300,
            "num_total_samples": 30000,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/rolling_dice_top/rolling_dice_top_parquet_summary.json"
        },
        "rolling_dice_sum": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/rolling_dice_sum",
            "num_files": 100,
            "num_total_samples": 10000,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/rolling_dice_sum/rolling_dice_sum_parquet_summary.json"
        },
        "Mira_electric_charge_3q": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/Mira_electric_charge_3q",
            "num_files": 20,
            "num_total_samples": 2000,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/Mira_electric_charge_3q/Mira_electric_charge_3q_parquet_summary.json"
        },
        "Mira_electric_charge_square": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/Mira_electric_charge_square",
            "num_files": 30,
            "num_total_samples": 3000,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/Mira_electric_charge_square/Mira_electric_charge_square_parquet_summary.json"
        },
        "gear_rotation_1_step": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/gear_rotation_1_step",
            "num_files": 100,
            "num_total_samples": 10000,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/gear_rotation_1_step/gear_rotation_1_step_parquet_summary.json"
        },
        # Mental Tracking
        "realunify_mt_edit_1_3_1_5_5000_same_10": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/realunify_mt_edit_1_3_1_5_5000_same_10",
            "num_files": 50,
            "num_total_samples": 5000,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/realunify_mt_edit_1_3_1_5_5000_same_10/realunify_mt_edit_1_3_1_5_5000_same_10_parquet_summary.json"
        },
        "realunify_mt_edit_multi_step_1_3_1_5_5000_seq_10": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/realunify_mt_edit_multi_step_1_3_1_5_5000_seq_10",
            "num_files": 50,
            "num_total_samples": 5000,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/realunify_mt_edit_multi_step_1_3_1_5_5000_seq_10/realunify_mt_edit_multi_step_1_3_1_5_5000_seq_10_parquet_summary.json"
        },
        "SFT_RealUnify_Maze_260211": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/SFT_RealUnify_Maze_260211",
            "num_files": 40,
            "num_total_samples": 4000,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/SFT_RealUnify_Maze_260211/SFT_RealUnify_Maze_260211_parquet_summary.json"
        },
        "SFT_UniMMMMU_Maze_260210": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/SFT_UniMMMMU_Maze_260210",
            "num_files": 24,
            "num_total_samples": 2400,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/SFT_UniMMMMU_Maze_260210/SFT_UniMMMMU_Maze_260210_parquet_summary.json"
        },
        "RealUnify_MR_260211": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/RealUnify_MR_260211",
            "num_files": 100,
            "num_total_samples": 10000,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/RealUnify_MR_260211/RealUnify_MR_260211_parquet_summary.json"
        },
        "unimmmu_sliding_le11_5k": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/unimmmu_sliding_le11_5k",
            "num_files": 50,
            "num_total_samples": 5000,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/unimmmu_sliding_le11_5k/unimmmu_sliding_le11_5k_parquet_summary.json"
        },
        "Unimmmu_geometry_260211": {
            "data_dir": "/mnt/afs/shangziyi/Data/WorkSpace/training_data/unimmmu_geo/260211/Unimmmu_geometry_260211",
            "num_files": 96,
            "num_total_samples": 9523,
            "parquet_info_path": "/mnt/afs/shangziyi/Data/WorkSpace/training_data/unimmmu_geo/260211/Unimmmu_geometry_260211/Unimmmu_geometry_260211_parquet_summary.json"
        },
        "RealUnify_MR_edit_260228": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/RealUnify_MR_edit_260228",
            "num_files": 100,
            "num_total_samples": 10000,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/RealUnify_MR_edit_260228/RealUnify_MR_edit_260228_parquet_summary.json"
        },
        "RealUnify_MR_260227": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/RealUnify_MR_260227",
            "num_files": 50,
            "num_total_samples": 5000,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/RealUnify_MR_260227/RealUnify_MR_260227_parquet_summary.json"
        },
        "realunify_attentional_focusing_visualcot_gqa_260304": {
            "data_dir": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/realunify_attentional_focusing_visualcot_gqa_260304",
            "num_files": 71,
            "num_total_samples": 7062,
            "parquet_info_path": "/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify/realunify_attentional_focusing_visualcot_gqa_260304/realunify_attentional_focusing_visualcot_gqa_260304_parquet_summary.json"
        }
    },

}
