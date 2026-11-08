# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import random
import json

import numpy as np
import torch

from .data_utils import (
    get_flattened_position_ids_interpolate,
    get_flattened_position_ids_extrapolate, 
    len2weight,
    patchify, 
    prepare_attention_mask_per_sample, 
)
from .dataset_info import DATASET_INFO, DATASET_REGISTRY
from .transforms import ImageTransform
from .video_utils import FrameSampler


class DataConfig:
    def __init__(
        self,
        grouped_datasets,
        text_cond_dropout_prob=0.1,
        vit_cond_dropout_prob=0.4,
        vae_cond_dropout_prob=0.1,
        vae_image_downsample=16,
        max_latent_size=32,
        vit_patch_size=14,
        max_num_patch_per_side=70,
        n_copy=1,
        enable_qformer_distill=False,
    ):
        self.grouped_datasets = grouped_datasets
        self.text_cond_dropout_prob = text_cond_dropout_prob
        self.vit_cond_dropout_prob = vit_cond_dropout_prob
        self.vit_patch_size = vit_patch_size
        self.max_num_patch_per_side = max_num_patch_per_side
        self.vae_cond_dropout_prob = vae_cond_dropout_prob
        self.vae_image_downsample = vae_image_downsample
        self.max_latent_size = max_latent_size
        self.n_copy = n_copy
        self.enable_qformer_distill = enable_qformer_distill


class PackedDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        data_config,
        tokenizer,
        special_tokens,
        local_rank,
        world_size,
        num_workers,
        expected_num_tokens=32768,
        max_num_tokens_per_sample=16384,
        max_num_tokens=36864,
        prefer_buffer_before=16384,
        max_buffer_size=50,
        interpolate_pos=False,
        use_flex=False,
        data_status=None,
        use_cot_boundary_loss=False,
        use_stage_weighting=False,
    ):
        super().__init__()
        self.expected_num_tokens = expected_num_tokens
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.prefer_buffer_before = prefer_buffer_before
        self.max_num_tokens = max_num_tokens
        self.max_buffer_size = max_buffer_size
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.use_flex = use_flex
        self.use_cot_boundary_loss = use_cot_boundary_loss
        self.use_stage_weighting = use_stage_weighting
        for k, v in special_tokens.items():
            setattr(self, k, v)

        grouped_datasets, is_mandatory, grouped_weights = self.build_datasets(
            data_config.grouped_datasets, data_status
        )
        self.grouped_datasets = grouped_datasets
        self.dataset_iters = [iter(dataset) for dataset in grouped_datasets]
        self.is_mandatory = is_mandatory
        self.grouped_weights = grouped_weights
        self.data_config = data_config
        self.interpolate_pos = interpolate_pos
        self.enable_qformer_distill = data_config.enable_qformer_distill
        if self.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

    def build_datasets(self, datasets_metainfo, data_status):
        datasets = []
        is_mandatory = []
        grouped_weights = []
        for grouped_dataset_name, dataset_args in datasets_metainfo.items():
            is_mandatory.append(dataset_args.pop('is_mandatory', False))
            grouped_weights.append(dataset_args.pop('weight', 0.0))

            if 'frame_sampler_args' in dataset_args.keys():
                frame_sampler = FrameSampler(**dataset_args.pop('frame_sampler_args'))
                dataset_args['frame_sampler'] = frame_sampler
            if 'image_transform_args' in dataset_args.keys():
                transform = ImageTransform(**dataset_args.pop('image_transform_args'))
                dataset_args['transform'] = transform
            if 'vit_image_transform_args' in dataset_args.keys():
                vit_transform = ImageTransform(**dataset_args.pop('vit_image_transform_args'))
                dataset_args['vit_transform'] = vit_transform

            assert 'dataset_names' in dataset_args.keys()
            dataset_names = dataset_args.pop('dataset_names')
            dataset_args['data_dir_list'] = []
            for item in dataset_names:
                if self.local_rank == 0:
                    print(f'Preparing Dataset {grouped_dataset_name}/{item}')
                meta_info = DATASET_INFO[grouped_dataset_name][item]
                dataset_args['data_dir_list'].append(meta_info['data_dir'])

                if "parquet_info_path" in meta_info.keys():
                    if 'parquet_info' not in dataset_args.keys():
                        dataset_args['parquet_info'] = {}
                    with open(meta_info['parquet_info_path'], 'r') as f:
                        parquet_info = json.load(f)
                    dataset_args['parquet_info'].update(parquet_info)

                if 'json_dir' in meta_info.keys():
                    # parquet/tar with json
                    if 'json_dir_list' not in dataset_args.keys():
                        dataset_args['json_dir_list'] = [meta_info['json_dir']]
                    else:
                        dataset_args['json_dir_list'].append(meta_info['json_dir'])

                if 'jsonl_path' in meta_info.keys():
                    # jsonl with jpeg
                    if 'jsonl_path_list' not in dataset_args.keys():
                        dataset_args['jsonl_path_list'] = [meta_info['jsonl_path']]
                    else:
                        dataset_args['jsonl_path_list'].append(meta_info['jsonl_path'])

            resume_data_status = dataset_args.pop('resume_data_status', True)
            if data_status is not None and grouped_dataset_name in data_status.keys() and resume_data_status:
                data_status_per_group = data_status[grouped_dataset_name]
            else:
                data_status_per_group = None

            dataset_args['use_cot_boundary_loss'] = self.use_cot_boundary_loss
            dataset_args['use_stage_weighting'] = self.use_stage_weighting

            dataset = DATASET_REGISTRY[grouped_dataset_name](
                dataset_name=grouped_dataset_name,
                tokenizer=self.tokenizer,
                local_rank=self.local_rank,
                world_size=self.world_size,
                num_workers=self.num_workers,
                data_status=data_status_per_group,
                **dataset_args
            )
            datasets.append(dataset)

        return datasets, is_mandatory, grouped_weights

    def set_epoch(self, seed):
        for dataset in self.grouped_datasets:
            dataset.set_epoch(seed)

    def set_sequence_status(self):
        """Initialize the sequence_status dict that accumulates all token-level
        and image-level bookkeeping while packing one batch.

        The dict is populated incrementally by ``pack_sequence()`` as it walks
        through each item in ``sequence_plan``.  After packing, ``to_tensor()``
        converts the lists into tensors and feeds them to the model.

        ── Terminology ──────────────────────────────────────────────────────
        • *sample*   : one training example (may contain multiple images/texts)
        • *ViT image*: an image processed by SigLIP for understanding
        • *VAE image*: an image encoded by VAE for generation
        • *wo_loss*  : VAE images that are NOT the generation target (loss=0),
                       i.e. anchor / context images used as conditioning
        • *loss*     : VAE images that ARE the generation target (loss=1)
        • *image_id* : unique id linking a vit_image and vae_image that come
                       from the same source image (for Q-Former distillation)

        ── Example ──────────────────────────────────────────────────────────
        Suppose a batch has 2 samples:

          Sample 0 (interleaved generation):
            items: [text, vae_image(loss=0,id=0), vit_image(id=0),
                    text, vae_image(loss=1)]
            → 1 context ViT image, 1 wo_loss VAE image, 1 loss VAE image

          Sample 1 (pure understanding):
            items: [vit_image, text]
            → 1 ViT image, 0 VAE images

        After packing (assuming no dropout):
          vit_cnt              = [0, 0]          # original vit_idx per surviving ViT
          vit_token_seqlens    = [49, 64]         # token count per ViT image
          vit_cnt_group        = [1, 1]           # ViT images per sample
          vit_image_ids        = [0]              # only sample 0's ViT has image_id

          vae_idx              = [0, 1]           # original vae_idx per surviving VAE
          vae_token_seqlens    = [16, 24]         # token count per VAE image
          vae_idx_group        = [2, 0]           # VAE images per sample

          wo_loss_vae_idx      = [0]              # vae_idx of wo_loss images
          wo_loss_vae_token_seqlens = [16]        # token count per wo_loss VAE
          wo_loss_vae_idx_group    = [1, 0]       # wo_loss count per sample
          wo_loss_vae_image_ids    = [0]          # image_id=0, matches vit_image_ids[0]

          loss_vae_idx         = [1]              # vae_idx of loss images
          loss_vae_token_seqlens   = [24]
          loss_vae_idx_group       = [1, 0]

        If vit_cond_dropout drops sample 0's ViT image:
          vit_cnt              = [0]              # only sample 1's ViT survives
          vit_token_seqlens    = [64]
          vit_cnt_group        = [0, 1]           # sample 0 has 0 ViT now
          vit_image_ids        = []               # no image_id (sample 1 has none)
          → Q-Former distillation skips sample 0 (no matching ViT for image_id=0)

        ── Field Reference ──────────────────────────────────────────────────

        Sequence-level:
          curr                       : int   - current position in packed sequence
          sample_lens                : [int] - total token length per sample
          packed_position_ids        : [int] - RoPE position id per token
          nested_attention_masks     : [T]   - pre-built attention masks (non-flex)
          split_lens / attn_modes    : [int]/[str] - per-split lengths and modes (flex)

        Text tokens:
          packed_text_ids            : [int] - token ids (with BOS/EOS)
          packed_text_indexes        : [int] - position in packed sequence
          packed_label_ids           : [int] - target label ids for CE loss
          ce_loss_indexes            : [int] - positions that contribute to CE loss
          ce_loss_weights            : [float] - per-token CE loss weight
          stage_weights              : [float] - CoT stage weight per CE token
          cot_boundary_indexes       : [int] - CoT boundary token positions
          cot_boundary_labels        : [int] - boundary type (0=think_start, etc.)

        VAE (generation) tokens:
          vae_image_tensors          : [Tensor] - raw image tensors for VAE encoding
          vae_latent_shapes          : [(h,w)]  - patchified latent shape per VAE image
          packed_vae_token_indexes   : [int] - position in packed sequence
          packed_latent_position_ids : [int] - 2D position ids for latent tokens
          vae_token_seqlens          : [int] - token count per VAE image
          packed_timesteps           : [float] - flow matching timestep per token
          mse_loss_indexes           : [int] - positions for MSE loss (loss=1 images)
          vae_idx                    : [int] - original vae_idx per surviving VAE image
          vae_idx_group              : [int] - VAE image count per sample

        VAE wo_loss subset (anchor/conditioning, loss=0):
          wo_loss_packed_vae_token_indexes : [int] - packed positions
          wo_loss_vae_token_seqlens        : [int] - token count per wo_loss image
          wo_loss_vae_idx                  : [int] - original vae_idx
          wo_loss_vae_idx_group            : [int] - wo_loss count per sample
          wo_loss_vae_image_ids            : [int] - image_id for ViT<->VAE matching

        VAE loss subset (generation target, loss=1):
          loss_vae_token_seqlens     : [int] - token count per loss image
          loss_vae_idx               : [int] - original vae_idx
          loss_vae_idx_group         : [int] - loss image count per sample

        ViT (understanding) tokens:
          packed_vit_tokens          : [Tensor] - patchified ViT input tensors
          vit_token_seqlens          : [int] - token count per ViT image
          packed_vit_position_ids    : [Tensor] - 2D position ids for ViT patches
          packed_vit_token_indexes   : [int] - position in packed sequence
          vit_cnt                    : [int] - original vit_idx per surviving ViT image
          vit_cnt_group              : [int] - ViT image count per sample
          vit_image_ids              : [int] - image_id for ViT<->VAE matching

        N-Copy (multi-timestep training):
          n_copy_group_ids           : [int] - copy group id per token
        """
        sequence_status = dict(
            curr                        = 0,
            sample_lens                 = list(),
            packed_position_ids         = list(),
            nested_attention_masks      = list(),
            split_lens                  = list(),
            attn_modes                  = list(),
            packed_text_ids             = list(),
            packed_text_indexes         = list(),
            packed_label_ids            = list(),
            ce_loss_indexes             = list(),
            ce_loss_weights             = list(),
            stage_weights               = list(),
            cot_boundary_indexes        = list(),
            cot_boundary_labels         = list(), 
            vae_image_tensors           = list(),
            packed_latent_position_ids  = list(),
            vae_latent_shapes           = list(),
            packed_vae_token_indexes    = list(),
            packed_timesteps            = list(),
            mse_loss_indexes            = list(),
            packed_vit_tokens           = list(),
            vit_token_seqlens           = list(),
            vae_token_seqlens           = list(),
            packed_vit_position_ids     = list(),
            packed_vit_token_indexes    = list(),
            vit_cnt                     = list(), 
            vae_idx                     = list(),
            vit_cnt_group               = list(),     
            vae_idx_group               = list(),   
            wo_loss_packed_vae_token_indexes  =  list(),
            wo_loss_vae_token_seqlens   = list(),
            wo_loss_vae_idx_group       = list(),
            wo_loss_vae_idx             = list(),
            wo_loss_vae_image_ids       = list(),  # image_id for each wo_loss VAE image
            vit_image_ids               = list(),  # image_id for each surviving ViT image
            global_image_id_offset      = 0,       # offset to make image_ids globally unique across packed samples
            loss_vae_token_seqlens      = list(),
            loss_vae_idx_group          = list(),
            loss_vae_idx                = list(),
            n_copy_group_ids            = list(),
        )
        return sequence_status

    def to_tensor(self, sequence_status):
        data = dict(
            sequence_length=sum(sequence_status['sample_lens']),
            sample_lens=sequence_status['sample_lens'],
            packed_text_ids=torch.tensor(sequence_status['packed_text_ids']),
            packed_text_indexes=torch.tensor(sequence_status['packed_text_indexes']),
            packed_position_ids=torch.tensor(sequence_status['packed_position_ids']),
        )
        if not self.use_flex:
            data['nested_attention_masks'] = sequence_status['nested_attention_masks']
        else:
            sequence_len = data['sequence_length']
            pad_len = self.max_num_tokens - sequence_len
            data['split_lens'] = sequence_status['split_lens'] + [pad_len]
            data['attn_modes'] = sequence_status['attn_modes'] + ['causal']
            data['sample_lens'] += [pad_len]

        # if the model has a convnet vae (e.g., as visual tokenizer)
        if len(sequence_status['vae_image_tensors']) > 0:
            image_tensors = sequence_status.pop('vae_image_tensors')
            image_sizes = [item.shape for item in image_tensors]
            max_image_size = [max(item) for item in list(zip(*image_sizes))]
            padded_images = torch.zeros(size=(len(image_tensors), *max_image_size))
            for i, image_tensor in enumerate(image_tensors):
                padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

            data['padded_images'] = padded_images
            data['patchified_vae_latent_shapes'] = sequence_status['vae_latent_shapes']
            data['packed_latent_position_ids'] = torch.cat(sequence_status['packed_latent_position_ids'], dim=0)
            data['packed_vae_token_indexes'] = torch.tensor(sequence_status['packed_vae_token_indexes'])
            data['wo_loss_packed_vae_token_indexes'] = torch.tensor(sequence_status['wo_loss_packed_vae_token_indexes'])
            data['vae_token_seqlens'] = torch.tensor(sequence_status['vae_token_seqlens'])
            data['wo_loss_vae_token_seqlens'] = torch.tensor(sequence_status['wo_loss_vae_token_seqlens'])
            data['loss_vae_token_seqlens'] = torch.tensor(sequence_status['loss_vae_token_seqlens'])

        # if the model has a vit (e.g., as visual tokenizer)
        if len(sequence_status['packed_vit_tokens']) > 0:
            data['packed_vit_tokens'] = torch.cat(sequence_status['packed_vit_tokens'], dim=0)
            data['packed_vit_position_ids'] = torch.cat(sequence_status['packed_vit_position_ids'], dim=0)
            data['packed_vit_token_indexes'] = torch.tensor(sequence_status['packed_vit_token_indexes'])
            data['vit_token_seqlens'] = torch.tensor(sequence_status['vit_token_seqlens'])

        # if the model is required to perform visual generation
        if len(sequence_status['packed_timesteps']) > 0:
            data['packed_timesteps'] = torch.tensor(sequence_status['packed_timesteps'])
            data['mse_loss_indexes'] = torch.tensor(sequence_status['mse_loss_indexes'])
            if len(sequence_status['n_copy_group_ids']) > 0:
                data['n_copy_group_ids'] = torch.tensor(sequence_status['n_copy_group_ids'], dtype=torch.long)

        # if the model is required to perform text generation
        if len(sequence_status['packed_label_ids']) > 0:
            data['packed_label_ids'] = torch.tensor(sequence_status['packed_label_ids'])
            data['ce_loss_indexes'] = torch.tensor(sequence_status['ce_loss_indexes'])
            data['ce_loss_weights'] = torch.tensor(sequence_status['ce_loss_weights'])

            # NEW: 添加阶段权重 (用于方案2)
            if len(sequence_status['stage_weights']) > 0:
                data['stage_weights'] = torch.tensor(sequence_status['stage_weights'])

            # NEW: 添加CoT边界信息 (用于方案1)
            if len(sequence_status['cot_boundary_indexes']) > 0:
                data['cot_boundary_indexes'] = torch.tensor(sequence_status['cot_boundary_indexes'])
                data['cot_boundary_labels'] = torch.tensor(sequence_status['cot_boundary_labels'], dtype=torch.long)
        
        if len(sequence_status['vit_cnt']) > 0:
            data['vit_cnt'] = torch.tensor(sequence_status['vit_cnt'], dtype=torch.long)
        
        if len(sequence_status['vae_idx']) > 0:
            data['vae_idx'] = torch.tensor(sequence_status['vae_idx'], dtype=torch.long)
            data['wo_loss_vae_idx'] = torch.tensor(sequence_status['wo_loss_vae_idx'], dtype=torch.long)
            data['loss_vae_idx'] = torch.tensor(sequence_status['loss_vae_idx'], dtype=torch.long)

        # Q-Former ViT<->VAE image_id mapping (for handling dropout correctly)
        if len(sequence_status['wo_loss_vae_image_ids']) > 0:
            data['wo_loss_vae_image_ids'] = torch.tensor(sequence_status['wo_loss_vae_image_ids'], dtype=torch.long)
        if len(sequence_status['vit_image_ids']) > 0:
            data['vit_image_ids'] = torch.tensor(sequence_status['vit_image_ids'], dtype=torch.long)

        # Group 计数通常是标量，pack_sequence 结束时赋值
        data['vit_cnt_group'] = torch.tensor(sequence_status['vit_cnt_group'], dtype=torch.long)
        data['vae_idx_group'] = torch.tensor(sequence_status['vae_idx_group'], dtype=torch.long)
        data['wo_loss_vae_idx_group'] = torch.tensor(sequence_status['wo_loss_vae_idx_group'], dtype=torch.long)
        data['loss_vae_idx_group'] = torch.tensor(sequence_status['loss_vae_idx_group'], dtype=torch.long)

        return data

    def _batch_has_matching_qformer_ids(self, sequence_status):
        """Check if the packed batch has matching ViT and wo_loss VAE image_ids.

        After packing and ViT dropout, the batch-level lists may be empty or
        have no overlapping IDs. Returns False if Q-Former forward would fail.
        """
        vit_ids = sequence_status['vit_image_ids']
        vae_ids = sequence_status['wo_loss_vae_image_ids']
        if len(vit_ids) == 0 or len(vae_ids) == 0:
            return False
        return len(set(vit_ids) & set(vae_ids)) > 0

    def __iter__(self):
        total_weights = sum(self.grouped_weights)
        assert total_weights > 0.0
        group_cumprobs = [sum(self.grouped_weights[:i + 1]) / total_weights 
                          for i in range(len(self.grouped_weights))]
        sequence_status = self.set_sequence_status()
        batch_data_indexes = []

        buffer = []
        while True:
            # Ensure at least one sample from each group
            if sequence_status['curr'] == 0:
                for group_index, group_iter in enumerate(self.dataset_iters):
                    if self.is_mandatory[group_index]:
                        while True:
                            sample = next(group_iter)
                            # if a sample is too long, skip it
                            num_tokens = sample['num_tokens'] + 2 * len(sample['sequence_plan'])
                            if num_tokens >= self.max_num_tokens_per_sample:
                                print(f"skip a sample with length {num_tokens}")
                                continue
                            sequence_status = self.pack_sequence(sample, sequence_status)
                            batch_data_indexes.append(sample['data_indexes'])
                            break

            if sequence_status['curr'] < self.prefer_buffer_before and len(buffer) > 0:
                sample = buffer.pop(0)
                sample_from_buffer = True
            else:
                # sample normally across all groups
                n = random.random()
                group_index = 0
                for i, cumprob in enumerate(group_cumprobs):
                    if n < cumprob:
                        group_index = i
                        break
                sample = next(self.dataset_iters[group_index])
                sample_from_buffer = False

            # if a sample is too long, skip it
            num_tokens = sample['num_tokens'] + 2 * len(sample['sequence_plan'])
            if num_tokens > self.max_num_tokens_per_sample:
                print(f"skip a sample with length {num_tokens}")
                continue

            if sequence_status['curr'] + num_tokens > self.max_num_tokens:
                if len(buffer) < self.max_buffer_size and not sample_from_buffer:
                    buffer.append(sample)
                else:
                    print(f"Yielding data with length {sum(sequence_status['sample_lens'])}")
                    if len(sequence_status['packed_vit_tokens']) == 0 or len(sequence_status['packed_label_ids']) == 0:
                        print(f"Skipping yielding data with length {sum(sequence_status['sample_lens'])}")
                        print(len(sequence_status['packed_vit_tokens']), len(sequence_status['packed_label_ids']))
                        sequence_status = self.set_sequence_status()
                        batch_data_indexes = []
                        continue
                    if self.enable_qformer_distill and not self._batch_has_matching_qformer_ids(sequence_status):
                        print(f"Skipping batch: no matching Q-Former image_ids in packed batch")
                        sequence_status = self.set_sequence_status()
                        batch_data_indexes = []
                        continue
                    data = self.to_tensor(sequence_status)
                    data['batch_data_indexes'] = batch_data_indexes
                    # print(f"The input data of wo_loss_vae_image_ids is :{data['wo_loss_vae_image_ids']}; The input of vit_image_ids is {data['vit_image_ids']}")
                    yield data
                    sequence_status = self.set_sequence_status()
                    batch_data_indexes = []
                continue

            sequence_status = self.pack_sequence(sample, sequence_status)
            batch_data_indexes.append(sample['data_indexes'])

            if sequence_status['curr'] >= self.expected_num_tokens:
                if len(sequence_status['packed_vit_tokens']) == 0 or len(sequence_status['packed_label_ids']) == 0:
                    print(f"Skipping yielding data with length {sum(sequence_status['sample_lens'])}")
                    print(len(sequence_status['packed_vit_tokens']), len(sequence_status['packed_label_ids']))
                    sequence_status = self.set_sequence_status()
                    batch_data_indexes = []
                    continue
                if self.enable_qformer_distill and not self._batch_has_matching_qformer_ids(sequence_status):
                    print(f"Skipping batch: no matching Q-Former image_ids in packed batch")
                    sequence_status = self.set_sequence_status()
                    batch_data_indexes = []
                    continue
                data = self.to_tensor(sequence_status)
                data['batch_data_indexes'] = batch_data_indexes
                # print(f"The input data of wo_loss_vae_image_ids is :{data['wo_loss_vae_image_ids']}; The input of vit_image_ids is {data['vit_image_ids']}")
                yield data
                sequence_status = self.set_sequence_status()
                batch_data_indexes = []

    def pack_sequence(self, sample, sequence_status):
        image_tensor_list = sample['image_tensor_list']
        text_ids_list = sample['text_ids_list']
        sequence_plan = sample['sequence_plan']

        split_lens, attn_modes = list(), list()
        curr = sequence_status['curr']
        curr_rope_id = 0
        sample_lens = 0
        vit_idx = -1
        vae_idx = -2
        vae_tag = 1

        # Compute per-sample image_id offset so IDs are globally unique across packed samples.
        # Find the max image_id in this sample's plan to update the offset after packing.
        image_id_offset = sequence_status['global_image_id_offset']
        max_image_id_in_sample = -1
        for item in sequence_plan:
            if 'image_id' in item:
                max_image_id_in_sample = max(max_image_id_in_sample, item['image_id'])

        for item in sequence_plan:
            split_start = item.get('split_start', True)
            
            if split_start:
                curr_split_len = 0

            if item['type'] == 'text':
                text_ids = text_ids_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.text_cond_dropout_prob:
                    continue

                shifted_text_ids = [self.bos_token_id] + text_ids
                sequence_status['packed_text_ids'].extend(shifted_text_ids)
                sequence_status['packed_text_indexes'].extend(range(curr, curr + len(shifted_text_ids)))

                # NEW: 获取阶段权重（用于方案2）
                stage_weight = item.get('stage_weight', 1.0)

                if item['loss'] == 1:
                    sequence_status['ce_loss_indexes'].extend(range(curr, curr + len(shifted_text_ids)))

                    # 应用 stage_weight 到 ce_loss_weights
                    base_weights = [len2weight(len(shifted_text_ids))] * len(shifted_text_ids)
                    weighted_values = [w * stage_weight for w in base_weights]
                    sequence_status['ce_loss_weights'].extend(weighted_values)
                    sequence_status['stage_weights'].extend([stage_weight] * len(shifted_text_ids))

                    sequence_status['packed_label_ids'].extend(text_ids + [self.eos_token_id])

                curr += len(shifted_text_ids)
                curr_split_len += len(shifted_text_ids)

                # add a <|im_end|> token
                sequence_status['packed_text_ids'].append(self.eos_token_id)
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1: # <|im_end|> may have loss
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['stage_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])

                # NEW: 处理CoT边界（用于方案1）
                if item.get('cot_boundaries'):
                    # CoT边界标签映射
                    boundary_label_map = {
                        'think_start': 0,
                        'think_end': 1,
                        'answer_start': 2,
                        'answer_end': 3,
                    }
                    # 将字符级别的边界转换为token级别
                    for char_start, char_end, boundary_type in item['cot_boundaries']:
                        # 简化处理：在边界位置记录标签
                        boundary_label = boundary_label_map.get(boundary_type, -1)
                        if boundary_label >= 0:
                            # 这里curr已经是token位置，记录边界
                            sequence_status['cot_boundary_indexes'].append(curr)
                            sequence_status['cot_boundary_labels'].append(boundary_label)

                curr += 1
                curr_split_len += 1

                # update sequence status
                attn_modes.append("causal")
                sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                curr_rope_id += curr_split_len

            elif item['type'] == 'vit_image':
                vit_idx += 1
                image_tensor = image_tensor_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.vit_cond_dropout_prob:
                    curr_rope_id += 1
                    continue
                
                sequence_status["vit_cnt"].append(vit_idx)
                # Record image_id for Q-Former ViT<->VAE correspondence (if available)
                if 'image_id' in item:
                    sequence_status['vit_image_ids'].append(item['image_id'] + image_id_offset)
                # add a <|startofimage|> token
                sequence_status['packed_text_ids'].append(self.start_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                # preprocess image
                vit_tokens = patchify(image_tensor, self.data_config.vit_patch_size)
                num_img_tokens = vit_tokens.shape[0]
                sequence_status['packed_vit_token_indexes'].extend(range(curr, curr + num_img_tokens))
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                sequence_status['packed_vit_tokens'].append(vit_tokens)
                sequence_status['vit_token_seqlens'].append(num_img_tokens)
                sequence_status['packed_vit_position_ids'].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1), image_tensor.size(2),
                        self.data_config.vit_patch_size, 
                        max_num_patches_per_side=self.data_config.max_num_patch_per_side
                    )
                )

                # add a <|endofimage|> token
                sequence_status['packed_text_ids'].append(self.end_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1: # <|endofimage|> may have loss
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # update sequence status
                attn_modes.append("full")
                sequence_status['packed_position_ids'].extend([curr_rope_id] * curr_split_len)
                curr_rope_id += 1

            elif item['type'] == 'vae_image':
                if item['loss'] == 1:
                    vae_idx += 1
                    vae_tag = -1
                else:
                    if vae_tag == -1:
                        vae_tag = 1
                    else:
                        vae_idx += 1
                image_tensor = image_tensor_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.vae_cond_dropout_prob:
                    # FIXME fix vae dropout in video2video setting.
                    curr_rope_id += 1
                    continue

                # Determine N-Copy count: only for loss-bearing images
                n_copy = self.data_config.n_copy if item['loss'] == 1 else 1
                # Assign a unique group id for this image's copies
                if item['loss'] == 1 and n_copy > 1:
                    n_copy_group_id = len(sequence_status['n_copy_group_ids'])
                    # Sample N timesteps and sort descending (high noise -> low noise)
                    raw_timesteps = sorted([np.random.randn() for _ in range(n_copy)], reverse=True)
                else:
                    n_copy_group_id = -1
                    raw_timesteps = None

                for copy_idx in range(n_copy):
                    # For copies after the first, finalize previous split and start new one
                    if copy_idx > 0 and item['loss'] == 1:
                        # Finalize the previous copy's split
                        split_lens.append(curr_split_len)
                        sample_lens += curr_split_len
                        curr_split_len = 0
                        attn_modes.append("noise")

                    sequence_status["vae_idx"].append(vae_idx)
                    # add a <|startofimage|> token
                    sequence_status['packed_text_ids'].append(self.start_of_image)
                    sequence_status['packed_text_indexes'].append(curr)
                    curr += 1
                    curr_split_len += 1

                    # preprocess image (all copies share the same image tensor)
                    sequence_status['vae_image_tensors'].append(image_tensor)

                    latent_pos_ids = self.get_flattened_position_ids(
                        image_tensor.size(1), image_tensor.size(2),
                        self.data_config.vae_image_downsample,
                        max_num_patches_per_side=self.data_config.max_latent_size
                    )
                    sequence_status['packed_latent_position_ids'].append(latent_pos_ids)

                    H, W = image_tensor.shape[1:]
                    h = H // self.data_config.vae_image_downsample
                    w = W // self.data_config.vae_image_downsample
                    sequence_status['vae_latent_shapes'].append((h, w))

                    num_img_tokens = w * h
                    sequence_status['packed_vae_token_indexes'].extend(range(curr, curr + num_img_tokens))
                    sequence_status['vae_token_seqlens'].append(num_img_tokens)

                    if item['loss'] != 1:
                        sequence_status['wo_loss_packed_vae_token_indexes'].extend(range(curr, curr + num_img_tokens))
                        sequence_status['wo_loss_vae_token_seqlens'].append(num_img_tokens)
                        sequence_status["wo_loss_vae_idx"].append(vae_idx)
                        # Record image_id for Q-Former ViT<->VAE correspondence (if available)
                        if 'image_id' in item:
                            sequence_status['wo_loss_vae_image_ids'].append(item['image_id'] + image_id_offset)

                    if item['loss'] == 1:
                        sequence_status['mse_loss_indexes'].extend(range(curr, curr + num_img_tokens))
                        sequence_status['loss_vae_token_seqlens'].append(num_img_tokens)
                        sequence_status["loss_vae_idx"].append(vae_idx)
                        if split_start or copy_idx > 0:
                            if raw_timesteps is not None:
                                timestep = raw_timesteps[copy_idx]
                            else:
                                timestep = np.random.randn()
                        # Track N-Copy group
                        sequence_status['n_copy_group_ids'].extend([n_copy_group_id] * num_img_tokens)
                    else:
                        timestep = float('-inf')

                    sequence_status['packed_timesteps'].extend([timestep] * num_img_tokens)
                    curr += num_img_tokens
                    curr_split_len += num_img_tokens

                    # add a <|endofimage|> token
                    sequence_status['packed_text_ids'].append(self.end_of_image)
                    sequence_status['packed_text_indexes'].append(curr)
                    # <|endofimage|> may have loss (only for first copy)
                    if item['special_token_loss'] == 1 and copy_idx == 0:
                        sequence_status['ce_loss_indexes'].append(curr)
                        sequence_status['ce_loss_weights'].append(1.0)
                        sequence_status['packed_label_ids'].append(item['special_token_label'])
                    curr += 1
                    curr_split_len += 1

                    # update sequence status for first copy
                    if copy_idx == 0 and split_start:
                        if item['loss'] == 1 and 'frame_delta' not in item.keys():
                            attn_modes.append("noise")
                        else:
                            attn_modes.append("full")

                    sequence_status['packed_position_ids'].extend([curr_rope_id] * (num_img_tokens + 2))

                # Position ID update (once after all copies)
                if 'frame_delta' in item.keys():
                    curr_rope_id += item['frame_delta']
                elif item['loss'] == 0:
                    curr_rope_id += 1

            if item.get('split_end', True):
                split_lens.append(curr_split_len)
                sample_lens += curr_split_len

        sequence_status['curr'] = curr
        sequence_status['sample_lens'].append(sample_lens)
        # Update global image_id offset so the next packed sample gets unique IDs
        if max_image_id_in_sample >= 0:
            sequence_status['global_image_id_offset'] = image_id_offset + max_image_id_in_sample + 1
        sequence_status['vit_cnt_group'].append(len(sequence_status["vit_cnt"]) - sum(sequence_status['vit_cnt_group']) if sequence_status['vit_cnt_group'] else len(sequence_status["vit_cnt"]))
        sequence_status['vae_idx_group'].append(len(sequence_status["vae_idx"]) - sum(sequence_status['vae_idx_group']) if sequence_status['vae_idx_group'] else len(sequence_status["vae_idx"]))
        sequence_status['wo_loss_vae_idx_group'].append(len(sequence_status["wo_loss_vae_idx"]) - sum(sequence_status['wo_loss_vae_idx_group']) if sequence_status['wo_loss_vae_idx_group'] else len(sequence_status["wo_loss_vae_idx"]))
        sequence_status['loss_vae_idx_group'].append(len(sequence_status["loss_vae_idx"]) - sum(sequence_status['loss_vae_idx_group']) if sequence_status['loss_vae_idx_group'] else len(sequence_status["loss_vae_idx"]))
        # prepare attention mask
        if not self.use_flex:
            sequence_status['nested_attention_masks'].append(
                prepare_attention_mask_per_sample(split_lens, attn_modes)
            )
        else:
            sequence_status['split_lens'].extend(split_lens)
            sequence_status['attn_modes'].extend(attn_modes)

        return sequence_status


class SimpleCustomBatch:
    def __init__(self, batch):
        data = batch[0]
        self.batch_data_indexes = data['batch_data_indexes']
        self.sequence_length = data["sequence_length"]
        self.sample_lens = data["sample_lens"]
        self.packed_text_ids = data["packed_text_ids"]
        self.packed_text_indexes = data["packed_text_indexes"]
        self.packed_position_ids = data["packed_position_ids"]

        self.use_flex = "nested_attention_masks" not in data.keys()

        if self.use_flex:
            self.split_lens = data["split_lens"]
            self.attn_modes = data["attn_modes"]
        else:
            self.nested_attention_masks = data["nested_attention_masks"]

        if "padded_images" in data.keys():
            self.padded_images = data["padded_images"]
            self.patchified_vae_latent_shapes = data["patchified_vae_latent_shapes"]
            self.packed_latent_position_ids = data["packed_latent_position_ids"]
            self.packed_vae_token_indexes = data["packed_vae_token_indexes"]
            self.wo_loss_packed_vae_token_indexes = data["wo_loss_packed_vae_token_indexes"]
            self.vae_token_seqlens = data["vae_token_seqlens"]
            self.wo_loss_vae_token_seqlens = data["wo_loss_vae_token_seqlens"]
            self.loss_vae_token_seqlens = data["loss_vae_token_seqlens"]

        if "packed_vit_tokens" in data.keys():
            self.packed_vit_tokens = data["packed_vit_tokens"]
            self.packed_vit_position_ids = data["packed_vit_position_ids"]
            self.packed_vit_token_indexes = data["packed_vit_token_indexes"]
            self.vit_token_seqlens = data["vit_token_seqlens"]

        if "packed_timesteps" in data.keys():
            self.packed_timesteps = data["packed_timesteps"]
            self.mse_loss_indexes = data["mse_loss_indexes"]
            if "n_copy_group_ids" in data.keys():
                self.n_copy_group_ids = data["n_copy_group_ids"]

        if "packed_label_ids" in data.keys():
            self.packed_label_ids = data["packed_label_ids"]
            self.ce_loss_indexes = data["ce_loss_indexes"]
            self.ce_loss_weights = data["ce_loss_weights"]

            # NEW: 添加阶段权重和CoT边界信息
            if "stage_weights" in data.keys():
                self.stage_weights = data["stage_weights"]
            if "cot_boundary_indexes" in data.keys():
                self.cot_boundary_indexes = data["cot_boundary_indexes"]
                self.cot_boundary_labels = data["cot_boundary_labels"]
        
        if "vae_token_seqlens" in data.keys():
            self.vae_token_seqlens = data["vae_token_seqlens"]
            self.wo_loss_vae_token_seqlens = data["wo_loss_vae_token_seqlens"]
            self.loss_vae_token_seqlens = data["loss_vae_token_seqlens"]
        
        if "vit_cnt" in data.keys():
            self.vit_cnt = data["vit_cnt"]
        
        if "vae_idx" in data.keys():
            self.vae_idx = data["vae_idx"]
            self.wo_loss_vae_idx = data["wo_loss_vae_idx"]
            self.loss_vae_idx = data["loss_vae_idx"]
            
        if "vit_cnt_group" in data.keys():
            self.vit_cnt_group = data["vit_cnt_group"]
            
        if "vae_idx_group" in data.keys():
            self.vae_idx_group = data["vae_idx_group"]
            self.wo_loss_vae_idx_group = data["wo_loss_vae_idx_group"]
            self.loss_vae_idx_group = data["loss_vae_idx_group"]

        # Q-Former ViT<->VAE image_id mapping
        if "wo_loss_vae_image_ids" in data.keys():
            self.wo_loss_vae_image_ids = data["wo_loss_vae_image_ids"]
        if "vit_image_ids" in data.keys():
            self.vit_image_ids = data["vit_image_ids"]

    def pin_memory(self):
        self.packed_text_ids = self.packed_text_ids.pin_memory()
        self.packed_text_indexes = self.packed_text_indexes.pin_memory()
        self.packed_position_ids = self.packed_position_ids.pin_memory()

        if not self.use_flex:
            self.nested_attention_masks = [item.pin_memory() for item in self.nested_attention_masks]

        if hasattr(self, 'padded_images'):
            self.padded_images = self.padded_images.pin_memory()
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.pin_memory()
            self.wo_loss_packed_vae_token_indexes = self.wo_loss_packed_vae_token_indexes.pin_memory()
            self.packed_latent_position_ids = self.packed_latent_position_ids.pin_memory()
            self.vae_token_seqlens = self.vae_token_seqlens.pin_memory()
            self.wo_loss_vae_token_seqlens = self.wo_loss_vae_token_seqlens.pin_memory()
            self.loss_vae_token_seqlens = self.loss_vae_token_seqlens.pin_memory()

        if hasattr(self, 'packed_timesteps'):
            self.packed_timesteps = self.packed_timesteps.pin_memory()
            self.mse_loss_indexes = self.mse_loss_indexes.pin_memory()
            if hasattr(self, 'n_copy_group_ids'):
                self.n_copy_group_ids = self.n_copy_group_ids.pin_memory()

        if hasattr(self, 'packed_vit_tokens'):
            self.packed_vit_tokens = self.packed_vit_tokens.pin_memory()
            self.packed_vit_position_ids = self.packed_vit_position_ids.pin_memory()
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.pin_memory()
            self.vit_token_seqlens = self.vit_token_seqlens.pin_memory()

        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.pin_memory()
            self.ce_loss_indexes = self.ce_loss_indexes.pin_memory()
            self.ce_loss_weights = self.ce_loss_weights.pin_memory()

            # NEW: pin memory for new fields
            if hasattr(self, 'stage_weights'):
                self.stage_weights = self.stage_weights.pin_memory()
            if hasattr(self, 'cot_boundary_indexes'):
                self.cot_boundary_indexes = self.cot_boundary_indexes.pin_memory()
                self.cot_boundary_labels = self.cot_boundary_labels.pin_memory()
        
        if hasattr(self, 'vae_token_seqlens'):
            self.vae_token_seqlens = self.vae_token_seqlens.pin_memory()
            self.wo_loss_vae_token_seqlens = self.wo_loss_vae_token_seqlens.pin_memory()
            self.loss_vae_token_seqlens = self.loss_vae_token_seqlens.pin_memory()
            
        if hasattr(self, 'vit_cnt'):
            self.vit_cnt = self.vit_cnt.pin_memory()
            
        if hasattr(self, 'vae_idx'):
            self.vae_idx = self.vae_idx.pin_memory()
            self.wo_loss_vae_idx = self.wo_loss_vae_idx.pin_memory()
            self.loss_vae_idx = self.loss_vae_idx.pin_memory()
            
        if hasattr(self, 'vit_cnt_group'):
            self.vit_cnt_group = self.vit_cnt_group.pin_memory()
            
        if hasattr(self, 'vae_idx_group'):
            self.vae_idx_group = self.vae_idx_group.pin_memory()
            self.wo_loss_vae_idx_group = self.wo_loss_vae_idx_group.pin_memory()
            self.loss_vae_idx_group = self.loss_vae_idx_group.pin_memory()

        if hasattr(self, 'wo_loss_vae_image_ids'):
            self.wo_loss_vae_image_ids = self.wo_loss_vae_image_ids.pin_memory()
        if hasattr(self, 'vit_image_ids'):
            self.vit_image_ids = self.vit_image_ids.pin_memory()

        return self

    def cuda(self, device):
        self.packed_text_ids = self.packed_text_ids.to(device)
        self.packed_text_indexes = self.packed_text_indexes.to(device)
        self.packed_position_ids = self.packed_position_ids.to(device)

        if not self.use_flex:
            self.nested_attention_masks = [item.to(device) for item in self.nested_attention_masks]

        if hasattr(self, 'padded_images'):
            self.padded_images = self.padded_images.to(device)
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.to(device)
            self.wo_loss_packed_vae_token_indexes = self.wo_loss_packed_vae_token_indexes.to(device)
            self.packed_latent_position_ids = self.packed_latent_position_ids.to(device)
            self.vae_token_seqlens = self.vae_token_seqlens.to(device)
            self.wo_loss_vae_token_seqlens = self.wo_loss_vae_token_seqlens.to(device)
            self.loss_vae_token_seqlens = self.loss_vae_token_seqlens.to(device)

        if hasattr(self, 'packed_timesteps'):
            self.packed_timesteps = self.packed_timesteps.to(device)
            self.mse_loss_indexes = self.mse_loss_indexes.to(device)
            if hasattr(self, 'n_copy_group_ids'):
                self.n_copy_group_ids = self.n_copy_group_ids.to(device)

        if hasattr(self, 'packed_vit_tokens'):
            self.packed_vit_tokens = self.packed_vit_tokens.to(device)
            self.packed_vit_position_ids = self.packed_vit_position_ids.to(device)
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.to(device)
            self.vit_token_seqlens = self.vit_token_seqlens.to(device)

        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.to(device)
            self.ce_loss_indexes = self.ce_loss_indexes.to(device)
            self.ce_loss_weights = self.ce_loss_weights.to(device)

            # NEW: move new fields to device
            if hasattr(self, 'stage_weights'):
                self.stage_weights = self.stage_weights.to(device)
            if hasattr(self, 'cot_boundary_indexes'):
                self.cot_boundary_indexes = self.cot_boundary_indexes.to(device)
                self.cot_boundary_labels = self.cot_boundary_labels.to(device)
        
        if hasattr(self, 'vae_token_seqlens'):
            self.vae_token_seqlens = self.vae_token_seqlens.to(device)
            self.wo_loss_vae_token_seqlens = self.wo_loss_vae_token_seqlens.to(device)
            self.loss_vae_token_seqlens = self.loss_vae_token_seqlens.to(device)
            
        if hasattr(self, 'vit_cnt'):
            self.vit_cnt = self.vit_cnt.to(device)
            
        if hasattr(self, 'vae_idx'):
            self.vae_idx = self.vae_idx.to(device)
            self.wo_loss_vae_idx = self.wo_loss_vae_idx.to(device)
            self.loss_vae_idx = self.loss_vae_idx.to(device)
            
        if hasattr(self, 'vit_cnt_group'):
            self.vit_cnt_group = self.vit_cnt_group.to(device)
            
        if hasattr(self, 'vae_idx_group'):
            self.vae_idx_group = self.vae_idx_group.to(device)
            self.wo_loss_vae_idx_group = self.wo_loss_vae_idx_group.to(device)
            self.loss_vae_idx_group = self.loss_vae_idx_group.to(device)

        if hasattr(self, 'wo_loss_vae_image_ids'):
            self.wo_loss_vae_image_ids = self.wo_loss_vae_image_ids.to(device)
        if hasattr(self, 'vit_image_ids'):
            self.vit_image_ids = self.vit_image_ids.to(device)

        return self

    def to_dict(self):
        data = dict(
            sequence_length = self.sequence_length,
            sample_lens = self.sample_lens,
            packed_text_ids = self.packed_text_ids,
            packed_text_indexes = self.packed_text_indexes,
            packed_position_ids = self.packed_position_ids,
            batch_data_indexes = self.batch_data_indexes,
        )

        if not self.use_flex:
            data['nested_attention_masks'] = self.nested_attention_masks
        else:
            data['split_lens'] = self.split_lens
            data['attn_modes'] = self.attn_modes

        if hasattr(self, 'padded_images'):
            data['padded_images'] = self.padded_images
            data['patchified_vae_latent_shapes'] = self.patchified_vae_latent_shapes
            data['packed_latent_position_ids'] = self.packed_latent_position_ids
            data['packed_vae_token_indexes'] = self.packed_vae_token_indexes
            data['wo_loss_packed_vae_token_indexes'] = self.wo_loss_packed_vae_token_indexes
            data['vae_token_seqlens'] = self.vae_token_seqlens
            data['wo_loss_vae_token_seqlens'] = self.wo_loss_vae_token_seqlens
            data['loss_vae_token_seqlens'] = self.loss_vae_token_seqlens

        if hasattr(self, 'packed_vit_tokens'):
            data['packed_vit_tokens'] = self.packed_vit_tokens
            data['packed_vit_position_ids'] = self.packed_vit_position_ids
            data['packed_vit_token_indexes'] = self.packed_vit_token_indexes
            data['vit_token_seqlens'] = self.vit_token_seqlens

        if hasattr(self, 'packed_timesteps'):
            data['packed_timesteps'] = self.packed_timesteps
            data['mse_loss_indexes'] = self.mse_loss_indexes
            if hasattr(self, 'n_copy_group_ids'):
                data['n_copy_group_ids'] = self.n_copy_group_ids

        if hasattr(self, 'packed_label_ids'):
            data['packed_label_ids'] = self.packed_label_ids
            data['ce_loss_indexes'] = self.ce_loss_indexes
            data['ce_loss_weights'] = self.ce_loss_weights

            # NEW: add new fields to dict
            if hasattr(self, 'stage_weights'):
                data['stage_weights'] = self.stage_weights
            if hasattr(self, 'cot_boundary_indexes'):
                data['cot_boundary_indexes'] = self.cot_boundary_indexes
                data['cot_boundary_labels'] = self.cot_boundary_labels

        if hasattr(self, 'vae_token_seqlens'):
            data['vae_token_seqlens'] = self.vae_token_seqlens
            
        if hasattr(self, 'vit_cnt'):
            data['vit_cnt'] = self.vit_cnt
            
        if hasattr(self, 'vae_idx'):
            data['vae_idx'] = self.vae_idx
            data['wo_loss_vae_idx'] = self.wo_loss_vae_idx
            data['loss_vae_idx'] = self.loss_vae_idx
            
        if hasattr(self, 'vit_cnt_group'):
            data['vit_cnt_group'] = self.vit_cnt_group
            
        if hasattr(self, 'vae_idx_group'):
            data['vae_idx_group'] = self.vae_idx_group
            data['wo_loss_vae_idx_group'] = self.wo_loss_vae_idx_group
            data['loss_vae_idx_group'] = self.loss_vae_idx_group

        if hasattr(self, 'wo_loss_vae_image_ids'):
            data['wo_loss_vae_image_ids'] = self.wo_loss_vae_image_ids
        if hasattr(self, 'vit_image_ids'):
            data['vit_image_ids'] = self.vit_image_ids

        return data


def collate_wrapper():
    def collate_fn(batch):
        return SimpleCustomBatch(batch)
    return collate_fn
