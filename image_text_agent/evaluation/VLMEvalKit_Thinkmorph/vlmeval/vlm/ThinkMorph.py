import os
import random
import time
import uuid

import numpy as np
import torch
import torch.distributed as dist
from accelerate import (
    infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch, 
    load_checkpoint_in_model, dispatch_model)
from PIL import Image
from .thinkmorph import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
    Qwen2Tokenizer,
    load_ae,
    add_special_tokens,
    ImageTransform,
    InterleaveInferencer,
)

from .base import BaseModel
import datetime

def get_rank():
    # 如果分布式环境未初始化，直接返回 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


class ThinkMorph(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='ThinkMorph/ThinkMorph-7B', think=True, understanding_output=True, save_dir=None, temperature=0.3, max_think_token_n=4096,
                 enable_style_consistency=False, anchor_gen=False,
                 cfg_zero_init_steps=0, cfg_zero_optimal_rescale=0.0,
                 cfg_guidance_mode='renorm', apg_text_eta=0.5, apg_img_eta=0.5,
                 enable_qformer_distill=False, qformer_num_layers=4, qformer_num_heads=16, qformer_alpha=0.0,
                 qformer_version='v2', qformer_num_ca_blocks=4, qformer_num_sa_blocks=8,
                 num_timesteps=50, verbose=False,
                 **kwargs):
        # self.check_install()
        # Directory holding the base BAGEL-7B-MoT config files (llm_config.json,
        # vit_config.json, ae.safetensors). Override with the BAGEL_CONFIG_PATH
        # environment variable. See README "Model Download".
        base_model_path = os.environ.get('BAGEL_CONFIG_PATH', 'YOUR_BAGEL_7B_MOT_PATH')
        assert model_path is not None
        if not understanding_output:
            assert save_dir is not None
        self.model_path = model_path
        self.understanding_output = understanding_output
        self.save_dir = save_dir
        self.think = think
        self.temperature = temperature
        self.max_think_token_n = max_think_token_n
        self.num_timesteps = num_timesteps
        self.verbose = verbose

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        llm_config = Qwen2Config.from_json_file(os.path.join(base_model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        llm_config.use_adaln = enable_style_consistency
        llm_config.condition_dim = llm_config.hidden_size

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(base_model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        vae_model, vae_config = load_ae(local_path=os.path.join(base_model_path, "ae.safetensors"))

        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
            latent_patch_size=2,
            max_latent_size=64,
            enable_style_consistency=enable_style_consistency,
            anchor_und=not anchor_gen,
            anchor_gen=anchor_gen,
            enable_qformer_distill=enable_qformer_distill,
            qformer_num_layers=qformer_num_layers,
            qformer_num_heads=qformer_num_heads,
            qformer_alpha=qformer_alpha,
            qformer_version=qformer_version,
            qformer_num_ca_blocks=qformer_num_ca_blocks,
            qformer_num_sa_blocks=qformer_num_sa_blocks,
        )

        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        # tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(base_model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        # transforms
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

        # device map
        max_mem_per_gpu = "80GiB"
        device_map = infer_auto_device_map(
            model,
            max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        # print("Before: ", device_map, torch.cuda.device_count())

        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
            'vit_pos_embed',
            'style_extractor_block',
            'qformer',
        ]

        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
                else:
                    device_map[k] = "cuda:0"
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device

        # Only to local device
        # for k, v in device_map.items():
        #     device_map[k] = get_rank()
        # print("Rank: ", get_rank())

        # Debug code
        # model = model.to(torch.bfloat16)
        # import pdb; pdb.set_trace()
        if dist.is_initialized():
            local_rank = get_rank()
            world_size = dist.get_world_size()
            torch.cuda.set_device(local_rank)
            torch.cuda.empty_cache()
            # Resolve checkpoint path. Priority order:
            #   1. ema.safetensors  — BAGEL official EMA weights (single file,
            #      preferred for inference; this is what BAGEL publishes)
            #   2. model.safetensors — single-file checkpoint (ThinkMorph training)
            #   3. model.safetensors.index.json — sharded index (filter out ae/ema)
            ema_path   = os.path.join(model_path, "ema.safetensors")
            ckpt_path  = os.path.join(model_path, "model.safetensors")
            ckpt_index = os.path.join(model_path, "model.safetensors.index.json")
            if os.path.isfile(ema_path):
                load_ckpt = ema_path
                if local_rank == 0:
                    print(f"[checkpoint] Using BAGEL EMA weights: {ema_path}")
            elif os.path.isfile(ckpt_path):
                load_ckpt = ckpt_path
            elif os.path.isfile(ckpt_index):
                # Filter out non-model shards (e.g. ae.safetensors) from the index
                import json as _json
                with open(ckpt_index, "r") as _f:
                    index_data = _json.load(_f)
                weight_map = index_data.get("weight_map", {})
                bad_files = {f for f in set(weight_map.values()) if not f.startswith("model")}
                if bad_files:
                    tmp_index = os.path.join(model_path, ".model_only.safetensors.index.json")
                    if local_rank == 0:
                        filtered_map = {k: v for k, v in weight_map.items() if v not in bad_files}
                        filtered_index = {**index_data, "weight_map": filtered_map}
                        with open(tmp_index, "w") as _f:
                            _json.dump(filtered_index, _f)
                        print(f"[checkpoint] Filtered out non-model files from index: {bad_files}")
                    dist.barrier(device_ids=[local_rank])
                    load_ckpt = tmp_index
                else:
                    load_ckpt = ckpt_index
            else:
                load_ckpt = model_path

            # --- Serialized loading to avoid CephFS I/O contention ---
            # Loading a large checkpoint (e.g. 58 GB) simultaneously on all 8
            # ranks causes severe mmap/I/O contention on CephFS, making the
            # process appear stuck after the safetensors metadata WARNING.
            # Solution: load one rank at a time (serialized).  Each rank takes
            # ~30s to load 58 GB, so 8 ranks finish in ~4 min total.
            import time as _time
            for loading_rank in range(world_size):
                if local_rank == loading_rank:
                    print(f"[checkpoint] Rank {local_rank}/{world_size} loading from: {load_ckpt}", flush=True)
                    _t0 = _time.time()
                    load_checkpoint_in_model(model, load_ckpt, dtype=torch.bfloat16, device_map=None)
                    _t1 = _time.time()
                    print(f"[checkpoint] Rank {local_rank} loaded in {_t1 - _t0:.1f}s", flush=True)
                dist.barrier(device_ids=[local_rank])

            # Materialize any meta-device parameters (missing keys from checkpoint,
            # e.g. new Q-Former/AdaLN layers not in the saved checkpoint)
            meta_params = []
            for name, module in model.named_modules():
                for attr_name, param in list(module.named_parameters(recurse=False)):
                    if param.device.type == 'meta':
                        full_name = f"{name}.{attr_name}" if name else attr_name
                        meta_params.append(f"{full_name} {list(param.shape)}")
                        new_param = torch.nn.Parameter(
                            torch.zeros(param.shape, dtype=torch.bfloat16, device='cpu'),
                            requires_grad=param.requires_grad,
                        )
                        setattr(module, attr_name, new_param)
                for attr_name, buf in list(module.named_buffers(recurse=False)):
                    if buf.device.type == 'meta':
                        full_name = f"{name}.{attr_name}" if name else attr_name
                        meta_params.append(f"{full_name} {list(buf.shape)} (buffer)")
                        module.register_buffer(attr_name, torch.zeros(buf.shape, dtype=buf.dtype, device='cpu'))
            if meta_params:
                print(f"[Rank {local_rank}][Warning] {len(meta_params)} meta parameters materialized (missing from checkpoint):")
                for p in meta_params:
                    print(f"  - {p}")
            else:
                if local_rank == 0:
                    print("[Info] All parameters loaded from checkpoint, no meta parameters.")

            print(f"[Rank {local_rank}] Moving model to cuda:{local_rank} ...", flush=True)
            model = model.to(torch.bfloat16).to(f"cuda:{local_rank}")
            model = dispatch_model(model, device_map={"": f"cuda:{local_rank}"}, force_hooks=True)
            torch.cuda.empty_cache()
            dist.barrier(device_ids=[local_rank])  # ensure all ranks finish before inference
        else:
            # Original code
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=os.path.join(model_path, "model.safetensors"),
                device_map=device_map,
                offload_buffers=True,
                dtype=torch.bfloat16,
                force_hooks=True,
                offload_folder="/tmp/offload"
            )
        # import pdb; pdb.set_trace()
        model = model.eval()
        print(f'Model loaded successfully')

        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids

        self.inferencer = InterleaveInferencer(
            model=self.model,
            vae_model=self.vae_model,
            tokenizer=self.tokenizer,
            vae_transform=self.vae_transform,
            vit_transform=self.vit_transform,
            new_token_ids=self.new_token_ids,
        )

        if understanding_output:
            # For understanding tasks: fix seed for reproducibility
            seed = 42
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            # For generation tasks: use random seed so multiple images per prompt differ
            seed = int(time.time() * 1000) % (2**31)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

        if understanding_output:
            inference_hyper = dict(
                max_think_token_n=self.max_think_token_n,
                do_sample=True,
                text_temperature=self.temperature,
            )
        else:
            inference_hyper = dict(
                max_think_token_n=self.max_think_token_n,
                do_sample=True,
                text_temperature=self.temperature,
                cfg_text_scale=4.0,
                cfg_img_scale=2.0,
                cfg_interval=[0.0, 1.0],
                timestep_shift=3.0,
                num_timesteps=self.num_timesteps,
                cfg_renorm_min=0.0,
                cfg_renorm_type="text_channel",
                cfg_zero_init_steps=cfg_zero_init_steps,
                cfg_zero_optimal_rescale=cfg_zero_optimal_rescale,
                cfg_guidance_mode=cfg_guidance_mode,
                apg_text_eta=apg_text_eta,
                apg_img_eta=apg_img_eta,
                verbose=self.verbose,
            )

        self.inference_hyper = inference_hyper


    def build_thinkmorph_input(self, message):
        # according to https://github.com/ByteDance-Seed/Bagel/issues/83

        images = []
        text_parts = []
        image_counter = 1  

        for m in message:
            if m['type'] == 'image':
                val = m['value']
                if isinstance(val, str):
                    img = Image.open(val).convert("RGB")
                elif isinstance(val, Image.Image):
                    img = val
                else:
                    raise TypeError(f"Unsupported image input type {type(val)}")
                
                images.append(img)
                text_parts.append(f"<img><|image_{image_counter}|></img>")
                image_counter += 1

            elif m['type'] == 'text':
                text_parts.append(m['value'])
            else:
                raise ValueError(f"Unsupported message type {m['type']}")

        if not images:
            # Pure text-to-image generation (e.g. GenEval, T2I-CompBench):
            # no input image is required. The inferencer handles text-only input.
            final_text = " ".join(text_parts)
            return [final_text]

        final_text = " ".join(text_parts)
        input_list = images + [final_text]
        return input_list
    
    def generate_inner(self, message, dataset=None, output_img=False, postprocess=True, gen_only=False):
        if postprocess:
            input_list = self.build_thinkmorph_input(message)
        else:
            input_list = message
        out_img_list = []

        # Set seed on every call so each image generation is independent.
        # Understanding tasks: fixed seed for reproducibility.
        # Generation tasks: random seed so multiple images per prompt differ.
        if gen_only:
            if self.understanding_output:
                seed = 42
            else:
                seed = int(time.time() * 1000 + id(message)) % (2**31)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        hash_id = uuid.uuid4().hex[:8]
        task_name = getattr(self, 'current_task', None)
        sample_idx = getattr(self, 'current_sample_idx', None)
        if self.understanding_output:
            output_dict = self.inferencer(input_list=input_list, think=self.think, 
                                        understanding_output=True, **self.inference_hyper)
            final_output = output_dict[0]
            
        else:
            if get_rank()==0:
                print(f"******[RANK{get_rank()}]] INPUT_CONTEXT is:******",input_list)
                current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"----[START_TIME SINGLE SAMPLE][{current_time}]")
            output_list = self.inferencer(input_list=input_list, think=self.think, gen_only=gen_only, **self.inference_hyper)
            if get_rank()==0:
                current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"----[END_TIME SINGLE SAMPLE][{current_time}]")
            results = []
            text_round = 0
            img_step = 0
            if get_rank() == 0:
                print(f"The output_list length is {len(output_list)}")
            for idx, out_item in enumerate(output_list):
                if isinstance(out_item, str):
                    results.append(out_item)
                    text_round += 1
                elif isinstance(out_item, Image.Image):
                    if task_name is not None and sample_idx is not None:
                        out_img_path = os.path.join(self.save_dir, f"{task_name}_{sample_idx}_{img_step}_{hash_id}.jpg")
                    else:
                        out_img_path = os.path.join(self.save_dir, f"thinkmorph_out_{hash_id}_{img_step}.jpg")
                    if get_rank() == 0:
                        print(f'******[RANK{get_rank()}]-[ROUND{img_step}] GEN_IMG is******:\n', out_img_path)
                    out_item.save(out_img_path)
                    results.append(f"[Image: {out_img_path}]")
                    out_img_list.append(out_img_path)
                    img_step += 1

            final_output = "\n".join(results)
        if output_img:
            return final_output, out_img_list
        else:
            return final_output

    def internal_infer(self, message, think=None, understanding_output=None, postprocess=True, 
        max_rounds=3, image_shapes=(1024, 1024), gen_only=False):
        pre_think, pre_understanding_output = self.think, self.understanding_output
        self.think = think if think is not None else self.think
        self.understanding_output = understanding_output if understanding_output is not None else self.understanding_output
        self.inference_hyper["max_rounds"] = max_rounds
        self.inference_hyper["image_shapes"] = image_shapes
        final_output, out_img_list = self.generate_inner(message, output_img=True, postprocess=postprocess, gen_only=gen_only)

        self.think, self.understanding_output = pre_think, pre_understanding_output
        return final_output, out_img_list