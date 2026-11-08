# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any

from PIL import Image
import torch

from .data.data_utils import pil_img2rgb
from .data.transforms import ImageTransform
from .modeling.bagel.qwen2_navit import NaiveCache

import torch.distributed as dist
import datetime

def get_rank():
    # 如果分布式环境未初始化，直接返回 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()



VLM_THINK_SYSTEM_PROMPT = '''
Let's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.
'''

GEN_THINK_SYSTEM_PROMPT = '''
Let's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.
'''

GEN_ONLY_SYSTEM_PROMPT = '''
Please generate an natural image based on the caption below. Carefully follow the described attributes of each object, including their colors, quantities, and relative positions (e.g., spatial relationships such as left, right, above, below).
'''



class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        
    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference, 

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True,
                              precomputed_vae_latent=None, precomputed_image_shape=None):
        # used for interleave data, currently only support 1 data inference,

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        # Check if Q-Former replaces ViT (alpha >= 1.0)
        use_qformer = (
            getattr(self.model.config, 'enable_qformer_distill', False)
            and getattr(self.model, 'qformer_alpha', 0.0) >= 1.0
        )

        # Cache VAE clean latent from vae path to avoid double encoding in Q-Former path
        vae_clean_latent_info = None

        if vae:
            ## update vae
            extra_vae_kwargs = {}
            if precomputed_vae_latent is not None and precomputed_image_shape is not None and use_qformer:
                print(f'use former is {use_qformer}')
                # Q-Former skip-decode path: build metadata from image_shape
                # instead of running VAE transforms on an actual image.
                generation_input, kv_lens, ropes = self.model.prepare_vae_from_latent(
                    curr_kvlens=kv_lens,
                    curr_rope=ropes,
                    image_shape=precomputed_image_shape,
                    new_token_ids=self.new_token_ids,
                )
                extra_vae_kwargs['precomputed_packed_latent_clean'] = precomputed_vae_latent
            else:
                generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                    curr_kvlens=kv_lens,
                    curr_rope=ropes,
                    images=[image],
                    transforms=self.vae_transform,
                    new_token_ids=self.new_token_ids,
                )
                print(f'use former is {use_qformer}')
            result = self.model.forward_cache_update_vae(
                self.vae_model, past_key_values, **generation_input, **extra_vae_kwargs,
            )
            # forward_cache_update_vae returns (past_key_values, style_features, vae_clean_latent_info)
            past_key_values, style_features, vae_clean_latent_info = result
            if style_features is not None:
                gen_context['style_features'] = style_features

        if vit:
            if use_qformer:
                # Q-Former mode: skip ViT, use VAE latent features directly.
                if precomputed_vae_latent is not None and precomputed_image_shape is not None and use_qformer:
                    # No real image available (skip-decode path): create a dummy
                    # image of the correct size so prepare_vit_images can compute
                    # metadata (position IDs, seqlens). The pixel values are never
                    # used because Q-Former replaces all ViT features.
                    H, W = precomputed_image_shape
                    vit_image_input = Image.new("RGB", (W, H))
                else:
                    vit_image_input = image
                generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                    curr_kvlens=kv_lens,
                    curr_rope=ropes,
                    images=[vit_image_input],
                    transforms=self.vit_transform,
                    new_token_ids=self.new_token_ids,
                )
                # Priority order for Q-Former input latent:
                # 1. precomputed_vae_latent (from gen_image, skip decode→re-encode)
                # 2. vae_clean_latent_info (from this call's VAE path, skip double encode)
                # 3. _prepare_qformer_latent(image) (fallback: fresh VAE encode)
                if precomputed_vae_latent is not None:
                    # From gen_image: raw denoised latent, no decode needed.
                    H, W = precomputed_image_shape
                    h = H // self.model.latent_downsample
                    w = W // self.model.latent_downsample
                    device = precomputed_vae_latent.device
                    qformer_kwargs = {
                        'packed_clean_vae_latent': precomputed_vae_latent,
                        'clean_vae_token_seqlens': torch.tensor(
                            [h * w], dtype=torch.int, device=device,
                        ),
                    }
                    if get_rank() == 0:
                        print(f"[Q-Former] using precomputed denoised latent (shape={tuple(precomputed_vae_latent.shape)})")
                elif vae_clean_latent_info is not None:
                    qformer_kwargs = vae_clean_latent_info
                    if get_rank() == 0:
                        print(f"late vae_clean_info is :{vae_clean_latent_info['packed_clean_vae_latent'][0]}")
                else:
                    qformer_kwargs = self._prepare_qformer_latent(image)
                past_key_values = self.model.forward_cache_update_vit(
                    past_key_values, **generation_input, **qformer_kwargs
                )
            else:
                # Standard ViT path
                generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                    curr_kvlens=kv_lens,
                    curr_rope=ropes,
                    images=[image],
                    transforms=self.vit_transform,
                    new_token_ids=self.new_token_ids,
                )
                past_key_values = self.model.forward_cache_update_vit(
                    past_key_values, **generation_input
                )

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values

        return gen_context

    def _prepare_qformer_latent(self, image):
        """Encode image to patchified VAE latent for Q-Former inference.

        Returns dict with packed_clean_vae_latent and clean_vae_token_seqlens
        ready to be passed as kwargs to forward_cache_update_vit.
        """
        image_tensor = self.vae_transform(image)
        H, W = image_tensor.shape[1:]
        h = H // self.model.latent_downsample
        w = W // self.model.latent_downsample
        p = self.model.latent_patch_size

        # Encode to VAE latent
        padded = image_tensor.unsqueeze(0)  # (1, C, H, W)
        with torch.no_grad():
            latent = self.vae_model.encode(padded.to(next(self.vae_model.parameters()).device))
        latent = latent[0]  # (z_channels, H', W')

        # Patchify: (z_channels, h*p, w*p) -> (h*w, p*p*z_channels)
        latent = latent[:, :h * p, :w * p].reshape(
            self.model.latent_channel, h, p, w, p
        )
        latent = torch.einsum("chpwq->hwpqc", latent).reshape(
            -1, p * p * self.model.latent_channel
        )

        return {
            'packed_clean_vae_latent': latent,
            'clean_vae_token_seqlens': torch.tensor([h * w], dtype=torch.int, device=latent.device),
        }

    @torch.no_grad()
    def gen_image(
        self,
        image_shape,
        gen_context,
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,

        cfg_text_precontext=None,
        cfg_img_precontext=None,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",

        num_timesteps=50,
        timestep_shift=3.0,
        enable_taylorseer=False,
        # CFG-Zero* + APG
        cfg_zero_init_steps=0,
        cfg_zero_optimal_rescale=0.0,
        cfg_guidance_mode="renorm",
        apg_text_eta=0.5,
        apg_img_eta=0.5,
        verbose=False,
    ):
        # print(cfg_renorm_type)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
        ) 
        
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            enable_taylorseer=enable_taylorseer,
            prev_img_global_feats=gen_context.get('style_features', None),
            cfg_zero_init_steps=cfg_zero_init_steps,
            cfg_zero_optimal_rescale=cfg_zero_optimal_rescale,
            cfg_guidance_mode=cfg_guidance_mode,
            apg_text_eta=apg_text_eta,
            apg_img_eta=apg_img_eta,
            verbose=verbose,
        )

        # if get_rank()==0:
        #     current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        #     print(f"\n----[AFTER IMAGE ENCODE][{current_time}]----")
        #     print(f"before vae_clean_info is :{unpacked_latent[0][0]}")

        raw_latent = unpacked_latent[0]  # (h*w, p*p*z_channels), patchified

        # When Q-Former is enabled, skip the expensive VAE decode since the
        # raw latent will be passed directly to Q-Former for understanding.

        use_qformer = (
            getattr(self.model.config, 'enable_qformer_distill', False)
            and getattr(self.model, 'qformer_alpha', 0.0) >= 1.0
        )
        if use_qformer:
            if get_rank() == 0:
                current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"\n----[SKIP VAE DECODE (Q-Former)][{current_time}]----")
            return None, raw_latent, image_shape

        image = self.decode_image(raw_latent, image_shape)

        if get_rank()==0:
            current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"\n----[AFTER IMAGE DECODE][{current_time}]----")
        return image, None, image_shape

        
    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output
        
    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,

        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        enable_taylorseer=False,
        max_rounds:int=3,
        # CFG-Zero* + APG
        cfg_zero_init_steps=0,
        cfg_zero_optimal_rescale=0.0,
        cfg_guidance_mode="renorm",
        apg_text_eta=0.5,
        apg_img_eta=0.5,
        verbose=False,
        gen_only=False,
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)
        
        if get_rank()==0:
            current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"\n----[INFERENCE BEFORE CONTEXT INIT][{current_time}]----")

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    if gen_only:
                        system_prompt = GEN_ONLY_SYSTEM_PROMPT
                    else:
                        system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")
            

            if get_rank()==0:
                current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"\n----[INFERENCE AFTER CONTEXT INIT][{current_time}]----")
            
            if gen_only:
                gen_context = self.update_context_text("<image_start>", gen_context)
                cfg_img_context = self.update_context_text("<image_start>", cfg_img_context)

            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)
            
            else:
                rounds = 0
                while rounds < max_rounds:
                    if not gen_only:
                        gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                        if get_rank() == 0:
                            print(f'******[RANK{get_rank()}]-[ROUND{rounds}] GEN_TEXT is:******\n', gen_text)
                        output_list.append(gen_text)
                        gen_context = self.update_context_text(gen_text, gen_context)
                    else:
                        gen_text = "<image_start>"

                    if get_rank()==0:
                        current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        print(f"\n----[TURN_{rounds} AFTER TEXT UPDATE][{current_time}]----")
                    
                    if "<image_start>" in gen_text:
                        result = self.gen_image(
                            image_shapes,
                            gen_context,
                            cfg_text_precontext=cfg_text_context,
                            cfg_img_precontext=cfg_img_context,
                            cfg_text_scale=cfg_text_scale,
                            cfg_img_scale=cfg_img_scale,
                            cfg_interval=cfg_interval,
                            timestep_shift=timestep_shift,
                            num_timesteps=num_timesteps,
                            cfg_renorm_min=cfg_renorm_min,
                            cfg_renorm_type=cfg_renorm_type,
                            cfg_zero_init_steps=cfg_zero_init_steps,
                            cfg_zero_optimal_rescale=cfg_zero_optimal_rescale,
                            cfg_guidance_mode=cfg_guidance_mode,
                            apg_text_eta=apg_text_eta,
                            apg_img_eta=apg_img_eta,
                            verbose=verbose,
                        )
                        img, raw_latent, img_shape = result

                        if img is not None:
                            output_list.append(img)

                        if get_rank()==0:
                            current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                            print(f"\n----[TURN_{rounds} AFTER IMAGE UPDATE][{current_time}]----")

                        if img is not None:
                            img_input = self.vae_transform.resize_transform(pil_img2rgb(img))
                        else:
                            # Q-Former skip-decode path: create a dummy image of the
                            # correct size so prepare_vit_images can compute metadata
                            # (position IDs, seqlens). Pixel values are unused.
                            H, W = img_shape
                            img_input = Image.new("RGB", (W, H))
                        gen_context = self.update_context_image(
                            img_input, gen_context, vae=not understanding_output,
                            precomputed_vae_latent=raw_latent,
                            precomputed_image_shape=img_shape,
                        )
                        rounds += 1

                        if get_rank()==0:
                            current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                            print(f"\n----[TURN_{rounds} FINISH IMG CACHE][{current_time}]----")

                        # Pure T2I: one image is enough, break out of the loop
                        if gen_only:
                            break
                    else:
                        break

        return output_list
    
    def __call__(
        self, 
        image: Optional[Image.Image] = None, 
        text: Optional[str] = None, 
        input_list = None, 
        **kargs
    ) -> Dict[str, Any]:
    
        if input_list is None:
            output_dict = {'image': None, 'text': None}

            if image is None and text is None:
                print('Please provide at least one input: either an image or text.')
                return output_dict

            input_list = []
            if image is not None:
                input_list.append(image)
            if text is not None:
                input_list.append(text)
        
        # print(f"the current input is {input_list}")

        output_list = self.interleave_inference(input_list, **kargs)

        return output_list
