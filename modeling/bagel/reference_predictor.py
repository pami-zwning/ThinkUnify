# Copyright 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Reference Index Prediction for StoryFlow
Allows the model to learn which reference image to use during generation.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from PIL import Image


class ReferenceIndexPredictor:
    """
    ✨ 新增：在推理时预测参考图索引。

    模型在训练时学习到：给定当前context，应该基于哪张前面的图生成新图。
    在推理时，模型可以自动预测出reference_idx，无需手动指定。
    """

    def __init__(self, model, device: torch.device):
        """
        Args:
            model: Bagel model with reference_predictor head
            device: Device to run on
        """
        self.model = model
        self.device = device

    @staticmethod
    def convert_logits_to_reference_idx(
        reference_logits: torch.Tensor,
        num_images_generated: int,
        default_to_anchor: bool = True,
    ) -> int:
        """
        将logits转换为reference_idx。

        Args:
            reference_logits: [num_classes] - reference predictor的输出
            num_images_generated: 已生成的图像数量
            default_to_anchor: ✨ NEW: 当不确定时是否默认基于anchor（第一张图）

        Returns:
            reference_idx:
                - 0: 参考第一张图（anchor）
                - 1, 2, ...: 参考具体的某张图
                - -1: 无参考（纯文本生成）
        """
        # 最后一个类代表"无参考"
        num_classes = reference_logits.shape[0]
        num_reference_classes = num_classes - 1

        # 获取预测的类别
        pred_class = reference_logits.argmax(dim=-1).item()

        # 如果预测的是最后一个类（无参考）
        if pred_class == num_reference_classes:
            # ✨ NEW: 默认情况下，改为返回0（anchor）而不是-1
            # 这提高了鲁棒性，确保至少有一个参考
            if default_to_anchor:
                return 0  # 默认基于第一张图（anchor）
            else:
                return -1  # 坚持无参考

        # 否则，预测的是哪张前面的图
        # 需要确保不超过已生成的图像数量
        reference_idx = min(pred_class, num_images_generated - 1)
        return reference_idx

    @staticmethod
    def apply_temperature_to_logits(
        logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        对logits应用temperature，用于控制预测的确定性。

        Args:
            logits: 原始logits
            temperature:
                - 1.0: 正常
                - < 1.0: 更确定（peak更尖锐）
                - > 1.0: 更不确定（分布更平坦）

        Returns:
            温度调节后的logits
        """
        return logits / temperature

    @staticmethod
    def get_reference_probability_distribution(
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        获取参考图选择的概率分布。

        Returns:
            probs: [num_classes] - softmax后的概率分布
        """
        return torch.softmax(logits, dim=-1)


class ReferenceAwareImageGenerator:
    """
    ✨ 完整的推理管道：自动预测参考图并生成新图。
    """

    def __init__(self, model, vit_model, vae_model, tokenizer, device: torch.device):
        """
        Args:
            model: Bagel model with both style consistency and reference prediction
            vit_model: Vision Transformer for feature extraction
            vae_model: VAE for encoding/decoding
            tokenizer: Tokenizer for text processing
            device: Device to run on
        """
        self.model = model
        self.vit_model = vit_model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.device = device
        self.predictor = ReferenceIndexPredictor(model, device)

        # 保存生成历史
        self.generated_images: List[Image.Image] = []
        self.image_features: List[torch.Tensor] = []
        self.generation_history: List[dict] = []

    @torch.no_grad()
    def generate_with_reference_prediction(
        self,
        prompt: str,
        temperature: float = 1.0,
        verbose: bool = True,
        default_to_anchor: bool = True,  # ✨ NEW
    ) -> Tuple[Image.Image, int, torch.Tensor]:
        """
        生成下一张图像，自动预测参考图。

        Args:
            prompt: 生成指令
            temperature: 参考预测的temperature
            verbose: 是否打印预测信息
            default_to_anchor: ✨ NEW: 当不确定时是否默认基于anchor（第一张图）

        Returns:
            (generated_image, predicted_reference_idx, reference_logits)
        """

        # Step 1: 前向传播获取reference logits
        # 这里需要准备输入，进行前向传播
        reference_logits = self._forward_to_get_reference_logits(prompt)

        # Step 2: 应用temperature
        if temperature != 1.0:
            reference_logits = self.predictor.apply_temperature_to_logits(
                reference_logits,
                temperature=temperature
            )

        # Step 3: 获取参考图索引
        # ✨ NEW: 传入default_to_anchor参数
        predicted_ref_idx = self.predictor.convert_logits_to_reference_idx(
            reference_logits,
            num_images_generated=len(self.generated_images),
            default_to_anchor=default_to_anchor,
        )

        # Step 4: 获取参考特征
        if predicted_ref_idx >= 0 and predicted_ref_idx < len(self.image_features):
            ref_features = self.image_features[predicted_ref_idx]
            validity_mask = torch.ones(1, dtype=torch.bool, device=self.device)  # 有效
        else:
            ref_features = None
            validity_mask = torch.zeros(1, dtype=torch.bool, device=self.device)  # 无效

        # Step 5: 生成图像
        # 这里调用generate_image，使用pred_ref_idx对应的特征
        generated_image = self._generate_image_with_reference(
            prompt=prompt,
            ref_features=ref_features,
            validity_mask=validity_mask,
        )

        # Step 6: 保存到历史
        self.generated_images.append(generated_image)
        gen_features = self._extract_features(generated_image)
        self.image_features.append(gen_features)

        self.generation_history.append({
            'prompt': prompt,
            'predicted_reference_idx': predicted_ref_idx,
            'reference_logits': reference_logits.cpu(),
            'reference_probs': self.predictor.get_reference_probability_distribution(
                reference_logits
            ).cpu(),
        })

        if verbose:
            probs = self.predictor.get_reference_probability_distribution(reference_logits)
            print(f"✓ Generated image based on figure #{predicted_ref_idx}")
            print(f"  Reference probabilities: {probs.cpu().numpy()[:min(5, len(probs))]}")

        return generated_image, predicted_ref_idx, reference_logits

    def _forward_to_get_reference_logits(self, prompt: str) -> torch.Tensor:
        """
        执行前向传播获取参考logits。

        这需要与实际的forward实现配合。
        """
        # 这是伪代码，实际实现需要根据generate_image的结构来调整
        raise NotImplementedError(
            "这个方法需要在你的推理管道中具体实现。"
            "通常在generate_image中添加一个hook来提取reference_logits。"
        )

    def _generate_image_with_reference(
        self,
        prompt: str,
        ref_features: Optional[torch.Tensor],
        validity_mask: Optional[torch.Tensor],
    ) -> Image.Image:
        """
        使用reference特征生成图像。
        """
        # 调用model.generate_image，传入ref_features和validity_mask
        # 这里省略具体实现细节
        pass

    def _extract_features(self, image: Image.Image) -> torch.Tensor:
        """
        从图像提取全局特征。
        """
        # 使用ViT提取特征
        pass

    def show_generation_history(self):
        """显示生成历史和参考关系"""
        print("\n=== Generation History ===")
        for i, record in enumerate(self.generation_history):
            ref_idx = record['predicted_reference_idx']
            probs = record['reference_probs']

            status = "✓" if ref_idx >= 0 else "✗"
            ref_str = f"figure #{ref_idx}" if ref_idx >= 0 else "none"
            conf = probs.max().item() if len(probs) > 0 else 0

            print(f"{i}: [{status}] Based on {ref_str} (confidence: {conf:.2%})")
            print(f"   Prompt: {record['prompt']}")

    def plot_reference_graph(self):
        """
        绘制参考关系图。

        显示每张生成的图像基于哪张图。
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # 构建图关系
        refs = [h['predicted_reference_idx'] for h in self.generation_history]
        num_images = len(self.generated_images)

        # 创建有向图
        fig, ax = plt.subplots(figsize=(12, 8))

        # 绘制节点
        x_pos = np.arange(num_images)
        y_pos = np.zeros(num_images)

        for i in range(num_images):
            # 第一个是anchor
            if i == 0:
                ax.plot(i, 0, 'go', markersize=15, label='Anchor')
            # 有参考的
            elif i - 1 < len(refs) and refs[i - 1] >= 0:
                ax.plot(i, 0, 'bo', markersize=15)
            # 无参考的
            else:
                ax.plot(i, 0, 'r^', markersize=15)

        # 绘制箭头表示参考关系
        for i, ref_idx in enumerate(refs):
            if ref_idx >= 0:
                ax.arrow(ref_idx, 0.05, i + 1 - ref_idx - 0.1, -0.05,
                        head_width=0.05, head_length=0.05, fc='gray', ec='gray', alpha=0.5)

        # 设置标签
        ax.set_xlim(-0.5, num_images - 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Img {i}" for i in range(num_images)])
        ax.set_yticks([])
        ax.set_title("Reference Dependency Graph")
        ax.legend()

        plt.tight_layout()
        return fig


# 简化的推理API
def infer_with_reference_prediction(
    model,
    initial_image: Image.Image,
    prompts: List[str],
    temperature: float = 1.0,
) -> Tuple[List[Image.Image], List[int]]:
    """
    多轮生成，自动预测参考图。

    Args:
        model: Bagel model
        initial_image: 初始图像
        prompts: 一系列生成指令
        temperature: 参考预测的temperature

    Returns:
        (generated_images, reference_indices)
    """
    generator = ReferenceAwareImageGenerator(
        model=model,
        vit_model=model.vit_model if hasattr(model, 'vit_model') else None,
        vae_model=None,  # 从model中获取
        tokenizer=None,  # 需要传入
        device=next(model.parameters()).device,
    )

    # 添加初始图像
    generator.generated_images.append(initial_image)
    gen_features = generator._extract_features(initial_image)
    generator.image_features.append(gen_features)

    # 多轮生成
    reference_indices = []
    for prompt in prompts:
        _, ref_idx, _ = generator.generate_with_reference_prediction(
            prompt=prompt,
            temperature=temperature,
        )
        reference_indices.append(ref_idx)

    return generator.generated_images, reference_indices
