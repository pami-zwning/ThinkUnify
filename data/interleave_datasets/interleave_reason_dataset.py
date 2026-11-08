from io import BytesIO
import random
import traceback
import re
import PIL
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class InterleaveReasoningIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def __init__(self, *args, use_cot_boundary_loss=False, use_stage_weighting=False, **kwargs):
        # Call parent class initialization
        super().__init__(*args, **kwargs)

        # CoT improvement flags
        self.use_cot_boundary_loss = use_cot_boundary_loss
        self.use_stage_weighting = use_stage_weighting

        # CoT token IDs (added during tokenizer setup)
        self.think_start_token = '<think>'
        self.think_end_token = '</think>'
        self.answer_start_token = '<answer>'
        self.answer_end_token = '</answer>'

    def _identify_cot_boundaries(self, text):
        """
        识别text中的CoT边界和推理阶段
        Returns: list of tuples (char_start, char_end, token_type, stage)
        token_type: 'think_start', 'think_end', 'answer_start', 'answer_end'
        stage: 'setup', 'intermediate', 'final'
        """
        boundaries = []

        # Find all think and answer tags
        think_starts = [(m.start(), m.end(), 'think_start') for m in re.finditer(r'<think>', text)]
        think_ends = [(m.start(), m.end(), 'think_end') for m in re.finditer(r'</think>', text)]
        answer_starts = [(m.start(), m.end(), 'answer_start') for m in re.finditer(r'<answer>', text)]
        answer_ends = [(m.start(), m.end(), 'answer_end') for m in re.finditer(r'</answer>', text)]

        all_boundaries = think_starts + think_ends + answer_starts + answer_ends
        all_boundaries.sort(key=lambda x: x[0])  # Sort by start position

        return all_boundaries

    def _parse_reasoning_stage(self, text):
        """
        解析推理文本中的阶段（setup, intermediate, final）
        """
        # 简化启发式：检查特定的词语来确定阶段
        setup_keywords = ['given', 'let', 'assume', 'consider', 'suppose']
        final_keywords = ['therefore', 'thus', 'finally', 'conclude', 'answer', 'result', 'output']

        text_lower = text.lower()

        # 检查是否包含setup关键词
        if any(kw in text_lower for kw in setup_keywords):
            return 'setup'
        # 检查是否包含final关键词
        elif any(kw in text_lower for kw in final_keywords):
            return 'final'
        else:
            return 'intermediate'

    def _add_text_with_cot(self, data, text, need_loss, enable_cfg=True, section_type="solution"):
        """
        增强版 _add_text，添加CoT边界和阶段信息
        """
        text_ids = self.tokenizer.encode(text)
        data['num_tokens'] += len(text_ids)
        data['text_ids_list'].append(text_ids)

        # 基础信息
        plan_item = {
            'type': 'text',
            'enable_cfg': int(enable_cfg),
            'loss': int(need_loss),
            'special_token_loss': 0,
            'special_token_label': None,
        }

        # 新增：CoT边界和阶段信息
        if section_type == "solution" and (self.use_cot_boundary_loss or self.use_stage_weighting):
            # 识别CoT边界
            boundaries = self._identify_cot_boundaries(text)
            plan_item['cot_boundaries'] = boundaries

            # 识别推理阶段
            stage = self._parse_reasoning_stage(text)
            plan_item['reasoning_stage'] = stage

            # 计算阶段权重
            if self.use_stage_weighting:
                if stage == 'setup':
                    plan_item['stage_weight'] = 0.8
                elif stage == 'intermediate':
                    plan_item['stage_weight'] = 1.5
                elif stage == 'final':
                    plan_item['stage_weight'] = 2.0
                else:
                    plan_item['stage_weight'] = 1.0
            else:
                plan_item['stage_weight'] = 1.0
        else:
            plan_item['cot_boundaries'] = []
            plan_item['reasoning_stage'] = None
            plan_item['stage_weight'] = 1.0

        data['sequence_plan'].append(plan_item)
        return data

    def _get_pil_image(self, image_data):
        """
        Elegantly handles loading an image whether it's in bytes format or already a PIL Image object.
        """
        if isinstance(image_data, bytes):
            # If the data is bytes, open it from a byte stream
            return Image.open(BytesIO(image_data))
        elif isinstance(image_data, Image.Image) or isinstance(image_data, PIL.PngImagePlugin.PngImageFile):
            # If it's already a PIL Image object, just return it
            return image_data
        elif isinstance(image_data, dict) and 'bytes' in image_data:
            # Handle the corner case where image is a dict with a 'bytes' key
            return Image.open(BytesIO(image_data['bytes']))
        else:
            # If the type is unexpected, raise an error for easier debugging
            raise TypeError(f"Unsupported image data type: {type(image_data)}")

    def parse_row(self, row):
        """Parse a single parquet row into the required format"""
        try:
            # Extract basic fields
            answer = row.get('answer', '')
            question_interleave = row.get('question_interleave', [])
            solution_interleave = row.get('solution_interleave', [])
            image_list = row.get('image_list', [])

            if len(question_interleave) == 0 or len(solution_interleave) == 0:
                print(question_interleave, solution_interleave, row.keys())
                return {}
            
            data = self._init_data()
            
            # 1. Process question part (question_interleave)
            success = self._add_interleave_sequence(
                data, 
                question_interleave,
                image_list,
                section_type="question"
            )
            if not success:  # Image loading failed
                return {}
            
            # 2. Process solution part (solution_interleave)  
            success = self._add_interleave_sequence(
                data,
                solution_interleave,
                image_list,
                section_type="solution"
            )
            if not success:  # Image loading failed
                return {}
            
            return data
            
        except Exception as e:
            traceback.print_exc()
            print(f"Error parsing row: {e}")
            return {}

    def _add_interleave_sequence(self, data, interleave_list, images_list, section_type):
        """Add interleaved text and image sequence in original order"""

        try:
            image_count = 0

            for i, item in enumerate(interleave_list):
                if item['type'] == 'text':
                    text_content = item['content']
                    if text_content and text_content.strip():
                        # Determine if this text should predict special token
                        special_token_label = None

                        if section_type == "question":
                            # Question texts don't need loss, no special token prediction
                            need_loss = False
                        else:  # solution
                            # Solution texts need loss and may predict special tokens
                            need_loss = True

                            # Check if next item is an image
                            has_following_image = (i + 1 < len(interleave_list)) and (interleave_list[i + 1]['type'] == 'image')

                            # Check if this is the last item in the sequence
                            is_last_item = (i == len(interleave_list) - 1)

                            # if has_following_image:
                            #     special_token_label = self.start_of_image
                            # elif is_last_item:
                            #     # Last text in solution predicts end_of_text
                            #     special_token_label = self.end_of_text
                            # else:
                            #     special_token_label = self.im_start

                        # 使用新的方法来处理文本和CoT信息
                        if self.use_cot_boundary_loss or self.use_stage_weighting:
                            data = self._add_text_with_cot(
                                data,
                                text_content.strip(),
                                need_loss=need_loss,
                                enable_cfg=True,
                                section_type=section_type,
                            )
                        else:
                            data = self._add_text(
                                data,
                                text_content.strip(),
                                need_loss=need_loss,
                                enable_cfg=True,
                            )

                elif item['type'] == 'image':
                    image_count += 1

                    try:
                        image_index = int(item['index'])
                        if image_index < len(images_list):
                            image_data = images_list[image_index]
                            image = self._get_pil_image(image_data)
                            image = pil_img2rgb(image)
                        else:
                            print(f"Image index {image_index} out of bounds for images_list of length {len(images_list)}")
                            return False

                        if section_type == "question":
                            # Question images: no loss, but need both VAE and VIT for potential editing and understanding
                            data = self._add_image(
                                data,
                                image,
                                need_loss=False,
                                need_vae=True,
                                need_vit=True,
                                enable_cfg=True,
                            )
                        else:  # solution
                            # Solution images: always need loss, VAE, and VIT for train-inference consistency
                            data = self._add_image(
                                data,
                                image,
                                need_loss=True,
                                need_vae=True,
                                need_vit=True,
                                enable_cfg=True,
                            )

                    except Exception as e:
                        print(f"Failed to load image with index {item.get('index', 'N/A')}: {e}")
                        traceback.print_exc()
                        return False

            return True

        except Exception as e:
            traceback.print_exc()
            print(f"Error processing interleave sequence: {e}")
            return False