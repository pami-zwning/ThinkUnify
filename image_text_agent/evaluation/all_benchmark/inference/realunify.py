import os
import re
from PIL import Image

def data_to_step1(item):
    messages = [
        dict(type='image', value=item["input_image"]),
        dict(type='text', value=item["evaluation_prompt"] if item["task_type"]=="mental_reconstruction" else item["edit_prompt"]),
    ]
    return messages

def data_to_step2(item, image):
    question = item["question"]

    final_text = re.sub(r"(Question: ).*?(\nOption:)", rf"\1{question}\2", item["evaluation_prompt"], flags=re.DOTALL)
    messages = [
        dict(type='image', value=image),
        dict(type='text', value=final_text),
    ]
    return messages

def all_data_to_step(item):
    messages = [
        dict(type='image', value=item["input_image"]),
        dict(type='text', value=item["evaluation_prompt"] if item["task_type"]=="mental_reconstruction" else item["edit_prompt"] + item["evaluation_prompt"]),
    ]
    return messages 

# step-wise的推理流程
def bagel_pipeline_step(vlm, item):
    # reproduce
    final_output, image_list = vlm.internal_infer(
        all_data_to_step(item), understanding_output=False, think=True, max_rounds=6)
    if len(image_list) == 0:
        return final_output, [None, None], final_output
    edited_image = image_list[-1]
    origin_image = edited_image.replace(".", "_ori.")
    Image.open(item["input_image"]).convert("RGB").save(origin_image)

    return final_output, [origin_image, edited_image], final_output

    # # Step1: Generate image
    # step1_output, image_list = vlm.internal_infer(
    #     data_to_step1(item), understanding_output=False, think=True, max_rounds=1)

    # if len(image_list) == 0:
    #     return None, None, None
    # edited_image = image_list[-1]
    # origin_image = edited_image.replace(".", "_ori.")
    # Image.open(item["input_image"]).convert("RGB").save(origin_image)
    # final_output, _ = vlm.internal_infer(
    #     data_to_step2(item, image=edited_image), understanding_output=False, think=False, max_rounds=3)
    # return final_output, [origin_image, edited_image], step1_output


# 将数据转化成messages, 由于realunify direct/step都可以从step的data得到, 这样较为方便
def to_messages(item):
    return item