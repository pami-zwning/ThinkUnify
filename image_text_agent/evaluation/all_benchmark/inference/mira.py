import os
import re
from PIL import Image

def message_to_step(item):
    assert item["messages"][0]["role"] == "user"
    content = item["messages"][0]["content"]
    
    messages = []
    for value in content:
        if value["type"] == "text":
            messages.append(dict(type='text', value=value["text"]))
        elif value["type"] == "image_url":
            messages.append(dict(type='image', value=value["image_url"]["url"]))
        else:
            raise ValueError(f"{value}")
    return messages

# step-wise的推理流程
def bagel_pipeline(vlm, item):
    final_output, image_list = vlm.internal_infer(
        message_to_step(item), understanding_output=False, think=True, max_rounds=6)
    return final_output, image_list, None           # Final output, Generated image, Mid output