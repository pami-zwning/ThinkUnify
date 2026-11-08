from PIL import Image
from qwen_vl_utils import process_vision_info # Ensure this utility script is available
import re
import os
import uuid
import copy
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.utils import *


MAX_ITERATIONS = 5  # Safety break for the loop
MAX_NEW_TOKENS_PER_STEP = 32768 # Max tokens for each generation step

SYSTEM_PROMPT = '''Reason step by step and place the thought process within the <think></think> tags, and provide the final conclusion at the end.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "crop_image", "description": "Crop the image based on the bounding box coordinates to zoom in on specific regions for detailed analysis.", "parameters": { "type": "object", "properties": { "bbox": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4, "description": "Normalized bounding box [x1, y1, x2, y2], where 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0. (x1,y1) is top-left corner, (x2,y2) is bottom-right corner." }, "image_index": { "type": "number", "minimum": 1, "description": "Index of the image to crop: 1 for original input image, 2 for first cropped image, 3 for second cropped image, etc." }, "scale": {"type": "number","description": "Scaling factor for the crop region. Default is 1.0. If scale > 1.0, the crop region is enlarged proportionally. If scale < 0, no cropping is performed; instead, the bounding box is drawn on the original image."}}, "required": ["bbox", "image_index"]}}}
{"type": "function", "function": {"name": "point_image", "description": "Draw points on an image given normalized 2D coordinates [x, y]. Each point is drawn as a small red circle.", "parameters": { "type": "object", "properties": { "points": { "type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}, "description": "List of normalized points [(x1, y1), (x2, y2), ...], where 0.0 <= x, y <= 1.0." }, "image_index": { "type": "number", "minimum": 1, "description": "Index of the image to draw points on: 1 for original input image, 2 for first processed image, etc." }}, "required": ["points", "image_index"]}}}
{"type": "function", "function": {"name": "locate_character_position", "description": "Locates the position (row and column) of a visually distinct character in a dense grid image. It analyzes the grid structure, identifies the target character that differs from the background, marks it on the image, and returns the precise coordinates in [row, column] format. Use this tool for position-finding questions.", "parameters": { "type": "object", "properties": { "image_index": { "type": "number", "minimum": 1, "description": "Index of the image to analyze: 1 for the original input image." }}, "required": ["image_index"]}}}
{"type": "function", "function": {"name": "count_grid_items", "description": "Counts the occurrences of distinct visual elements in a grid image. Useful for word counting questions. It uses structural difference analysis (template matching) to distinguish similar characters and returns the count for each group.", "parameters": { "type": "object", "properties": { "image_index": { "type": "number", "minimum": 1, "description": "Index of the image to analyze: 1 for the original input image." }}, "required": ["image_index"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>'''

def remove_unpickable_values(dictionary):
    import pickle

    def is_pickable(obj):
        try:
            pickle.dumps(obj)
            return True
        except (pickle.PicklingError, TypeError, AttributeError):
            return False
            
def generate_prompt_final_qa(user_question, user_image_path):
    # Construct the prompt based on the given requirements
    try:
        pil_image = Image.open(user_image_path)
        image_info = str(pil_image)
        info = f"<image>(img_path={user_image_path}, img_info={image_info}\n"
        user_prompt = f"{info}{user_question}"
    except Exception as e:
        user_image_size = "Unable to determine (error reading image)"
    return user_prompt

SPECIAL_STRING_LIST=["</tool_call>","</answer>"]

def run_evaluation(user_question: str, initial_image_url_or_path: str, model, processor):
    import json, ast

    current_image_path_for_code = initial_image_url_or_path
    conversation_history = []

    # System prompt
    conversation_history.append({
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}]
    })

    # First user message
    initial_prompt_text = f"<image>\n{user_question}"
    conversation_history.append({
        "role": "user",
        "content": [
            {"type": "image", "image": initial_image_url_or_path},
            {"type": "text", "text": initial_prompt_text}
        ]
    })

    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1} ---")

        # ---- Build prompt ----
        text_prompt_for_model = processor.apply_chat_template(
            conversation_history,
            tokenize=False,
            add_generation_prompt=(iteration == 0)
        )

        # if iteration != 0 and text_prompt_for_model.endswith("<|im_end|>\n"):
        #     text_prompt_for_model = text_prompt_for_model[:-len("<|im_end|>\n")]
        print(text_prompt_for_model)
        image_inputs, video_inputs = process_vision_info(conversation_history)

        inputs = processor(
            text=text_prompt_for_model,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        print("Generating model response...")

        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS_PER_STEP,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id,
            stop_strings=SPECIAL_STRING_LIST,
            tokenizer=processor.tokenizer
        )

        input_token_len = inputs.input_ids.shape[1]
        new_ids = generated_ids[0, input_token_len:]
        generated_text_segment = processor.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        print(f"LLM Output:\n{generated_text_segment}\n")

        # Temporary container for this assistant turn
        generated_content = []

        # -------------------------
        # 1) Found <answer>
        # -------------------------
        if "</answer>" in generated_text_segment:
            generated_content.append({"type": "text", "text": generated_text_segment.strip()})
            conversation_history.append({"role": "assistant", "content": generated_content})
            print("\033[32m--- Final answer found, stopping. ---\033[0m")
            break

        # -------------------------
        # 2) Found <tool_call>
        # -------------------------
        tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.IGNORECASE | re.DOTALL)
        m = tool_call_regex.search(generated_text_segment)

        if m:
            tool_json_text = m.group(1).strip()
            try:
                tool_call = json.loads(tool_json_text)
            except:
                print("ERROR: Tool call JSON 解析失败：\n", tool_json_text)
                break

            generated_content.append({"type": "text", "text": generated_text_segment.strip()})
            conversation_history.append({"role": "assistant", "content": generated_content})

            func_name = tool_call["name"]
            args = tool_call["arguments"]

            # Normalize args
            if "bbox" in args and isinstance(args["bbox"], str):
                args["bbox"] = ast.literal_eval(args["bbox"])

            # Generate output path
            fname = os.path.basename(current_image_path_for_code)
            name, ext = os.path.splitext(fname)
            new_name = f"{name}_{uuid.uuid4().hex[:6]}{ext}"
            output_path = f"./evaluation/{new_name}"

            # Execute tool
            try:
                idx = args.pop("image_index") - 1
                base_img = image_inputs[idx]
            except Exception:
                print("ERROR：image_index 不合法")
                break
            extra_tool_text = ""
            if func_name == "crop_image":
                result = crop_image(image=base_img, bbox=args["bbox"])
            elif func_name == "point_image":
                result = point_image(image=base_img, points_2d=args["points"])
            elif func_name == "locate_character_position":
                result = locate_character_position(image=base_img)
            elif func_name == "count_grid_items":
                result = count_grid_items(image=base_img)
            else:
                print("ERROR: 未知函数名", func_name)
                break

            if isinstance(result, str):
                print(f"ERROR: {func_name} 执行失败: {result}")
                break
            
            if isinstance(result, tuple):
                new_img, extra_tool_text = result
                if isinstance(new_img, bool):
                    print(f"ERROR: {func_name} 执行失败: {result}")
                    break
            else:
                new_img = result

            new_img.save(output_path)
            current_image_path_for_code = output_path

            tool_response_content = [
                {"type": "text", "text": "<tool_response>\nHere is the processed image:"},
                {"type": "image", "image": output_path}
            ]
            
            if extra_tool_text:
                tool_response_content.append({
                    "type": "text", 
                    "text": f"\n {extra_tool_text}"
                })

            tool_response_content.append({"type": "text", "text": "\n</tool_response>"})

            conversation_history.append({
                "role": "user",
                "content": tool_response_content
            })
            continue

        # -------------------------
        # 3) No tool_call, No answer → Unexpected output
        # -------------------------
        print("WARNING: No <tool_call> and no <answer>, stopping.")
        break

    # -------------------------
    # END LOOP
    # -------------------------

    # Summarize final assistant response
    final_response = ""
    for msg in conversation_history:
        if msg["role"] == "assistant":
            buf = ""
            for item in msg["content"]:
                if item["type"] == "text":
                    buf += item["text"]
            final_response = buf  # last one is final
    m = re.search(r"<answer>(.*?)</answer>", final_response, re.DOTALL)
    final_answer = m.group(1).strip() if m else "No answer tag found."

    return final_response, final_answer ,conversation_history