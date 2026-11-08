import os
import json
from PIL import Image
from desc import TOOLS
from openai import OpenAI
from utils import run_toolcall, encode_image_with_format, toolcall_meesage_convert

TOOL_DIY=[{
            "type": "function",
            "function": {
                  "name": "point_image",
                  "description": "Draw points on the image at the specified coordinates, which are normalized in the range [0, 1000]. Each point is represented as a small red circle on the image.",
                  "parameters": {
                        "type": "object",
                        "properties": {
                        "points_2d": {
                        "type": "array",
                        "items": {
                              "type": "array",
                              "items": { "type": "number" },
                              "minItems": 2,
                              "maxItems": 2
                        },
                          "description": "A list of coordinate pairs [(x, y)], where each pair represents a point to be drawn on the image. The coordinates are in the range 0 <= x, y <= 1000, where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner. Each point will be rendered as a small red circle on the image."
                        },
                        "image_index": {
                        "type": "integer",
                        "minimum": -1,
                        "description": "Index of the image to draw the points on: 1 for original input image, 2 for first modified image, 3 for second modified image, etc. If the value is -1, it will use the most recently processed image from the previous step."
                        }
                        },
                        "required": ["points_2d", "image_index"]
                  }
            }
      }]
def main():
      all_fn_names = [item['function']['name'] for item in TOOL_DIY]
      print(f"Testing fn names: {all_fn_names}")

      image_path = './test.jpg'
      save_image_root = './results'
      os.makedirs(save_image_root, exist_ok=True)

      image_list = [Image.open(image_path)]

      host_ip = "10.119.26.108"
      client = OpenAI(api_key="YOUR_API_KEY", base_url=f"http://{host_ip}:8000/v1")  # thinking
      model_name = client.models.list().data[0].id

      for fn_name in all_fn_names:
            content_list = []
            
            img_url = encode_image_with_format(image_path)
            content_list.append({
                  "type": "image_url",
                  "image_url": img_url
            })
            content_list.append({"type": "text", "text": f"请你正确地调用两次工具 {fn_name} 并使得调用后的结果与原图有明显差异。第一次调用："})
            messages = [{"role": "user", "content": content_list}]

            for i in range(2):
                  response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        tools=TOOL_DIY,
                        top_p=0.25,
                        temperature=0.5,
                        max_tokens=1024*16,
                        extra_body={
                        "repetition_penalty": 1.05,
                        "chat_template_kwargs": {"enable_thinking": True}  # default to True
                        },
                  )
                  import pdb;pdb.set_trace()
                  #print(response.choices[0].message.model_dump())
                  messages.append(response.choices[0].message.model_dump())
                  if tool_calls := messages[-1].get("tool_calls", None):
                        for tool_call in tool_calls:
                              call_id: str = tool_call["id"]
                              if fn_call := tool_call.get("function"):
                                    #import pdb;pdb.set_trace()

                                    fn_name: str = fn_call["name"]
                                    print(f"running {fn_name}")
                                    fn_args: dict = json.loads(fn_call["arguments"])
                                    function_return = run_toolcall(fn_name, fn_args, image_list)
                                    #import pdb;pdb.set_trace()
                                    function_return_content = toolcall_meesage_convert(function_return)
                                    messages.append({
                                          "tool_call_id": call_id,
                                          "role": "tool",
                                          "name": fn_name,
                                          "content": function_return_content,
                                    })

                                    if 'image' in function_return:
                                          for image_idx, image_item in enumerate(function_return['image']):
                                                img_name = os.path.basename(image_path).split('.')[0]
                                                img_format = os.path.basename(image_path).split('.')[-1]
                                                save_path = os.path.join(save_image_root, f"{img_name}_{fn_name}_{image_idx}.{img_format}")
                                                image_item.save(save_path)
                                                image_list.append(image_item)
                                                print(f"{fn_name} result save to {save_path}")

                                    if 'text' in function_return:
                                          print(function_return['text'])

                  else:
                        print(f"model calls {fn_name} tool fail")

      print("finish!")
            

if __name__ == "__main__":
    main()