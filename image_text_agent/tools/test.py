import os
import json
from PIL import Image
from desc import TOOLS
from openai import OpenAI
from utils import run_toolcall, encode_image_with_format, toolcall_meesage_convert

def main():
      all_fn_names = [item['function']['name'] for item in TOOLS]
      print(f"Testing fn names: {all_fn_names}")

      image_path = './demo.jpg'
      save_image_root = './results'
      os.makedirs(save_image_root, exist_ok=True)

      image_list = [Image.open(image_path)]

      host_ip = "10.119.16.206"
      client = OpenAI(api_key="YOUR_API_KEY", base_url=f"http://{host_ip}:8000/v1")  # thinking
      model_name = client.models.list().data[0].id

      for fn_name in all_fn_names:
            content_list = []
            
            img_url = encode_image_with_format(image_path)
            content_list.append({
                  "type": "image_url",
                  "image_url": img_url
            })
            content_list.append({"type": "text", "text": f"请你正确地调用工具 {fn_name} 并使得调用后的结果与原图有明显差异。"})
            messages = [{"role": "user", "content": content_list}]

            response = client.chat.completions.create(
                  model=model_name,
                  messages=messages,
                  tools=TOOLS,
                  top_p=0.25,
                  temperature=0.5,
                  max_tokens=1024*16,
                  extra_body={
                  "repetition_penalty": 1.05,
                  "chat_template_kwargs": {"enable_thinking": True}  # default to True
                  },
            )
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
                                          print(f"{fn_name} result save to {save_path}")

                              if 'text' in function_return:
                                    print(function_return['text'])

            else:
                  print(f"model calls {fn_name} tool fail")

      print("finish!")
            

if __name__ == "__main__":
    main()