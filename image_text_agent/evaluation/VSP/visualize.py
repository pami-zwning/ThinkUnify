import json
import os


def read_jsonl_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [json.loads(line) for line in lines]
def read_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

json_path = "/mnt/afs/mashengnan/o3like/Test_Code/image_text_agent/evaluation/Visual-Spatial-Planning/VSP-main/maze/task-main/Results/Jsons/qwen3vl_sft_maze_20260108.json"

json_data = read_json_file(json_path)
vis_jsonl_path = json_path.replace('.json', "_vis.jsonl")
#vis_jsonl_path = '/mnt/afs/mashengnan/o3like/Test_Code/image_text_agent/evaluation/TIR-Bench/Results/Extract_Answer/qwen3vl-sft-xiangli-1222_sft_final_vis.jsonl'
with open(vis_jsonl_path, 'w') as wfile:
    for item in json_data:
        images = item['all_images_list']
        question = item['prompt']
        final_response = item['model_response']

        jsonl_item = {'messages':[{"role": "user", "content": f"<image>\n{question}"}]}

        messages = item['mid_process']
        for message in messages:
            if message['role'] == 'user':
                continue

            elif message['role'] == 'assistant':
                assistent_content = message['content']
                
                if message.get("tool_calls", None):
                    toolcalls = message['tool_calls']
                    for tool_call in toolcalls:
                        func_name = tool_call['function']['name']
                        fn_args = json.loads(tool_call['function']['arguments'])
                        #tooltype = fn_args['name']
                        assistent_content+=f"\n{json.dumps(fn_args, ensure_ascii=False)}"

                jsonl_item['messages'].append({"role": "assistant", "content":assistent_content})
                
            elif message['role'] == 'tool':
                tool_content_str = ''
                for tool_content in message['content']:
                    if tool_content['type'] == 'text':
                        tool_content_str += str(tool_content['text'])
                    if tool_content['type'] == 'image_url':
                        tool_content_str += "<image>"
                tool_message = {"role": "user", "content":tool_content_str}
                jsonl_item['messages'].append(tool_message)

        jsonl_item['images'] = images
        wfile.write(json.dumps(jsonl_item, ensure_ascii=False)+'\n')