import yaml
from desc import TOOLS

FUNCTION_TO_CLASS_MAP = {
    "crop_image": "CropImageTool",
    "draw_bbox": "DrawBboxTool",
    "point_image": "PointImageTool",
    "rotate_image": "RotateImageTool",
    "brighten_image": "BrightenImageTool",
    "mirror_image": "MirrorImageTool",
    "route_drawer": "ImageRouteDrawerTool", 
    "draw_line": "DrawLineTool",
    "locate_character_position": "LocateCharacterTool", 
    "count_grid_items": "CountGridItemsTool",
    "rearrange_tiles": "RearrangeTileTool", 
    "get_color_distribution": "ColorDistributionTool"
}

def tools_to_verl_yaml_format(tools_list):
    """
    将 OpenAI 格式的 tools 列表转换为 Verl YAML 配置文件所需的字典结构
    """
    verl_tools_config = []

    for tool_def in tools_list:
        func_name = tool_def['function']['name']
        
        # 1. 获取对应的类名
        if func_name in FUNCTION_TO_CLASS_MAP:
            class_name = FUNCTION_TO_CLASS_MAP[func_name]
        else:
            # 默认 fallback：snake_case -> CamelCase + Tool
            # 例如: my_tool -> MyToolTool
            class_name = ''.join(word.title() for word in func_name.split('_')) + "Tool"
            print(f"Warning: No explicit mapping for '{func_name}', using generated class name '{class_name}'")

        full_class_path = f"verl.tools.Interleave_tools.{class_name}"

        # 2. 构建单个 Tool 的配置项
        tool_entry = {
            "class_name": full_class_path,
            "config": {
                "type": "native"
            },
            "tool_schema": tool_def # 直接复用原始的 tool 定义
        }
        
        verl_tools_config.append(tool_entry)

    # 3. 包装在 "tools" key 下
    return {"tools": verl_tools_config}

def dict_to_yaml(data: dict, path: str):
    """
    将字典保存为 YAML 文件，处理格式细节
    """
    # 自定义 Dumper 以避免 YAML 别名锚点 (aliases/anchors, 如 &id001)
    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                data,
                f,
                allow_unicode=True, # 允许中文等 unicode 字符
                sort_keys=False,    # 保持字典顺序
                width=1000,         # 避免过早换行
                Dumper=NoAliasDumper 
            )
        print(f"Successfully saved YAML configuration to: {path}")
    except Exception as e:
        print(f"Error saving YAML: {e}")

# ==========================================
# 3. 执行
# ==========================================

if __name__ == "__main__":
    # 1. 转换数据结构
    yaml_data = tools_to_verl_yaml_format(TOOLS)
    
    # 2. 保存为 config.yaml
    dict_to_yaml(yaml_data, "config.yaml")