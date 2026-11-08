# image_text_agent
tool_call图文交错代码管理

## LangChain Agent

Install dependencies:

* `python -m pip install -r requirements.txt`
* `langchain-opanai` or other provider packages for API models.
* `transformers` for the local models.

Example configs:

* `configs/react_agent_api.json` for API models, and filling the API keys before using.
* `configs/react_agent_local.json` for local transformer models, and ensuring the model path before using.

Example to chat with agent by the command line:

```shell
PYTHONPATH=. python fvg/utils/agent_chat.py \
    -c {your/config/path} \
    -q file:///mnt/afs/wuzehuan/Documents/image_text_agent/evaluation/0_0.png \
       请在工具的帮助下确认手套的材质:A.橡胶,B.棉,C.塑料,D.皮革 \
    -o output.json
```

Example to chat with the agent by Python code:

```python
import fvg.common
import json

with open("path/to/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

agent = fvg.common.make_object_from_config(config["agent"])
output = agent.invoke(
    input={
        "messages": [
            {"role": "user", "content": "第40个斐波那契数是多少"},
        ],
    },
    config={
        "configurable": {
            "thread_id": "main"
        },
        "recursion_limit": 8
    })
```

### Bridge the image processing functions

The [ImageToolWrapper](/fvg/tools/image_tool_wrapper.py) is the LangChain tool wrapper for image processing functions, converting the parameter of PIL image in the schema to image index number for model, and converting the image index from the model's tool call to the PIL image to the underlying function, without extra effort.

The processing function example:
```python
from PIL import Image

def crop_image(image: Image.Image, bbox: list):
    """Crop the image based on the bounding box coordinates to zoom in on
    specific regions for detailed analysis.

    Args:
        image: The input image.
        bbox: Normalized bounding box [x1, y1, x2, y2], where
            0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0. (x1,y1) is
            top-left corner, (x2,y2) is bottom-right corner.
    """

    img_x, img_y = image.size
    x1, y1, x2, y2 = [float(coord) for coord in bbox]
    return image.crop((
        int(x1 * img_x), int(y1 * img_y),
        int(x2 * img_x), int(y2 * img_y)
    ))
```

The tool example:
```python
from fvg.tools.image_tool_wrapper import ImageToolWrapper
from fvg.functions.image import crop_image

# make the tool
crop_image_tool = ImageToolWrapper(
    name="crop_image",
    underlying_func=crop_image,
    input_parameters=["image"],
)

# get the tool schema
crop_image_tool_schema = crop_image_tool.tool_call_schema.model_json_schema()

# dumped result of the tool schema
# {
#   "description": "Process the image from the message list with given function.",
#   "properties": {
#     "bbox": {
#       "description": "Normalized bounding box [x1, y1, x2, y2], where 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0. (x1,y1) is top-left corner, (x2,y2) is bottom-right corner.",
#       "items": {},
#       "title": "Bbox",
#       "type": "array"
#     },
#     "image_index": {
#       "description": "The image index of the message contents for the purpose of 'The input image.'.",
#       "title": "Image Index",
#       "type": "integer"
#     }
#   },
#   "required": [
#     "bbox",
#     "image_index"
#   ],
#   "title": "crop_image",
#   "type": "object"
# }
```