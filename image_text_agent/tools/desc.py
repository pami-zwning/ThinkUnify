import json
TOOLS = [
      {
            "type": "function",
            "function": {
                  "name": "crop_image",
                  "description": "Crop the image based on the bounding box coordinates, which are normalized in the range [0, 1000], to focus on specific regions for detailed analysis.",
                  "parameters": {
                        "type": "object",
                        "properties": {
                              "bbox": {
                                    "type": "array",
                                    "items": { "type": "number" },
                                    "minItems": 4,
                                    "maxItems": 4,
                                    "description": "A bounding box represented as an array of four integers: [x1, y1, x2, y2], where 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000. [x1, y1] is the top-left corner and [x2, y2] is the bottom-right corner of the box."
                              },
                              "image_index": {
                                    "type": "integer",
                                    "minimum": -1,
                                    "description": "Index of the image to crop: 1 for original input image, 2 for first cropped image, 3 for second cropped image, etc. If the value is -1, it will use the most recently processed image from the previous step."
                              },
                        },
                        "required": ["bbox", "image_index"]
                        }
            }
      },
      {
            "type": "function",
            "function": {
                  "name": "draw_bbox",
                  "description": "Draw a bounding box on the image based on the given coordinates, which are normalized in the range [0, 1000], to visualize the object or region of interest.",
                  "parameters": {
                        "type": "object",
                        "properties": {
                        "bbox": {
                              "type": "array",
                              "items": {
                                    "type": "number"
                              },
                              "minItems": 4,
                              "maxItems": 4,
                              "description": "Normalized bounding box [x1, y1, x2, y2], where 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000. (x1,y1) is the top-left corner, (x2,y2) is the bottom-right corner."
                        },
                        "image_index": {
                              "type": "integer",
                              "minimum": -1,
                              "description": "Index of the image to draw the bounding box on: 1 for original input image, 2 for first cropped image, 3 for second cropped image, etc. If the value is -1, it will use the most recently processed image from the previous step."
                        }
                        },
                        "required": ["bbox","image_index"]
                  }
            }
      },
      {
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
                          "description": "A list of coordinate pairs [[x, y]], where each pair represents a point to be drawn on the image. The coordinates are in the range 0 <= x, y <= 1000, where [0, 0] is the top-left corner and [1000, 1000] is the bottom-right corner. Each point will be rendered as a small red circle on the image."
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
      },
      {
            "type": "function",
            "function": {
                  "name": "rotate_image",
                  "description": "Rotate the input image by a specified angle in degrees. Positive values rotate the image clockwise.",
                  "parameters": {
                        "type": "object",
                        "properties": {
                        "angle": {
                        "type": "number",
                        "description": "Rotation angle in degrees. Positive values rotate clockwise. Can also be a string that can be converted to a float."
                        },
                        "image_index": {
                        "type": "integer",
                        "minimum": -1,
                        "description": "Index of the image to rotate: 1 for the original input image, 2 for the first modified image, 3 for the second modified image, etc. If the value is -1, it will use the most recently processed image from the previous step."
                        }
                        },
                        "required": ["angle", "image_index"]
                  }
            }
      },
      {
      "type": "function",
            "function": {
            "name": "brighten_image",
            "description": "Adjust the brightness of the input image by a specified factor. Values >1 make the image brighter, values <1 make it darker.",
            "parameters": {
                  "type": "object",
                  "properties": {
                  "factor": {
                  "type": "number",
                  "description": "Brightness adjustment factor. Must be a numeric value. Greater than 1 increases brightness, between 0 and 1 decreases brightness."
                  },
                  "image_index": {
                  "type": "integer",
                  "minimum": -1,
                  "description": "Index of the image to brighten: 1 for the original input image, 2 for the first modified image, 3 for the second modified image, etc. If the value is -1, it will use the most recently processed image from the previous step."
                  }
                  },
                  "required": ["factor", "image_index"]
            }
            }
      },
      {
            "type": "function",
            "function": {
            "name": "mirror_image",
            "description": "Mirror (flip) the input image in the specified direction.",
            "parameters": {
                  "type": "object",
                  "properties": {
                  "direction": {
                  "type": "string",
                  "description": "Direction to flip the image. Options are 'horizontal' (left-right), 'vertical' (top-bottom), or 'both' (flip both horizontally and vertically)."
                  },
                  "image_index": {
                  "type": "integer",
                  "minimum": -1,
                  "description": "Index of the image to mirror: 1 for the original input image, 2 for the first modified image, 3 for the second modified image, etc. If the value is -1, it will use the most recently processed image from the previous step."
                  }
                  },
                  "required": ["direction", "image_index"]
            }
            }
      },
      {
            "type": "function",
            "function": {
                  "name": "route_drawer",
                  "description": "Draws a red line connecting the specified grid points in the image. The grid points are provided as a list of [x, y] pairs representing the grid coordinates in an NxN grid, where 1 <= x, y <= N. The image will be divided into N x N grid cells, and the points will be plotted accordingly.",
                  "parameters": {
                        "type": "object",
                        "properties": {
                        "grid_points": {
                              "type": "array",
                              "items": {
                                    "type": "array",
                                    "items": { "type": "integer" },
                                    "minItems": 2,
                                    "maxItems": 2,
                                    "description": "A list of grid points where each point is represented as a two-element array or tuple [row, column]. The first element (row/x) is the **row index (height direction)**, and the second element (column/y) is the **column index (width direction)**. The **top-left corner is [1, 1]**, and both indices range from 1 to N. The points should be connected in sequence."
                              },
                              "minItems": 2,
                              "description": "A list of grid points to connect with lines."
                        },
                        "N": {
                              "type": "integer",
                              "minimum": 1,
                              "description": "The number of rows and columns in the grid, which divides the image into an N x N grid. The grid indices for each point must be between 1 and N, inclusive."
                        },
                        "image_index": {
                              "type": "integer",
                              "minimum": -1,
                              "description": "Index of the image to crop: 1 for original input image, 2 for first modified image, 3 for second modified image, etc. If the value is -1, it will use the most recently processed image from the previous step."
                        }
                        },
                        "required": ["grid_points", "N", "image_index"]
                  }
            }
      },
      {
            "type": "function",
            "function": {
            "name": "draw_line",
            "description": "Draw a thick red line on the image using normalized coordinates. Optionally extends the line from the second point outward as a dashed line.",
            "parameters": {
                  "type": "object",
                  "properties": {
                  "coords": {
                  "type": "array",
                  "items": {
                        "type": "integer"
                  },
                  "minItems": 4,
                  "maxItems": 4,
                  "description": "Normalized coordinates [x1, y1, x2, y2] in range [0,1000]. Origin is top-left; x increases rightward, y downward."
                  },
                  "image_index": {
                  "type": "integer",
                  "minimum": -1,
                  "description": "Index of the image to draw on. -1 means the most recently processed image."
                  },
                  "extend": {
                  "type": "boolean",
                  "description": "If true, extend the line from the second point outward as a dashed line. If false, draw only the solid line segment.",
                  "default": False
                  }
                  },
                  "required": [
                  "coords",
                  "image_index"
                  ]
            }
            }
      },
      {
            "type": "function",
            "function": {
                  "name": "locate_character_position",
                  "description": "Locates the position (row and column) of a visually distinct character in a dense grid image. It analyzes the grid structure, identifies the target character that differs from the background, marks it on the image, and returns the precise coordinates in [row, column] format. Use this tool for position-finding questions.",  
                  "parameters": {
                        "type": "object",
                        "properties": {
                              "image_index": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "description": "Index of the image to analyze: 1 for the original input image."
                              }
                        },
                        "required": ["image_index"]
                  }
            }
      },
      {
            "type": "function",
            "function": {
                  "name": "count_grid_items",
                  "description": "Counts the occurrences of distinct visual elements in a grid image. Useful for word counting questions. It uses structural difference analysis (template matching) to distinguish similar characters and returns the count for each group.",
                  "parameters": {
                        "type": "object",
                        "properties": {
                              "image_index": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "description": "Index of the image to analyze: 1 for the original input image."
                              }
                        },
                        "required": ["image_index"]
                  }
            }
      },
      {
            "type": "function",
            "function": {
                "name": "rearrange_tiles",
                "description": "Rearrange the image tiles from a current state to a target state.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "row": {
                            "type": "integer",
                            "minimum": 1, 
                            "description": "Number of rows."
                        },
                        "col": {
                            "type": "integer", 
                            "minimum": 1, 
                            "description": "Number of columns."
                        },
                        "state_curr": {
                            "type": "array", 
                            "items": {
                                "type": "integer"
                              }, 
                              "description": "Current layout tile IDs."
                        },
                        "state_next": {
                            "type": "array", 
                            "items": {
                                "type": "integer"
                              }, 
                            "description": "Target layout tile IDs."
                        },
                        "image_index": {
                              "type": "integer",
                              "minimum": -1,
                              "description": "Index of the image to draw on: 1 for original input image, 2 for first modified image, 3 for second modified image, etc. If the value is -1, it will use the most recently processed image from the previous step."
                        }
                  },
                  "required": [
                  "row", 
                  "col", 
                  "state_curr", 
                  "state_next", 
                  "image_index"
                  ]
                }
            }
      },
      {
      "type": "function",
      "function": {
            "name": "get_color_distribution",
            "description": "Analyzes the color distribution of an image by quantizing it to the most dominant colors and calculating their proportions.",
            "parameters": {
                  "type": "object",
                  "properties": {
                  "image_index": {
                        "type": "number",
                        "minimum": 1,
                        "description": "Index of the image to analyze: 1 for original input image, 2 for first processed image, etc."
                  }
                  },
                  "required": [
                  "image_index"
                  ]
            }
      }
      }
]

SYSTEM_TOOL_PROMPT = """\n\n# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_descs}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

SYSTEM_ROLE_PROMPT = """# Role
You are a step-by-step reasoning assistant. 
Given a question, your task is to solve the problem **one substep at a time**.  

## Guiding Principles  
At each turn, you must **either**:  
1. Issue **one specific tool** enclosed in <tool_call> </tool_call> tags,  
2. Or provide the **final answer** enclosed in <answer> </answer> tags.  

All outputs **must begin with a thought** enclosed in <thinking> </thinking> tags, explaining your current reasoning and what to do next. Then, depending on the case, output one of the following:

1. If reasoning continues:  
<thinking> Your current reasoning and next plan </thinking>  
<tool_call> One precise, tool call to assist your reasoning </tool_call>

2. If ready to conclude:  
<thinking> Summarize all reasoning and derive the answer </thinking>  
<answer> Final answer </answer>"""

SYSTEM_PROMPT = SYSTEM_ROLE_PROMPT + SYSTEM_TOOL_PROMPT


def main():
    tool_descs = '\n'.join([json.dumps(f, ensure_ascii=False) for f in TOOLS])
    tool_system = SYSTEM_PROMPT.format(tool_descs=tool_descs)
    print(repr(tool_system))
    print(tool_system)
    #print(repr(SYSTEM_ROLE_PROMPT))

if __name__ == "__main__":
    main()