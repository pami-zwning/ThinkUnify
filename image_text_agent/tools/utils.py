import os
import base64
import math
import ast
import sys
from io import BytesIO, StringIO
from desc import TOOLS
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
import numpy as np


def encode_image_with_format(image_path): 
      with Image.open(image_path) as img:

            img = img.convert("RGB")

            image_format = img.format
            if image_format is None:
                  image_format = "JPEG"

            image_format = image_format.lower()
            if image_format == "jpg":
                  image_format = "jpeg"

            # 保存到内存
            buffer = BytesIO()
            img.save(buffer, format=image_format.upper())
            image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

      data_url = {
            "url": f"data:image/{image_format};base64,{image_base64}"
      }
      return data_url


def encode_pil_image_with_format(img_obj, max_size=0):
      try:
            # 1. 复制对象以避免修改原始图像
            img = img_obj.copy()

            img = img.convert("RGB")
            
            if max_size > 0:
                  img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # 4. 保存到内存 buffer
            buffered = BytesIO()
            # 注意：img.save() 需要大写的格式名 (如 PNG, JPEG)
            img.save(buffered, format="JPEG")
            
            # 5. Base64 编码f
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # 6. 构造 Data URL
            data_url = f"data:image/jpeg;base64,{image_base64}"
            return data_url

      except Exception as e:
            print(f"Error encoding PIL Image: {e}")
            return None

def get_function_by_name(name):
      # 需要同时在desc.py的TOOLS定义和utils.py有对应实现
      all_fn_names = [item['function']['name'] for item in TOOLS]
      if name in globals() and name in all_fn_names:
            return globals()[name]
      else:
            raise ValueError(f"Function {name} not found!")

def toolcall_message_convert(return_value, max_size=0):
      tool_message = []
      
      # 1. 优先处理文本内容
      text_content = return_value.get('text')
      if text_content:
            tool_message.append({"type": "text", "text": text_content})

      # 2. 处理图像内容
      images = return_value.get('image')
      if images:
            # 类型校验（防御性编程）
            if not isinstance(images, list):
                  raise ValueError(
                  f"Image must be a list, but got {type(images)}. "
                  "Wrap single images in a list: [image]."
                  )

            # 补充缺失的文本描述（如果只有图没有文字）
            if not text_content:
                  tool_message.append({"type": "text", "text": "Here is the processed image:"})

            # 编码图像
            for img in images:
                  encoded_url = encode_pil_image_with_format(img, max_size)
                  tool_message.append({
                  "type": "image_url",
                  "image_url": {"url": encoded_url}
                  })

            
      return tool_message


def run_toolcall(fn_name, fn_args, image_list):
      try:
            ## toolcall中image_index 起始为1
            image_index = fn_args.get("image_index", 1)
            if not isinstance(image_index, int):
                  error_msg = f"Invalid image_index: {image_index}. It must be an integer."
                  raise ValueError(error_msg)  # 抛出异常

            if image_index < -1 or image_index > len(image_list) or image_index == 0 :
                  error_msg = f"Invalid image_index: {image_index}. It must be between -1 and {len(image_list)}, excluding 0."
                  raise ValueError(error_msg)  # 抛出异常

            if image_index == -1:
                  fn_image = image_list[image_index].copy()
            else:
                  fn_image = image_list[image_index-1].copy()

            fn_image = fn_image.convert("RGB")

            input_args = fn_args.copy()
            input_args['image'] = fn_image
            del input_args['image_index']

            return_value = {}
            function_return = get_function_by_name(fn_name)(**input_args)
            if isinstance(function_return, str):
                  return_value['text'] = function_return

            elif isinstance(function_return, Image.Image):
                  return_value['image'] = [function_return]

            elif isinstance(function_return, tuple) or isinstance(function_return, list):
                  content = ""
                  return_image_list = []
                  for item in function_return:
                        if isinstance(item, str):
                              content+=item
                        elif isinstance(item, Image.Image):
                              return_image_list.append(item)
                        else:
                              error_msg = f"Returns that do not meet the type requirements {item}"
                              raise ValueError(error_msg)  # 抛出异常
                  
                  # <image>数量匹配
                  img_placehold_num = content.count("<image>\n")
                  if img_placehold_num > len(return_image_list):
                        error_msg = "The number of placeholders is incorrect"
                        raise ValueError(error_msg)  # 抛出异常
                  elif img_placehold_num < len(return_image_list):
                        content+='<image>\n'*(len(return_image_list)-img_placehold_num)

                  return_value = {'text':content, 'image':return_image_list}
            else:
                  error_msg = f"Returns that do not meet the type requirements {function_return}"
                  raise ValueError(error_msg)  # 抛出异常
            
            return return_value
      
      except Exception as e:
            fc_arg_info = {"name":fn_name, "arguments":fn_args}
            error_msg = f"**Error occurred while executing {fn_name}**\nError Message:\n{str(e)}\nFunction Call Arguments:\n{fc_arg_info}"
            # 为了和RL对齐 因此将上面注释，直接返回报错
            return_value = {'text':str(e)}
            return return_value  # 返回 None 或者其他你希望的值


def crop_image(image: Image.Image, bbox):
      if isinstance(bbox, str):
            try:
                  bbox = ast.literal_eval(bbox)
            except (ValueError, SyntaxError):
                  raise ValueError(f"Invalid bbox format: {bbox}")

      if not (isinstance(bbox, list) and len(bbox) == 4):
            raise ValueError(f"Invalid bbox format: {bbox}")

      img_x, img_y = image.size

      try:
            x1, y1, x2, y2 = [float(coord) for coord in bbox]
      except (ValueError, TypeError):
            raise ValueError(f"Bbox coordinates must be numeric: {bbox}")

      if not (0 <= x1 <= 1000 and 0 <= y1 <= 1000 and 0 <= x2 <= 1000 and 0 <= y2 <= 1000):
            raise ValueError(f"In relative mode, all coordinates must be between 0 and 1. Got: {bbox}")

      if not (x1 < x2 and y1 < y2):
            raise ValueError(f"Invalid bbox coordinates: x1 must be < x2 and y1 must be < y2. Got: {bbox}")

      # Crop without padding
      cropped_img = image.crop((
            int(x1/1000 * img_x), int(y1/1000 * img_y),
            int(x2/1000 * img_x), int(y2/1000 * img_y)
      ))
      w, h = cropped_img.size
      w, h = max(w, 32), max(h, 32)
      cropped_img = cropped_img.resize((w, h), Image.Resampling.LANCZOS)

      return cropped_img

def draw_bbox(image: Image.Image, bbox):
      if isinstance(bbox, str):
            try:
                  bbox = ast.literal_eval(bbox)
            except (ValueError, SyntaxError):
                  raise ValueError(f"Invalid bbox format: {bbox}")

      if not (isinstance(bbox, list) and len(bbox) == 4):
            raise ValueError(f"Invalid bbox format: {bbox}")

      img_x, img_y = image.size

      try:
            x1, y1, x2, y2 = [float(coord) for coord in bbox]
      except (ValueError, TypeError):
            raise ValueError(f"Bbox coordinates must be numeric: {bbox}")

      if not (0 <= x1 <= 1000 and 0 <= y1 <= 1000 and 0 <= x2 <= 1000 and 0 <= y2 <= 1000):
            raise ValueError(f"In relative mode, all coordinates must be between 0 and 1000. Got: {bbox}")

      if not (x1 < x2 and y1 < y2):
            raise ValueError(f"Invalid bbox coordinates: x1 must be < x2 and y1 must be < y2. Got: {bbox}")

      # Draw bounding box on the image
      draw = ImageDraw.Draw(image)

      # Convert normalized coordinates to actual pixel coordinates
      x1_px, y1_px = int(x1 / 1000 * img_x), int(y1 / 1000 * img_y)
      x2_px, y2_px = int(x2 / 1000 * img_x), int(y2 / 1000 * img_y)

      # Draw the rectangle on the image
      draw.rectangle([x1_px, y1_px, x2_px, y2_px], outline="red", width=3)

      return image


def point_image(image: Image.Image, points_2d):
      if not isinstance(points_2d, (list, tuple)):
            raise ValueError(f"points_2d must be a list or tuple, got {type(points_2d)}")
      
      try:
            width, height = image.size
            draw = ImageDraw.Draw(image)
            for pt in points_2d:
                  # Validate each point format
                  if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                        raise ValueError(f"Invalid point format: {pt}. Points should be in [x, y] format.")
                  
                  x_norm, y_norm = pt
                  # Convert normalized coordinates to pixel coordinates
                  x = int(x_norm / 1000 * width)
                  y = int(y_norm / 1000 * height)
                  
                  # Draw the point as a red circle with radius 5
                  base = min(width, height)
                  r = max(3, int(base * 0.002)) 
                  draw.ellipse((x - r, y - r, x + r, y + r), fill='red', outline='red')

      except Exception as e:
            raise RuntimeError(f"Error occurred while drawing points on the image: {e}")
      
      return image

def rotate_image(image: Image.Image, angle):
      """
      默认顺时针旋转
      """
      if isinstance(angle, str):
            try:
                  angle = float(angle)
            except ValueError:
                  raise ValueError(f"Cannot convert angle '{angle}' to float")

      if not isinstance(angle, (int, float)):
            raise f"angle must be numeric, got {type(angle)}"

      try:
            rotated = image.rotate(-angle, expand=True, resample=Image.Resampling.BICUBIC,fillcolor=(255,255,255))
            
      except Exception as e:
            raise RuntimeError(f"Error while rotating image: {str(e)}")
      
      return rotated

def brighten_image(image: Image.Image, factor):
      if isinstance(factor, str):
            try:
                  factor = float(factor)
            except ValueError:
                  raise f"Cannot convert factor '{factor}' to float"

      if not isinstance(factor, (int, float)):
            raise f"factor must be numeric, got {type(factor)}"

      try:
            enhancer = ImageEnhance.Brightness(image)
            brightened = enhancer.enhance(factor)
      except Exception as e:
            raise str(e)

      return brightened

def mirror_image(image: Image.Image, direction):
      if not isinstance(direction, str):
            raise f"direction must be a string, got {type(direction)}"

      try:
            if direction.lower() == "horizontal":
                  mirrored = image.transpose(Image.FLIP_LEFT_RIGHT)
            elif direction.lower() == "vertical":
                  mirrored = image.transpose(Image.FLIP_TOP_BOTTOM)
            elif direction.lower() == "both":
                  mirrored = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
            else:
                  raise f"Invalid direction '{direction}', choose from 'horizontal', 'vertical', 'both'"
      except Exception as e:
            raise str(e)

      return mirrored

def route_drawer(image: Image.Image, grid_points, N):
      if not isinstance(N, int) or N <= 0:
            raise ValueError(f"Grid size N must be a positive integer, got {N}")

      img_w, img_h = image.size 
      x_spacing_for_row = img_w / N
      y_spacing_for_col = img_h / N

      try:
            img_copy = image.copy()
            draw = ImageDraw.Draw(img_copy)

            prev_x, prev_y = None, None
            for point in grid_points:
                  if not (isinstance(point, (list, tuple)) and len(point) == 2):
                        raise TypeError(f"Point is not a two-element tuple/list: {point}")

                  row_i, col_j = point 

                  # 检查坐标范围是否在 [1, N] 之间
                  if not (1 <= row_i <= N and 1 <= col_j <= N):
                        raise ValueError(
                              f"Point ({row_i}, {col_j}) is out of bounds [1, {N}]. "
                              f"Coordinates must be between 1 and N."
                        )
                  x = row_i * x_spacing_for_row - x_spacing_for_row // 2
                  y = col_j * y_spacing_for_col - y_spacing_for_col // 2

                  if prev_x is not None and prev_y is not None:
                        draw.line((prev_y,prev_x,y, x), fill="red", width=3)
                  prev_x, prev_y = x, y

      except Exception as e:
            raise RuntimeError(f"Path drawing failed during execution: {e}")
            
      return img_copy

def draw_line(image: Image.Image, coords, extend=False):
    # =========================
    # 1. 解析 coords
    # =========================
    if isinstance(coords, str):
        try:
            coords = ast.literal_eval(coords)
        except (ValueError, SyntaxError) as e:
            # 修改点：使用 raise 抛出异常
            raise ValueError(f"Invalid coords format: {coords}. Error: {e}") 

    if not (isinstance(coords, list) and len(coords) == 4):
        # 修改点：使用 raise 抛出异常
        raise ValueError(f"Invalid coords format, expected [x1, y1, x2, y2], got: {coords}")

    try:
        x1_n, y1_n, x2_n, y2_n = map(float, coords)
    except ValueError as e:
        raise ValueError(f"Coordinates must be numbers. Error: {e}")

    # =========================
    # 2. 检查归一化范围
    # =========================
    for c in (x1_n, y1_n, x2_n, y2_n):
        if not (0 <= c <= 1000):
            # 修改点：使用 raise 抛出异常
            raise ValueError(f"Normalized coordinate {c} out of range [0, 1000].")

    img_w, img_h = image.size

    # =========================
    # 3. 归一化 → 像素
    # =========================
    x1 = x1_n / 1000 * img_w
    y1 = y1_n / 1000 * img_h
    x2 = x2_n / 1000 * img_w
    y2 = y2_n / 1000 * img_h

    if x1 == x2 and y1 == y2:
        # 修改点：使用 raise 抛出异常
        raise ValueError("Start and end points cannot be the same.")

    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    base = min(img_w, img_h)
    line_width = max(4, int(base * 0.004))

    # =========================
    # 4. 画 C-D 原线段（实线）
    # =========================
    if not extend:
        draw.line((x1, y1, x2, y2), fill="red", width=line_width)

    # =========================
    # 5. 延长线（仅从 D 向外，虚线）
    # =========================
    if extend:
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)

        ux = dx / length
        uy = dy / length

        # 延长到图像外
        max_len = math.hypot(img_w, img_h)
        x_end = x2 + ux * max_len
        y_end = y2 + uy * max_len

        # ---- 画虚线（分段）----
        dash_len = 12
        gap_len = 8

        total_len = math.hypot(x_end - x2, y_end - y2)
        steps = int(total_len // (dash_len + gap_len))

        for i in range(steps):
            start = i * (dash_len + gap_len)
            end = start + dash_len

            xs = x2 + ux * start
            ys = y2 + uy * start
            xe = x2 + ux * end
            ye = y2 + uy * end

            draw.line((xs, ys, xe, ye), fill="red", width=line_width)

    return img_copy

def _get_projection_segments(projection, threshold=0):
    """
    calculate grid split points based on projection
    """
    segments = []
    start = -1
    for i, val in enumerate(projection):
        if val > threshold:
            if start == -1:
                start = i
        else:
            if start != -1:
                segments.append((start, i))
                start = -1
    if start != -1:
        segments.append((start, len(projection)))
    return segments

def locate_character_position(image: Image.Image):
    """
    定位密集网格中的异常元素，返回标记后的图片和文本描述。
    Returns: (True, (result_image, result_text))
    """
    try:
        # 1. image to grayscale
        gray = image.convert('L')
        inverted = ImageOps.invert(gray)
        binary = inverted.point(lambda x: 255 if x > 128 else 0, mode='1')
        arr = np.array(binary, dtype=np.uint8) 
        h, w = arr.shape

        # 2. projection and grid split
        row_proj = np.sum(arr > 0, axis=1)
        col_proj = np.sum(arr > 0, axis=0)
        row_segments = _get_projection_segments(row_proj, threshold=1)
        col_segments = _get_projection_segments(col_proj, threshold=1)

        if len(row_segments) < 2 or len(col_segments) < 2:
            return False, "Failed to detect grid structure."

        # 3. feature extraction
        features = []
        for r_idx, (r_start, r_end) in enumerate(row_segments):
            for c_idx, (c_start, c_end) in enumerate(col_segments):
                cell = arr[r_start:r_end, c_start:c_end]
                pixel_count = np.count_nonzero(cell)
                center_y = (r_start + r_end) // 2
                center_x = (c_start + c_end) // 2
                
                features.append({
                    "r_idx": r_idx + 1,
                    "c_idx": c_idx + 1,
                    "val": pixel_count,
                    "cx": center_x,
                    "cy": center_y,
                    "bbox": [c_start, r_start, c_end, r_end]
                })

        if not features: return False, "No cells extracted."

        vals = np.array([f["val"] for f in features])
        median_val = np.median(vals)
        diffs = np.abs(vals - median_val)
        max_diff_idx = np.argmax(diffs)
        target = features[max_diff_idx]

        if diffs[max_diff_idx] == 0:
             return False, "All cells identical."

        result_image = image.copy()
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        cell_w = target["bbox"][2] - target["bbox"][0]
        cell_h = target["bbox"][3] - target["bbox"][1]
        r = max(min(cell_w, cell_h) // 2, 10)
        
        overlay_draw.ellipse(
            (target['cx']-r, target['cy']-r, target['cx']+r, target['cy']+r),
            outline=(255, 100, 100, 150),
            width=3
        )
        # 合成到原图
        result_image = Image.alpha_composite(result_image.convert('RGBA'), overlay).convert('RGB')
        
        # construct precise text description
        result_text = (
            f"A distinct character is located at Row {target['r_idx']}, Column {target['c_idx']}. "
        )

        return (result_image, result_text)

    except Exception as e:
        return str(e)

def _extract_grid_cells(image, binary_arr):
    """辅助函数：网格切分和标准化"""
    # 1. 投影计算
    row_proj = np.sum(binary_arr > 0, axis=1)
    col_proj = np.sum(binary_arr > 0, axis=0)
    
    # 2. 网格切分 (动态阈值)
    threshold_row = max(2, np.max(row_proj) * 0.05)
    threshold_col = max(2, np.max(col_proj) * 0.05)
    
    row_segments = _get_projection_segments(row_proj, threshold=threshold_row)
    col_segments = _get_projection_segments(col_proj, threshold=threshold_col)
        
    if len(row_segments) < 2 or len(col_segments) < 2:
        return None, f"Failed to detect grid. Found {len(row_segments)} rows x {len(col_segments)} cols."

    cells = []
    bboxes = []
    std_size = (32, 32)
    
    for r_idx, (r_start, r_end) in enumerate(row_segments):
        for c_idx, (c_start, c_end) in enumerate(col_segments):
            box = (c_start, r_start, c_end, r_end)
            # 过滤过小的噪点
            if (r_end - r_start) < 4 or (c_end - c_start) < 4:
                continue
                
            cell_img = image.crop(box).convert('L')
            cell_resized = cell_img.resize(std_size, Image.Resampling.BILINEAR)
            cell_np = np.array(cell_resized, dtype=np.float32) / 255.0
            
            cells.append(cell_np)
            bboxes.append({
                "bbox": box, "cx": (c_start + c_end) // 2, "cy": (r_start + r_end) // 2
            })
            
    if not cells: return None, "No valid cells extracted."
    return (np.array(cells), bboxes), None

def count_grid_items(image: Image.Image):
    """
    通过结构差异统计网格元素 (处理 U vs V, 鱼 vs 鲤)。
    Returns: (True, (result_image, result_text))
    """
    try:
        # 1. 预处理
        gray = image.convert('L')
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(2.0)
        inverted = ImageOps.invert(gray)
        binary = inverted.point(lambda x: 255 if x > 100 else 0, mode='1')
        arr = np.array(binary, dtype=np.uint8) 

        # 2. 提取单元格
        data, error = _extract_grid_cells(image, arr)
        if error: return False, error
        cells_np, meta_info = data
        
        # 3. 模板匹配 (Reference Subtraction)
        ref_cell = np.median(cells_np, axis=0) # 计算“平均脸”
        diffs = np.sum(np.abs(cells_np - ref_cell), axis=(1, 2)) # 计算差异
        
        # 4. Max Gap
        vals = diffs
        sorted_vals = np.sort(vals)
        diff_of_diffs = np.diff(sorted_vals)
        
        if len(diff_of_diffs) > 0:
            max_gap_idx = np.argmax(diff_of_diffs)
            max_gap = diff_of_diffs[max_gap_idx]
            split_threshold = (sorted_vals[max_gap_idx] + sorted_vals[max_gap_idx+1]) / 2
        else:
            max_gap, split_threshold = 0, 0
            
        is_single_type = max_gap < 5.0 
        
        result_image = image.copy()
        
        if is_single_type:
            result_text = f"Only 1 structural type detected (Max Gap {max_gap:.2f}). Total count: {len(cells_np)}."
        else:
            group_A = [i for i, v in enumerate(vals) if v <= split_threshold] # Common/Background
            group_B = [i for i, v in enumerate(vals) if v > split_threshold]  # Distinct/Foreground
            
            # 圈出少数派
            if len(group_B) < len(group_A):
                target_indices = group_B
                target_label = "High Difference (Type B)"
            else:
                target_indices = group_A
                target_label = "Low Difference (Type A)"
            
            # 创建半透明图层
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            for idx in target_indices:
                info = meta_info[idx]
                cx, cy = info['cx'], info['cy']
                r = max(8, (info['bbox'][2] - info['bbox'][0]) // 2)
                # 绘制半透明浅红色圆圈
                overlay_draw.ellipse(
                    (cx-r, cy-r, cx+r, cy+r),
                    outline=(255, 100, 100, 150),
                    width=3
                )
            # 合成到原图
            result_image = Image.alpha_composite(result_image.convert('RGBA'), overlay).convert('RGB')
            
            result_text = (
                f"Detected distinct item types:\n"
                f"- Low Difference (Type A): {len(group_A)} items\n"
                f"- High Difference (Type B): {len(group_B)} items\n"
                f"-> Minority group ({len(target_indices)} items) marked with red circles."
            )
        return (result_image, result_text)

    except Exception as e:
        return str(e)

def rearrange_tiles(image: Image.Image, row: int, col: int, state_curr: list, state_next: list):
      """
      Rearrange image tiles from state_curr to state_next.
      
      Args:
            image (Image.Image): The visual representation of state_curr.
            row (int): Number of rows.
            col (int): Number of columns.
            state_curr (list): The list of tile IDs currently displayed in the image.
                              e.g., [5, 2, 9, ...]
            state_next (list): The target list of tile IDs.
                              e.g., [1, 2, 3, ...]
      
      Returns:
            tuple: (success (bool), result (Image.Image or str error message))
      """
      if not isinstance(state_curr, list) or not isinstance(state_next, list):
            raise "state_curr and state_next must be lists."
      
      total_tiles = row * col
      
      if len(state_curr) != total_tiles or len(state_next) != total_tiles:
            raise f"State length must match grid size ({row}x{col}={total_tiles})."

      width, height = image.size
      cell_w = width // col
      cell_h = height // row

      
      tile_id_to_image_patch = {}
      
      for i, tile_id in enumerate(state_curr):
            r = i // col
            c = i % col
            
            left = c * cell_w
            top = r * cell_h
            right = left + cell_w
            bottom = top + cell_h
            
            patch = image.crop((left, top, right, bottom)).copy()
            
            tile_id_to_image_patch[tile_id] = patch

      new_image = Image.new('RGB', (width, height))
      
      for i, target_tile_id in enumerate(state_next):
            if target_tile_id not in tile_id_to_image_patch:
                  raise f"Tile ID {target_tile_id} in state_next not found in state_curr."
            
            patch_to_paste = tile_id_to_image_patch[target_tile_id]
            
            target_r = i // col
            target_c = i % col
            
            paste_x = target_c * cell_w
            paste_y = target_r * cell_h
            
            new_image.paste(patch_to_paste, (paste_x, paste_y))
            
      return new_image

def get_color_distribution(image: Image.Image):
      """
      执行图像颜色分析，并返回格式化后的结果字符串
      参数:
            image (Image.Image): PIL 图像对象
      返回:
            output (str): 颜色分布分析结果或错误信息
      """
      try:
            # 1. 确保图像是 RGB 模式
            img = image.convert("RGB")

            # 2. 量化到 20 种主色
            # 注意：这里需要确保 PIL 已导入
            quantized = img.quantize(
                  colors=20, 
                  method=Image.Quantize.MAXCOVERAGE
            ).convert("RGB")

            total_pixels = quantized.size[0] * quantized.size[1]

            # 3. 获取颜色分布
            colors = quantized.getcolors(maxcolors=total_pixels)

            if colors is None:
                  return "执行错误: getcolors 返回了 None"

            # 4. 按像素数量降序排序
            colors = sorted(colors, key=lambda x: x[0], reverse=True)

            # 5. 构建输出字符串
            results = []
            results.append(f"Unique colors count: {len(colors)}")
            
            for count, rgb in colors:
                  ratio = count / total_pixels
                  results.append(f"Color: {rgb}, Ratio: {ratio:.4f}")

            return "\n".join(results)

      except Exception as e:
            return f"执行错误: {type(e).__name__}: {e}"