#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import json
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm
from openai import OpenAI

from desc import TOOLS, SYSTEM_ROLE_PROMPT
from utils import run_toolcall, toolcall_message_convert

from unimmmu_prompts import (
    JIGSAW_USER_PROMPT,
    MAZE_USER_PROMPT,
    GEOMETRY_TOOLCALL_OVERLAY_PREFIX,
    GEOMETRY_TOOLCALL_REASONING_PREFIX,
)

# -----------------------
# utils
# -----------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def atomic_write_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def safe_id(s: str) -> str:
    return str(s).replace("/", "_")

def try_link_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def encode_image_path(image_path: str, max_size: int = 0) -> Optional[str]:
    try:
        p = Path(image_path)
        if not p.exists():
            return None
        with Image.open(p) as img:
            fmt = (img.format or "PNG").lower()
            if fmt == "jpg":
                fmt = "jpeg"
            if max_size > 0:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buf = BytesIO()
            img.save(buf, format=fmt.upper())
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/{fmt};base64,{b64}"
    except Exception:
        return None

def build_user_message(prompt: str, image_paths: List[str], max_size: int = 0) -> Dict[str, Any]:
    content = [{"type": "text", "text": prompt}]
    for p in image_paths:
        url = encode_image_path(p, max_size=max_size) or "<MISSING_IMAGE>"
        content.append({"type": "image_url", "image_url": {"url": url}})
    return {"role": "user", "content": content}

def scrub_data_urls(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "url" and isinstance(v, str) and v.startswith("data:image"):
                out[k] = "<IMAGE_BASE64_OMITTED>"
            else:
                out[k] = scrub_data_urls(v)
        return out
    if isinstance(obj, list):
        return [scrub_data_urls(x) for x in obj]
    return obj

def clean_message_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    import copy
    return scrub_data_urls(copy.deepcopy(messages))

def build_client(base_urls: List[str]) -> Tuple[OpenAI, str]:
    base_url = random.choice(base_urls)
    client = OpenAI(api_key="EMPTY", base_url=base_url)
    return client, base_url

def infer_model_name(client: OpenAI, fallback: str = "default-model") -> str:
    try:
        return client.models.list().data[0].id
    except Exception:
        return fallback

def is_done(sample_dir: Path) -> bool:
    p = sample_dir / "result.json"
    if not p.exists():
        return False
    try:
        x = json.loads(p.read_text(encoding="utf-8"))
        return x.get("status") == "ok"
    except Exception:
        return False

# -----------------------
# core loop (for all tasks)
# -----------------------
def run_one_sample_toolcall(
    *,
    base_urls: List[str],
    model_name_arg: str,
    task: str,
    sample: Dict[str, Any],
    out_root: Path,
    max_turns: int,
    tool_return_max_size: int,
    input_image_max_size: int,
    save_images: bool,
    force_rerun: bool,
) -> Tuple[str, str]:
    """
    Returns (task, id)
    Writes per-sample folder:
      result.json
      messages_clean.json
      images/...
    """
    sid = safe_id(sample.get("id"))
    sample_dir = out_root / "tool_call" / task / sid
    images_dir = sample_dir / "images"
    ensure_dir(images_dir)

    if (not force_rerun) and is_done(sample_dir):
        return task, sid

    # ---- choose prompt + images (from dataset record) ----
    if task == "jigsaw":
        prompt = JIGSAW_USER_PROMPT
        image_paths = list(sample["reasoning_imgs"])  # [ref, cand0, cand1]
    elif task == "maze":
        prompt = MAZE_USER_PROMPT
        image_paths = list(sample["reasoning_imgs"])  # [maze_step_0000]
    elif task == "geometry":
        reasoning_qs = sample.get("reasoning_qs", "")
        auxiliary_qs = sample.get("auxiliary_qs", "")
        prompt = (
            f"{GEOMETRY_TOOLCALL_OVERLAY_PREFIX}\n\n"
            f"{GEOMETRY_TOOLCALL_REASONING_PREFIX}\n\n"
            f"PROBLEM:\n{reasoning_qs}\n\n"
            f"AUXILIARY LINES TO DRAW:\n{auxiliary_qs}\n"
        )
        image_paths = list(sample["reasoning_imgs"][:1])  # base only
    else:
        raise ValueError(f"Unknown task: {task}")

    # ---- copy/link input images into sample folder for all images ----
    input_copied: List[str] = []
    for i, p in enumerate(image_paths):
        src = Path(p)
        dst = images_dir / f"input_{i:02d}{src.suffix.lower() or '.png'}"
        if src.exists():
            try_link_or_copy(src, dst)
            input_copied.append(str(dst.resolve()))
        else:
            input_copied.append(p)  # fallback (missing)

    # ---- init messages ----
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_ROLE_PROMPT},
        build_user_message(prompt, input_copied, max_size=input_image_max_size),
    ]

    # ---- tool image context: PIL list ----
    fn_image_list: List[Image.Image] = []
    for p in input_copied:
        if Path(p).exists():
            fn_image_list.append(Image.open(p))

    tool_images_saved: List[str] = []
    start_t = time.time()

    client, base_url = build_client(base_urls)
    model_name = model_name_arg.strip() or infer_model_name(client)

    status = "error"
    final_text = ""
    err_msg = None

    try:
        for turn_idx in range(max_turns):
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=TOOLS,
                temperature=0.5,
                top_p=0.25,
                max_tokens=4096,
                extra_body={
                    "repetition_penalty": 1.05,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            msg = resp.choices[0].message
            messages.append(msg.model_dump())

            # stop: no tool call => final answer
            if not msg.tool_calls:
                status = "ok"
                final_text = msg.content or ""
                break

            # 有 tool_calls：每轮只执行一次工具调用
            tool_call = msg.tool_calls[0]
            func_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            fn_ret = run_toolcall(func_name, fn_args, fn_image_list)
            tool_content = toolcall_message_convert(fn_ret, max_size=tool_return_max_size)

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": func_name,
                    "content": tool_content,
                }
            )

            # save returned images
            if save_images and isinstance(fn_ret, dict) and "image" in fn_ret and isinstance(fn_ret["image"], list):
                for i, im in enumerate(fn_ret["image"]):
                    if isinstance(im, Image.Image):
                        fn_image_list.append(im)
                        out_path = images_dir / f"tool_t{turn_idx:02d}_{func_name}_{i}.png"
                        im.save(out_path)
                        tool_images_saved.append(str(out_path.resolve()))

        if status != "ok":
            final_text = final_text or f"[ERROR] Max turns exceeded ({max_turns})"

    except Exception as e:
        err_msg = repr(e)
        status = "error"
        final_text = f"[ERROR] {err_msg}"

    dur = time.time() - start_t
    messages_clean = clean_message_history(messages)

    # ---- write per-sample files ----
    result = {
        "task": task,
        "id": sid,
        "status": status,
        "model": model_name,
        "base_url": base_url,
        "gt_text": sample.get("gt_text"),
        "reasoning_qs": sample.get("reasoning_qs"),
        "auxiliary_qs": sample.get("auxiliary_qs"),
        "prompt_used": prompt,
        "input_images": input_copied,
        "tool_images": tool_images_saved,
        "model_response": final_text,
        "error": err_msg,
        "runtime_sec": round(dur, 4),
        "ts": time.time(),
    }
    atomic_write_json(result, sample_dir / "result.json")
    atomic_write_json(messages_clean, sample_dir / "messages_clean.json")

    return task, sid


# -----------------------
# main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, default="/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/Uni-MMMU")
    ap.add_argument("--result_dir", type=str, required=True)
    ap.add_argument("--ip_list", type=str, nargs="+", required=True)
    ap.add_argument("--model_name", type=str, default="", help="empty => auto from /v1/models")

    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_turns", type=int, default=10)
    ap.add_argument("--tool_return_max_size", type=int, default=0)
    ap.add_argument("--input_image_max_size", type=int, default=0, help="shrink input image when encoding to base64; 0=keep")
    ap.add_argument("--save_images", action="store_true", default=True)
    ap.add_argument("--force_rerun", action="store_true", default=False)
    ap.add_argument("--max_samples", type=int, default=0, help="0 means all")
    args = ap.parse_args()

    base_urls = [f"http://{ip}:8000/v1" for ip in args.ip_list]
    dataset_root = Path(args.dataset_root)
    out_root = Path(args.result_dir)
    ensure_dir(out_root)

    for task in ["geometry", "jigsaw", "maze"]:
        data = read_json(dataset_root / task / "dataset.json")
        if args.max_samples and args.max_samples > 0:
            data = data[: args.max_samples]

        # filter not done
        jobs = []
        for s in data:
            sid = safe_id(s.get("id"))
            sample_dir = out_root / "tool_call" / task / sid
            if args.force_rerun or (not is_done(sample_dir)):
                jobs.append(s)

        print(f"[tool_call/{task}] total={len(data)} todo={len(jobs)} workers={args.workers}")

        if not jobs:
            continue

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [
                ex.submit(
                    run_one_sample_toolcall,
                    base_urls=base_urls,
                    model_name_arg=args.model_name,
                    task=task,
                    sample=s,
                    out_root=out_root,
                    max_turns=args.max_turns,
                    tool_return_max_size=args.tool_return_max_size,
                    input_image_max_size=args.input_image_max_size,
                    save_images=args.save_images,
                    force_rerun=args.force_rerun,
                )
                for s in jobs
            ]

            for _ in tqdm(as_completed(futs), total=len(futs), desc=f"run_{task}"):
                pass

    print("DONE.")


if __name__ == "__main__":
    main()
