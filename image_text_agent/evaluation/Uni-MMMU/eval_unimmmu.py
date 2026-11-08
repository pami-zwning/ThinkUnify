#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from io import BytesIO
import base64

from tqdm import tqdm
from openai import AzureOpenAI


# ===================== Azure judge =====================
def get_azure_client() -> Tuple[AzureOpenAI, str]:
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://duomotai.openai.azure.com/")
    AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "YOUR_AZURE_OPENAI_API_KEY")
    AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-2024-11-20")
    AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

    client = AzureOpenAI(
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
    )
    return client, AZURE_DEPLOYMENT_NAME


# ===================== Templates =====================
SYSTEM_OVERLAY = (
    "You are a strict yet fair judge for geometry diagram overlays.\n"
    "Compare images to decide if the candidate overlay correctly draws the REQUIRED auxiliary lines.\n"
    "Input images:\n"
    "  (1) base: original (no overlays), the clean figure.\n"
    "  (2) gt  : ground-truth overlay (what SHOULD be drawn).\n"
    "  (3) pred: ONLY ONE candidate overlay to judge.\n"
    "Also given the English text listing auxiliary lines to draw.\n\n"
    "Rules:\n"
    "  - Focus on whether the REQUIRED lines exist (correct endpoints/relations) in the candidate.\n"
    "  - Ignore extra small ticks/labels/harmless marks if they do not distort the meaning.\n"
    "  - Tolerate minor pixel/position noise if geometry intent is clear (parallel/perpendicular/through-point).\n"
    "  - ALL specified lines must be present; otherwise mark 0.\n"
    '  - Output MUST be a compact JSON: {"overlay_ok":0|1,"overlay_reason":"<short>"}\n'
)

SYSTEM_TEXT = (
    "You are a rigorous grader for geometry reasoning.\n"
    "Given: problem statement, ground-truth solution text (reference), and a candidate solution text.\n"
    "Candidate may contain <thinking>/<answer> tags; ignore formatting and judge the math.\n"
    "Decide two things: (i) is the reasoning rigorous (no major logical gaps or false claims), "
    "(ii) is the final conclusion correct.\n"
    "For CALCULATION problems: conclusion correctness means the final NUMERIC result matches the ground-truth "
    "(ignore formatting/units; radicals/π are okay if numerically equivalent).\n"
    "For PROVING problems: conclusion correctness means the claim is indeed established.\n"
    "Only if (rigorous AND correct) → text_ok=1, else 0.\n"
    'Output MUST be a compact JSON: {"reasoning_rigorous":0|1,"conclusion_correct":0|1,'
    '"calc_numeric_match":0|1|null,"text_ok":0|1,"text_reason":"<short>"}\n'
)

OVERLAY_USER_TMPL = (
    "Task: Check if the candidate overlay draws the REQUIRED auxiliary lines.\n\n"
    "Auxiliary lines to draw (English; draw ALL):\n"
    "{aux_en}\n\n"
    "Important:\n"
    "- Evaluate ONLY ONE candidate overlay image (pred_1).\n"
    "- Allow extra harmless marks/letters if they don't distort meaning.\n"
    "- Focus on presence/placement/direction of REQUIRED lines.\n\n"
    "Images in order:\n"
    "- base (original, no overlays)\n"
    "- gt (ground-truth overlay)\n"
    "- pred_1 (the ONLY candidate to judge)\n\n"
    "Return strictly-formatted JSON only. Do Not Use any markers like ```json ... ```.\n"
)

TEXT_USER_TMPL = (
    "Task: Grade the candidate solution text for rigor and correctness.\n\n"
    "Type: {task_type}\n"
    "Problem:\n{problem}\n\n"
    "Ground-truth solution text (reference):\n{gt_text}\n\n"
    "Candidate solution text (to grade):\n{cand_text}\n\n"
    "Rules:\n"
    "- Only accept text_ok=1 if (reasoning_rigorous=1 AND conclusion_correct=1).\n"
    "- For calculation: calc_numeric_match=1 if numeric answer equals reference.\n"
    "- Return strictly-formatted JSON only. Do Not Use any markers like ```json ... ```.\n"
)


# ===================== Basic IO =====================
def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def encode_img_to_data_url(img_path: Path, max_size: int = 0) -> str:
    with Image.open(img_path) as im:
        fmt = (im.format or "PNG").lower()
        if fmt == "jpg":
            fmt = "jpeg"
        if max_size > 0:
            im.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buf = BytesIO()
        im.save(buf, format=fmt.upper())
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/{fmt};base64,{b64}"


# ===================== Parse last assistant text =====================
def load_last_text(sample_dir: Path) -> str:
    # 优先用 messages_clean 的最后 assistant；否则用 result.json 的 model_response
    mc = sample_dir / "messages_clean.json"
    if mc.exists():
        try:
            msgs = read_json(mc)
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    c = m.get("content", "")
                    if isinstance(c, str):
                        return c
                    if isinstance(c, list):
                        parts = []
                        for it in c:
                            if isinstance(it, dict) and it.get("type") == "text":
                                parts.append(it.get("text", ""))
                        return "\n".join([p for p in parts if p])
        except Exception:
            pass

    rj = sample_dir / "result.json"
    if rj.exists():
        try:
            r = read_json(rj)
            return str(r.get("model_response", "") or "")
        except Exception:
            return ""
    return ""


# ===================== Regex extractors =====================
ANSWER_TAG_RE = re.compile(r"<answer>\s*([^<]+?)\s*</answer>", re.IGNORECASE | re.DOTALL)

def extract_maze_moves(text: str) -> str:
    # 1) <answer>...</answer>
    m = ANSWER_TAG_RE.search(text or "")
    if m:
        s = m.group(1).strip().upper()
        s = re.sub(r"[^RLUD]", "", s)
        return s
    # 2) fallback: last RLUD chunk
    chunks = re.findall(r"[RLUD]+", (text or "").upper())
    return chunks[-1] if chunks else ""

def extract_jigsaw_choice(text: str) -> Optional[int]:
    # 1) <answer>0</answer> or <answer>1</answer>
    m = ANSWER_TAG_RE.search(text or "")
    if m:
        s = m.group(1).strip()
        m2 = re.search(r"\b([01])\b", s)
        if m2:
            return int(m2.group(1))
    # 2) fallback: last standalone 0/1
    ms = list(re.finditer(r"(?<!\d)([01])(?!\d)", text or ""))
    if ms:
        return int(ms[-1].group(1))
    return None


# ===================== Maze metric =====================
def evaluate_text_moves(pred: List[str], gt: List[str]) -> Dict[str, float]:
    if not gt:
        return {"text_exact": 0, "text_frame_acc": 0.0}
    match_count = sum(1 for p, g in zip(pred, gt) if p == g)
    return {
        "text_exact": 1 if pred == gt else 0,
        "text_frame_acc": match_count / len(gt),
    }


# ===================== Azure call helpers =====================
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def parse_json_from_text(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        m = JSON_OBJ_RE.search(s)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}

def azure_chat(azure_client: AzureOpenAI, deployment: str, messages: List[Dict[str, Any]], max_tokens: int, retries: int) -> Tuple[str, Dict[str, Any]]:
    last_err = None
    for _ in range(max(1, retries)):
        try:
            resp = azure_client.chat.completions.create(
                model=deployment,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            txt = resp.choices[0].message.content or ""
            data = parse_json_from_text(txt)
            return txt, data
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    return f"[AZURE_ERROR] {repr(last_err)}", {}

def judge_overlay(
    azure_client: AzureOpenAI,
    deployment: str,
    base_img: Path,
    gt_img: Path,
    pred_img: Path,
    aux_en: str,
    *,
    max_tokens: int = 512,
    retries: int = 2,
    vision_max_size: int = 0,
) -> Dict[str, Any]:
    user_parts = [
        {"type": "text", "text": OVERLAY_USER_TMPL.format(aux_en=aux_en or "(none)")},
        {"type": "image_url", "image_url": {"url": encode_img_to_data_url(base_img, vision_max_size)}},
        {"type": "image_url", "image_url": {"url": encode_img_to_data_url(gt_img, vision_max_size)}},
        {"type": "image_url", "image_url": {"url": encode_img_to_data_url(pred_img, vision_max_size)}},
    ]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_OVERLAY}]},
        {"role": "user", "content": user_parts},
    ]
    raw_text, data = azure_chat(azure_client, deployment, messages, max_tokens=max_tokens, retries=retries)
    ok = 1 if int(data.get("overlay_ok", 0) or 0) == 1 else 0
    reason = str(data.get("overlay_reason", "")).strip() or ""
    return {"overlay_ok": ok, "overlay_reason": reason, "raw_text": raw_text, "raw_json": data}

def judge_text(
    azure_client: AzureOpenAI,
    deployment: str,
    task_type: str,
    problem: str,
    gt_text: str,
    cand_text: str,
    *,
    max_tokens: int = 1024,
    retries: int = 2,
) -> Dict[str, Any]:
    user_text = TEXT_USER_TMPL.format(
        task_type=task_type, problem=problem, gt_text=gt_text, cand_text=cand_text
    )
    messages = [
        {"role": "system", "content": SYSTEM_TEXT},
        {"role": "user", "content": user_text},
    ]
    raw_text, data = azure_chat(azure_client, deployment, messages, max_tokens=max_tokens, retries=retries)

    def _b(x):  # to 0/1
        try:
            return 1 if int(x) == 1 else 0
        except Exception:
            return 0

    out = {
        "reasoning_rigorous": _b(data.get("reasoning_rigorous", 0)),
        "conclusion_correct": _b(data.get("conclusion_correct", 0)),
        "calc_numeric_match": data.get("calc_numeric_match", None),
        "text_ok": _b(data.get("text_ok", 0)),
        "text_reason": str(data.get("text_reason", "")).strip() or "",
        "raw_text": raw_text,
        "raw_json": data,
        "request": {
            "task_type": task_type,
            "problem": problem,
            "gt_text": gt_text,
            "cand_text": cand_text,
        },
    }
    return out


# ===================== Dataset index =====================
def index_dataset(dataset_root: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for task in ["geometry", "jigsaw", "maze"]:
        p = dataset_root / task / "dataset.json"
        items = read_json(p)
        out[task] = {str(x["id"]): x for x in items}
    return out


# ===================== Eval job =====================
@dataclass
class Job:
    mode: str        # "direct" | "tool_call"
    task: str        # geometry|jigsaw|maze
    sid: str         # sample id
    sample_dir: Path # directory containing result.json/messages_clean.json


def eval_one(
    job: Job,
    dataset_idx: Dict[str, Dict[str, Dict[str, Any]]],
    azure_client: AzureOpenAI,
    azure_deployment: str,
    *,
    max_retries: int,
    vision_max_size: int,
) -> Tuple[str, str, str]:
    """
    Returns (mode, task, sid). Writes files into sample_dir/eval/
    """
    eval_dir = job.sample_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # skip if eval already exists
    eval_json = eval_dir / "eval.json"
    if eval_json.exists():
        try:
            x = read_json(eval_json)
            if x.get("status") == "ok":
                return job.mode, job.task, job.sid
        except Exception:
            pass

    gt = dataset_idx.get(job.task, {}).get(job.sid)
    if gt is None:
        write_json({"status": "error", "error": "missing_gt", "id": job.sid, "task": job.task}, eval_json)
        return job.mode, job.task, job.sid

    cand_text = load_last_text(job.sample_dir)

    # ---------------- maze / jigsaw: regex metrics ----------------
    if job.task == "maze":
        pred_str = extract_maze_moves(cand_text)
        gt_str = str(gt.get("gt_text", "") or "").strip().upper()
        pred = list(pred_str)
        gt_moves = list(gt_str)
        m = evaluate_text_moves(pred, gt_moves)

        out = {
            "status": "ok",
            "mode": job.mode,
            "task": "maze",
            "id": job.sid,
            "pred_text": cand_text,
            "pred_moves": pred_str,
            "gt_moves": gt_str,
            "text_exact": m["text_exact"],
            "text_frame_acc": m["text_frame_acc"],
        }
        write_json(out, eval_json)
        return job.mode, job.task, job.sid

    if job.task == "jigsaw":
        pred_choice = extract_jigsaw_choice(cand_text)
        gt_choice = int(gt.get("gt_text"))
        acc = 1 if (pred_choice is not None and int(pred_choice) == gt_choice) else 0
        out = {
            "status": "ok",
            "mode": job.mode,
            "task": "jigsaw",
            "id": job.sid,
            "pred_text": cand_text,
            "pred_choice": pred_choice,
            "gt_choice": gt_choice,
            "acc": acc,
        }
        write_json(out, eval_json)
        return job.mode, job.task, job.sid

    # ---------------- geometry: Azure judge ----------------
    # reasoning judge (both modes)
    task_type_raw = (gt.get("task_type") or "calculation").lower()
    task_type = "CALCULATION" if task_type_raw.startswith("calc") else "PROVING"
    problem = gt.get("reasoning_qs", "") or ""
    gt_text = gt.get("gt_text", "") or ""

    text_req_path = eval_dir / "judge_text_request.json"
    text_rsp_path = eval_dir / "judge_text_response.json"

    write_json(
        {
            "task_type": task_type,
            "problem": problem,
            "gt_text": gt_text,
            "cand_text": cand_text,
            "system": SYSTEM_TEXT,
            "user_template": TEXT_USER_TMPL,
        },
        text_req_path,
    )

    text_eval = judge_text(
        azure_client,
        azure_deployment,
        task_type=task_type,
        problem=problem,
        gt_text=gt_text,
        cand_text=cand_text,
        max_tokens=2048,
        retries=max_retries,
    )
    write_json(text_eval, text_rsp_path)

    # overlay judge only for tool_call mode
    overlay_eval = None
    if job.mode == "tool_call":
        rj = read_json(job.sample_dir / "result.json")
        input_imgs = list(rj.get("input_images", []) or [])
        tool_imgs = list(rj.get("tool_images", []) or [])
        merged = input_imgs + tool_imgs

        base_img = Path(merged[0]) if merged else Path(gt["reasoning_imgs"][0])
        pred_img = Path(merged[-1]) if merged else Path(gt["reasoning_imgs"][0])
        gt_overlay_img = Path(gt["gt_auxiliary_imgs"][0])

        aux_en = gt.get("auxiliary_qs", "") or ""

        overlay_req_path = eval_dir / "judge_overlay_request.json"
        overlay_rsp_path = eval_dir / "judge_overlay_response.json"
        write_json(
            {
                "aux_en": aux_en,
                "base_img": str(base_img),
                "gt_overlay_img": str(gt_overlay_img),
                "pred_img": str(pred_img),
                "system": SYSTEM_OVERLAY,
                "user_template": OVERLAY_USER_TMPL,
                "note": "Images are encoded to data URLs at runtime (not stored here).",
            },
            overlay_req_path,
        )

        overlay_eval = judge_overlay(
            azure_client,
            azure_deployment,
            base_img=base_img,
            gt_img=gt_overlay_img,
            pred_img=pred_img,
            aux_en=aux_en,
            max_tokens=512,
            retries=max_retries,
            vision_max_size=vision_max_size,
        )
        write_json(overlay_eval, overlay_rsp_path)

    out = {
        "status": "ok",
        "mode": job.mode,
        "task": "geometry",
        "id": job.sid,
        "text_ok": int(text_eval.get("text_ok", 0) == 1),
        "reasoning_rigorous": int(text_eval.get("reasoning_rigorous", 0) == 1),
        "conclusion_correct": int(text_eval.get("conclusion_correct", 0) == 1),
        "calc_numeric_match": text_eval.get("calc_numeric_match", None),
        "overlay_ok": (overlay_eval.get("overlay_ok") if overlay_eval else None),
        "overlay_reason": (overlay_eval.get("overlay_reason") if overlay_eval else None),
    }
    write_json(out, eval_json)
    return job.mode, job.task, job.sid


# ===================== Summaries =====================
def summarize(eval_root: Path, mode: str, task: str) -> Dict[str, Any]:
    # scan eval.json under run_root/{mode}/{task}/*/eval/eval.json
    base = eval_root / mode / task
    if not base.exists():
        return {"mode": mode, "task": task, "total": 0}

    eval_files = sorted(base.glob("*/eval/eval.json"))
    recs = []
    for p in eval_files:
        try:
            x = read_json(p)
            if x.get("status") == "ok":
                recs.append(x)
        except Exception:
            pass

    total = len(eval_files)
    scored = len(recs)

    out: Dict[str, Any] = {"mode": mode, "task": task, "total": total, "scored": scored}

    if task == "maze":
        out["text_exact"] = sum(r.get("text_exact", 0) for r in recs)
        out["avg_text_frame_acc"] = (sum(r.get("text_frame_acc", 0.0) for r in recs) / scored) if scored else 0.0
        out["exact_acc"] = (out["text_exact"] / scored) if scored else 0.0

    elif task == "jigsaw":
        ok = sum(r.get("acc", 0) for r in recs)
        out["acc"] = (ok / scored) if scored else 0.0

    else:  # geometry
        out["text_ok"] = sum(int(r.get("text_ok", 0) == 1) for r in recs)
        out["text_acc"] = (out["text_ok"] / scored) if scored else 0.0
        if mode == "tool_call":
            overlay_vals = [r.get("overlay_ok") for r in recs if r.get("overlay_ok") is not None]
            out["overlay_scored"] = len(overlay_vals)
            out["overlay_ok"] = sum(int(v == 1) for v in overlay_vals)
            out["overlay_acc"] = (out["overlay_ok"] / out["overlay_scored"]) if out["overlay_scored"] else 0.0

    return out


# ===================== Main =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", type=str, required=True, help="root that contains direct/ and tool_call/ outputs")
    ap.add_argument("--dataset_root", type=str, default="/mnt/afs/share_data/shangziyi/datasets/unimmmu")
    ap.add_argument("--modes", type=str, nargs="+", default=["direct", "tool_call"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--vision_max_size", type=int, default=0, help="optional downscale for overlay judge images; 0=no")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    dataset_root = Path(args.dataset_root)

    dataset_idx = index_dataset(dataset_root)

    azure_client, azure_deployment = get_azure_client()

    jobs: List[Job] = []
    for mode in args.modes:
        for task in ["geometry", "jigsaw", "maze"]:
            base = run_root / mode / task
            if not base.exists():
                continue
            for sample_dir in sorted(base.glob("*")):
                if not sample_dir.is_dir():
                    continue
                if not (sample_dir / "result.json").exists():
                    continue
                sid = sample_dir.name
                jobs.append(Job(mode=mode, task=task, sid=sid, sample_dir=sample_dir))

    print(f"[INFO] Found {len(jobs)} jobs under {run_root}")
    if not jobs:
        return

    # run eval with workers
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                eval_one,
                job,
                dataset_idx,
                azure_client,
                azure_deployment,
                max_retries=args.max_retries,
                vision_max_size=args.vision_max_size,
            )
            for job in jobs
        ]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="evaluating"):
            pass

    # write summaries
    summary_dir = run_root / "eval_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for mode in args.modes:
        for task in ["geometry", "jigsaw", "maze"]:
            s = summarize(run_root, mode, task)
            all_summaries.append(s)
            write_json(s, summary_dir / f"{mode}_{task}_summary.json")

    write_json(all_summaries, summary_dir / "all_summaries.json")
    print(json.dumps(all_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
