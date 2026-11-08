#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import hashlib
import hmac
import datetime
import uuid
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# ===================== Config & Templates =====================

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


# ===================== Internal API Client (GPT-as-judge) =====================
# Credentials are read from environment variables; no secrets are shipped.

_API_CONFIG = {
    "host":         os.getenv("JUDGE_API_HOST", "http://your-judge-api-host:8080"),
    "secret_id":    os.getenv("JUDGE_SECRET_ID", "YOUR_SECRET_ID"),
    "secret_key":   os.getenv("JUDGE_SECRET_KEY", "YOUR_SECRET_KEY"),
    "model_marker": os.getenv("JUDGE_MODEL_MARKER", "api_naci_default_gemini-3-flash-preview"),
}

_API_VERSION = "v2.03"


def _get_simple_auth(source: str, secret_id: str, secret_key: str) -> Tuple[str, str]:
    date_time = datetime.datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
    auth = f'hmac id="{secret_id}", algorithm="hmac-sha1", headers="date source", signature="'
    sign_str = f"date: {date_time}\nsource: {source}"
    sign = hmac.new(secret_key.encode(), sign_str.encode(), hashlib.sha1).digest()
    sign = base64.b64encode(sign).decode()
    return auth + sign + '"', date_time


def _get_header() -> dict:
    source = "xxxxxxx"
    cfg = _API_CONFIG
    auth, date_time = _get_simple_auth(source, cfg["secret_id"], cfg["secret_key"])
    return {
        "Apiversion": _API_VERSION,
        "Authorization": auth,
        "Date": date_time,
        "Source": source,
    }


def get_internal_client() -> dict:
    """Return the current API config (used in place of AzureOpenAI client)."""
    return dict(_API_CONFIG)


_tls = threading.local()


def get_thread_azure() -> Tuple[dict, str]:
    """Thread-local client config.  Returns (config_dict, model_marker)."""
    if getattr(_tls, "azure", None) is None:
        _tls.azure = (get_internal_client(), _API_CONFIG["model_marker"])
    return _tls.azure


def azure_chat(client_cfg: dict, model_marker: str,
               messages: List[Dict[str, Any]],
               max_tokens: int, retries: int) -> Tuple[str, Dict[str, Any]]:
    """
    Call the internal LLM API.  Signature kept identical to the old AzureOpenAI version
    so all call-sites require no changes.

    The API endpoint and auth follow the same pattern as geometry_judge_agent.py.
    The internal API expects 'system' as a top-level field (not inside messages),
    so we extract the system message from the messages list if present.
    """
    host = client_cfg.get("host", _API_CONFIG["host"])

    # Extract system prompt from messages (API expects it as a top-level field)
    system_content = ""
    user_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "system":
            # Extract plain text from system message
            if isinstance(content, list):
                system_content = " ".join(
                    part.get("text", "") for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            else:
                system_content = str(content)
        else:
            # Convert content to required list format:
            # [{"type": "text", "value": "..."}, {"type": "image_url", "value": "..."}]
            if isinstance(content, str):
                new_content = [{"type": "text", "value": content}]
            elif isinstance(content, list):
                new_content = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type", "text")
                    if ptype == "text":
                        new_content.append({"type": "text", "value": part.get("text", "")})
                    elif ptype == "image_url":
                        # image_url may be {"url": "data:image/...;base64,..."} or a plain str
                        img = part.get("image_url", {})
                        url = img.get("url", img) if isinstance(img, dict) else str(img)
                        new_content.append({"type": "image_url", "value": url})
                    else:
                        # passthrough for unknown types
                        new_content.append({"type": ptype, "value": str(part)})
            else:
                new_content = [{"type": "text", "value": str(content)}]

            user_messages.append({"role": role, "content": new_content})

    data = {
        "request_id": str(uuid.uuid4()),
        "model_marker": model_marker,
        "system": system_content,
        "messages": user_messages,
        "params": {
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": max_tokens,
            },
        },
        "timeout": 30,
    }

    last_err: Optional[Exception] = None

    for attempt in range(max(1, retries)):
        try:
            resp = requests.post(
                f"{host}/api/v1/data_eval",
                headers=_get_header(),
                json=data,
                timeout=120,
            )
            resp_json = resp.json()
            resp.raise_for_status()
            # Response structure: {"code": 0, "answer": [{"type": "text", "value": "..."}], ...}
            answer = resp_json.get("answer") or []
            if answer:
                txt = answer[0].get("value", "")
            else:
                txt = resp_json.get("content") or resp_json.get("text") or str(resp_json)
            return txt, parse_json_from_text(txt)
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    print(f"[API_ERROR] {repr(last_err)}")
    return f"[API_ERROR] {repr(last_err)}", {}


# ===================== IO & Utils =====================

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def last_assistant_text_from_mid_process(mid_process: Any) -> str:
    if not isinstance(mid_process, list):
        return ""
    for m in reversed(mid_process):
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
    return ""

def get_candidate_text(rec: Dict[str, Any]) -> str:
    t = last_assistant_text_from_mid_process(rec.get("mid_process"))
    return t if t.strip() else str(rec.get("model_response", "") or "")

def make_hash_uid(task: str, mode: str, sid: str, n_hex: int = 16) -> str:
    h = hashlib.sha1(f"{task}|{mode}|{sid}".encode("utf-8")).hexdigest()
    return h[:n_hex]

ANSWER_TAG_RE = re.compile(r"<answer>\s*([^<]+?)\s*</answer>", re.IGNORECASE | re.DOTALL)
FINAL_JSON_TAG_RE = re.compile(r"<FINAL_ANSWER_JSON>\s*(\{.*?\})\s*</FINAL_ANSWER_JSON>", re.IGNORECASE | re.DOTALL)
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_maze_moves(text: str) -> str:
    m = ANSWER_TAG_RE.search(text or "")
    if m:
        return re.sub(r"[^RLUD]", "", m.group(1).strip().upper())
    chunks = re.findall(r"[RLUD]+", (text or "").upper())
    return chunks[-1] if chunks else ""

def extract_jigsaw_choice(text: str) -> Optional[int]:
    m = FINAL_JSON_TAG_RE.search(text or "")
    if m:
        try:
            obj = json.loads(m.group(1))
            if "choice" in obj:
                return int(obj["choice"])
        except Exception:
            pass
    m = ANSWER_TAG_RE.search(text or "")
    if m:
        mm = re.search(r"\b([01])\b", m.group(1).strip())
        if mm:
            return int(mm.group(1))
    mm = re.search(r'\"choice\"\s*:\s*([01])', text or "")
    if mm:
        return int(mm.group(1))
    ms = list(re.finditer(r"(?<!\d)([01])(?!\d)", text or ""))
    return int(ms[-1].group(1)) if ms else None

def evaluate_text_moves(pred: List[str], gt: List[str]) -> Dict[str, float]:
    if not gt:
        return {"text_exact": 0, "text_frame_acc": 0.0}
    match_count = sum(1 for p, g in zip(pred, gt) if p == g)
    return {"text_exact": 1 if pred == gt else 0, "text_frame_acc": match_count / len(gt)}

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

def judge_overlay(client_cfg: dict, deployment: str, base_img: Path, gt_img: Path, pred_img: Path, aux_en: str,
                 max_tokens: int, retries: int, vision_max_size: int) -> Dict[str, Any]:
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
    raw_text, data = azure_chat(client_cfg, deployment, messages, max_tokens=max_tokens, retries=retries)
    ok = 1 if int(data.get("overlay_ok", 0) or 0) == 1 else 0
    reason = str(data.get("overlay_reason", "")).strip() or ""
    return {"overlay_ok": ok, "overlay_reason": reason, "raw_text": raw_text, "raw_json": data}

def judge_text(client_cfg: dict, deployment: str, task_type: str, problem: str, gt_text: str, cand_text: str,
              max_tokens: int, retries: int) -> Dict[str, Any]:
    user_text = TEXT_USER_TMPL.format(task_type=task_type, problem=problem, gt_text=gt_text, cand_text=cand_text)
    messages = [{"role": "system", "content": SYSTEM_TEXT}, {"role": "user", "content": user_text}]
    raw_text, data = azure_chat(client_cfg, deployment, messages, max_tokens=max_tokens, retries=retries)
    

    def _b(x):
        try:
            return 1 if int(x) == 1 else 0
        except Exception:
            return 0

    return {
        "reasoning_rigorous": _b(data.get("reasoning_rigorous", 0)),
        "conclusion_correct": _b(data.get("conclusion_correct", 0)),
        "calc_numeric_match": data.get("calc_numeric_match", None),
        "text_ok": _b(data.get("text_ok", 0)),
        "text_reason": str(data.get("text_reason", "")).strip() or "",
        "raw_text": raw_text,
        "raw_json": data,
    }

def pick_overlay_images_from_record(rec: Dict[str, Any]) -> Tuple[Optional[Path], Optional[Path], Optional[Path], str, List[str]]:
    merged: List[str] = []

    if isinstance(rec.get("all_images_list"), list) and rec["all_images_list"]:
        merged = [str(x) for x in rec["all_images_list"]]
    else:
        input_imgs = rec.get("input_images", [])
        tool_imgs = rec.get("tool_images", [])
        if isinstance(input_imgs, list) and isinstance(tool_imgs, list) and (input_imgs or tool_imgs):
            merged = [str(x) for x in (input_imgs + tool_imgs)]

    if not merged:
        ri = rec.get("reasoning_imgs") or []
        if isinstance(ri, list) and ri:
            merged = [str(ri[0])]

    base_img = Path(merged[0]) if merged else None
    pred_img = Path(merged[-1]) if merged else None

    gt_aux = rec.get("gt_auxiliary_imgs")
    gt_img = None
    if isinstance(gt_aux, list) and gt_aux:
        gt_img = Path(str(gt_aux[0]))

    aux_en = str(rec.get("auxiliary_qs", "") or "")
    return base_img, gt_img, pred_img, aux_en, merged


# ===================== Evaluation Logic =====================

@dataclass
class EvalCfg:
    mode: str
    judge_logs_root: Path
    max_retries: int
    vision_max_size: int
    hash_len: int


def eval_record(rec: Dict[str, Any], cfg: EvalCfg) -> Dict[str, Any]:
    task = str(rec.get("task", "") or "").strip()
    sid = str(rec.get("id", "") or "").strip()
    rec["mode"] = cfg.mode

    if not rec.get("hash_uid"):
        rec["hash_uid"] = make_hash_uid(task, cfg.mode, sid, n_hex=cfg.hash_len)

    # Allow resume: if already evaluated OK, return immediately
    if isinstance(rec.get("eval"), dict) and rec["eval"].get("status") == "ok":
        return rec

    cand_text = get_candidate_text(rec)
    case_dir = cfg.judge_logs_root / task / rec["hash_uid"]
    case_dir.mkdir(parents=True, exist_ok=True)

    # --- Maze Task ---
    if task == "maze":
        gt_str = str(rec.get("gt_text", "") or "").strip().upper()
        pred_str = extract_maze_moves(cand_text)
        m = evaluate_text_moves(list(pred_str), list(gt_str))
        out = {
            "status": "ok", "task": task, "mode": cfg.mode, "id": sid, "hash_uid": rec["hash_uid"],
            "pred_moves": pred_str, "gt_moves": gt_str,
            "text_exact": m["text_exact"], "text_frame_acc": m["text_frame_acc"],
        }
        write_json(out, case_dir / "eval.json")
        rec["eval"] = out
        return rec

    # --- Jigsaw Task ---
    if task == "jigsaw":
        gt_choice = rec.get("gt_text", None)
        gt_choice = int(gt_choice) if gt_choice is not None else None
        pred_choice = extract_jigsaw_choice(cand_text)
        acc = 1 if (gt_choice is not None and pred_choice is not None and int(pred_choice) == int(gt_choice)) else 0
        out = {
            "status": "ok", "task": task, "mode": cfg.mode, "id": sid, "hash_uid": rec["hash_uid"],
            "pred_choice": pred_choice, "gt_choice": gt_choice, "acc": acc,
        }
        write_json(out, case_dir / "eval.json")
        rec["eval"] = out
        return rec

    # --- Geometry Task (LLM Judge) ---
    if task != "geometry":
        out = {"status": "error", "error": f"unknown_task:{task}", "task": task, "id": sid, "mode": cfg.mode}
        write_json(out, case_dir / "eval.json")
        rec["eval"] = out
        return rec

    task_type_raw = str(rec.get("task_type", "calculation") or "calculation").lower()
    task_type = "CALCULATION" if task_type_raw.startswith("calc") else "PROVING"
    problem = str(rec.get("reasoning_qs", "") or "")
    gt_text = str(rec.get("gt_text", "") or "")

    client_cfg, azure_deployment = get_thread_azure()

    write_json(
        {"task_type": task_type, "problem": problem, "gt_text": gt_text, "cand_text": cand_text,
         "system": SYSTEM_TEXT, "user_template": TEXT_USER_TMPL},
        case_dir / "judge_text_request.json",
    )
    text_eval = judge_text(
        client_cfg, azure_deployment,
        task_type=task_type, problem=problem, gt_text=gt_text, cand_text=cand_text,
        max_tokens=16384, retries=cfg.max_retries
    )
    write_json(text_eval, case_dir / "judge_text_response.json")

    overlay_eval = None
    if cfg.mode == "tool_call":
        base_img, gt_img, pred_img, aux_en, merged = pick_overlay_images_from_record(rec)

        write_json(
            {"aux_en": aux_en, "base_img": str(base_img) if base_img else None,
             "gt_overlay_img": str(gt_img) if gt_img else None,
             "pred_img": str(pred_img) if pred_img else None,
             "merged_images": merged,
             "system": SYSTEM_OVERLAY, "user_template": OVERLAY_USER_TMPL,
             "note": "Images are encoded to data URLs at runtime."},
            case_dir / "judge_overlay_request.json",
        )

        # Check file existence
        files_ok = all(
            img and img.exists() 
            for img in [base_img, gt_img, pred_img]
        )
        
        if not files_ok:
            overlay_eval = {"overlay_ok": 0, "overlay_reason": "missing_overlay_images_or_not_exist"}
        else:
            overlay_eval = judge_overlay(
                client_cfg, azure_deployment,
                base_img=base_img, gt_img=gt_img, pred_img=pred_img, aux_en=aux_en,
                max_tokens=512, retries=cfg.max_retries, vision_max_size=cfg.vision_max_size
            )
        write_json(overlay_eval, case_dir / "judge_overlay_response.json")

    out = {
        "status": "ok", "task": task, "mode": cfg.mode, "id": sid, "hash_uid": rec["hash_uid"],
        "text_ok": int(text_eval.get("text_ok", 0) == 1),
        "reasoning_rigorous": int(text_eval.get("reasoning_rigorous", 0) == 1),
        "conclusion_correct": int(text_eval.get("conclusion_correct", 0) == 1),
        "calc_numeric_match": text_eval.get("calc_numeric_match", None),
        "overlay_ok": (overlay_eval.get("overlay_ok") if overlay_eval else None),
        "overlay_reason": (overlay_eval.get("overlay_reason") if overlay_eval else None),
    }
    write_json(out, case_dir / "eval.json")
    rec["eval"] = out
    return rec


def compute_summary(records: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    def ok_evs(task: str) -> List[Dict[str, Any]]:
        evs = []
        for r in records:
            if str(r.get("task")) != task:
                continue
            ev = r.get("eval")
            if isinstance(ev, dict) and ev.get("status") == "ok":
                evs.append(ev)
        return evs

    out: Dict[str, Any] = {"mode": mode, "tasks": {}}

    # Maze Summary
    evs = ok_evs("maze")
    n = len(evs)
    out["tasks"]["maze"] = {
        "scored": n,
        "exact_acc": (sum(e.get("text_exact", 0) for e in evs) / n) if n else 0.0,
        "avg_text_frame_acc": (sum(e.get("text_frame_acc", 0.0) for e in evs) / n) if n else 0.0,
    }

    # Jigsaw Summary
    evs = ok_evs("jigsaw")
    n = len(evs)
    out["tasks"]["jigsaw"] = {
        "scored": n,
        "acc": (sum(e.get("acc", 0) for e in evs) / n) if n else 0.0,
    }

    # Geometry Summary
    evs = ok_evs("geometry")
    n = len(evs)
    geom = {
        "scored": n,
        "text_acc": (sum(e.get("text_ok", 0) for e in evs) / n) if n else 0.0,
    }
    if mode == "tool_call":
        overlay_vals = [e.get("overlay_ok") for e in evs if e.get("overlay_ok") is not None]
        geom["overlay_scored"] = len(overlay_vals)
        geom["overlay_acc"] = (sum(1 for v in overlay_vals if int(v) == 1) / len(overlay_vals)) if overlay_vals else 0.0
    out["tasks"]["geometry"] = geom

    return out


# ===================== Public API =====================

def calculate_score(
    input_data: Union[str, Path, List[Dict[str, Any]]],
    use_tools: bool,
    output_dir: Union[str, Path],
    workers: int = 8,
    max_retries: int = 2,
    vision_max_size: int = 0,
    hash_len: int = 16,
    skip_if_eval_ok: bool = False,
    # Internal API overrides (optional)
    api_host: Optional[str] = None,
    api_secret_id: Optional[str] = None,
    api_secret_key: Optional[str] = None,
    api_model_marker: Optional[str] = None,
    # Legacy Azure params kept for backward compatibility (ignored)
    azure_endpoint: Optional[str] = None,
    azure_api_key: Optional[str] = None,
    azure_deployment: Optional[str] = None,
    azure_api_version: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Main function to calculate scores for benchmark records.

    Args:
        input_data: Path to JSON list file OR the list of records itself.
        use_tools: Whether to use tool_call mode (True) or direct mode (False).
        output_dir: Directory to save evaluation logs and summary.
        workers: Number of threads for parallel processing.
        max_retries: Retry count for API calls.
        vision_max_size: Resize logic for images (0 means no resize).
        hash_len: Length of UID hash.
        skip_if_eval_ok: If True, skip records that already contain a successful 'eval' field.
        api_host: Override internal API host.
        api_secret_id: Override internal API secret_id.
        api_secret_key: Override internal API secret_key.
        api_model_marker: Override internal API model_marker.

    Returns:
        Tuple containing:
        1. Summary dictionary (aggregated stats).
        2. List of detailed records with evaluation results added.
    """

    # 1. Update internal API config if overrides provided
    if api_host:         _API_CONFIG["host"] = api_host
    if api_secret_id:    _API_CONFIG["secret_id"] = api_secret_id
    if api_secret_key:   _API_CONFIG["secret_key"] = api_secret_key
    if api_model_marker: _API_CONFIG["model_marker"] = api_model_marker
    # Reset thread-local client so new config is picked up
    _tls.azure = None

    if use_tools:
        mode = 'tool_call'
    else:
        mode = 'direct'

    # 2. Prepare Data
    if isinstance(input_data, (str, Path)):
        in_path = Path(input_data)
        if not in_path.exists():
            raise FileNotFoundError(f"input not found: {in_path}")
        records = read_json(in_path)
    else:
        records = input_data

    if not isinstance(records, list):
        raise ValueError("input_data must be a list of records")

    # 3. Setup Paths
    out_root = Path(output_dir) / mode
    out_root.mkdir(parents=True, exist_ok=True)
    judge_logs_root = out_root / "judge_logs"

    # 4. Config Object
    cfg = EvalCfg(
        mode=mode,
        judge_logs_root=judge_logs_root,
        max_retries=max_retries,
        vision_max_size=vision_max_size,
        hash_len=hash_len,
    )

    out_records: List[Dict[str, Any]] = [None] * len(records)  # type: ignore

    def _work(i: int) -> Tuple[int, Dict[str, Any]]:
        r = records[i]
        # Check if we should skip
        if skip_if_eval_ok and isinstance(r.get("eval"), dict) and r["eval"].get("status") == "ok":
            r["mode"] = mode
            if not r.get("hash_uid"):
                r["hash_uid"] = make_hash_uid(str(r.get("task", "")), mode, str(r.get("id", "")), n_hex=hash_len)
            return i, r
        return i, eval_record(r, cfg)

    # 5. Execute
    print(f"Starting evaluation: {len(records)} records, mode={mode}, workers={workers}")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_work, i) for i in range(len(records))]
        for fu in tqdm(as_completed(futs), total=len(futs), desc=f"evaluating[{mode}]"):
            i, rr = fu.result()
            out_records[i] = rr

    # 6. Summarize
    summary = compute_summary(out_records, mode)
    
    # Save main outputs to disk as well
    write_json(out_records, out_root / "detailed_eval.json")
    write_json(summary, out_root / "summary.json")

    print("UniMMMU Acc:\n", summary['tasks'])


# ===================== CLI Entry Point =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=["direct", "tool_call"])
    ap.add_argument("--input_json", type=str, required=True, help="JSON list file containing responses + gt fields")
    ap.add_argument("--eval_root", type=str, required=True, help="output root dir")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_retries", type=int, default=1)
    ap.add_argument("--vision_max_size", type=int, default=0)
    ap.add_argument("--hash_len", type=int, default=16)
    ap.add_argument("--skip_if_eval_ok", action="store_true")
    # Internal API overrides
    ap.add_argument("--api_host",         type=str, default=None)
    ap.add_argument("--api_secret_id",    type=str, default=None)
    ap.add_argument("--api_secret_key",   type=str, default=None)
    ap.add_argument("--api_model_marker", type=str, default=None)

    args = ap.parse_args()

    summary, _ = calculate_score(
        input_data=args.input_json,
        use_tools=(args.mode == "tool_call"),
        output_dir=args.eval_root,
        workers=args.workers,
        max_retries=args.max_retries,
        vision_max_size=args.vision_max_size,
        hash_len=args.hash_len,
        skip_if_eval_ok=args.skip_if_eval_ok,
        api_host=args.api_host,
        api_secret_id=args.api_secret_id,
        api_secret_key=args.api_secret_key,
        api_model_marker=args.api_model_marker,
    )

    print(f"[DONE] Mode: {args.mode}")
    print(f"[DONE] Summary: {json.dumps(summary, indent=2)}")
    print(f"[DONE] Logs saved to: {args.eval_root}/{args.mode}")


if __name__ == "__main__":
    main()