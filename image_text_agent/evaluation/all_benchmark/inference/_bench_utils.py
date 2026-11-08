"""Shared helpers for general understanding benchmarks.

Many understanding benchmarks share the same basic structure:
  - Load an HF dataset
  - Cache PIL images to disk (to fit run_bagel.py's JSONL-path interface)
  - Format a multiple-choice prompt with <answer> tag instruction
  - Extract a single letter (A/B/C/D) from the response

This module provides small helpers reused by the per-benchmark inference
and scoring modules under inference/ and calculate_score/.
"""

import os
import re
from PIL import Image


LETTER_LABELS = [chr(ord("A") + i) for i in range(26)]  # A-Z


def save_pil_image(pil_image, out_path):
    """Save a PIL image to out_path, converting to RGB if needed."""
    if pil_image is None:
        return None
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    if not os.path.exists(out_path):
        pil_image.save(out_path)
    return out_path


def format_mc_prompt(question, options, image_desc="image"):
    """Format a 4-choice (or N-choice) multiple-choice prompt with <answer> tag."""
    choice_str = ""
    for i, opt in enumerate(options):
        letter = LETTER_LABELS[i]
        choice_str += f"\n({letter}) {opt}"
    letter_range = f"A, B, C, or D" if len(options) == 4 else f"one of {', '.join(LETTER_LABELS[: len(options)])}"
    return (
        f"{question}{choice_str}\n\n"
        f"Please analyze the {image_desc} carefully, conduct step-by-step reasoning, "
        f"and provide the final answer as a single letter ({letter_range}) "
        f"inside <answer></answer> tags."
    )


def format_openended_prompt(question, image_desc="image"):
    """Format an open-ended prompt that asks for an answer inside <answer> tags."""
    return (
        f"{question}\n\n"
        f"Please analyze the {image_desc} carefully, conduct step-by-step reasoning, "
        f"and provide the final answer inside <answer></answer> tags."
    )


def extract_answer_tag(response):
    """Return content inside the first <answer>...</answer> tag, or stripped response."""
    if not response:
        return ""
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response.strip()


def extract_letter(response, num_choices=4):
    """Extract a single MC letter (A, B, ...) from the response.

    Priority:
      1. Inside <answer>...</answer>
      2. First matching letter in the full response
    """
    if not response:
        return ""
    candidate = extract_answer_tag(response)
    valid = "".join(LETTER_LABELS[:num_choices])
    m = re.search(f"[{valid}]", candidate, re.IGNORECASE)
    if m:
        return m.group(0).upper()
    return ""


def is_numeric(s):
    if s is None:
        return False
    try:
        cleaned = str(s).replace(",", "").replace("%", "").replace("$", "").strip()
        float(cleaned)
        return True
    except (ValueError, TypeError):
        return False


def parse_numeric(s):
    cleaned = str(s).replace(",", "").replace("%", "").replace("$", "").strip()
    return float(cleaned)


def numeric_match(pred, gold, rel_tol=0.05, abs_tol=1e-3):
    """Return True if pred ~= gold as floats (abs or rel tolerance)."""
    if not (is_numeric(pred) and is_numeric(gold)):
        return False
    p = parse_numeric(pred)
    g = parse_numeric(gold)
    if abs(p - g) <= abs_tol:
        return True
    if g == 0:
        return p == 0
    return abs(p - g) / abs(g) <= rel_tol
