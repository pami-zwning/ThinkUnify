#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared client for close-source model evaluation (Gemini, Doubao, etc.)

Supports:
  - gemini-3-pro-preview
  - gemini-3.1-pro-preview
  - doubao-seed-2.0 (via openai_proxy)

Usage:
    from close_source_client import create_client

    client = create_client("gemini-3-pro")
    response_text = client.generate(prompt="Hello", image_paths=["img.png"])

    client = create_client("doubao-seed-2.0")
    response_text = client.generate(prompt="Hello", image_paths=["img.png"])
"""

import base64
import getpass
import json
import sys
import time
import os
from io import BytesIO
from pathlib import Path
from typing import List, Optional
import uuid

import requests
from PIL import Image

# Add openai_proxy to path
_EVAL_ROOT = str(Path(__file__).resolve().parent)
if _EVAL_ROOT not in sys.path:
    sys.path.insert(0, _EVAL_ROOT)


# ── Model registry ──────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "gemini-3-pro": {
        "backend": "gemini",
        "api_host": "https://api.ppchat.vip",
        "model_name": "gemini-3-pro-preview",
        "api_key": os.getenv("CLOSE_SOURCE_API_KEY", "YOUR_API_KEY"),
    },
    "gemini-3.1-pro": {
        "backend": "gemini",
        "api_host": "https://api.ppchat.vip",
        "model_name": "gemini-3.1-pro-preview",
        "api_key": os.getenv("CLOSE_SOURCE_API_KEY", "YOUR_API_KEY"),
    },
    "doubao-seed-2.0": {
        "backend": "doubao",
        "model_name": "doubao-seed-2.0-260215",
        "api_key": os.getenv("DOUBAO_API_KEY", "YOUR_API_KEY"),
        "channel_code": "doubao",
    },
}


def list_models():
    """Return available model names."""
    return list(MODEL_CONFIGS.keys())


# ── Image encoding ──────────────────────────────────────────────────────────

def encode_image_to_base64(image_path: str, max_size: int = 0) -> Optional[dict]:
    """
    Encode an image file to Gemini inline_data format.

    Returns:
        {"mime_type": "image/png", "data": "<base64>"}  or None on failure.
    """
    try:
        p = Path(image_path)
        if not p.exists():
            print(f"[WARN] Image not found: {image_path}")
            return None

        with Image.open(p) as img:
            fmt = (img.format or "PNG").lower()
            if fmt == "jpg":
                fmt = "jpeg"
            if max_size > 0:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            # Ensure RGB for JPEG
            if fmt == "jpeg" and img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format=fmt.upper())
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        mime = f"image/{fmt}"
        return {"mime_type": mime, "data": b64}
    except Exception as e:
        print(f"[WARN] Failed to encode image {image_path}: {e}")
        return None


def encode_image_to_data_url(image_path: str, max_size: int = 0) -> Optional[str]:
    """
    Encode an image file to a data URL for OpenAI-style vision APIs.

    Returns:
        "data:image/png;base64,<base64>" or None on failure.
    """
    result = encode_image_to_base64(image_path, max_size=max_size)
    if result is None:
        return None
    return f"data:{result['mime_type']};base64,{result['data']}"


# ── Gemini client ───────────────────────────────────────────────────────────

class GeminiClient:
    """
    Client for Gemini-style REST API through a proxy.

    API format:
        POST {api_host}/v1beta/models/{model}:generateContent
        Headers: x-goog-api-key, Content-Type: application/json
        Body: {"contents": [{"parts": [...]}], "generationConfig": {...}}
    """

    def __init__(self, api_host: str, model_name: str, api_key: str,
                 max_retries: int = 3, retry_delay: float = 5.0,
                 timeout: int = 300):
        self.api_host = api_host.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

        self.url = f"{self.api_host}/v1beta/models/{self.model_name}:generateContent"
        self.headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    @staticmethod
    def _extract_response_text(data: dict) -> str:
        """
        Extract the actual response text from Gemini API response,
        handling the thinking/thought parts properly.

        Gemini 3 models return multiple parts:
          [{"thought": true, "text": "thinking..."}, {"text": "actual answer"}]
        We only want the non-thought parts as the final response.
        """
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError):
            if "candidates" not in data or len(data.get("candidates", [])) == 0:
                block_reason = data.get("promptFeedback", {}).get("blockReason", "UNKNOWN")
                return f"[ERROR] Blocked by safety filter: {block_reason}"
            return f"[ERROR] Unexpected response format: {json.dumps(data, ensure_ascii=False)[:500]}"

        # Collect non-thought text parts
        answer_texts = []
        for part in parts:
            if part.get("thought", False):
                continue
            if "text" in part:
                answer_texts.append(part["text"])

        if answer_texts:
            return "\n".join(answer_texts)

        # Fallback: if no non-thought parts found, return all text parts
        all_texts = [part["text"] for part in parts if "text" in part]
        if all_texts:
            return "\n".join(all_texts)

        return f"[ERROR] No text in response parts: {json.dumps(data, ensure_ascii=False)[:500]}"

    def generate(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        max_tokens: int = 16384,
        temperature: float = 0.5,
        top_p: float = 0.25,
        image_max_size: int = 0,
        thinking_budget: int = 8192,
    ) -> str:
        """
        Send a request and return the generated text.

        Args:
            prompt: Text prompt.
            image_paths: Optional list of image file paths to include.
            max_tokens: Maximum output tokens (includes thinking tokens).
            temperature: Sampling temperature.
            top_p: Top-p sampling.
            image_max_size: If > 0, resize images to fit within this box.
            thinking_budget: Token budget for thinking. Set 0 to disable thinking.

        Returns:
            Generated text string. Returns "[ERROR] ..." on failure.
        """
        # Build parts
        parts = []
        # Add images first (Gemini expects images before text typically)
        if image_paths:
            for img_path in image_paths:
                encoded = encode_image_to_base64(img_path, max_size=image_max_size)
                if encoded:
                    parts.append({"inline_data": encoded})
        # Add text
        parts.append({"text": prompt})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                "topP": top_p,
                "thinkingConfig": {
                    "thinkingBudget": thinking_budget,
                },
            },
        }

        # Retry loop
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    self.url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return self._extract_response_text(data)
                elif resp.status_code == 429:
                    # Rate limited
                    wait = self.retry_delay * attempt
                    print(f"[WARN] Rate limited (429), waiting {wait:.1f}s (attempt {attempt}/{self.max_retries})")
                    time.sleep(wait)
                    continue
                elif resp.status_code >= 500:
                    # Server error, retry
                    wait = self.retry_delay * attempt
                    print(f"[WARN] Server error {resp.status_code}, waiting {wait:.1f}s (attempt {attempt}/{self.max_retries})")
                    time.sleep(wait)
                    continue
                else:
                    return f"[ERROR] HTTP {resp.status_code}: {resp.text[:500]}"
            except requests.exceptions.Timeout:
                wait = self.retry_delay * attempt
                print(f"[WARN] Timeout, waiting {wait:.1f}s (attempt {attempt}/{self.max_retries})")
                time.sleep(wait)
                continue
            except Exception as e:
                return f"[ERROR] Request failed: {repr(e)}"

        return f"[ERROR] Max retries ({self.max_retries}) exceeded"

    def __repr__(self):
        return f"GeminiClient(model={self.model_name}, host={self.api_host})"


# ── Doubao client (via openai_proxy) ────────────────────────────────────────

class DoubaoClient:
    """
    Client for Doubao models via openai_proxy.GptProxy.

    Uses OpenAI-style messages format with vision support.
    """

    def __init__(self, model_name: str, api_key: str,
                 channel_code: str = "doubao",
                 max_retries: int = 3, retry_delay: float = 5.0,
                 timeout: int = 300):
        self.model_name = model_name
        self.api_key = api_key
        self.channel_code = channel_code
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

    def _get_proxy_client(self):
        """Create a GptProxy client instance."""
        import openai_proxy
        openai_proxy.generate.default_url = "http://api.schedule.mtc.sensetime.com:80"
        from openai_proxy import GptProxy
        return GptProxy(api_key=self.api_key, )

    @staticmethod
    def _extract_response_text(data: dict) -> str:
        """
        Extract the actual response text from doubao proxy API response.

        Response format from openai_proxy.GptProxy.generate():
        {
            "code": 10000, "msg": "成功",
            "data": {
                "task_id": ...,
                "response_content": {
                    "choices": [{"message": {"content": "...", "role": "assistant"}}],
                    ...
                },
                "request_content": [...]
            }
        }
        """
        try:
            # Check API-level error
            code = data.get("code", -1)
            if code != 10000:
                return f"[ERROR] API error code={code}: {data.get('msg', 'unknown')}"

            inner = data.get("data", {})

            # Path 1: data.data.response_content.choices[0].message.content
            resp_content = inner.get("response_content", {})
            if isinstance(resp_content, dict):
                choices = resp_content.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    text = msg.get("content", "")
                    if text:
                        return text

            # Path 2: response_content is a string directly
            if isinstance(resp_content, str) and resp_content:
                return resp_content

            # Path 3: top-level choices (unlikely but defensive)
            choices = data.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                text = msg.get("content", "")
                if text:
                    return text

            # Dump for debugging (strip base64 images to keep it readable)
            debug_str = json.dumps(data, ensure_ascii=False)
            # Remove base64 data from debug output
            import re
            debug_str = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', '<BASE64_IMG>', debug_str)
            return f"[ERROR] Cannot extract text from response: {debug_str[:1000]}"
        except Exception as e:
            return f"[ERROR] Failed to parse response: {repr(e)}"

    def generate(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        max_tokens: int = 16384,
        temperature: float = 0.5,
        top_p: float = 0.25,
        image_max_size: int = 0,
        thinking_budget: int = 8192,
    ) -> str:
        """
        Send a request and return the generated text.

        Same interface as GeminiClient.generate() for compatibility.
        """
        # Build OpenAI-style message content
        content = []
        if image_paths:
            for img_path in image_paths:
                data_url = encode_image_to_data_url(img_path, max_size=image_max_size)
                if data_url:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    })
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        user = getpass.getuser()
        # transaction_id = f"{user}-{self.model_name}"
        transaction_id = f"visualbeamsearch_gpt_{uuid.uuid4().hex[:8]}"

        # Retry loop
        for attempt in range(1, self.max_retries + 1):
            try:
                client = self._get_proxy_client()
                rsp = client.generate(
                    messages=messages,
                    model=self.model_name,
                    channel_code=self.channel_code,
                    transaction_id=transaction_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                if rsp.ok:
                    data = rsp.json()
                    return self._extract_response_text(data)
                else:
                    status_code = rsp.status_code
                    if status_code == 429:
                        wait = self.retry_delay * attempt
                        print(f"[WARN] Rate limited (429), waiting {wait:.1f}s (attempt {attempt}/{self.max_retries})")
                        time.sleep(wait)
                        continue
                    elif status_code >= 500:
                        wait = self.retry_delay * attempt
                        print(f"[WARN] Server error {status_code}, waiting {wait:.1f}s (attempt {attempt}/{self.max_retries})")
                        time.sleep(wait)
                        continue
                    else:
                        return f"[ERROR] HTTP {status_code}: {rsp.text[:500]}"
            except Exception as e:
                if attempt < self.max_retries:
                    wait = self.retry_delay * attempt
                    print(f"[WARN] Request failed: {repr(e)}, waiting {wait:.1f}s (attempt {attempt}/{self.max_retries})")
                    time.sleep(wait)
                    continue
                return f"[ERROR] Request failed: {repr(e)}"

        return f"[ERROR] Max retries ({self.max_retries}) exceeded"

    def __repr__(self):
        return f"DoubaoClient(model={self.model_name})"


# ── Factory ─────────────────────────────────────────────────────────────────

def create_client(model_key: str, **kwargs):
    """
    Create a client by model key name.

    Args:
        model_key: One of the keys in MODEL_CONFIGS (e.g. "gemini-3-pro", "doubao-seed-2.0").
        **kwargs: Override any client constructor parameter.

    Returns:
        GeminiClient or DoubaoClient instance.
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{model_key}'. Available: {list(MODEL_CONFIGS.keys())}"
        )
    cfg = MODEL_CONFIGS[model_key]
    backend = cfg.get("backend", "gemini")

    if backend == "doubao":
        params = {
            "model_name": cfg["model_name"],
            "api_key": cfg["api_key"],
            "channel_code": cfg.get("channel_code", "doubao"),
        }
        params.update(kwargs)
        return DoubaoClient(**params)
    else:
        params = {
            "api_host": cfg["api_host"],
            "model_name": cfg["model_name"],
            "api_key": cfg["api_key"],
        }
        params.update(kwargs)
        return GeminiClient(**params)
