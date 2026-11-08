"""
Gemini API wrapper via a configurable GPT-as-judge gateway.
Used as a judge for MathVista and other benchmarks that need LLM evaluation.
Configure the endpoint/credentials via the GEMINI_API_HOST / GEMINI_SECRET_ID /
GEMINI_SECRET_KEY environment variables.
"""
import base64
import datetime
import hashlib
import hmac
import os
import time
import uuid

import requests


API_VERSION = "v2.03"


def _get_auth(secret_id, secret_key):
    source = "xxxxxxx"
    dt = datetime.datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
    sign_str = f"date: {dt}\nsource: {source}"
    sig = base64.b64encode(
        hmac.new(secret_key.encode(), sign_str.encode(),
                 hashlib.sha1).digest()
    ).decode()
    auth = (f'hmac id="{secret_id}", algorithm="hmac-sha1", '
            f'headers="date source", signature="{sig}"')
    return {"Apiversion": API_VERSION, "Authorization": auth,
            "Date": dt, "Source": source, "Content-Type": "application/json"}


class GeminiJudge:
    """
    A judge model wrapper that calls Gemini via the internal trpc gateway.
    Compatible with VLMEvalKit's judge interface (has .generate() and .working() methods).
    """

    def __init__(
        self,
        model_version="gemini-2.5-flash",
        host=None,
        secret_id=None,
        secret_key=None,
        model_marker=None,
        max_tokens=1024,
        temperature=0,
        timeout=120,
        retry=3,
        verbose=False,
        **kwargs,
    ):
        self.host = host or os.environ.get(
            "GEMINI_API_HOST", "http://your-judge-api-host:8080"
        )
        self.secret_id = secret_id or os.environ.get(
            "GEMINI_SECRET_ID", "YOUR_SECRET_ID"
        )
        self.secret_key = secret_key or os.environ.get(
            "GEMINI_SECRET_KEY", "YOUR_SECRET_KEY"
        )
        self.model_marker = model_marker or os.environ.get(
            "GEMINI_MODEL_MARKER", "api_naci_default_gemini-2.5-flash"
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.retry = retry
        self.verbose = verbose
        self.model = model_version
        # Bypass corporate proxy for internal Polaris endpoints
        self.proxies = {"http": None, "https": None}
        # But if user explicitly sets a proxy for eval, use that
        eval_proxy = os.environ.get('EVAL_HTTP_PROXY', '')
        if eval_proxy:
            self.proxies = {"http": eval_proxy, "https": eval_proxy}

    def working(self):
        """Test if the API is reachable."""
        try:
            resp = self.generate_inner([dict(type='text', value='Say OK.')])
            if self.verbose:
                print(f"[GeminiJudge] working() check: code={resp[0]}, answer={resp[1][:50]}")
            return resp[0] == 0  # code == 0 means success
        except Exception as e:
            if self.verbose:
                print(f"[GeminiJudge] working() failed: {e}")
            return False

    def generate_inner(self, msgs, **kwargs):
        """
        Generate a response from Gemini.

        Args:
            msgs: list of dicts with 'type' and 'value' keys, or list of strings.

        Returns:
            (code, answer, raw_response)
            code: 0 for success, -1 for failure
        """
        # Convert VLMEvalKit message format to internal trpc gateway format:
        # messages = [{"role": "user", "content": [{"type": "text", "value": "..."}]}]
        content_parts = []
        for msg in msgs:
            if isinstance(msg, dict):
                if msg.get('type') == 'text':
                    content_parts.append({"type": "text", "value": msg['value']})
                elif msg.get('type') == 'image':
                    # Skip images for text-only judge
                    pass
            elif isinstance(msg, str):
                content_parts.append({"type": "text", "value": msg})

        messages = [{"role": "user", "content": content_parts}]

        data = {
            "request_id": str(uuid.uuid4()),
            "model_marker": self.model_marker,
            "system": "You are a helpful assistant for evaluating model outputs.",
            "messages": messages,
            "params": {
                "generationConfig": {
                    "maxOutputTokens": self.max_tokens,
                    "temperature": self.temperature,
                }
            },
            "timeout": self.timeout,
        }

        for attempt in range(self.retry):
            try:
                headers = _get_auth(self.secret_id, self.secret_key)
                resp = requests.post(
                    f"{self.host}/api/v1/data_eval",
                    headers=headers,
                    json=data,
                    timeout=self.timeout,
                    proxies=self.proxies,
                )
                if resp.status_code == 200:
                    result = resp.json()
                    answer = result.get("answer", [{}])[-1].get("value", "")
                    if self.verbose:
                        print(f"[GeminiJudge] Response: {answer[:200]}")
                    return 0, answer, resp.text
                else:
                    if self.verbose:
                        print(f"[GeminiJudge] HTTP {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                if self.verbose:
                    print(f"[GeminiJudge] Attempt {attempt+1}/{self.retry} failed: {e}")
                time.sleep(2 ** attempt)

        return -1, "Failed to get response from Gemini API", ""

    def generate(self, msgs, temperature=None, **kwargs):
        """
        High-level generate method compatible with VLMEvalKit judge interface.
        Returns the answer string directly.

        Args:
            msgs: str or list of message dicts
            temperature: optional temperature override
        """
        if isinstance(msgs, str):
            msgs = [dict(type='text', value=msgs)]

        # Override temperature if provided
        old_temp = self.temperature
        if temperature is not None:
            self.temperature = temperature

        code, answer, _ = self.generate_inner(msgs, **kwargs)

        self.temperature = old_temp

        if code == 0:
            return answer
        return ""
