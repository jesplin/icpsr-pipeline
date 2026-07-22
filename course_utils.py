"""
course_utils.py — shared helpers for the ICPSR Social Science Data Pipeline course.

Everything LLM-related in this course goes through this file, for two reasons:
1. You only have to understand the plumbing once.
2. Every call gets logged to outputs/llm_call_log.csv — and on Day 5 that log
   becomes a teaching device: it is the reproducibility record of every model,
   prompt, and dollar this course spent.

There is no magic here. Read this file; it is course material.
"""

import base64
import csv
import datetime
import hashlib
import json
import os
import re
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Our workhorse text model: a mixture-of-experts model (30B parameters, ~3B
# active per token), which makes it fast and cheap while still a capable
# classifier. The vision model is used in Lab 6.
DEFAULT_MODEL = "qwen/qwen3-30b-a3b"
VISION_MODEL = "qwen/qwen3-vl-30b-a3b-instruct"  # see book/multimodal.qmd for why

# Prices in USD per MILLION tokens, from openrouter.ai/models (checked
# 2026-06-10). These feed the cost column of the call log. If you use a model
# not listed here, the cost is logged as blank, not zero — unknown is not the
# same as free.
PRICES = {
    "qwen/qwen3-30b-a3b":             {"input": 0.12, "output": 0.50},
    "qwen/qwen3-8b":                  {"input": 0.05, "output": 0.40},
    "qwen/qwen3-vl-30b-a3b-instruct": {"input": 0.13, "output": 0.52},
}

LOG_PATH = os.path.join("outputs", "llm_call_log.csv")

_client = None  # created lazily on first use, so importing this file is free


def get_client():
    """Create (once) and return the OpenAI-format client, pointed at OpenRouter.

    The `openai` package is just a generic client for any OpenAI-compatible
    API. The key comes from the OPENROUTER_API_KEY environment variable; if
    it isn't set, we prompt for it rather than crashing.
    """
    global _client
    if _client is None:
        from openai import OpenAI  # imported here so pandas-only labs don't need it

        if "OPENROUTER_API_KEY" not in os.environ:
            import getpass
            os.environ["OPENROUTER_API_KEY"] = getpass.getpass(
                "Paste your OpenRouter API key: ")
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    return _client


# ---------------------------------------------------------------------------
# Call logging — the log IS part of your research record
# ---------------------------------------------------------------------------

def _log_call(model, prompt, usage):
    """Append one row per API call: when, what model, a hash of the prompt
    (so you can tell whether two runs used the same prompt without storing
    full text thousands of times), token counts, and computed cost."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    new_file = not os.path.exists(LOG_PATH)

    in_tok = getattr(usage, "prompt_tokens", None)
    out_tok = getattr(usage, "completion_tokens", None)
    price = PRICES.get(model)
    if price and in_tok is not None and out_tok is not None:
        cost = (in_tok * price["input"] + out_tok * price["output"]) / 1_000_000
        cost = f"{cost:.8f}"
    else:
        cost = ""  # unknown model or missing usage: blank, not zero

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "model", "prompt_hash",
                             "input_tokens", "output_tokens", "cost_usd"])
        writer.writerow([
            datetime.datetime.now().isoformat(timespec="seconds"),
            model,
            hashlib.sha256(prompt.encode()).hexdigest()[:12],
            in_tok, out_tok, cost,
        ])


# ---------------------------------------------------------------------------
# The two calls you'll actually make
# ---------------------------------------------------------------------------

def chat(prompt, system=None, model=DEFAULT_MODEL, temperature=0.0, max_retries=3):
    """Send one prompt, return the text of the response.

    - temperature=0.0 makes output (mostly!) deterministic — right for
      classification, wrong for creative tasks. Day 5 complicates the word
      "deterministic" considerably.
    - Retries with exponential backoff, because networks fail and so do
      providers.
    - Strips <think>...</think> blocks: Qwen3 is a "hybrid reasoning" model
      that sometimes thinks out loud before answering.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return _complete(messages, prompt, model, temperature, max_retries)


def chat_image(prompt, image_path, system=None, model=VISION_MODEL,
               temperature=0.0, max_retries=3):
    """Send one prompt plus one image (Lab 6). The image is read from disk
    and sent base64-encoded inside the message — no upload step, no URL."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(image_path)[1].lstrip(".").lower().replace("jpg", "jpeg")

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url",
         "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
    ]
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return _complete(messages, prompt, model, temperature, max_retries)


def _complete(messages, prompt_for_log, model, temperature, max_retries):
    """Shared request/retry/log loop behind chat() and chat_image()."""
    client = get_client()
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature)
            _log_call(model, prompt_for_log, resp.usage)
            text = resp.choices[0].message.content
            return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"Error ({e}), retrying in {wait}s...")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Parsing and validating structured output
# ---------------------------------------------------------------------------

def parse_json(raw, required_keys=None, valid_values=None):
    """Extract and validate a JSON object from a model response.

    Returns the parsed dict, or None on ANY failure — your pipeline's
    "I broke" signal. Valid JSON can still be garbage, so you can also check:
      required_keys: keys that must be present, e.g. ["label", "confidence"]
      valid_values:  {key: set_of_allowed_values}, e.g.
                     {"label": {"civilian_harm", "other"}}
    """
    if raw is None:
        return None
    # Models sometimes wrap JSON in markdown fences or add stray prose;
    # grab the first {...} block we can find.
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if required_keys and not all(k in obj for k in required_keys):
        return None
    if valid_values:
        for key, allowed in valid_values.items():
            if obj.get(key) not in allowed:
                return None
    return obj


# ---------------------------------------------------------------------------
# Cost estimation — do this BEFORE every batch run
# ---------------------------------------------------------------------------

def estimate_cost(prompt, docs, output_tokens, input_price, output_price):
    """Estimate the cost of classifying `docs` (a list of strings) with
    `prompt` prepended to each. Prices are USD per million tokens.

    Uses the tokens ≈ words × 1.3 rule of thumb (English; worse for many
    other languages). Prints a small report and returns total estimated USD.
    """
    prompt_tokens = len(prompt.split()) * 1.3
    avg_doc_tokens = sum(len(d.split()) for d in docs) / max(len(docs), 1) * 1.3
    per_doc = ((prompt_tokens + avg_doc_tokens) * input_price
               + output_tokens * output_price) / 1_000_000
    total = per_doc * len(docs)
    print(f"Estimated cost per document: ${per_doc:.6f}")
    print(f"This corpus ({len(docs)} docs):  ${total:.4f}")
    print(f"A 100,000-doc corpus:          ${per_doc * 100_000:.2f}")
    return total
