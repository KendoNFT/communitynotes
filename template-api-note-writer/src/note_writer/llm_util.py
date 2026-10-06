"""
LLM helpers for the AI Note Writer, using Claude (Anthropic API) instead of Grok.

The function names (get_grok_response, etc.) are kept so the rest of the
template works unchanged. Set ANTHROPIC_API_KEY as a GitHub Actions secret.
Optionally set CLAUDE_MODEL to change the model.
"""

import os
import time

import dotenv
import requests

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = os.getenv("CLAUDE_MODEL") or "claude-sonnet-5-5"
SYSTEM_PROMPT = (
    "You help write X Community Notes. Be accurate, neutral and concise. "
    "Only state facts you can support with reliable, linkable sources."
)


def _headers() -> dict:
    return {
        "content-type": "application/json",
        "x-api-key": os.getenv("ANTHROPIC_API_KEY", ""),
        "anthropic-version": "2023-06-01",
    }


def _post(payload: dict) -> dict:
    """POST to the Messages API with simple retries for rate limits / overload."""
    for attempt in range(4):
        response = requests.post(API_URL, headers=_headers(), json=payload, timeout=300)
        if response.status_code == 200:
            return response.json()
        if response.status_code in (429, 500, 502, 503, 529) and attempt < 3:
            time.sleep(5 * (attempt + 1))
            continue
        raise Exception(f"Error making request: {response.status_code} {response.text}")


def _text_of(content: list) -> str:
    return "".join(block.get("text", "") for block in content if block.get("type") == "text").strip()


def _make_request(payload: dict) -> str:
    """Send a request; if a server tool (web search) pauses the turn, continue it."""
    data = _post(payload)
    content = list(data.get("content", []))
    for _ in range(5):
        if data.get("stop_reason") != "pause_turn":
            break
        payload = {
            **payload,
            "messages": payload["messages"] + [{"role": "assistant", "content": content}],
        }
        data = _post(payload)
        content = content + list(data.get("content", []))
    return _text_of(content)


def get_grok_response(prompt: str, temperature: float = 0.8, model: str = DEFAULT_MODEL):
    payload = {
        "model": model,
        "max_tokens": 2048,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    return _make_request(payload)


def grok_describe_image(image_url: str, temperature: float = 0.01, model: str = DEFAULT_MODEL):
    payload = {
        "model": model,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "url", "url": image_url}},
                    {"type": "text", "text": "Describe this image factually, including any visible text."},
                ],
            }
        ],
        "temperature": temperature,
    }
    return _make_request(payload)


def get_grok_live_search_response(prompt: str, temperature: float = 0.8, model: str = DEFAULT_MODEL):
    """Research with Claude's server-side web search tool."""
    payload = {
        "model": model,
        "max_tokens": 4096,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
        "temperature": temperature,
    }
    return _make_request(payload)


# Clearer aliases if you edit the rest of the code later.
get_claude_response = get_grok_response
claude_describe_image = grok_describe_image
get_claude_search_response = get_grok_live_search_response


if __name__ == "__main__":
    dotenv.load_dotenv()
    print(
        get_grok_live_search_response(
            "Give me a short digest of today's top world news, with a link next to each claim."
        )
    )
