"""Gemini generation with system-control tool calling."""

import re
import time

from google import genai
from google.genai import types

from audit_log import log_tool_call
from system_control import SYSTEM_TOOLS, TOOL_EXECUTORS

GEMINI_MODEL = __import__("os").getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_RETRIES = 3


def _safe_response_text(response) -> str:
    if not response.candidates:
        return ""
    content = response.candidates[0].content
    if not content or not content.parts:
        return response.text or ""
    return response.text or ""


def _parse_retry_seconds(error: Exception) -> float:
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(error), re.I)
    if match:
        return float(match.group(1)) + 1
    return 20


def _is_quota_error(error: Exception) -> bool:
    text = str(error)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower()


def _call_gemini(client, contents, config):
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            )
        except Exception as e:
            last_error = e
            if _is_quota_error(e) and attempt < MAX_RETRIES - 1:
                wait = _parse_retry_seconds(e)
                print(f"[Gemini] Rate limit hit, retrying in {wait:.0f}s...")
                time.sleep(wait)
                continue
            raise
    raise last_error


def generate_with_tools(client: genai.Client, contents, system_instruction, max_rounds=5) -> str:
    """Call Gemini and execute any requested system tools before returning final text."""
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=SYSTEM_TOOLS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    working = list(contents)
    last_response = None

    for _ in range(max_rounds):
        last_response = _call_gemini(client, working, config)

        if not last_response.candidates:
            return "I didn't get a response. Please try again."

        candidate = last_response.candidates[0].content
        if not candidate or not candidate.parts:
            return _safe_response_text(last_response) or "I didn't get a response. Please try again."

        function_calls = [p.function_call for p in candidate.parts if p.function_call]

        if not function_calls:
            return _safe_response_text(last_response) or "I didn't get a response. Please try again."

        working.append(candidate)
        response_parts = []
        for part in candidate.parts:
            if not part.function_call:
                continue
            fc = part.function_call
            name = fc.name
            args = dict(fc.args) if fc.args else {}
            print(f"[System] {name}({args})")

            executor = TOOL_EXECUTORS.get(name)
            if executor:
                try:
                    result = executor(**args)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {name}"

            confirmed = args.get("confirmed") if isinstance(args, dict) else None
            if confirmed is not None and not isinstance(confirmed, bool):
                confirmed = bool(confirmed)
            log_tool_call(name, args, result, confirmed=confirmed)

            preview = result[:120] + ("..." if len(result) > 120 else "")
            print(f"[System] -> {preview}")
            response_parts.append(types.Part.from_function_response(
                name=name,
                response={"result": result},
            ))

        working.append(types.Content(role="user", parts=response_parts))

    return _safe_response_text(last_response) if last_response else "I couldn't complete that action."


def friendly_error(error: Exception) -> str:
    if _is_quota_error(error):
        wait = int(_parse_retry_seconds(error))
        return (
            f"My Gemini API free tier limit is reached. "
            f"Wait about {wait} seconds and try again, or upgrade at ai.google.dev. "
            f"Simple commands like open apps still work without the API."
        )
    return "I'm having trouble thinking right now. Please try again."
