"""LLM API routes: provider connectivity test and ad-hoc / per-finding queries."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from fastapi import APIRouter, Form, Request, status
from fastapi.responses import JSONResponse

from ...llm_client import _call_llm, analyze_findings_with_llm
from ...models import ModuleLLMConfig, Severity
from ...notifications import add_notification
from ..auth import get_current_user
from ..db import (
    get_finding_by_id,
    update_finding_llm_data,
    update_finding_llm_error,
)
from ..state import STATE
from ..shared import finding_excerpt_preview as _finding_excerpt_preview
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# Lightweight stdlib HTTP helpers for the provider-test / scan-docs endpoints.
# These mirror the transport style in logtriage/llm_client.py (urllib, ~10s
# timeouts) rather than pulling in httpx/requests, keeping behavior consistent
# with the rest of the LLM stack. JSON POST bodies make them CSRF-exempt (the
# csrf middleware only guards form/multipart content types), matching the
# existing POST /api/rag/reindex/{repo_id} route.
# ---------------------------------------------------------------------------

def _http_get(url: str, headers: Dict[str, str], timeout: float = 10.0):
    """GET a URL; return (status_code, body_text). HTTP error responses are
    returned as (code, body) rather than raised, so callers can branch on the
    status code for the common 401/404 cases."""
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:
            body = ""
        return exc.code, body


def _ollama_root(api_base: str) -> str:
    """Derive the Ollama server root from any base, stripping trailing /v1 and
    /api so the native /api/tags endpoint can be reached."""
    root = (api_base or "").rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")].rstrip("/")
    if root.endswith("/api"):
        root = root[: -len("/api")].rstrip("/")
    return root or "http://127.0.0.1:11434"


def _parse_model_ids(text: str) -> Optional[List[str]]:
    """Best-effort parse of an OpenAI/Anthropic/Ollama models listing."""
    try:
        data = json.loads(text)
    except Exception:
        return None
    items = None
    if isinstance(data, dict):
        items = data.get("data") or data.get("models")
    elif isinstance(data, list):
        items = data
    if not isinstance(items, list):
        return None
    out: List[str] = []
    for it in items:
        if isinstance(it, dict):
            mid = it.get("id") or it.get("name") or it.get("model")
            if isinstance(mid, str) and mid:
                out.append(mid)
        elif isinstance(it, str):
            out.append(it)
    return out or None


def _model_note(model: str, models: Optional[List[str]]) -> Optional[str]:
    if not model or not models:
        return None
    if model in models:
        return f"Model '{model}' is available."
    return f"Model '{model}' not listed ({len(models)} models returned)."


@router.post("/api/llm/test")
async def api_llm_test(request: Request):
    """Lightweight LLM provider auth/reachability check.

    Accepts a JSON body {name, provider_type, api_base, api_key, api_key_env,
    model}. Always returns HTTP 200 with {ok, message, detail, latency_ms,
    models}; logical failures are reported via ok=false. CSRF-exempt (JSON),
    auth required (mirrors /api/rag/reindex/{repo_id})."""
    username = get_current_user(request, STATE.settings)
    if not username:
        return JSONResponse({"error": "Unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED)

    try:
        body = await request.json()
    except Exception:
        body = {}

    name = (body.get("name") or "").strip()
    provider_type = (body.get("provider_type") or "openai").strip().lower()
    api_base = (body.get("api_base") or "").strip()
    api_key_env = (body.get("api_key_env") or "").strip()
    api_key_literal = (body.get("api_key") or "").strip()
    model = (body.get("model") or "").strip()

    # Resolve key: literal wins, else env var.
    api_key = api_key_literal or (os.environ.get(api_key_env) if api_key_env else None)

    state = {"latency_ms": None}
    t0 = time.monotonic()

    def _ok(message, detail=None, models=None):
        return JSONResponse({
            "ok": True, "message": message, "detail": detail,
            "latency_ms": state["latency_ms"], "models": models,
        })

    def _err(message, detail=None):
        return JSONResponse({
            "ok": False, "message": message, "detail": detail,
            "latency_ms": state["latency_ms"], "models": None,
        })

    def _generation_fallback():
        """Confirm credentials via a minimal 1-token generation through
        _call_llm, so transport matches production exactly. Used when the models
        listing endpoint is unsupported / returns a non-2xx, non-auth status."""
        try:
            from ..models import LLMProviderConfig
            prov = LLMProviderConfig(
                name=name or "test",
                api_base=(("https://api.anthropic.com" if provider_type == "anthropic" else api_base) or ""),
                api_key_env=api_key_env or None,
                api_key=api_key_literal or None,
                model=model or "",
                provider_type=provider_type,
                request_timeout=10.0,
                max_output_tokens=1,
                temperature=0.0,
            )
            payload = {
                "model": model or "",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "temperature": 0.0,
            }
            _call_llm(prov, payload)
            state["latency_ms"] = int((time.monotonic() - t0) * 1000)
            return _ok("Generation succeeded.")
        except Exception as exc:
            state["latency_ms"] = int((time.monotonic() - t0) * 1000)
            return _err("Provider test failed", f"{type(exc).__name__}: {exc}")

    try:
        if provider_type == "anthropic":
            # Address fixed; ignore submitted api_base.
            if not api_key:
                return _err("No API key", f"Set an API key or the {api_key_env or 'configured'} env var.")
            url = "https://api.anthropic.com/v1/models"
            headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
            code, text = _http_get(url, headers, timeout=10.0)
            state["latency_ms"] = int((time.monotonic() - t0) * 1000)
            if 200 <= code < 300:
                models = _parse_model_ids(text)
                return _ok("Reachable; credentials accepted.", _model_note(model, models), models)
            if code in (401, 403):
                return _err("Authentication failed", f"HTTP {code} from /v1/models")
            return _generation_fallback()
        elif provider_type == "ollama":
            root = _ollama_root(api_base) if api_base else "http://127.0.0.1:11434"
            url = f"{root}/api/tags"
            code, text = _http_get(url, {}, timeout=10.0)
            state["latency_ms"] = int((time.monotonic() - t0) * 1000)
            if 200 <= code < 300:
                models = _parse_model_ids(text)
                return _ok("Reachable.", _model_note(model, models), models)
            return _err("Ollama not reachable", f"HTTP {code} from /api/tags")
        else:
            # openai / openai-compatible
            if not api_base:
                return _err("No API base", "Set the API base URL (e.g. https://api.openai.com/v1).")
            if not api_key:
                return _err("No API key", f"Set an API key or the {api_key_env or 'configured'} env var.")
            url = f"{api_base.rstrip('/')}/models"
            headers = {"Authorization": f"Bearer {api_key}"}
            code, text = _http_get(url, headers, timeout=10.0)
            state["latency_ms"] = int((time.monotonic() - t0) * 1000)
            if 200 <= code < 300:
                models = _parse_model_ids(text)
                return _ok("Reachable; credentials accepted.", _model_note(model, models), models)
            if code in (401, 403):
                return _err("Authentication failed", f"HTTP {code} from /models")
            return _generation_fallback()
    except Exception as exc:
        state["latency_ms"] = int((time.monotonic() - t0) * 1000)
        return _err("Connection failed", f"{type(exc).__name__}: {exc}")




def _finding_excerpt_preview(finding, max_lines: int) -> str:
    excerpt_lines = (getattr(finding, "excerpt", "") or "").splitlines()
    if max_lines > 0:
        excerpt_lines = excerpt_lines[:max_lines]
    return "\n".join(excerpt_lines)


@router.post("/llm/query_finding", name="llm_query_finding")
async def llm_query_finding(
    request: Request,
    finding_id: str = Form(...),
    provider: str = Form(...),
):
    """Query LLM for a specific finding with RAG context."""
    username = get_current_user(request, STATE.settings)
    if not username:
        return JSONResponse(
            {"error": "Unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED
        )

    try:
        # Validate finding_id parameter
        try:
            finding_id_int = int(finding_id)
            if finding_id_int <= 0:
                return JSONResponse({"error": "Invalid finding ID"}, status_code=400)
        except (ValueError, TypeError):
            return JSONResponse({"error": "Invalid finding ID format"}, status_code=400)
        
        # Get the finding from database
        finding = get_finding_by_id(finding_id_int)
        if not finding:
            return JSONResponse({"error": "Finding not found"}, status_code=404)
        
        # Get module name from finding
        module_name = getattr(finding, "module_name", None)
        if not module_name:
            return JSONResponse({"error": "Module name not found for finding"}, status_code=400)
        
        # Get provider configuration
        provider_config = STATE.llm_defaults.providers.get(provider)
        if not provider_config:
            return JSONResponse({"error": f"Provider '{provider}' not found"}, status_code=400)
        
        # Create a temporary module config for LLM analysis
        temp_module_llm = ModuleLLMConfig(
            enabled=True,
            provider_name=provider,
            emit_llm_payloads_dir=None,
            min_severity=Severity.WARNING,  # Use lowest severity for manual requests
            max_excerpt_lines=provider_config.max_excerpt_lines,  # Use provider's setting
        )
        
        # Analyze finding with LLM (including RAG context)
        # Create a compatible Finding object from the FindingRecord
        from ..models import Finding
        compatible_finding = Finding(
            file_path=finding.file_path_obj,
            pipeline_name=finding.pipeline_name,
            finding_index=finding.finding_index,
            severity=finding.severity_enum,
            message=finding.message,
            line_start=finding.line_start,
            line_end=finding.line_end,
            rule_id=finding.rule_id,
            excerpt=finding.excerpt_as_list,
            needs_llm=True,
        )
        
        analyze_findings_with_llm(
            [compatible_finding], 
            llm_defaults, 
            temp_module_llm,
            rag_client=STATE.rag_client,
            module_name=module_name
        )
        
        # Update the database record with the LLM response
        if compatible_finding.llm_response:
            update_finding_llm_data(
                finding.id,
                provider=compatible_finding.llm_response.provider,
                model=compatible_finding.llm_response.model,
                content=compatible_finding.llm_response.content,
                prompt_tokens=compatible_finding.llm_response.prompt_tokens,
                completion_tokens=compatible_finding.llm_response.completion_tokens,
            )
            # Update the original finding object with the response for the return value
            finding.llm_response = compatible_finding.llm_response
        else:
            llm_error = getattr(compatible_finding, "llm_error", None)
            if llm_error:
                update_finding_llm_error(
                    finding.id,
                    error=str(llm_error),
                    provider=provider_config.name,
                    model=provider_config.model,
                )
                setattr(finding, "llm_error", str(llm_error))
        
        # Return the LLM response
        if finding.llm_response:
            response_data = {
                "provider": finding.llm_response.provider,
                "model": finding.llm_response.model,
                "content": finding.llm_response.content,
                "usage": {
                    "prompt_tokens": finding.llm_response.prompt_tokens,
                    "completion_tokens": finding.llm_response.completion_tokens,
                },
                "citations": finding.llm_response.citations,
            }
            return JSONResponse(response_data)
        llm_error = getattr(finding, "llm_error", None)
        if llm_error:
            return JSONResponse({"error": str(llm_error)}, status_code=502)
        else:
            return JSONResponse({"error": "No LLM response generated"}, status_code=500)
            
    except Exception as e:
        logger.error(f"LLM query failed for finding_id {finding_id}: {e}", exc_info=True)
        add_notification("error", "LLM query failed", f"Finding {finding_id}: {str(e)}")
        return JSONResponse({"error": f"LLM query failed: {str(e)}"}, status_code=500)


@router.post("/llm/query", name="llm_query")
async def llm_query(
    request: Request,
    provider: str = Form(...),
    prompt: str = Form(...),
):
    username = get_current_user(request, STATE.settings)
    if not username:
        return JSONResponse(
            {"error": "Unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED
        )

    provider_cfg = STATE.llm_defaults.providers.get(provider)
    if provider_cfg is None:
        return JSONResponse({"error": f"Unknown provider '{provider}'"}, status_code=400)

    prompt_text = (prompt or "").strip()
    if not prompt_text:
        return JSONResponse({"error": "Prompt cannot be empty."}, status_code=400)

    chat_payload = {
        "model": provider_cfg.model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You are a log triage assistant that summarizes log snippets succinctly and "
                    "suggests follow-up actions when appropriate. Respond to the following prompt:\n\n"
                    f"{prompt_text}"
                ),
            }
        ],
        "temperature": provider_cfg.temperature,
        "top_p": provider_cfg.top_p,
        "max_tokens": provider_cfg.max_output_tokens,
    }

    try:
        response_data = _call_llm(provider_cfg, chat_payload)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    message = response_data.get("choices", [{}])[0].get("message", {})
    usage = response_data.get("usage", {}) or {}
    content = message.get("content", "").strip()

    return JSONResponse(
        {
            "provider": provider_cfg.name,
            "model": response_data.get("model", provider_cfg.model),
            "content": content,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            },
        }
    )


