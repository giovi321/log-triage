import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

from .llm_payload import render_llm_payload
from .notifications import add_notification
from .models import (
    Finding,
    GlobalLLMConfig,
    LLMProviderConfig,
    LLMResponse,
    ModuleLLMConfig,
)

# Import RAG client (optional import to avoid circular dependencies)
try:
    from .rag import RAGClient
except ImportError:
    RAGClient = None


def resolve_provider(llm_defaults: GlobalLLMConfig, module_llm: ModuleLLMConfig) -> Optional[LLMProviderConfig]:
    if not llm_defaults.enabled or not module_llm.enabled:
        return None

    if module_llm.provider_name:
        provider = llm_defaults.providers.get(module_llm.provider_name)
        if provider is None:
            raise ValueError(
                f"Unknown LLM provider '{module_llm.provider_name}'. Define it under llm.providers."
            )
        return provider

    if llm_defaults.default_provider:
        provider = llm_defaults.providers.get(llm_defaults.default_provider)
        if provider is None:
            raise ValueError(
                f"Default LLM provider '{llm_defaults.default_provider}' was not found in llm.providers"
            )
        return provider

    if len(llm_defaults.providers) == 1:
        return next(iter(llm_defaults.providers.values()))

    return None


def _select_max_tokens(module_llm: ModuleLLMConfig, provider: LLMProviderConfig, llm_defaults: GlobalLLMConfig) -> int:
    if module_llm.max_output_tokens is not None:
        return module_llm.max_output_tokens
    if provider.max_output_tokens:
        return provider.max_output_tokens
    return 512


def _chat_completion_url(api_base: str) -> str:
    normalized_base = api_base.rstrip("/")
    if normalized_base.endswith("/v1"):
        return f"{normalized_base}/chat/completions"
    return f"{normalized_base}/v1/chat/completions"


def _normalize_messages_for_strict_alternation(messages: List[dict]) -> List[dict]:
    system_parts: List[str] = []
    normalized: List[dict] = []

    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip().lower()
        content = m.get("content")
        if content is None:
            continue
        if not isinstance(content, str):
            content = str(content)
        if not content.strip():
            continue

        if role == "system":
            system_parts.append(content)
            continue

        if role not in ("user", "assistant"):
            role = "user"

        if normalized and normalized[-1].get("role") == role:
            normalized[-1]["content"] = f"{normalized[-1].get('content', '').rstrip()}\n\n{content.lstrip()}"
            continue

        normalized.append({"role": role, "content": content})

    if system_parts:
        system_block = "\n\n".join(system_parts)
        if not normalized:
            normalized = [{"role": "user", "content": system_block}]
        elif normalized[0].get("role") != "user":
            normalized.insert(0, {"role": "user", "content": system_block})
        else:
            normalized[0]["content"] = f"{system_block}\n\n{normalized[0].get('content', '')}"

    if normalized and normalized[0].get("role") == "assistant":
        normalized[0]["role"] = "user"

    if not normalized:
        return []

    out: List[dict] = []
    for m in normalized:
        if out and out[-1].get("role") == m.get("role"):
            out[-1]["content"] = f"{out[-1].get('content', '').rstrip()}\n\n{m.get('content', '').lstrip()}"
        else:
            out.append(m)

    return out


def _call_chat_completion(provider: LLMProviderConfig, payload: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    if provider.api_key_env:
        api_key = os.environ.get(provider.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {provider.api_key_env} is required to call provider {provider.name}"
            )
        headers["Authorization"] = f"Bearer {api_key}"

    url = _chat_completion_url(provider.api_base)
    data = json.dumps(payload).encode("utf-8")
    if provider.organization:
        headers["OpenAI-Organization"] = provider.organization

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    def _do_request(req_obj: urllib.request.Request) -> dict:
        with urllib.request.urlopen(req_obj, timeout=provider.request_timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)

    try:
        return _do_request(req)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if exc.fp else exc.reason
        if exc.code == 400 and isinstance(payload, dict):
            detail_text = detail if isinstance(detail, str) else str(detail)
            if "Conversation roles must alternate" in detail_text and "messages" in payload:
                messages = payload.get("messages")
                if isinstance(messages, list):
                    normalized = _normalize_messages_for_strict_alternation(messages)
                    if normalized and normalized != messages:
                        retry_payload = dict(payload)
                        retry_payload["messages"] = normalized
                        retry_data = json.dumps(retry_payload).encode("utf-8")
                        retry_req = urllib.request.Request(url, data=retry_data, headers=headers, method="POST")
                        try:
                            return _do_request(retry_req)
                        except Exception:
                            pass

                    roles = [m.get("role") for m in messages if isinstance(m, dict)]
                    raise RuntimeError(
                        f"LLM provider {provider.name} HTTP {exc.code}: {detail} (roles={roles})"
                    )

        raise RuntimeError(f"LLM provider {provider.name} HTTP {exc.code}: {detail}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach LLM provider {provider.name}: {exc.reason}")


def analyze_findings_with_llm(
    findings: List[Finding], 
    llm_defaults: GlobalLLMConfig, 
    module_llm: ModuleLLMConfig,
    rag_client: Optional["RAGClient"] = None,
    module_name: str = None
) -> None:
    provider = resolve_provider(llm_defaults, module_llm)
    if provider is None:
        return

    for f in findings:
        if not f.needs_llm:
            continue

        # Retrieve RAG context if available
        rag_context = ""
        citations = []
        
        if rag_client and module_name and RAGClient:
            retrieval_result = rag_client.retrieve_for_finding(f, module_name)
            if retrieval_result and retrieval_result.chunks:
                rag_context = "\n\n--- Relevant Documentation ---\n"
                for i, chunk in enumerate(retrieval_result.chunks, 1):
                    rag_context += f"\n{i}. {chunk.heading}\n{chunk.content}\n"
                    citations.append(f"[{i}] {chunk.heading} ({Path(chunk.file_path).name})")
        
        payload_text = render_llm_payload(f, module_llm, rag_context=rag_context)
        if not payload_text:
            continue

        max_tokens = _select_max_tokens(module_llm, provider, llm_defaults)

        # Add citation instruction to prompt
        system_message = (
            "You are a log triage assistant that summarizes log snippets succinctly "
            "and suggests follow-up actions when appropriate. Use the provided documentation "
            "to ground your analysis. Cite relevant documentation using the reference numbers "
            f"in brackets.{' Include citations in your response.' if citations else ''}"
        )

        # For VLLM providers, avoid system messages as some models don't support them properly
        # Instead, include the system instruction as part of the user message
        if "vllm" in provider.api_base.lower():
            messages = [
                {
                    "role": "user",
                    "content": f"{system_message}\n\n{payload_text}",
                }
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user", 
                    "content": payload_text,
                }
            ]

        chat_payload = {
            "model": provider.model,
            "messages": messages,
            "temperature": provider.temperature,
            "top_p": provider.top_p,
            "max_tokens": max_tokens,
        }

        try:
            response_data = _call_chat_completion(provider, chat_payload)
        except Exception as exc:
            add_notification(
                "error",
                "LLM call failed",
                f"{provider.name}: {exc}",
            )
            continue

        message = response_data.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "")
        usage = response_data.get("usage", {}) or {}
        f.llm_response = LLMResponse(
            provider=provider.name,
            model=response_data.get("model", provider.model),
            content=content,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            citations=citations if citations else None,
        )

        if module_llm.emit_llm_payloads_dir:
            module_llm.emit_llm_payloads_dir.mkdir(parents=True, exist_ok=True)
            out_path = module_llm.emit_llm_payloads_dir / (
                f"{f.pipeline_name}_{f.severity.name}_finding{f.finding_index}_response.json"
            )
            with out_path.open("w", encoding="utf-8") as resp_file:
                json.dump(
                    {
                        "provider": f.llm_response.provider,
                        "model": f.llm_response.model,
                        "content": f.llm_response.content,
                        "prompt_tokens": f.llm_response.prompt_tokens,
                        "completion_tokens": f.llm_response.completion_tokens,
                    },
                    resp_file,
                    indent=2,
                )
