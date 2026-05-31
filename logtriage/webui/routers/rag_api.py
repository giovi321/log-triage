"""RAG service API routes: status, progress, doc-glob scan, reindex."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from ...llm_client import _call_llm
from ..auth import get_current_user
from ..live import get_rag_monitor_status
from ..state import STATE

router = APIRouter()


@router.get("/api/rag/status")
async def get_rag_status(request: Request):
    """Get RAG service status for AJAX calls (authenticated).

    Returns operator-facing infra detail (repo URLs, commit hashes, chunk counts)
    that the dashboard displays — this is intentional for a signed-in operator;
    the security fix is requiring authentication, since this was previously open.
    """
    if not get_current_user(request, STATE.settings):
        return JSONResponse({"error": "Unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED)
    monitor_data = get_rag_monitor_status()

    if monitor_data["detailed_status"] is None:
        monitor_data["detailed_status"] = {
            "initialization": {
                "started": False, "completed": False, "updating": False,
                "error": None, "current_phase": "unavailable",
                "progress": {
                    "current_step": 0, "total_steps": 5,
                    "step_description": "RAG service not available", "percentage": 0.0,
                },
                "repository_updates": {
                    "current_repo": None, "total_repos": 0,
                    "completed_repos": 0, "current_progress": 0.0,
                },
            }
        }

    if not monitor_data["rag_available"]:
        return {
            "enabled": False, "message": "RAG service unavailable",
            "service_available": False, "service_ready": False,
            "detailed_status": monitor_data["detailed_status"], "monitor": monitor_data,
        }
    if not monitor_data["rag_ready"]:
        return {
            "enabled": False, "message": "RAG service initializing",
            "service_available": True, "service_ready": False,
            "detailed_status": monitor_data["detailed_status"], "monitor": monitor_data,
        }

    rag_client = STATE.rag_client
    try:
        if rag_client:
            status_data = rag_client.get_status()
            vector_stats = status_data.get("vector_store_stats", {}) or {
                "total_chunks": 0, "persist_directory": "Service running but no data"
            }
            return {
                "enabled": status_data.get("enabled", False),
                "service_available": True, "service_ready": True,
                "total_repositories": status_data.get("total_repositories", 0),
                "vector_store_stats": vector_stats,
                "repositories": status_data.get("repositories", []),
                "monitor": monitor_data,
            }
        return {
            "enabled": False, "message": "RAG client not initialized",
            "service_available": True, "service_ready": False, "monitor": monitor_data,
        }
    except Exception as e:
        return {
            "enabled": False, "message": f"Error getting RAG status: {e}",
            "service_available": False, "service_ready": False, "monitor": monitor_data,
        }


@router.get("/api/rag/progress", name="rag_progress")
async def get_rag_progress(request: Request):
    if not get_current_user(request, STATE.settings):
        return JSONResponse({"error": "Unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED)
    monitor_data = get_rag_monitor_status()
    rag_client = STATE.rag_client
    if rag_client is None or not hasattr(rag_client, "_make_request"):
        return {
            "service_available": monitor_data.get("rag_available", False),
            "service_ready": monitor_data.get("rag_ready", False),
            "progress": None, "monitor": monitor_data,
        }
    try:
        progress = rag_client._make_request("GET", "/progress", max_retries=0)
        return {
            "service_available": monitor_data.get("rag_available", False),
            "service_ready": monitor_data.get("rag_ready", False),
            "progress": progress, "monitor": monitor_data,
        }
    except Exception as e:
        return {
            "service_available": False, "service_ready": False,
            "progress": None, "error": str(e), "monitor": monitor_data,
        }


# ---- doc-glob detection (for the scan-docs helper) ------------------------

def _looks_like_git_target(repo_url) -> bool:
    if not isinstance(repo_url, str):
        return False
    u = repo_url.strip()
    if not u:
        return False
    return u.startswith(("http://", "https://", "git@", "ssh://"))


def _detect_doc_globs(root: Path) -> List[str]:
    """Heuristic documentation glob detection over a cloned repo tree."""
    patterns: List[str] = []

    def _add(p: str) -> None:
        if p and p not in patterns:
            patterns.append(p)

    skip = {".git", "node_modules", "vendor", ".venv"}
    doc_dir_names = {"docs", "doc", "documentation", "wiki", "guide", "guides", "manual"}

    found_dirs: List[Path] = []
    try:
        for d in root.rglob("*"):
            if d.is_dir() and d.name.lower() in doc_dir_names:
                rel_parts = {part.lower() for part in d.relative_to(root).parts}
                if rel_parts & skip:
                    continue
                found_dirs.append(d)
    except OSError:
        pass
    found_dirs.sort(key=lambda p: len(p.relative_to(root).parts))
    for d in found_dirs[:6]:
        rel = d.relative_to(root).as_posix()
        _add(f"{rel}/**/*.md")
        _add(f"{rel}/**/*.rst")

    for mk in ("mkdocs.yml", "mkdocs.yaml"):
        mkpath = root / mk
        if mkpath.is_file():
            docs_dir = "docs"
            try:
                if yaml is not None:
                    mk_cfg = yaml.safe_load(mkpath.read_text(encoding="utf-8")) or {}
                    if isinstance(mk_cfg, dict) and isinstance(mk_cfg.get("docs_dir"), str):
                        docs_dir = mk_cfg["docs_dir"].strip("/") or "docs"
            except Exception:
                docs_dir = "docs"
            _add(f"{docs_dir}/**/*.md")
            break

    try:
        for conf in root.rglob("conf.py"):
            rel_parts = {part.lower() for part in conf.relative_to(root).parts}
            if rel_parts & skip:
                continue
            cdir = conf.parent.relative_to(root).as_posix()
            prefix = f"{cdir}/" if cdir not in ("", ".") else ""
            _add(f"{prefix}**/*.rst")
            _add(f"{prefix}**/*.md")
            break
    except OSError:
        pass

    try:
        if any(p.name.lower().startswith("readme") for p in root.iterdir() if p.is_file()):
            _add("README*")
    except OSError:
        pass

    try:
        root_md = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".md"]
        if len(root_md) >= 2:
            _add("*.md")
    except OSError:
        pass

    if not patterns:
        patterns = ["**/*.md", "**/*.rst", "**/*.txt"]
    return patterns[:12]


def _llm_refine_doc_globs(root: Path, heuristic: List[str]):
    """Best-effort LLM refinement of doc globs. Never raises; returns
    (patterns, used_llm)."""
    llm_defaults = STATE.llm_defaults
    try:
        if not getattr(llm_defaults, "enabled", False):
            return heuristic, False
        provider = llm_defaults.providers.get(llm_defaults.default_provider or "")
        if provider is None:
            return heuristic, False
        files: List[str] = []
        for p in root.rglob("*"):
            if len(files) >= 400:
                break
            if p.is_file():
                rel = p.relative_to(root).as_posix()
                if rel.startswith(".git/"):
                    continue
                files.append(rel)
        if not files:
            return heuristic, False
        prompt = (
            "You are given the file list of a git repository. Return a JSON array "
            "of glob patterns (relative to the repo root) that capture the "
            "human-readable documentation (markdown/rst/text guides, manuals, "
            "READMEs). Return ONLY a JSON array of strings, no prose.\n\n"
            "Files:\n" + "\n".join(files)
        )
        resp = _call_llm(provider, {
            "model": provider.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512, "temperature": 0.0,
        })
        content = ""
        if isinstance(resp, dict):
            content = resp.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        if not content:
            return heuristic, False
        st = content.find("[")
        en = content.rfind("]")
        if st == -1 or en == -1 or en <= st:
            return heuristic, False
        arr = json.loads(content[st: en + 1])
        llm_globs = [g.strip() for g in arr if isinstance(g, str) and g.strip()]
        if not llm_globs:
            return heuristic, False
        merged = list(heuristic)
        contributed = False
        for g in llm_globs:
            if g not in merged:
                merged.append(g)
                contributed = True
        return merged[:24], contributed
    except Exception:
        return heuristic, False


@router.post("/api/rag/scan-docs")
async def api_scan_docs(request: Request):
    """Shallow-clone a repo and detect documentation glob patterns."""
    if not get_current_user(request, STATE.settings):
        return JSONResponse({"error": "Unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED)

    try:
        body = await request.json()
    except Exception:
        body = {}
    repo_url = (body.get("repo_url") or "").strip()
    branch = (body.get("branch") or "main").strip() or "main"

    if not _looks_like_git_target(repo_url):
        return JSONResponse({
            "ok": False, "include_paths": [],
            "message": "Invalid repo URL (expected http(s)://, git@ or ssh://).",
            "used_llm": False,
        })

    tmpdir = tempfile.mkdtemp(prefix="logtriage-scandocs-")
    try:
        def _clone(with_branch: bool):
            cmd = ["git", "clone", "--depth", "1"]
            if with_branch:
                cmd += ["--branch", branch]
            cmd += [repo_url, tmpdir]
            return subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        try:
            proc = _clone(with_branch=True)
            if proc.returncode != 0:
                shutil.rmtree(tmpdir, ignore_errors=True)
                os.makedirs(tmpdir, exist_ok=True)
                proc = _clone(with_branch=False)
        except subprocess.TimeoutExpired:
            return JSONResponse({
                "ok": False, "include_paths": [],
                "message": "clone failed", "detail": "git clone timed out", "used_llm": False,
            })

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()[:500]
            return JSONResponse({
                "ok": False, "include_paths": [],
                "message": "clone failed", "detail": detail, "used_llm": False,
            })

        root = Path(tmpdir)
        heuristic = _detect_doc_globs(root)
        include_paths, used_llm = _llm_refine_doc_globs(root, heuristic)
        msg = f"Detected {len(include_paths)} path pattern(s)"
        if used_llm:
            msg += " (LLM-refined)"
        return JSONResponse({
            "ok": True, "include_paths": include_paths, "message": msg, "used_llm": used_llm,
        })
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@router.post("/api/rag/reindex/{repo_id}")
async def reindex_rag_repo(repo_id: str, request: Request):
    if not get_current_user(request, STATE.settings):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    monitor_data = get_rag_monitor_status()
    if not monitor_data.get("rag_available"):
        raise HTTPException(status_code=503, detail="RAG service unavailable")

    rag_client = STATE.rag_client
    if rag_client is None or not hasattr(rag_client, "_make_request"):
        raise HTTPException(status_code=503, detail="RAG client not available")

    try:
        result = rag_client._make_request("POST", f"/reindex/{repo_id}", json={"refresh": True}, max_retries=0)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not result:
        raise HTTPException(status_code=502, detail="Failed to start reindex")
    return result
