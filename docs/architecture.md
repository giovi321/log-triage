# Technical architecture (how it works)

This page describes how `log-triage` works end-to-end and what each major file/module is responsible for.

The codebase is organized around three entry points:

- `logtriage` (CLI)
- `logtriage-webui` (Web UI)
- `logtriage-rag` (RAG service)

## High-level data flow

### Batch analysis (CLI)

1. **Load config** (`logtriage/config.py`)
2. **Build pipelines/modules** (`logtriage/config.py`)
3. **Analyze a file or directory** (`logtriage/engine.py`)
4. **Group log lines** (`logtriage/grouping/*`)
5. **Classify grouped chunks** (`logtriage/classifiers/*`) to produce `Finding` objects
6. **Optionally call an LLM** (`logtriage/llm_client.py`) using a payload rendered by `logtriage/llm_payload.py`
7. **Optionally store findings** (database helpers in `logtriage/webui/db.py`)
8. **Optionally send alerts** (`logtriage/alerts.py` / `logtriage/notifications.py`)

### Follow/tail analysis (CLI)

1. Follow a file like `tail -F` with rotation handling (`logtriage/stream.py`)
2. For each batch of appended lines:
   - classify
   - optionally enrich with LLM + RAG
   - optionally store/send

### Web UI

- The Web UI is a FastAPI app (`logtriage/webui/app.py`) that:
  - reads the same YAML configuration
  - shows findings (in-memory or via database)
  - can edit `config.yaml`
  - can test/save regexes
  - can trigger LLM calls
  - can optionally integrate with the RAG service

### RAG (Retrieval-Augmented Generation)

- The RAG service (`logtriage/rag/service.py`) builds and serves a documentation index.
- The CLI/Web UI optionally call the service via `logtriage/rag/service_client.py`.
- When available, the client retrieves documentation snippets relevant to a finding and appends them to the LLM prompt (`logtriage/llm_client.py`, `logtriage/llm_payload.py`).

## Key concepts

- **Pipeline**: a reusable set of grouping + classifier rules applied to matching log files.
- **Module**: binds a pipeline to an on-disk path and a runtime mode (`batch` or `follow`).
- **Finding**: a structured result representing a problem detected in logs.

## Repository structure / role of each file

### Packaging and entry points

- `setup.py`
  - Packaging metadata and extras (`webui`, `alerts`, `rag`).
  - Defines console scripts:
    - `logtriage=logtriage.cli:main`
    - `logtriage-webui=logtriage.webui.__main__:main`
    - `logtriage-rag=logtriage.rag.service:main`
- `logtriage/__init__.py`
  - Exposes `main` and `__version__`.
- `logtriage/__main__.py`
  - Allows `python -m logtriage` to run the CLI.
- `logtriage/cli.py`
  - CLI implementation:
    - argument parsing and command dispatch
    - module execution in batch/follow modes
    - optional config reload (`--reload-on-change`)
    - optional DB initialization and retention cleanup
    - starts a background RAG monitor to detect service readiness
- `logtriage/version.py`
  - Stores the package version string.

### Core domain model

- `logtriage/models.py`
  - Dataclasses and enums used across the application:
    - `Severity`, `Finding`
    - pipeline/module/LLM configuration models
    - RAG configuration models and retrieval result types

### Configuration loading and validation

- `logtriage/config.py`
  - Loads `config.yaml`.
  - Compiles regexes and builds:
    - `PipelineConfig` list (`build_pipelines`)
    - `ModuleConfig` list (`build_modules`)
    - LLM config (`build_llm_config`)
    - RAG config (`build_rag_config`, `build_module_rag_config`)

### Log analysis engine

- `logtriage/engine.py`
  - Batch analysis:
    - `analyze_file`: read file, group lines, classify groups
    - `analyze_path`: apply the appropriate pipeline(s) across a file or directory
- `logtriage/utils.py`
  - Utility helpers:
    - enumerate log files (`iter_log_files`)
    - pick a pipeline for a file (`select_pipeline`)

### Streaming / follow mode

- `logtriage/stream.py`
  - Implements `tail -F`-like follow mode with rotation detection.
  - For each appended batch:
    - classify
    - optional LLM
    - optional persistence/alerts

### Grouping strategies (`logtriage/grouping/`)

- `logtriage/grouping/__init__.py`
  - Exposes grouping dispatch.
- `logtriage/grouping/marker.py`
  - Marker-based grouping (start/end regex delimit chunks).
- `logtriage/grouping/separator.py`
  - Separator-based grouping (split on a regex separator line; supports `only_last`).
- `logtriage/grouping/whole_file.py`
  - Whole-file grouping (treats the entire input as one chunk).

### Classifiers (`logtriage/classifiers/`)

- `logtriage/classifiers/__init__.py`
  - Exposes classifier dispatch.
- `logtriage/classifiers/regex_counter.py`
  - Regex-based classification:
    - apply ignore patterns first
    - emit a `Finding` for each match
- `logtriage/classifiers/*` (other files in the folder)
  - Additional classifier strategies (for example heuristics tailored to specific log formats).

### LLM integration

- `logtriage/llm_payload.py`
  - Renders a plain-text prompt payload for the LLM.
  - Appends RAG context when provided.
- `logtriage/llm_client.py`
  - Selects provider config and calls chat-completions style APIs.
  - Handles provider auth via environment variables.
  - Optionally retrieves RAG context and adds citations.

### Alerts, notifications, and logging

- `logtriage/alerts.py`
  - Dispatches outbound alerts (webhook and/or MQTT) based on severity thresholds.
- `logtriage/notifications.py`
  - In-process notification collection used by the Web UI and RAG service endpoints.
- `logtriage/logging_setup.py`
  - Logging configuration from config.

### Web UI (`logtriage/webui/`)

- `logtriage/webui/__main__.py`
  - Web UI entry point; loads config for logging and starts uvicorn.
- `logtriage/webui/app.py`
  - Main FastAPI application:
    - routing for dashboard, log explorer, config editor, regex tools
    - session middleware setup
    - optional background RAG monitor
- `logtriage/webui/auth.py`
  - Password verification and session-token helpers.
- `logtriage/webui/config.py`
  - Web UI-specific settings parsing (`webui.*` section of config).
- `logtriage/webui/db.py`
  - SQLAlchemy models + persistence helpers for findings and LLM results.
  - Provides retention cleanup and statistics queries.
- `logtriage/webui/ingestion_status.py`
  - Derives module “stale/active” status for the dashboard.
- `logtriage/webui/regex_utils.py`
  - Regex validation and sample preparation helpers for the regex lab.
- `logtriage/webui/assets/*` and `logtriage/webui/templates/*`
  - Static assets and server-rendered HTML templates.

### RAG implementation (`logtriage/rag/`)

- `logtriage/rag/service.py`
  - Standalone FastAPI service:
    - initializes the index in the background
    - exposes endpoints for health, status, repository progress and retrieval
- `logtriage/rag/service_client.py`
  - HTTP client used by CLI/Web UI to talk to the RAG service.
  - Provides a `NoOpRAGClient` fallback.
- `logtriage/rag/monitor.py`
  - Background thread helper used by CLI/Web UI to monitor whether the RAG service is up and ready.
- `logtriage/rag/rag_client.py`
  - In-process coordinator (used by the RAG service):
    - knowledge management
    - document processing
    - embeddings
    - vector store
    - retrieval
- `logtriage/rag/knowledge_manager.py`
  - Clones/updates Git repositories and enumerates documentation files.
- `logtriage/rag/document_processor.py`
  - Splits documentation files into chunks (by headings/paragraphs) with memory cleanup.
- `logtriage/rag/embeddings.py`
  - Wraps embedding generation (SentenceTransformers) and batching.
- `logtriage/rag/subprocess_embeddings.py`
  - Alternate embedding approach using subprocess isolation (when used).
- `logtriage/rag/vector_store.py`
  - Persistent FAISS index + SQLite metadata store.
- `logtriage/rag/retrieval.py`
  - Builds a query from a `Finding`, embeds it, queries vector store, filters by similarity.
- `logtriage/rag/__init__.py`
  - Exposes public RAG symbols for import by the rest of the package.

## Where to look for specific behavior

- **How findings are created**: `logtriage/classifiers/*` and `logtriage/engine.py`
- **How grouping works**: `logtriage/grouping/*`
- **How follow-mode works**: `logtriage/stream.py`
- **How LLM calls happen**: `logtriage/llm_client.py`
- **How RAG is appended to prompts**: `logtriage/llm_client.py` + `logtriage/llm_payload.py`
- **How the docs index is built**: `logtriage/rag/rag_client.py` + `logtriage/rag/service.py`
