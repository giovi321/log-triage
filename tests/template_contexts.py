"""Route-faithful render contexts for every Web UI page template.

Single source of truth for both:
  * tests/test_template_render.py — the template smoke test, and
  * _preview_render.py — the (gitignored) visual-QA scaffolding,

so the two never drift. ``build_pages()`` returns a list of
``(template_name, out_name, context)`` tuples whose context dicts mirror what
the real routes pass. Keep these in sync with the routers when page context
changes — the coverage guard in the smoke test fails if a new page template has
no entry here.
"""
from __future__ import annotations

import datetime
import json
import types

_NS = lambda **k: types.SimpleNamespace(**k)


def build_pages() -> list:
    """Return [(template_name, out_name, context), ...] for every page."""
    pages: list = []

    def page(template, **ctx):
        pages.append((template, template, ctx))

    now = datetime.datetime(2026, 5, 29, 14, 30, 0)
    day = datetime.timedelta(days=1)

    # ---- login ------------------------------------------------------------
    page("login.html", username=None, error=None,
         oidc_enabled=True, oidc_exclusive=False, _path="/login")

    # ---- shared module + status fixtures ----------------------------------
    modules = [
        _NS(name="homeassistant", mode="follow", path="/var/log/fluent-bit/homeassistant.log",
            pipeline_name="homeassistant", enabled=True),
        _NS(name="nextcloud", mode="follow", path="/var/log/fluent-bit/nextcloud.log",
            pipeline_name="nextcloud", enabled=True),
        _NS(name="authentik", mode="follow", path="/var/log/fluent-bit/authentik.log",
            pipeline_name="authentik", enabled=True),
        _NS(name="apache2_access", mode="follow", path="/var/log/apache2/vhosts",
            pipeline_name="apache2_access", enabled=False),
        _NS(name="rsnapshot", mode="batch", path="/var/log/rsnapshot.log",
            pipeline_name="rsnapshot", enabled=True),
    ]
    stats = {
        "homeassistant": _NS(last_severity="ERROR", last_log_update=now, errors_24h=12, warnings_24h=44),
        "nextcloud": _NS(last_severity="WARNING", last_log_update=now, errors_24h=0, warnings_24h=7),
        "authentik": _NS(last_severity="CRITICAL", last_log_update=now, errors_24h=3, warnings_24h=1),
        "rsnapshot": _NS(last_severity="OK", last_log_update=now, errors_24h=0, warnings_24h=0),
    }
    ingestion_status = _NS(stale_modules=["nextcloud"], state_class="warn", latest_log_update=now)
    db_status = {"configured": True, "connected": True, "error": None, "url": "sqlite:///./logtriage.db"}
    rag_status = {
        "enabled": True, "total_repositories": 4,
        "vector_store_stats": {"total_chunks": 18422, "persist_directory": "./rag_vector_store"},
        "repositories": [
            _NS(repo_id="r1", url="https://github.com/home-assistant/home-assistant.io",
                branch="current", last_commit_hash="a1b2c3d4e5f6", last_commit_at=now,
                last_indexed_hash="a1b2c3d4e5f6", last_indexed_at=now, chunk_count=12044,
                needs_reindexing=False),
            _NS(repo_id="r2", url="https://github.com/nextcloud/documentation",
                branch="master", last_commit_hash="99887766aabb", last_commit_at=now,
                last_indexed_hash="11223344ccdd", last_indexed_at=now, chunk_count=6378,
                needs_reindexing=True),
        ],
    }

    # ---- dashboard --------------------------------------------------------
    page("dashboard.html",
         username="admin", modules=modules, stats=stats, db_status=db_status,
         page_rendered_at=now, ingestion_status=ingestion_status,
         rag_status=rag_status, rag_service_available=True, rag_service_ready=True,
         rag_monitor={}, _path="/")

    # ---- account ----------------------------------------------------------
    preview_users = [
        _NS(username="admin", created_at=now, is_admin=True),
        _NS(username="ops", created_at=now, is_admin=True),
    ]
    page("account.html", username="admin", users=preview_users,
         is_admin=True, oidc_enabled=True, db_status=db_status, error=None, message=None,
         _path="/account")

    # ---- config editor ----------------------------------------------------
    sample_cfg = (
        "llm:\n  enabled: true\n  default_provider: claude\n  providers:\n"
        "    claude:\n      api_base: https://api.anthropic.com/v1\n"
        "      api_key_env: ANTHROPIC_API_KEY\n      model: claude-sonnet-4-6\n\n"
        "database:\n  url: sqlite:///./logtriage.db\n  retention_days: 30\n"
    )
    sample_cfg_obj = {
        "defaults": {"llm_enabled": True, "max_excerpt_lines": 20},
        "llm": {
            "enabled": True, "default_provider": "claude",
            "context_prefix_lines": 2, "context_suffix_lines": 2,
            "max_excerpt_lines": 200, "max_output_tokens": 512, "request_timeout": 30,
            "temperature": 0.0, "top_p": 1.0,
            "providers": {
                "claude": {"api_base": "https://api.anthropic.com/v1", "api_key_env": "ANTHROPIC_API_KEY",
                           "model": "claude-sonnet-4-6", "provider_type": "anthropic",
                           "max_output_tokens": 1024, "temperature": 0.2},
                "local_vllm": {"api_base": "http://127.0.0.1:8000/v1", "api_key_env": None,
                               "model": "Mistral-7B-Instruct", "provider_type": "openai"},
            },
        },
        "rag": {"enabled": True, "service_url": "http://127.0.0.1:8091", "cache_dir": "./rag_cache",
                "vector_store_dir": "./rag_vector_store",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu", "batch_size": 32, "top_k": 5, "similarity_threshold": 0.7, "max_chunks": 10},
        "pipelines": [
            {"name": "homeassistant", "match": {"filename_regex": "homeassistant.*\\.log"},
             "classifier": {"type": "regex_counter", "error_regexes": ["\\berror\\b", "Traceback"],
                            "warning_regexes": ["\\bwarning\\b"], "ignore_regexes": []},
             "grouping": {"type": "whole_file"}},
            {"name": "rsnapshot", "match": {"filename_regex": "rsnapshot.*\\.log"},
             "classifier": {"type": "rsnapshot_basic", "error_regexes": ["ERROR:"],
                            "warning_regexes": ["WARNING"], "ignore_regexes": []},
             "grouping": {"type": "separator", "separator_regex": "^rsnapshot ", "only_last": True}},
        ],
        "modules": [
            {"name": "homeassistant", "enabled": True, "path": "/var/log/fluent-bit/homeassistant.log",
             "mode": "follow", "pipeline": "homeassistant", "output_format": "text",
             "min_print_severity": "WARNING", "stale_after_minutes": 60,
             "llm": {"enabled": True, "min_severity": "WARNING", "provider": "claude",
                     "prompt_template": "./prompts/homeassistant.txt",
                     "context_prefix_lines": 2, "context_suffix_lines": 2},
             "alerts": {"webhook": {"enabled": False, "method": "POST", "url": "", "min_severity": "ERROR"},
                        "mqtt": {"enabled": False, "host": "localhost", "port": 1883,
                                 "topic": "logtriage/ha", "min_severity": "ERROR"}},
             "stream": {"from_beginning": False, "interval": 1.0},
             "rag": {"enabled": True, "knowledge_sources": [
                 {"repo_url": "https://github.com/home-assistant/home-assistant.io", "branch": "current",
                  "include_paths": ["source/**/*.md"]}]}},
        ],
        "database": {"url": "sqlite:///./logtriage.db", "retention_days": 30},
        "webui": {"enabled": True, "host": "192.168.1.10", "port": 8090, "base_path": "/",
                  "secret_key": "CHANGE_ME", "session_cookie_name": "logtriage_session",
                  "dark_mode_default": True, "csrf_enabled": True, "session_max_age_hours": 24,
                  "allowed_ips": ["127.0.0.1", "192.168.1.1"], "trusted_proxies": [],
                  "admin_users": [{"username": "admin", "password_hash": "$2b$12$abc..."}]},
        "logging": {"level": "INFO", "format": "%(asctime)s %(levelname)s %(name)s: %(message)s",
                    "file": "./logtriage.log",
                    "loggers": {"logtriage.rag": "DEBUG", "logtriage.stream": "WARNING"}},
    }
    page("config_edit.html",
         username="admin", config_text=sample_cfg, config_json=json.dumps(sample_cfg_obj),
         context_hints={"root": "Top-level sections mirror the README. "
                                "Move the cursor to a section to see details."},
         error=None, message=None, _path="/config/edit")

    # ---- regex lab --------------------------------------------------------
    sample_lines = [
        _NS(index=101, full="2026-05-29 14:31:02 INFO  homeassistant.setup: Setting up zwave_js"),
        _NS(index=102, full="2026-05-29 14:31:03 WARNING homeassistant.helpers: Entity sensor.foo is taking over 10s"),
        _NS(index=103, full="2026-05-29 14:31:05 ERROR homeassistant.components.mqtt: Unable to connect to MQTT broker"),
        _NS(index=104, full="2026-05-29 14:31:05 ERROR homeassistant.bootstrap: Traceback (most recent call last):"),
        _NS(index=105, full="2026-05-29 14:31:06 INFO  homeassistant.core: Bus:Handling <Event homeassistant_start>"),
        _NS(index=106, full="2026-05-29 14:31:09 ERROR homeassistant.components.http: Login attempt failed from 10.0.0.4"),
    ]
    page("regex.html",
         username="admin", modules=modules, current_module=modules[0],
         sample_lines=sample_lines,
         recent_findings=[_NS(line_start=103, severity="ERROR"),
                          _NS(line_start=104, severity="ERROR"),
                          _NS(line_start=106, severity="WARNING")],
         matches=[103, 104, 106], regex_kind="error", regex_value=r"\bERROR\b|Traceback",
         regex_issues=None, sample_source="tail",
         ingestion_status=_NS(message="active · last line 4s ago"), _path="/regex")

    # ---- regex generator (LLM from raw log lines) -------------------------
    gen_result = _NS(
        candidates=[
            _NS(pattern=r"Unable to connect to MQTT broker \S+", rationale="MQTT broker unreachable",
                valid=True, error=None, match_count=14, new_matches=14, over_match=0, safe=True,
                examples=["2026-05-29 14:31:05 ERROR mqtt: Unable to connect to MQTT broker 10.0.0.5:1883"]),
            _NS(pattern=r"Traceback", rationale="Python traceback", valid=True, error=None,
                match_count=27, new_matches=5, over_match=0, safe=True,
                examples=["2026-05-29 14:31:05 ERROR bootstrap: Traceback (most recent call last):"]),
            _NS(pattern=r"[unclosed", rationale="malformed", valid=False,
                error="missing ], unterminated subpattern", match_count=0, new_matches=0,
                over_match=0, safe=False, examples=[]),
        ],
        lines_sampled=1000, families=42, families_omitted=0,
        provider="ollama-local", model="qwen2.5", error=None,
        raw_response="REGEX: Unable to connect to MQTT broker \\S+ # broker unreachable\nREGEX: Traceback",
    )
    page("regex_generate.html",
         username="admin", modules=modules, current_module=modules[0], regex_kind="error",
         provider_options=[{"name": "ollama-local", "model": "qwen2.5", "type": "ollama"},
                           {"name": "claude", "model": "claude-sonnet-4-6", "type": "anthropic"}],
         selected_provider="ollama-local", sample_size=1000, sample_sizes=[500, 1000, 2000],
         valid_kinds=["ignore", "error", "warning"],
         result=gen_result, error=None, _path="/regex/generate")

    # ---- triage (issues + detail) -----------------------------------------
    issue1 = _NS(
        id=1, severity="CRITICAL", priority_score=132.0,
        title="Unable to connect to MQTT broker <IP>:<N>",
        module_name="homeassistant", signature="ERROR mqtt: Unable to connect to MQTT broker <IP>:<N>",
        fingerprint="a1b2c3d4e5f6a7b8", llm_analyzed_fingerprint="a1b2c3d4e5f6a7b8",
        llm_content="Home Assistant cannot reach the MQTT broker, so all MQTT-backed entities are "
                    "unavailable. Verify the broker is running and reachable, then check host/port/"
                    "password. [1]",
        citations=["[1] MQTT integration setup (mqtt.markdown)"], has_llm_analysis=True,
        occurrence_count=432, last_seen=now, first_seen=now - 3 * day, status="open",
        pipeline_name="homeassistant", rule_id=r"\berror\b",
        llm_error=None, llm_provider="claude", llm_model="claude-sonnet-4-6",
        llm_updated_at=now, llm_prompt_tokens=1840, llm_completion_tokens=96, llm_category="network",
        sample_excerpt="2026-05-29 14:31:05 ERROR homeassistant.components.mqtt: "
                       "Unable to connect to MQTT broker 10.0.0.5:1883",
    )
    issue2 = _NS(
        id=2, severity="ERROR", priority_score=78.0, title="Traceback (most recent call last):",
        module_name="homeassistant", signature="ERROR Traceback (most recent call last):",
        fingerprint="ff0011223344", llm_analyzed_fingerprint=None, llm_content=None, citations=[],
        has_llm_analysis=False, occurrence_count=27, last_seen=now, first_seen=now - 1 * day,
        status="open", pipeline_name="homeassistant", rule_id="Traceback", llm_error=None,
        llm_category=None,
        sample_excerpt="2026-05-29 14:31:05 ERROR homeassistant.bootstrap: Traceback (most recent call last):",
    )
    issue3 = _NS(
        id=3, severity="WARNING", priority_score=31.0, title="Login attempt failed from <IP>",
        module_name="authentik", signature="WARNING Login attempt failed from <IP>",
        fingerprint="99aabbcc", llm_analyzed_fingerprint="99aabbcc",
        llm_content="Repeated failed logins from a single source — likely a misconfigured client or a "
                    "brute-force attempt.",
        citations=[], has_llm_analysis=True, occurrence_count=8, last_seen=now, first_seen=now - 2 * day,
        status="acknowledged", pipeline_name="authentik", rule_id="Login failed",
        llm_error=None, llm_provider="claude", llm_model="claude-sonnet-4-6", llm_updated_at=now,
        llm_prompt_tokens=420, llm_completion_tokens=58, llm_category="other: brute force",
        sample_excerpt="2026-05-29 14:31:06 WARNING authentik: Login attempt failed from 203.0.113.9",
    )
    sparklines = {
        1: [0, 0, 1, 0, 2, 3, 5, 4, 8, 12, 9, 6, 4, 7, 11, 18, 22, 14, 9, 6, 3, 5, 8, 10],
        2: [0, 0, 0, 0, 1, 0, 0, 2, 1, 0, 0, 3, 1, 0, 2, 0, 1, 4, 0, 1, 0, 0, 2, 1],
        3: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 3, 0, 0, 1, 0, 0, 0],
    }
    page("issues.html",
         username="admin", issues=[issue1, issue2, issue3], grouped=None, sparklines=sparklines,
         counts={"open": 5, "acknowledged": 2, "resolved": 11, "muted": 1, "false_positive": 3},
         category_counts={"network": 1, "other": 1, "uncategorized": 1},
         category_filter="", group_by_category=False,
         modules=modules, current_module="", status_filter="active", severity_filter="", search="",
         severity_choices=["CRITICAL", "ERROR", "WARNING"],
         status_choices=["open", "acknowledged", "resolved", "muted", "false_positive"],
         db_status=db_status, message=None, error=None, _path="/issues")

    occurrences = [
        _NS(created_at=now, severity="ERROR", line_start=1043,
            message='Matched error pattern /\\berror\\b/ on "ERROR mqtt: Unable to connect"'),
        _NS(created_at=now, severity="ERROR", line_start=1041,
            message='Matched error pattern /\\berror\\b/ on "ERROR mqtt: Unable to connect"'),
    ]
    page("issue_detail.html",
         username="admin", issue=issue1,
         spark_24h=sparklines[1], spark_14d=[3, 9, 14, 22, 18, 11, 7, 9, 12, 16, 21, 19, 8, 12],
         occurrences=occurrences,
         status_choices=["open", "acknowledged", "resolved", "muted", "false_positive"],
         db_status=db_status, can_analyze=True,
         suggested_ignore=r"Unable to connect to MQTT broker \d+\.\d+\.\d+\.\d+:\d+",
         _path="/issues/1")

    return pages
