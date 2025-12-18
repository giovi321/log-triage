from __future__ import annotations

import datetime
import logging
import threading
from typing import Any, Callable, Dict, Optional


class RAGServiceMonitor:
    def __init__(
        self,
        *,
        status: Dict[str, Any],
        get_service_url: Callable[[], Optional[str]],
        create_client: Callable[[str], Any],
        get_client: Callable[[], Any],
        set_client: Callable[[Any], None],
        timestamp_mode: str = "unix",
        include_detailed_status: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._status = status
        self._get_service_url = get_service_url
        self._create_client = create_client
        self._get_client = get_client
        self._set_client = set_client
        self._timestamp_mode = timestamp_mode
        self._include_detailed_status = include_detailed_status
        self._logger = logger or logging.getLogger(__name__)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        t = None
        with self._lock:
            self._stop_event.set()
            t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=5)
        with self._lock:
            self._thread = None

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def get_ready_client(self) -> Any:
        with self._lock:
            if self._status.get("rag_ready"):
                return self._get_client()
            return None

    def _set_last_check(self) -> None:
        if self._timestamp_mode == "iso":
            self._status["last_check"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
        else:
            import time

            self._status["last_check"] = time.time()

    def _run(self) -> None:
        while True:
            interval = float(self._status.get("check_interval", 10) or 10)
            if self._stop_event.wait(interval):
                return

            try:
                service_url = self._get_service_url()
                if not service_url:
                    with self._lock:
                        self._set_client(None)
                        self._status["rag_available"] = False
                        self._status["rag_ready"] = False
                        if self._include_detailed_status:
                            self._status["detailed_status"] = None
                        self._set_last_check()
                    continue

                client = self._get_client()

                if client is None or not hasattr(client, "is_ready"):
                    try:
                        new_client = self._create_client(service_url)
                    except Exception as exc:
                        self._logger.debug(f"RAG service not yet available: {exc}")
                        with self._lock:
                            self._status["rag_available"] = False
                            self._status["rag_ready"] = False
                            if self._include_detailed_status:
                                self._status["detailed_status"] = None
                            self._set_last_check()
                        continue

                    try:
                        is_ready = bool(new_client and new_client.is_ready())
                        is_healthy = bool(
                            new_client
                            and (
                                new_client.is_healthy()
                                if hasattr(new_client, "is_healthy")
                                else is_ready
                            )
                        )
                    except Exception:
                        is_ready = False
                        is_healthy = False

                    with self._lock:
                        if is_ready or is_healthy:
                            self._set_client(new_client)
                        self._status["rag_available"] = bool(is_healthy)
                        self._status["rag_ready"] = bool(is_ready)

                        if (
                            self._include_detailed_status
                            and is_healthy
                            and hasattr(new_client, "_make_request")
                        ):
                            try:
                                detailed = new_client._make_request("GET", "/health")
                            except Exception:
                                detailed = None
                            if detailed:
                                self._status["detailed_status"] = detailed

                        self._set_last_check()
                    continue

                try:
                    is_ready = bool(client.is_ready())
                    is_healthy = bool(
                        client.is_healthy() if hasattr(client, "is_healthy") else is_ready
                    )

                    detailed = None
                    if self._include_detailed_status and hasattr(client, "_make_request"):
                        try:
                            detailed = client._make_request("GET", "/health")
                        except Exception:
                            detailed = None

                    with self._lock:
                        self._status["rag_available"] = is_healthy
                        self._status["rag_ready"] = is_ready
                        if self._include_detailed_status:
                            if detailed:
                                self._status["detailed_status"] = detailed
                        self._set_last_check()

                except Exception as exc:
                    self._logger.debug(f"RAG service check failed: {exc}")
                    with self._lock:
                        self._status["rag_available"] = False
                        self._status["rag_ready"] = False
                        self._set_client(None)
                        if self._include_detailed_status:
                            self._status["detailed_status"] = None
                        self._set_last_check()

            except Exception as exc:
                self._logger.error(f"RAG monitor worker error: {exc}")
                with self._lock:
                    self._status["rag_available"] = False
                    self._status["rag_ready"] = False
                    self._set_last_check()
