"""JSONL event log + span context manager."""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class EventLog:
    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex[:8]

    def emit(self, event: str, **fields: Any) -> None:
        row = {"ts": time.time(), "run_id": self.run_id, "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    @contextmanager
    def span(self, name: str, **fields: Any) -> Iterator[dict[str, Any]]:
        span_id = uuid.uuid4().hex[:8]
        t0 = time.perf_counter()
        self.emit("span.start", span=name, span_id=span_id, **fields)
        ctx: dict[str, Any] = {"span_id": span_id}
        try:
            yield ctx
        except Exception as e:
            self.emit(
                "span.error",
                span=name,
                span_id=span_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
        finally:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            self.emit(
                "span.end",
                span=name,
                span_id=span_id,
                elapsed_ms=elapsed_ms,
                **{k: v for k, v in ctx.items() if k != "span_id"},
            )

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
