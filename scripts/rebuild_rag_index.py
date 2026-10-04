"""Перестраивает локальный RAG-индекс с текущими параметрами последнего запуска."""

from __future__ import annotations

import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from document_index import RagIndexService


def main() -> int:
    service = RagIndexService(ROOT / "rag_documents", ROOT / "data" / "rag_index.sqlite3")
    latest = service.state().get("latest_run") or {}
    settings = latest.get("settings") or {
        "fixed_chunk_size": 350,
        "fixed_overlap": 50,
        "structural_max_size": 350,
        "structural_overlap": 50,
    }
    settings = {
        key: settings[key]
        for key in ("fixed_chunk_size", "fixed_overlap", "structural_max_size", "structural_overlap")
    }
    job = service.start_build(settings)
    print(f"job={job['id']}", flush=True)
    while service.state()["job"]["running"]:
        time.sleep(0.2)
    state = service.state()
    print(state["job"], flush=True)
    print(state["latest_run"], flush=True)
    return 0 if state["job"]["phase"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
