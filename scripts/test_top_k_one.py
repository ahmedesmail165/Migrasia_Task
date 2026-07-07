"""Quick live check for passport question with top_k=1."""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.rag_service import RAGService

QUESTION = "Can an employer keep a helper's passport?"


def main() -> int:
    print("=== RAGService.ask(top_k=1) ===")
    rag = RAGService()
    response = rag.ask(QUESTION, top_k=1)
    print(json.dumps(
        {
            "answer": response.answer,
            "confidence": response.confidence,
            "sources": [s.model_dump() for s in response.sources],
        },
        indent=2,
    ))

    try:
        body = json.dumps({"question": QUESTION, "top_k": 1}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:8000/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        api_payload = json.loads(urllib.request.urlopen(req, timeout=180).read().decode())
        print("\n=== POST /chat top_k=1 ===")
        print(json.dumps(api_payload, indent=2))
    except Exception as exc:
        print(f"\n(API skipped: {exc})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
