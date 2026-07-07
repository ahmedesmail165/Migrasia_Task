"""Source-balance sanity check for hybrid retrieval."""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "source_balance_results.json"

QUESTIONS = [
    "What are the rules for recruitment agencies?",
    "What should an employment agency charge a job seeker?",
    "Can an employer keep a helper's passport?",
    "What are the rights of foreign domestic helpers in Hong Kong?",
    "When should wages be paid to a domestic helper?",
    "What leave is a domestic helper entitled to?",
    "What happens if a helper is injured at work?",
    "Can a foreign domestic helper work part-time for another employer?",
    "Can a foreign domestic helper live outside the employer's home?",
]

BASE_URL = "http://127.0.0.1:8000/chat"


def main() -> int:
    results = []
    for question in QUESTIONS:
        body = json.dumps({"question": question, "top_k": 10}).encode()
        req = urllib.request.Request(
            BASE_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            payload = json.loads(urllib.request.urlopen(req, timeout=180).read().decode())
        except Exception as exc:
            print(f"ERROR: {question[:60]} -> {exc}", file=sys.stderr)
            return 1

        sources = [s["source_file"] for s in payload.get("sources", [])]
        counts = dict(Counter(sources))
        cop_count = counts.get("CoP_Eng.pdf", 0)
        results.append(
            {
                "question": question,
                "confidence": payload.get("confidence"),
                "answer_summary": (payload.get("answer") or "")[:300],
                "source_counts": counts,
                "top_sources": sources[:5],
                "cop_count": cop_count,
                "cop_share": round(cop_count / max(len(sources), 1), 2),
            }
        )
        print(f"Q: {question}")
        print(f"  confidence={payload.get('confidence')} cop={cop_count}/{len(sources)}")
        print(f"  sources={counts}")
        print(f"  answer={(payload.get('answer') or '')[:160]}...")
        print()
        time.sleep(0.5)

    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
