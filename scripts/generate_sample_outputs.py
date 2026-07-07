"""Generate sample_outputs.md from live chat validation."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings_or_raise
from app.services.rag_service import RAGService

QUESTIONS = [
    (
        "1. Rights of foreign domestic helpers",
        "What are the rights of foreign domestic helpers in Hong Kong?",
    ),
    (
        "2. Recruitment agency rules",
        "What are the rules for recruitment agencies?",
    ),
    (
        "3. Passport retention",
        "Can an employer keep a helper's passport?",
    ),
    (
        "4. Wage payment timing",
        "When should wages be paid to a domestic helper?",
    ),
    (
        "5. Work injury",
        "What happens if a helper is injured at work?",
    ),
    (
        "6. Part-time work for another employer",
        "Can a foreign domestic helper work part-time for another employer?",
    ),
    (
        "7. Employment agency charges",
        "What should an employment agency charge a job seeker?",
    ),
    (
        "8. Arabic – passport retention",
        "هل يمكن لصاحب العمل الاحتفاظ بجواز سفر العاملة؟",
    ),
    (
        "9. Tagalog – passport retention",
        "Pwede bang kunin ng employer ang passport ng helper?",
    ),
]

FALLBACK = (
    "I could not find enough information in the provided documents to answer this "
    "confidently. Please rephrase the question or contact the relevant Hong Kong "
    "authority for confirmation."
)


def format_response(question: str, payload: dict) -> str:
    """Format one question block for markdown."""
    lines = [
        f"**Question:** {question}",
        "",
        f"**Confidence:** {payload['confidence']}",
        "",
        "**Answer:**",
        "",
        payload["answer"],
        "",
    ]
    if payload["sources"]:
        lines.append("**Sources:**")
        lines.append("")
        for source in payload["sources"]:
            lines.append(
                f"- `{source['source_file']}` "
                f"(pages {source['page_start']}-{source['page_end']}, "
                f"score={source['score']})"
            )
        lines.append("")
    else:
        lines.append("**Sources:** None (fallback triggered)")
        lines.append("")
    lines.append("<details><summary>Raw JSON</summary>")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("</details>")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Run chat validation and write sample_outputs.md."""
    try:
        get_settings_or_raise()
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    rag = RAGService()
    rag.reload_vector_store()
    if not rag.vector_store.is_loaded:
        print(
            "Vector store is not loaded. Run `python scripts/ingest.py` first.",
            file=sys.stderr,
        )
        return 2

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = [
        "# Sample PoBot Outputs",
        "",
        f"Generated: {generated_at}",
        "",
        "Live validation outputs from the indexed Hong Kong labour document corpus.",
        "",
        "## Low-confidence fallback reference",
        "",
        f"```\n{FALLBACK}\n```",
        "",
    ]

    for title, question in QUESTIONS:
        response = rag.ask(question)
        payload = {
            "answer": response.answer,
            "confidence": response.confidence,
            "sources": [source.model_dump() for source in response.sources],
        }
        sections.append(f"## {title}")
        sections.append("")
        sections.append(format_response(question, payload))

    output_path = PROJECT_ROOT / "sample_outputs.md"
    output_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
