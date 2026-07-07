"""Command-line interface for PoBot."""

from __future__ import annotations

import argparse
import sys

from app.config import get_settings_or_raise
from app.services.ingestion import run_ingestion
from app.services.rag_service import RAGService
from app.utils.logging import logger


def _cmd_ingest() -> int:
    """Run document ingestion."""
    result = run_ingestion()
    print("Ingestion complete.")
    print(f"  Files processed : {result.files_processed}")
    print(f"  Pages processed : {result.pages_processed}")
    print(f"  Chunks created  : {result.chunks_created}")
    print(f"  Index path      : {result.index_path}")
    print(f"  Chunks path     : {result.chunks_path}")
    return 0


def _cmd_ask(question: str) -> int:
    """Ask a single question."""
    rag = RAGService()
    response = rag.ask(question)
    print(f"\nConfidence: {response.confidence:.3f}\n")
    print(response.answer)
    if response.sources:
        print("\nSources:")
        for source in response.sources:
            print(
                f"  - {source.source_file} "
                f"(pages {source.page_start}-{source.page_end}, score={source.score:.3f})"
            )
    return 0


def _cmd_chat() -> int:
    """Run interactive chat mode."""
    rag = RAGService()
    print("PoBot interactive chat. Type 'exit', 'quit', or 'q' to leave.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            return 0

        try:
            response = rag.ask(question)
            print(f"\nPoBot (confidence {response.confidence:.3f}):")
            print(response.answer)
            if response.sources:
                print("\nSources:")
                for source in response.sources:
                    print(
                        f"  - {source.source_file} "
                        f"(pages {source.page_start}-{source.page_end})"
                    )
            print()
        except Exception as exc:
            logger.error("Chat error: %s", exc)
            print(f"Error: {exc}\n")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PoBot – AI Assistant for Migrant Support",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Ingest documents and build the vector index")

    ask_parser = subparsers.add_parser("ask", help="Ask a single question")
    ask_parser.add_argument("question", type=str, help="Question to ask PoBot")

    subparsers.add_parser("chat", help="Start interactive chat mode")

    args = parser.parse_args(argv)

    try:
        get_settings_or_raise()
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    if args.command == "ingest":
        return _cmd_ingest()
    if args.command == "ask":
        return _cmd_ask(args.question)
    if args.command == "chat":
        return _cmd_chat()

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
