"""Command line entry point:  python -m rag <command>"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import settings


def cmd_ingest(args) -> int:
    from .chunking import split_documents
    from .loaders import load_dir, load_path
    from .store import Store

    target = Path(args.path or settings.data_dir)
    print(f"Reading {target} ...")
    docs = load_dir(target) if target.is_dir() else load_path(target)
    if not docs:
        print("No readable documents found.", file=sys.stderr)
        return 1

    by_how: dict[str, int] = {}
    for d in docs:
        by_how[d.how] = by_how.get(d.how, 0) + 1
    for how, n in sorted(by_how.items()):
        print(f"  {how:18} {n} document(s)")

    chunks = split_documents(docs)
    print(f"\nEmbedding {len(chunks)} chunks with {settings.embed_model} ...")
    store = Store.build(chunks)
    where = store.save(args.out)
    print(f"Saved index to {where}  ({len(store)} chunks, {len(store.sources)} sources)")
    return 0


def cmd_ask(args) -> int:
    from .pipeline import ask
    from .store import Store

    store = Store.load(args.index)
    answer = ask(" ".join(args.question), store, args.k)

    print(answer.text)
    print()
    if answer.sources:
        print("Sources:")
        for s in answer.sources:
            print(f"  - {s}")
    if answer.provider:
        print(f"\n({answer.provider})")
    for note in answer.notes:
        print(f"note: {note}", file=sys.stderr)
    return 0 if answer.grounded else 2


def cmd_eval(args) -> int:
    from .evaluate import format_report, run
    from .store import Store

    store = Store.load(args.index)
    results, summary = run(store, args.set, answers=not args.retrieval_only, k=args.k)
    print(format_report(results, summary))

    floor = args.min_hit_rate
    if floor is not None and summary["hit_rate"] < floor:
        print(f"\nhit@k {summary['hit_rate']:.0%} is below the {floor:.0%} floor", file=sys.stderr)
        return 1
    return 0


def cmd_info(args) -> int:
    from .llm import available_providers
    from .loaders import find_tesseract
    from .store import Store

    print(f"embed model   {settings.embed_model}")
    print(f"index dir     {settings.index_dir}")
    print(f"hybrid        {'on' if settings.use_hybrid else 'off'}  (top_k={settings.top_k})")
    print(f"tesseract     {find_tesseract() or 'not installed'}")

    providers = available_providers()
    print(f"providers     {', '.join(p.name for p in providers) or 'NONE - check .env'}")

    if Store.exists():
        store = Store.load()
        print(f"index         {len(store)} chunks from {len(store.sources)} source(s)")
        for s in store.sources:
            print(f"  - {s}")
    else:
        print("index         none yet - run: python -m rag ingest")
    return 0


def cmd_serve(args) -> int:
    import subprocess

    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(Path(__file__).parent.parent / "app.py")]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description="RAG pipeline toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="read documents and build the index")
    p.add_argument("path", nargs="?", help="file or directory (default: data/)")
    p.add_argument("--out", help="index directory (default: storage/)")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("question", nargs="+")
    p.add_argument("--index")
    p.add_argument("-k", type=int)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="score the pipeline against eval/questions.json")
    p.add_argument("--set", help="path to a question set")
    p.add_argument("--index")
    p.add_argument("-k", type=int)
    p.add_argument("--retrieval-only", action="store_true", help="skip LLM calls")
    p.add_argument("--min-hit-rate", type=float, help="exit 1 below this (for CI)")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("info", help="show configuration and index status")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("serve", help="launch the Streamlit app")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
