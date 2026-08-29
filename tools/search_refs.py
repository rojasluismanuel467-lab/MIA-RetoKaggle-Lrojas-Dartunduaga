#!/usr/bin/env python3
"""Buscador full-text sobre `references/` — soporta .ipynb, .py, .md y .pdf.

Uso:
    python tools/search_refs.py "bounding box regression"
    python tools/search_refs.py -i "albumentations" --ext ipynb
    python tools/search_refs.py "transfer learning" --repo handson-ml3 -C 2

Muestra `ruta:linea: fragmento` para copiar/leer rápido.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent / "references"
DEFAULT_EXTS = ("ipynb", "py", "md", "pdf")


def iter_notebook_lines(path: Path) -> Iterable[tuple[int, str]]:
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return
    global_line = 0
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        for local, line in enumerate(src.splitlines(), start=1):
            global_line += 1
            yield global_line, line


def iter_text_lines(path: Path) -> Iterable[tuple[int, str]]:
    try:
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            yield i, line
    except Exception:
        return


def iter_pdf_lines(path: Path) -> Iterable[tuple[int, str]]:
    try:
        out = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True, text=True, timeout=60,
        )
        for i, line in enumerate(out.stdout.splitlines(), 1):
            yield i, line
    except Exception:
        return


def iter_lines(path: Path) -> Iterable[tuple[int, str]]:
    suf = path.suffix.lower()
    if suf == ".ipynb":
        yield from iter_notebook_lines(path)
    elif suf == ".pdf":
        yield from iter_pdf_lines(path)
    else:
        yield from iter_text_lines(path)


def main() -> int:
    p = argparse.ArgumentParser(description="Search across the reference books.")
    p.add_argument("query", help="regex to search (Python re)")
    p.add_argument("-i", "--ignore-case", action="store_true")
    p.add_argument("--ext", nargs="+", default=list(DEFAULT_EXTS),
                   help=f"extensions to include (default: {' '.join(DEFAULT_EXTS)})")
    p.add_argument("--repo", nargs="+", default=None,
                   help="restrict to one or more repo folder names")
    p.add_argument("-C", "--context", type=int, default=0,
                   help="context lines around each hit")
    p.add_argument("--max-per-file", type=int, default=5,
                   help="max hits shown per file (default 5)")
    args = p.parse_args()

    flags = re.IGNORECASE if args.ignore_case else 0
    try:
        pattern = re.compile(args.query, flags)
    except re.error as e:
        print(f"regex error: {e}", file=sys.stderr)
        return 2

    if not ROOT.is_dir():
        print(f"references/ not found at {ROOT}", file=sys.stderr)
        return 1

    repos = [ROOT / r for r in args.repo] if args.repo else list(ROOT.iterdir())
    exts = {"." + e.lstrip(".").lower() for e in args.ext}

    total_hits = 0
    for repo in repos:
        if not repo.is_dir():
            continue
        for path in sorted(repo.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in exts:
                continue
            hits_in_file = 0
            buffer = []
            for ln, line in iter_lines(path):
                buffer.append((ln, line))
                if len(buffer) > 2 * args.context + 1:
                    buffer.pop(0)
                if pattern.search(line):
                    if hits_in_file >= args.max_per_file:
                        continue
                    rel = path.relative_to(ROOT.parent)
                    header = f"\n\x1b[36m{rel}:{ln}\x1b[0m"
                    print(header)
                    for cln, cline in buffer:
                        marker = ">" if cln == ln else " "
                        print(f"  {marker} {cln:>5}: {cline.rstrip()}")
                    hits_in_file += 1
                    total_hits += 1

    print(f"\n== {total_hits} hits ==", file=sys.stderr)
    return 0 if total_hits else 1


if __name__ == "__main__":
    raise SystemExit(main())
