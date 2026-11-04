"""
Check whether PDFs contain extractable text before adding them to the corpus.

A scanned PDF looks normal in a viewer but yields no text to PyMuPDF, so it
contributes nothing to a retrieval index. media/textbooks/1_CAT26.pdf is 400
pages of exactly this: every page extracts empty. Run this on candidate files
first.

Usage
-----
    venv/Scripts/python.exe tools/check_pdfs.py <file-or-folder> [...]

    # vet a download folder before copying anything in
    venv/Scripts/python.exe tools/check_pdfs.py ~/Downloads/textbooks

    # check what is already in the corpus folder
    venv/Scripts/python.exe tools/check_pdfs.py media/textbooks

Verdicts
--------
    GOOD      plenty of text on nearly every page -- add it
    PARTIAL   some pages empty; usable but will contribute less than it looks
    SCANNED   no extractable text -- needs OCR, or skip it
    DUPLICATE byte-identical to a file already seen in this run
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# A page needs at least this many words to count as having real text.
MIN_WORDS_PER_PAGE = 20
# Fraction of pages with text required for a GOOD verdict.
GOOD_TEXT_PAGE_RATIO = 0.80
# Estimated passages = words / stride, matching ir.corpus defaults
# (chunk_size=400, overlap=80 -> stride 320).
PASSAGE_STRIDE = 320


def collect_pdfs(targets: list[str]) -> list[Path]:
    """Expand files and folders into a sorted list of PDF paths."""
    found: list[Path] = []
    for target in targets:
        path = Path(target).expanduser()
        if path.is_dir():
            found.extend(sorted(path.rglob('*.pdf')))
        elif path.is_file() and path.suffix.lower() == '.pdf':
            found.append(path)
        else:
            print(f"  ! not a PDF or folder: {target}", file=sys.stderr)
    return found


def inspect(path: Path) -> dict:
    """Extract per-page word counts and return a verdict for one PDF."""
    import fitz

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        return {'path': path, 'error': str(exc), 'verdict': 'ERROR'}

    try:
        n_pages = len(doc)
        words_per_page = [
            len(doc[i].get_text("text").split()) for i in range(n_pages)
        ]
    finally:
        doc.close()

    total_words = sum(words_per_page)
    text_pages = sum(1 for w in words_per_page if w >= MIN_WORDS_PER_PAGE)
    ratio = text_pages / n_pages if n_pages else 0.0

    if total_words == 0:
        verdict = 'SCANNED'
    elif ratio >= GOOD_TEXT_PAGE_RATIO:
        verdict = 'GOOD'
    else:
        verdict = 'PARTIAL'

    return {
        'path': path,
        'n_pages': n_pages,
        'text_pages': text_pages,
        'ratio': ratio,
        'total_words': total_words,
        'words_per_text_page': total_words // max(text_pages, 1),
        'est_passages': total_words // PASSAGE_STRIDE,
        'verdict': verdict,
    }


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1

    pdfs = collect_pdfs(argv)
    if not pdfs:
        print("No PDFs found.")
        return 1

    print(f"Checking {len(pdfs)} PDF(s)\n")
    header = (f"{'file':<40} {'verdict':<10} {'pages':>6} {'w/text pg':>10} "
              f"{'words':>9} {'~passages':>10}")
    print(header)
    print("-" * len(header))

    seen: dict[str, str] = {}
    usable_passages = 0
    counts: dict[str, int] = {}

    for path in pdfs:
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen:
            counts['DUPLICATE'] = counts.get('DUPLICATE', 0) + 1
            print(f"{path.name[:39]:<40} {'DUPLICATE':<10} "
                  f"{'':>6} {'':>10} {'':>9} {'':>10}   "
                  f"same bytes as {seen[digest]}")
            continue
        seen[digest] = path.name

        info = inspect(path)
        verdict = info['verdict']
        counts[verdict] = counts.get(verdict, 0) + 1

        if verdict == 'ERROR':
            print(f"{path.name[:39]:<40} {'ERROR':<10}   {info['error'][:40]}")
            continue

        if verdict in ('GOOD', 'PARTIAL'):
            usable_passages += info['est_passages']

        print(f"{path.name[:39]:<40} {verdict:<10} {info['n_pages']:>6} "
              f"{info['words_per_text_page']:>10} {info['total_words']:>9} "
              f"{info['est_passages']:>10}")

    print("-" * len(header))
    summary = "  ".join(f"{v}={n}" for v, n in sorted(counts.items()))
    print(f"\n{summary}")
    print(f"estimated usable passages: {usable_passages}")

    # Guidance against the target collection size. Below roughly a thousand
    # passages, top-10 is too large a slice of the collection for retrieval
    # methods to separate: every system looks equally good.
    if usable_passages < 1000:
        print(f"\n  Too small. At {usable_passages} passages, a top-10 result "
              f"is {1000 / max(usable_passages, 1):.0f}x too large a share of "
              f"the collection\n  for systems to be distinguished. Aim for "
              f"3000+ passages (~1M words, ~2500 text pages).")
    elif usable_passages < 3000:
        print(f"\n  Workable but small ({usable_passages} passages). "
              f"Differences between systems will be\n  noisy and significance "
              f"tests will rarely fire. More documents would help.")
    else:
        print(f"\n  Good size ({usable_passages} passages) for a retrieval "
              f"evaluation.")

    if counts.get('SCANNED'):
        print(f"\n  {counts['SCANNED']} scanned file(s) contribute nothing "
              f"without OCR. Either replace them with\n  text-based editions, "
              f"or exclude them -- ir.corpus reports them as empty either way.")

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
