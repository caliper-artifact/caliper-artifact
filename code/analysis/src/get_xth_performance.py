
import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ScoreItem = Tuple[int, str]          # (task_score, paraphrase text)
PerStyle  = Dict[str, List[ScoreItem]]


def load_scores(path: Path, *, min_count: int) -> PerStyle:
    """
    Returns a mapping  instruct_type → list[(task_score, paraphrase_string)]
    – keeps only paraphrases with paraphrase_content_score 4 or 5
    – keeps only instruct_types with ≥ min_count such paraphrases
    """
    with path.open("r", encoding="utf‑8") as f:
        data = json.load(f)

    per_style: PerStyle = {}
    for prompt in data:                               # each original prompt
        for para in prompt.get("paraphrases", []):    # its paraphrases
            if para.get("paraphrase_content_score") not in (4, 5):
                continue
            style = para["instruct_type"]
            per_style.setdefault(style, []).append(
                (para["task_score"], para["paraphrase"])
            )

    return {s: items for s, items in per_style.items() if len(items) >= min_count}


def summarise(per_style: PerStyle):
    """Convert to list[(avg, median, std, min, max, style, 2 examples)]"""
    records = []
    for style, items in per_style.items():
        scores = [score for score, _ in items]
        avg = st.mean(scores)
        med = st.median(scores)
        std = st.stdev(scores) if len(scores) > 1 else 0.0
        mn, mx = min(scores), max(scores)
        examples = [p for _, p in items[:2]]
        records.append((avg, med, std, mn, mx, style, examples))

    records.sort(key=lambda r: r[0], reverse=True)
    return records


def header(title: str):
    line = "=" * len(title)
    print(f"{line}\n{title}\n{line}")


def print_chunk(records, start_idx: int, count: int = 5):
    for rank, rec in enumerate(records[start_idx:start_idx + count], start=1):
        avg, med, std, mn, mx, style, examples = rec
        print(f"{rank:2d}. {style}")
        print(f"    avg={avg:.3f}, med={med}, std={std:.3f},"
              f" min={mn}, max={mx}")
        for ex in examples:
            print(f"      » {ex}")
        print()


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyse paraphrase‑style robustness buckets.")
    p.add_argument("json_file", metavar="FILE",
                   help="Path to the data JSON")
    p.add_argument("-m", "--min-count", type=int, default=50,
                   help="Minimum number of qualifying paraphrases (default 50)")
    return p.parse_args()


def main():
    args = parse_args()
    path = Path(args.json_file).expanduser()
    if not path.exists():
        sys.exit(f"File not found: {path}")

    per_style = load_scores(path, min_count=args.min_count)
    if not per_style:
        sys.exit("No styles met the filtering criteria.")

    records = summarise(per_style)
    total_styles = len(records)

    header("GLOBAL TOP 5 PARAPHRASE STYLES")
    print_chunk(records, 0)

    for pct in range(10, 100, 10):
        cut = math.floor(total_styles * pct / 100)
        header(f"TOP 5 AFTER REMOVING TOP {pct}% (cut index {cut})")
        if cut >= total_styles:
            print("No styles remain after this cut.\n")
            continue
        print_chunk(records, cut)

    header("GLOBAL WORST 5 PARAPHRASE STYLES")
    worst_slice = sorted(records, key=lambda r: r[0])[:5]
    for rank, rec in enumerate(worst_slice, start=1):
        avg, med, std, mn, mx, style, examples = rec
        print(f"{rank:2d}. {style}")
        print(f"    avg={avg:.3f}, med={med}, std={std:.3f},"
              f" min={mn}, max={mx}")
        for ex in examples:
            print(f"      » {ex}")
        print()


if __name__ == "__main__":
    main()
