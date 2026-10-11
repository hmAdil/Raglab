import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATS_PATH = ROOT / "data" / "corpus_stats.json"
PROCESSED_DIR = ROOT / "data" / "processed"


def print_stats():
    if not STATS_PATH.exists():
        print("[ERROR] corpus_stats.json not found. Run: python src/collect_corpus.py first")
        return

    data = json.loads(STATS_PATH.read_text(encoding="utf-8"))

    print("\n" + "="*60)
    print("RAGLAB CORPUS STATISTICS")
    print("="*60)

    for s in data["sources"]:
        print(f"\n  Source:           {s['source'].upper()}")
        print(f"  Pages collected:  {s['num_pages']}")
        print(f"  Passages:         {s['num_passages']}")
        print(f"  Unique sections:  {s['num_sections']}")
        print(f"  Total words:      {s['total_words']:,}")
        print(f"  Total tokens:     {s['total_tokens']:,}")
        print(f"  Avg tokens/pass:  {s['avg_tokens_per_passage']}")

    c = data["combined"]
    print(f"\n  {'-'*40}")
    print(f"  COMBINED")
    print(f"  Total pages:      {c['total_pages']}")
    print(f"  Total passages:   {c['total_passages']}")
    print(f"  Total sections:   {c['total_sections']}")
    print(f"  Total words:      {c['total_words']:,}")
    print(f"  Total tokens:     {c['total_tokens']:,}")
    print("="*60)

    print("\nSECTION SAMPLES (first 5 per source):")
    for path in sorted(PROCESSED_DIR.glob("*.jsonl")):
        source = path.stem
        print(f"\n  [{source.upper()}]")
        seen = set()
        count = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line.strip())
                sec = row.get("section", "?")
                if sec not in seen:
                    print(f"    - {sec}")
                    seen.add(sec)
                    count += 1
                if count >= 5:
                    break


if __name__ == "__main__":
    print_stats()
