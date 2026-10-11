import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from loaders import collect_source, save_processed, compute_stats, CORPUS_SOURCES

all_stats = []
all_passages = []

for source_name in CORPUS_SOURCES:
    print(f"\n{'='*50}")
    print(f"Collecting: {source_name}")
    print(f"{'='*50}")
    passages = collect_source(source_name)
    save_processed(passages, CORPUS_SOURCES[source_name]["processed_file"])
    stats = compute_stats(passages, source_name)
    all_stats.append(stats)
    all_passages.extend(passages)

combined = {
    "sources": all_stats,
    "combined": {
        "total_pages": sum(s["num_pages"] for s in all_stats),
        "total_passages": sum(s["num_passages"] for s in all_stats),
        "total_sections": sum(s["num_sections"] for s in all_stats),
        "total_words": sum(s["total_words"] for s in all_stats),
        "total_tokens": sum(s["total_tokens"] for s in all_stats),
    },
}

stats_path = ROOT / "data" / "corpus_stats.json"
stats_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

print("\n" + "="*50)
print("CORPUS STATS SUMMARY")
print("="*50)
for s in all_stats:
    print(f"\n[{s['source']}]")
    print(f"  Pages:     {s['num_pages']}")
    print(f"  Passages:  {s['num_passages']}")
    print(f"  Sections:  {s['num_sections']}")
    print(f"  Words:     {s['total_words']:,}")
    print(f"  Tokens:    {s['total_tokens']:,}")
    print(f"  Avg tok/passage: {s['avg_tokens_per_passage']}")

print(f"\n[COMBINED]")
for k, v in combined["combined"].items():
    print(f"  {k}: {v:,}")

print(f"\nStats saved -> {stats_path}")
