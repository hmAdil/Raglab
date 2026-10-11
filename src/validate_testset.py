import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTSET_PATH = ROOT / "data" / "testset.jsonl"
PROCESSED_DIR = ROOT / "data" / "processed"

REQUIRED_FIELDS = ["id", "source_member", "question", "reference_answer",
                   "gold_evidence", "answerable", "type"]
VALID_TYPES = {"factual", "procedural", "conceptual", "unanswerable"}
VALID_SOURCES = {"fastapi", "pytorch"}

GOLD_EVIDENCE_FIELDS = ["source", "section", "span"]


def load_corpus_text() -> dict[str, str]:
    corpus = {}
    for path in PROCESSED_DIR.glob("*.jsonl"):
        source = path.stem
        passages = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                passages.append(json.loads(line)["text"])
        corpus[source] = "\n".join(passages)
    return corpus


def validate(testset_path: Path = TESTSET_PATH) -> bool:
    if not testset_path.exists():
        print(f"[ERROR] Testset not found: {testset_path}")
        print("Run: python src/merge_testset.py first")
        return False

    print(f"Validating: {testset_path}")
    corpus = load_corpus_text()
    if not corpus:
        print("[WARN] No processed corpus found in data/processed/. Span checks skipped.")

    rows = []
    errors = []
    ids_seen = set()

    for i, line in enumerate(testset_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {i}: invalid JSON — {e}")
            continue
        rows.append((i, row))

    for i, row in rows:
        prefix = f"Line {i} (id={row.get('id', '?')})"

        for field in REQUIRED_FIELDS:
            if field not in row:
                errors.append(f"{prefix}: missing field '{field}'")

        qid = row.get("id", "")
        if qid in ids_seen:
            errors.append(f"{prefix}: duplicate id '{qid}'")
        ids_seen.add(qid)

        if row.get("type") not in VALID_TYPES:
            errors.append(f"{prefix}: invalid type '{row.get('type')}'. Must be one of {VALID_TYPES}")

        ge = row.get("gold_evidence", {})
        for gf in GOLD_EVIDENCE_FIELDS:
            if gf not in ge:
                errors.append(f"{prefix}: gold_evidence missing field '{gf}'")

        answerable = row.get("answerable", True)
        span = ge.get("span")
        source = ge.get("source")

        if answerable:
            if not span:
                errors.append(f"{prefix}: answerable=true but gold_evidence.span is null/empty")
            elif corpus:
                if source not in corpus:
                    errors.append(f"{prefix}: source '{source}' not found in corpus")
                elif span not in corpus[source]:
                    errors.append(
                        f"{prefix}: gold span NOT FOUND verbatim in corpus '{source}'\n"
                        f"         Span: {span[:120]!r}..."
                    )
        else:
            if span is not None:
                errors.append(f"{prefix}: answerable=false but gold_evidence.span is not null")

    print(f"\nTotal questions: {len(rows)}")

    if errors:
        print(f"\n[FAIL] {len(errors)} error(s) found:\n")
        for e in errors:
            print(f"  [X] {e}")
        return False
    else:
        print("[PASS] All checks passed.")
        answerable_count = sum(1 for _, r in rows if r.get("answerable"))
        by_type = {}
        for _, r in rows:
            t = r.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
        print(f"  Answerable:   {answerable_count}/{len(rows)}")
        print(f"  Unanswerable: {len(rows) - answerable_count}/{len(rows)}")
        print(f"  By type: {by_type}")
        return True


if __name__ == "__main__":
    ok = validate()
    sys.exit(0 if ok else 1)
