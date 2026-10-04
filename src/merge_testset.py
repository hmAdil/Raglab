import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "data" / "questions"
OUTPUT = ROOT / "data" / "testset.jsonl"


def main():
    files = sorted(QUESTIONS.glob("*.jsonl"))
    rows = []
    for path in files:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line:
                continue
            json.loads(line)
            rows.append(line)
    OUTPUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"{len(rows)} questions from {len(files)} files")


if __name__ == "__main__":
    main()