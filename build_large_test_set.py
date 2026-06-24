# coding: utf-8
"""Build a larger clean test set for final evaluation.

The set is independent from the refined training set:
- Most samples are drawn from ChiFraud_t2023.
- A small held-out portion comes from Doubao-validated DeepSeek dialogues.
- Crawler weak-label data is intentionally excluded from the final test set.
"""

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def clean_text(text):
    text = str(text).replace("\ufeff", "")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def render_dialogue(items):
    rendered = []
    for speaker, utterance in items:
        role = "对方" if str(speaker) == "1" else "我方"
        utterance = clean_text(utterance)
        if utterance:
            rendered.append(f"{role}: {utterance}")
    return " ".join(rendered)


def summarize(rows):
    labels = Counter(row["label"] for row in rows)
    sources = Counter(row.get("source", "unknown") for row in rows)
    return {
        "total": len(rows),
        "normal": labels.get(0, 0),
        "fraud": labels.get(1, 0),
        "sources": dict(sorted(sources.items())),
    }


def stratified_split(rows, train_ratio, dev_ratio, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    rng = random.Random(seed)
    splits = {"train": [], "dev": [], "test": []}
    for label_rows in grouped.values():
        label_rows = list(label_rows)
        rng.shuffle(label_rows)
        total = len(label_rows)
        train_end = int(total * train_ratio)
        dev_end = train_end + int(total * dev_ratio)
        splits["train"].extend(label_rows[:train_end])
        splits["dev"].extend(label_rows[train_end:dev_end])
        splits["test"].extend(label_rows[dev_end:])
    for name in splits:
        rng.shuffle(splits[name])
    return splits


def load_validated(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = render_dialogue(obj["text"])
            if text:
                rows.append({
                    "label": int(obj["label"]),
                    "text": text,
                    "source": "validated_deepseek_doubao",
                    "source_id": f"validated:{idx}",
                })
    return rows


def load_chifraud(path):
    rows = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader, start=1):
            if "Label_id" not in row or "Text" not in row:
                continue
            raw_label = int(row["Label_id"])
            text = clean_text(row["Text"])
            if text:
                rows.append({
                    "label": 0 if raw_label == 0 else 1,
                    "raw_label": raw_label,
                    "text": text,
                    "source": path.stem,
                    "source_id": f"{path.stem}:{idx}",
                })
    return rows


def sample_chifraud(rows, target_total, fraud_ratio, seed):
    rng = random.Random(seed)
    normal_rows = [row for row in rows if row["label"] == 0]
    fraud_groups = defaultdict(list)
    for row in rows:
        if row["label"] == 1:
            fraud_groups[row.get("raw_label", 1)].append(row)

    fraud_target = int(round(target_total * fraud_ratio))
    normal_target = target_total - fraud_target
    sampled = []
    plan = {"normal": normal_target, "fraud_total": fraud_target, "fraud_by_raw_label": {}}

    fraud_labels = sorted(fraud_groups)
    base = fraud_target // len(fraud_labels)
    rem = fraud_target % len(fraud_labels)
    for idx, raw_label in enumerate(fraud_labels):
        target = base + (1 if idx < rem else 0)
        pool = fraud_groups[raw_label]
        chosen = rng.sample(pool, target) if len(pool) >= target else [pool[i % len(pool)] for i in range(target)]
        sampled.extend(dict(row, source=f"chifraud_{row['source']}") for row in chosen)
        plan["fraud_by_raw_label"][str(raw_label)] = target

    chosen = rng.sample(normal_rows, normal_target) if len(normal_rows) >= normal_target else [normal_rows[i % len(normal_rows)] for i in range(normal_target)]
    sampled.extend(dict(row, source=f"chifraud_{row['source']}") for row in chosen)
    rng.shuffle(sampled)
    return sampled, plan


def write_tsv(path, rows, include_source=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        if include_source:
            writer.writerow(["label", "text", "source", "source_id"])
            for row in rows:
                writer.writerow([row["label"], row["text"], row.get("source", ""), row.get("source_id", "")])
        else:
            writer.writerow(["label", "text"])
            for row in rows:
                writer.writerow([row["label"], row["text"]])


def main():
    parser = argparse.ArgumentParser(description="Build a larger clean final test set.")
    parser.add_argument("--dataset-dir", default="./dataset")
    parser.add_argument("--output-name", default="dialogue_binary_refined_large_test")
    parser.add_argument("--validated-file", default="validated_data.jsonl")
    parser.add_argument("--chifraud-test-file", default="ChiFraud_t2023.csv")
    parser.add_argument("--total", type=int, default=500)
    parser.add_argument("--fraud-ratio", type=float, default=0.25)
    parser.add_argument("--validated-total", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = dataset_dir / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    validated = load_validated(dataset_dir / args.validated_file)
    validated_test = stratified_split(validated, 0.8, 0.1, 2026)["test"]
    if args.validated_total > len(validated_test):
        args.validated_total = len(validated_test)
    rng = random.Random(args.seed)
    validated_part = rng.sample(validated_test, args.validated_total)

    chifraud_target = args.total - len(validated_part)
    chifraud_rows = load_chifraud(dataset_dir / args.chifraud_test_file)
    chifraud_part, chifraud_plan = sample_chifraud(chifraud_rows, chifraud_target, args.fraud_ratio, args.seed + 1)

    rows = validated_part + chifraud_part
    rng.shuffle(rows)

    write_tsv(output_dir / "test.tsv", rows)
    write_tsv(output_dir / "source_test.tsv", rows, include_source=True)
    (output_dir / "class.txt").write_text("0 正常\n1 诈骗\n", encoding="utf-8")

    metadata = {
        "dataset": args.output_name,
        "seed": args.seed,
        "policy": "large clean test; crawler weak labels are excluded",
        "input_files": {
            "validated": args.validated_file,
            "chifraud_test": args.chifraud_test_file,
        },
        "target_total": args.total,
        "target_fraud_ratio": args.fraud_ratio,
        "validated_part_summary": summarize(validated_part),
        "chifraud_part_summary": summarize(chifraud_part),
        "chifraud_sample_plan": chifraud_plan,
        "final_summary": summarize(rows),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata["final_summary"], ensure_ascii=False, indent=2))
    print(f"Large test set written to: {output_dir}")


if __name__ == "__main__":
    main()
