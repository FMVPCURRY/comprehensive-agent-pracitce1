# coding: utf-8
"""Build a refined binary dialogue dataset for SmartShield.

The new split uses:
- Doubao-validated DeepSeek dialogues as the trusted generated source.
- Sampled ChiFraud original data, keeping the original train/dev/test years.
- Weakly labeled crawler data for training only by default.

Output TSV files keep the format expected by run.py: label<TAB>text.
Extra source_*.tsv files are written for audit and reporting.
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
    train_rows, dev_rows, test_rows = [], [], []
    for label_rows in grouped.values():
        label_rows = list(label_rows)
        rng.shuffle(label_rows)
        total = len(label_rows)
        train_end = int(total * train_ratio)
        dev_end = train_end + int(total * dev_ratio)
        train_rows.extend(label_rows[:train_end])
        dev_rows.extend(label_rows[train_end:dev_end])
        test_rows.extend(label_rows[dev_end:])

    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)
    rng.shuffle(test_rows)
    return train_rows, dev_rows, test_rows


def load_validated_dialogues(path):
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
            if not text:
                continue
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
    if fraud_target > 0 and fraud_labels:
        base = fraud_target // len(fraud_labels)
        rem = fraud_target % len(fraud_labels)
        for i, raw_label in enumerate(fraud_labels):
            target = base + (1 if i < rem else 0)
            source = fraud_groups[raw_label]
            chosen = rng.sample(source, target) if len(source) >= target else [source[j % len(source)] for j in range(target)]
            sampled.extend(dict(row, source=f"chifraud_{row['source']}") for row in chosen)
            plan["fraud_by_raw_label"][str(raw_label)] = target

    if normal_target > 0:
        chosen = rng.sample(normal_rows, normal_target) if len(normal_rows) >= normal_target else [normal_rows[j % len(normal_rows)] for j in range(normal_target)]
        sampled.extend(dict(row, source=f"chifraud_{row['source']}") for row in chosen)

    rng.shuffle(sampled)
    return sampled, plan


def load_crawler(path, max_chars):
    rows = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        text_field = "text" if "text" in fieldnames else fieldnames[0]
        for idx, row in enumerate(reader, start=1):
            text = clean_text(row.get(text_field, ""))
            label = row.get("label", "")
            if label not in ("0", "1") or not text:
                continue
            rows.append({
                "label": int(label),
                "text": text[:max_chars],
                "source": "crawler_weak",
                "source_id": f"crawler:{idx}",
            })
    return rows


def sample_binary(rows, target_total, fraud_ratio, seed):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    fraud_target = int(round(target_total * fraud_ratio))
    normal_target = target_total - fraud_target
    sampled = []
    for label, target in [(1, fraud_target), (0, normal_target)]:
        source = by_label[label]
        if not source or target <= 0:
            continue
        chosen = rng.sample(source, target) if len(source) >= target else [source[i % len(source)] for i in range(target)]
        sampled.extend(chosen)
    rng.shuffle(sampled)
    return sampled


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
    parser = argparse.ArgumentParser(description="Build refined ChiFraud + validated dialogue dataset.")
    parser.add_argument("--dataset-dir", default="./dataset")
    parser.add_argument("--output-name", default="dialogue_binary_refined")
    parser.add_argument("--validated-file", default="validated_data.jsonl")
    parser.add_argument("--crawler-file", default="fraud_text_label.csv")
    parser.add_argument("--train-file", default="ChiFraud_train.csv")
    parser.add_argument("--dev-file", default="ChiFraud_t2022.csv")
    parser.add_argument("--test-file", default="ChiFraud_t2023.csv")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--original-multiplier", type=int, default=2)
    parser.add_argument("--crawler-train-total", type=int, default=400)
    parser.add_argument("--crawler-max-chars", type=int, default=600)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = dataset_dir / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    validated = load_validated_dialogues(dataset_dir / args.validated_file)
    gen_train, gen_dev, gen_test = stratified_split(validated, args.train_ratio, args.dev_ratio, args.seed)
    generated_splits = {"train": gen_train, "dev": gen_dev, "test": gen_test}
    fraud_ratio = summarize(gen_train)["fraud"] / max(1, len(gen_train))

    chifraud_sources = {
        "train": load_chifraud(dataset_dir / args.train_file),
        "dev": load_chifraud(dataset_dir / args.dev_file),
        "test": load_chifraud(dataset_dir / args.test_file),
    }

    split_rows = {}
    sample_plans = {}
    for offset, split in enumerate(["train", "dev", "test"]):
        target = len(generated_splits[split]) * args.original_multiplier
        sampled, plan = sample_chifraud(chifraud_sources[split], target, fraud_ratio, args.seed + 10 + offset)
        split_rows[split] = list(generated_splits[split]) + sampled
        sample_plans[split] = plan

    crawler_summary = None
    if args.crawler_train_total > 0:
        crawler_rows = load_crawler(dataset_dir / args.crawler_file, args.crawler_max_chars)
        sampled_crawler = sample_binary(crawler_rows, args.crawler_train_total, fraud_ratio, args.seed + 30)
        split_rows["train"].extend(sampled_crawler)
        crawler_summary = summarize(sampled_crawler)

    for offset, split in enumerate(["train", "dev", "test"]):
        random.Random(args.seed + 100 + offset).shuffle(split_rows[split])
        write_tsv(output_dir / f"{split}.tsv", split_rows[split])
        write_tsv(output_dir / f"source_{split}.tsv", split_rows[split], include_source=True)

    (output_dir / "class.txt").write_text("0 正常\n1 诈骗\n", encoding="utf-8")

    metadata = {
        "dataset": args.output_name,
        "seed": args.seed,
        "strategy": {
            "validated_dialogues": "stratified 8:1:1 split",
            "chifraud": "sampled from original train/2022/2023 separately",
            "crawler": "weak labels, train only by default",
            "test_policy": "dev/test do not include crawler weak-label data by default",
        },
        "input_files": {
            "validated": args.validated_file,
            "crawler": args.crawler_file,
            "chifraud_train": args.train_file,
            "chifraud_dev": args.dev_file,
            "chifraud_test": args.test_file,
        },
        "validated_summary": summarize(validated),
        "generated_split_summary": {k: summarize(v) for k, v in generated_splits.items()},
        "chifraud_sample_plans": sample_plans,
        "crawler_train_summary": crawler_summary,
        "final_summary": {k: summarize(v) for k, v in split_rows.items()},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metadata["final_summary"], ensure_ascii=False, indent=2))
    print(f"Dataset written to: {output_dir}")


if __name__ == "__main__":
    main()
