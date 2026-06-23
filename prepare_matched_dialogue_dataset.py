import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def normalize_text(text: str) -> str:
    return str(text).replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()


def write_tsv(path: Path, rows: Iterable[Tuple[int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("label\ttext\n")
        for label, text in rows:
            f.write(f"{label}\t{text}\n")


def summarize(rows: Sequence[Tuple[int, str]]) -> Dict[str, int]:
    counter = Counter(label for label, _ in rows)
    return {
        "total": len(rows),
        "normal": counter.get(0, 0),
        "fraud": counter.get(1, 0),
    }


def load_generated_dialogues(path: Path) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rendered = []
            for speaker, utterance in obj["text"]:
                speaker_name = "对方" if str(speaker) == "1" else "我方"
                utterance = normalize_text(utterance)
                if utterance:
                    rendered.append(f"{speaker_name}：{utterance}")
            rows.append((int(obj["label"]), " ".join(rendered)))
    return rows


def stratified_split(
    rows: Sequence[Tuple[int, str]],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[0]].append(row)

    rng = random.Random(seed)
    train_rows: List[Tuple[int, str]] = []
    dev_rows: List[Tuple[int, str]] = []
    test_rows: List[Tuple[int, str]] = []

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


def load_original_multiclass(path: Path) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label = int(row["Label_id"])
            text = normalize_text(row["Text"])
            if text:
                rows.append((label, text))
    return rows


def sample_original_rows(
    rows: Sequence[Tuple[int, str]],
    target_total: int,
    fraud_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[int, str]], Dict[str, object]]:
    rng = random.Random(seed)
    normal_rows = [row for row in rows if row[0] == 0]
    fraud_groups: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    for label, text in rows:
        if label != 0:
            fraud_groups[label].append((label, text))

    fraud_labels = sorted(fraud_groups.keys())
    fraud_target = max(1, int(round(target_total * fraud_ratio)))
    normal_target = target_total - fraud_target

    per_class_base = fraud_target // len(fraud_labels)
    remainder = fraud_target % len(fraud_labels)

    sampled_binary_rows: List[Tuple[int, str]] = []
    fraud_plan: Dict[int, int] = {}

    for index, label in enumerate(fraud_labels):
        target = per_class_base + (1 if index < remainder else 0)
        fraud_plan[label] = target
        source = list(fraud_groups[label])
        if len(source) >= target:
            chosen = rng.sample(source, target)
        else:
            chosen = [source[i % len(source)] for i in range(target)]
            rng.shuffle(chosen)
        sampled_binary_rows.extend((1, text) for _, text in chosen)

    if len(normal_rows) >= normal_target:
        normal_chosen = rng.sample(normal_rows, normal_target)
    else:
        normal_chosen = [normal_rows[i % len(normal_rows)] for i in range(normal_target)]
        rng.shuffle(normal_chosen)
    sampled_binary_rows.extend((0, text) for _, text in normal_chosen)
    rng.shuffle(sampled_binary_rows)

    metadata = {
        "target_total": target_total,
        "fraud_ratio": fraud_ratio,
        "normal_target": normal_target,
        "fraud_target": fraud_target,
        "fraud_labels_in_split": fraud_labels,
        "fraud_samples_per_label": fraud_plan,
    }
    return sampled_binary_rows, metadata


def build_variant(
    output_dir: Path,
    original_splits: Dict[str, List[Tuple[int, str]]],
    generated_splits: Dict[str, List[Tuple[int, str]]],
    multiplier: int,
    seed: int,
) -> None:
    generated_counts = {name: len(rows) for name, rows in generated_splits.items()}
    fraud_ratio = summarize(generated_splits["train"])["fraud"] / generated_counts["train"]

    sampled_original = {}
    sample_meta = {}
    for offset, split_name in enumerate(["train", "dev", "test"]):
        sampled_rows, metadata = sample_original_rows(
            original_splits[split_name],
            target_total=generated_counts[split_name] * multiplier,
            fraud_ratio=fraud_ratio,
            seed=seed + offset,
        )
        sampled_original[split_name] = sampled_rows
        sample_meta[split_name] = metadata

    combined = {}
    for offset, split_name in enumerate(["train", "dev", "test"]):
        rows = list(sampled_original[split_name]) + list(generated_splits[split_name])
        random.Random(seed + 100 + offset).shuffle(rows)
        combined[split_name] = rows

    write_tsv(output_dir / "train.tsv", combined["train"])
    write_tsv(output_dir / "dev.tsv", combined["dev"])
    write_tsv(output_dir / "test.tsv", combined["test"])
    write_tsv(output_dir / "sampled_original_train.tsv", sampled_original["train"])
    write_tsv(output_dir / "sampled_original_dev.tsv", sampled_original["dev"])
    write_tsv(output_dir / "sampled_original_test.tsv", sampled_original["test"])
    write_tsv(output_dir / "generated_train.tsv", generated_splits["train"])
    write_tsv(output_dir / "generated_dev.tsv", generated_splits["dev"])
    write_tsv(output_dir / "generated_test.tsv", generated_splits["test"])
    (output_dir / "class.txt").write_text("0 正常\n1 诈骗\n", encoding="utf-8")

    metadata = {
        "variant": output_dir.name,
        "multiplier": multiplier,
        "generated_ratio_reference": summarize(generated_splits["train"]),
        "sample_plan": sample_meta,
        "combined_summary": {split: summarize(rows) for split, rows in combined.items()},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(output_dir.name)
    for split_name in ["train", "dev", "test"]:
        print(split_name, summarize(combined[split_name]), sample_meta[split_name]["fraud_samples_per_label"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sampled original + generated dialogue datasets.")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--generated-file", default="data.jsonl")
    parser.add_argument("--train-file", default="ChiFraud_train.csv")
    parser.add_argument("--dev-file", default="ChiFraud_t2022.csv")
    parser.add_argument("--test-file", default="ChiFraud_t2023.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    generated_rows = load_generated_dialogues(dataset_dir / args.generated_file)
    gen_train, gen_dev, gen_test = stratified_split(generated_rows, 0.8, 0.1, args.seed)
    generated_splits = {"train": gen_train, "dev": gen_dev, "test": gen_test}

    original_splits = {
        "train": load_original_multiclass(dataset_dir / args.train_file),
        "dev": load_original_multiclass(dataset_dir / args.dev_file),
        "test": load_original_multiclass(dataset_dir / args.test_file),
    }

    build_variant(dataset_dir / "dialogue_binary_matched_1x", original_splits, generated_splits, multiplier=1, seed=args.seed)
    build_variant(dataset_dir / "dialogue_binary_matched_2x", original_splits, generated_splits, multiplier=2, seed=args.seed)


if __name__ == "__main__":
    main()
