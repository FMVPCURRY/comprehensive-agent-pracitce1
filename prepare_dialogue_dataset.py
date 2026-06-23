import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def normalize_text(text: str) -> str:
    return str(text).replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()


def convert_original_row(row: dict) -> Tuple[int, str]:
    label = int(row["Label_id"])
    text = normalize_text(row["Text"])
    return 0 if label == 0 else 1, text


def convert_dialogue_row(row: dict) -> Tuple[int, str]:
    rendered_turns: List[str] = []
    for speaker, utterance in row["text"]:
        speaker_name = "对方" if str(speaker) == "1" else "我方"
        utterance = normalize_text(utterance)
        if utterance:
            rendered_turns.append(f"{speaker_name}：{utterance}")
    return int(row["label"]), " ".join(rendered_turns)


def load_original_split(path: Path) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label, text = convert_original_row(row)
            if text:
                rows.append((label, text))
    return rows


def stratified_split(
    rows: Sequence[Tuple[int, str]],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row[0], []).append(row)

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


def load_generated_dialogues(
    path: Path,
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    rows: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            label, text = convert_dialogue_row(obj)
            if text:
                rows.append((label, text))
    return stratified_split(rows, train_ratio, dev_ratio, seed)


def write_tsv(path: Path, rows: Iterable[Tuple[int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("label\ttext\n")
        for label, text in rows:
            f.write(f"{label}\t{text}\n")


def summarize(name: str, rows: Sequence[Tuple[int, str]]) -> str:
    normal = sum(1 for label, _ in rows if label == 0)
    fraud = sum(1 for label, _ in rows if label == 1)
    return f"{name}: total={len(rows)}, normal={normal}, fraud={fraud}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a binary dialogue fraud dataset.")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="dataset/dialogue_binary")
    parser.add_argument("--generated-file", default="data.jsonl")
    parser.add_argument("--train-file", default="ChiFraud_train.csv")
    parser.add_argument("--dev-file", default="ChiFraud_t2022.csv")
    parser.add_argument("--test-file", default="ChiFraud_t2023.csv")
    parser.add_argument("--generated-train-ratio", type=float, default=0.8)
    parser.add_argument("--generated-dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    original_train = load_original_split(dataset_dir / args.train_file)
    original_dev = load_original_split(dataset_dir / args.dev_file)
    original_test = load_original_split(dataset_dir / args.test_file)

    generated_train, generated_dev, generated_test = load_generated_dialogues(
        dataset_dir / args.generated_file,
        args.generated_train_ratio,
        args.generated_dev_ratio,
        args.seed,
    )

    final_train = original_train + generated_train
    final_dev = original_dev + generated_dev
    final_test = original_test + generated_test

    random.Random(args.seed).shuffle(final_train)
    random.Random(args.seed + 1).shuffle(final_dev)
    random.Random(args.seed + 2).shuffle(final_test)

    write_tsv(output_dir / "train.tsv", final_train)
    write_tsv(output_dir / "dev.tsv", final_dev)
    write_tsv(output_dir / "test.tsv", final_test)
    write_tsv(output_dir / "generated_train.tsv", generated_train)
    write_tsv(output_dir / "generated_dev.tsv", generated_dev)
    write_tsv(output_dir / "generated_test.tsv", generated_test)
    write_tsv(output_dir / "original_train_binary.tsv", original_train)
    write_tsv(output_dir / "original_dev_binary.tsv", original_dev)
    write_tsv(output_dir / "original_test_binary.tsv", original_test)
    (output_dir / "class.txt").write_text("0 正常\n1 诈骗\n", encoding="utf-8")

    metadata = {
        "source": "ChiFraud + generated dialogue data",
        "seed": args.seed,
        "generated_split_ratio": {
            "train": args.generated_train_ratio,
            "dev": args.generated_dev_ratio,
            "test": 1 - args.generated_train_ratio - args.generated_dev_ratio,
        },
        "counts": {
            "train": len(final_train),
            "dev": len(final_dev),
            "test": len(final_test),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summarize("train", final_train))
    print(summarize("dev", final_dev))
    print(summarize("test", final_test))
    print(summarize("generated_train", generated_train))
    print(summarize("generated_dev", generated_dev))
    print(summarize("generated_test", generated_test))


if __name__ == "__main__":
    main()
