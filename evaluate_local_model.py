# coding: utf-8
"""Evaluate a saved BERT/ChineseBERT checkpoint on a custom TSV set."""

import argparse
import csv
import json
import os
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics


def load_gold_text(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0] in ("label", "label_id") or len(row) < 2:
                continue
            rows.append((int(row[0]), row[1]))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate local checkpoint on custom TSV.")
    parser.add_argument("--model", required=True, choices=["Bert", "Chinese_Bert"])
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--class-path", required=True)
    parser.add_argument("--bert-path", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="./result")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pad-size", type=int, default=256)
    args = parser.parse_args()

    if args.model == "Bert":
        from utils_bert import build_dataset, build_iterator
    else:
        from utils_chinesebert import build_dataset, build_iterator

    module = import_module("models." + args.model)
    config = module.Config(args.dataset_name, "random")
    config.train_path = args.eval_path
    config.dev_path = args.eval_path
    config.test_path = args.eval_path
    config.class_list = [line.strip() for line in open(args.class_path, encoding="utf-8")]
    config.num_classes = len(config.class_list)
    config.batch_size = args.batch_size
    config.pad_size = args.pad_size
    if args.bert_path and hasattr(config, "bert_path"):
        config.bert_path = args.bert_path
        if hasattr(config, "reload_tokenizer"):
            config.reload_tokenizer()
    checkpoint = args.checkpoint or config.save_path

    _, _, _, eval_data = build_dataset(config, False)
    eval_iter = build_iterator(eval_data, config)

    model = module.Model(config).to(config.device)
    model.load_state_dict(torch.load(checkpoint, map_location=config.device))
    model.eval()

    y_true = []
    y_pred = []
    probs_all = []
    start = time.time()
    with torch.no_grad():
        for texts, labels in eval_iter:
            logits = model(texts)
            probs = F.softmax(logits, dim=1)
            preds = torch.max(logits.data, 1)[1].cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.data.cpu().numpy().tolist())
            probs_all.extend(probs.cpu().numpy().tolist())

    rows = load_gold_text(args.eval_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{args.run_name}.predictions.tsv"
    summary_path = output_dir / f"{args.run_name}.summary.json"

    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["gold", "pred", "prob_normal", "prob_fraud", "text"])
        for idx, (gold, text) in enumerate(rows[:len(y_pred)]):
            prob = probs_all[idx]
            writer.writerow([gold, y_pred[idx], prob[0], prob[1], text])

    report = metrics.classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["normal", "fraud"],
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    summary = {
        "model": args.model,
        "dataset_name": args.dataset_name,
        "eval_path": args.eval_path,
        "checkpoint": checkpoint,
        "count": len(y_true),
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "fraud_precision": report["fraud"]["precision"],
        "fraud_recall": report["fraud"]["recall"],
        "fraud_f1": report["fraud"]["f1-score"],
        "normal_precision": report["normal"]["precision"],
        "normal_recall": report["normal"]["recall"],
        "normal_f1": report["normal"]["f1-score"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "confusion_matrix": metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "elapsed_seconds": round(time.time() - start, 3),
        "prediction_file": str(pred_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
