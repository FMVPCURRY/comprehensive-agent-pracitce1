import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from inference_backend import QwenApiPredictor


def load_tsv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0] in ("label", "label_id") or len(row) < 2:
                continue
            rows.append({"label": int(row[0]), "text": row[1]})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate current web Qwen API predictor.")
    parser.add_argument("--eval-path", default="./dataset/dialogue_binary_matched_2x/test.tsv")
    parser.add_argument("--output-dir", default="./result_qwen")
    parser.add_argument("--run-name", default="matched2x_test_qwen_api_current")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    rows = load_tsv(args.eval_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{args.run_name}.predictions.jsonl"
    summary_path = out_dir / f"{args.run_name}.summary.json"

    predictor = QwenApiPredictor()
    y_true = []
    y_pred = []
    failures = []

    with pred_path.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(rows, start=1):
            record = {
                "index": idx,
                "gold": row["label"],
                "text": row["text"],
            }
            try:
                result = predictor.predict(row["text"])
                pred = 1 if result["label"] == "fraud" else 0
                y_true.append(row["label"])
                y_pred.append(pred)
                record.update({
                    "pred": pred,
                    "raw_response": result.get("raw_response", ""),
                    "error": None,
                })
            except Exception as exc:
                failures.append({"index": idx, "gold": row["label"], "error": str(exc)})
                record.update({"pred": None, "raw_response": "", "error": str(exc)})

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            if idx % args.log_every == 0 or idx == len(rows):
                print(f"[{idx}/{len(rows)}] processed, valid={len(y_true)}, failed={len(failures)}", flush=True)

    if y_true:
        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["normal", "fraud"],
            zero_division=0,
            output_dict=True,
        )
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
        summary = {
            "model": "qwen_api",
            "eval_path": args.eval_path,
            "count_total": len(rows),
            "count_valid": len(y_true),
            "count_failed": len(failures),
            "accuracy": accuracy_score(y_true, y_pred),
            "fraud_precision": report["fraud"]["precision"],
            "fraud_recall": report["fraud"]["recall"],
            "fraud_f1": report["fraud"]["f1-score"],
            "normal_precision": report["normal"]["precision"],
            "normal_recall": report["normal"]["recall"],
            "normal_f1": report["normal"]["f1-score"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "confusion_matrix": matrix,
            "failures": failures,
            "prediction_file": str(pred_path),
        }
    else:
        summary = {
            "model": "qwen_api",
            "eval_path": args.eval_path,
            "count_total": len(rows),
            "count_valid": 0,
            "count_failed": len(failures),
            "failures": failures,
            "prediction_file": str(pred_path),
        }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
