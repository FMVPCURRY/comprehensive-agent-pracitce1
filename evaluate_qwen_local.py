import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_tsv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0] in ("label", "label_id") or len(row) < 2:
                continue
            rows.append({"label": int(row[0]), "text": row[1]})
    return rows


def shorten_text(text, max_chars):
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[TRUNCATED]"


def deterministic_example(example_pool, sample_text, salt):
    if not example_pool:
        return ""
    digest = hashlib.md5((salt + sample_text).encode("utf-8")).hexdigest()
    return example_pool[int(digest, 16) % len(example_pool)]["text"]


def build_messages(sample_text, mode, pos_example=None, neg_example=None):
    system_prompt = (
        "You are a Chinese online fraud detection assistant. "
        "Determine whether the given text should be classified as fraud. "
        "Label 1 means fraud. Label 0 means normal. "
        "Classify as fraud if the text contains scam intent, illegal trading, transfer inducement, "
        "private contact diversion, fake certificates, bank card trading, underground loans, "
        "gambling, prostitution, prohibited drugs, or other obviously fraudulent or illegal content. "
        "Classify as normal for ordinary conversation, benign information, or legitimate content. "
        "Return JSON only."
    )

    output_format = (
        'Output JSON in exactly this format: '
        '{"label": 0 or 1, "conclusion": "normal or fraud", "explanation": "short reason"}'
    )

    if mode == "zero-shot":
        user_prompt = (
            "Please complete a binary classification task.\n"
            "Labels: 0=normal, 1=fraud.\n"
            f"{output_format}\n"
            f"[Text]\n{sample_text}"
        )
    else:
        user_prompt = (
            "Please complete a binary classification task with in-context examples.\n"
            "Labels: 0=normal, 1=fraud.\n"
            f"{output_format}\n"
            f"[Context Sample 1]\nLabel: 1\nText: {pos_example}\n"
            f"[Context Sample 2]\nLabel: 0\nText: {neg_example}\n"
            f"[Text]\n{sample_text}"
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_label(raw_text):
    raw_text = raw_text.strip()
    try:
        data = json.loads(raw_text)
        return int(data["label"]), str(data.get("explanation", ""))
    except Exception:
        pass

    lowered = raw_text.lower()
    if '"label": 1' in lowered or '"label":1' in lowered:
        return 1, raw_text
    if '"label": 0' in lowered or '"label":0' in lowered:
        return 0, raw_text
    if "fraud" in lowered and "normal" not in lowered:
        return 1, raw_text
    if "normal" in lowered and "fraud" not in lowered:
        return 0, raw_text
    raise ValueError(f"Unable to parse label from response: {raw_text}")


def generate_response(model, tokenizer, messages, max_new_tokens):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


def evaluate(args):
    train_rows = load_tsv(args.train_path)
    eval_rows = load_tsv(args.eval_path)
    pos_pool = [row for row in train_rows if row["label"] == 1]
    neg_pool = [row for row in train_rows if row["label"] == 0]

    tokenizer_source = args.adapter_dir if args.adapter_dir else args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir, local_files_only=True) if args.adapter_dir else base_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{args.run_name}.predictions.jsonl"
    summary_path = output_dir / f"{args.run_name}.summary.json"

    y_true = []
    y_pred = []

    with pred_path.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(eval_rows, start=1):
            sample_text = shorten_text(row["text"], args.max_text_chars)
            if args.mode == "icl":
                pos_example = shorten_text(deterministic_example(pos_pool, row["text"], "pos"), args.max_example_chars)
                neg_example = shorten_text(deterministic_example(neg_pool, row["text"], "neg"), args.max_example_chars)
                messages = build_messages(sample_text, "icl", pos_example, neg_example)
            else:
                messages = build_messages(sample_text, "zero-shot")

            raw_response = generate_response(model, tokenizer, messages, args.max_new_tokens)
            pred_label, explanation = parse_label(raw_response)
            y_true.append(row["label"])
            y_pred.append(pred_label)

            fout.write(json.dumps({
                "index": idx,
                "gold": row["label"],
                "pred": pred_label,
                "text": row["text"],
                "raw_response": raw_response,
                "explanation": explanation,
            }, ensure_ascii=False) + "\n")

            if idx % args.log_every == 0 or idx == len(eval_rows):
                print(f"[{idx}/{len(eval_rows)}] processed")

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["normal", "fraud"],
        zero_division=0,
        output_dict=True,
    )
    summary = {
        "mode": args.mode,
        "model_dir": args.model_dir,
        "adapter_dir": args.adapter_dir,
        "eval_path": args.eval_path,
        "count": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "fraud_precision": report["fraud"]["precision"],
        "fraud_recall": report["fraud"]["recall"],
        "fraud_f1": report["fraud"]["f1-score"],
        "normal_precision": report["normal"]["precision"],
        "normal_recall": report["normal"]["recall"],
        "normal_f1": report["normal"]["f1-score"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "prediction_file": str(pred_path),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Evaluate local Qwen on ChiFraud-style binary detection.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--adapter-dir")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--output-dir", default="./result_qwen_local")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=["zero-shot", "icl"], default="icl")
    parser.add_argument("--max-text-chars", type=int, default=600)
    parser.add_argument("--max-example-chars", type=int, default=400)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
