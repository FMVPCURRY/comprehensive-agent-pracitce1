import argparse
import csv
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DEFAULT_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")


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


def sanitize_for_api(text):
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\d", "X", text)
    replacements = {
        "微信": "联系工具",
        "薇信": "联系工具",
        "薇芯": "联系工具",
        "威信": "联系工具",
        "qq": "联系号",
        "QQ": "联系号",
        "q号": "联系号",
        "vx": "联系号",
        "VX": "联系号",
        "v信": "联系号",
        "V信": "联系号",
        "电报": "联系工具",
        "Telegram": "联系工具",
        "飞机": "联系工具",
        "外围": "灰产服务",
        "上门": "线下服务",
        "保健": "灰产服务",
        "全套": "灰产服务",
        "小姐": "灰产对象",
        "美女": "人员",
        "陪睡": "违规服务",
        "嫖": "违规",
        "妓": "违规",
        "迷药": "违禁品",
        "药": "特殊物品",
        "枪": "危险物",
        "弹": "危险物",
        "银行卡": "金融工具",
        "信用卡": "金融工具",
        "贷款": "借贷",
        "博彩": "赌博",
        "赌": "博",
        "发票": "票据",
        "证件": "材料",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"(联系号|联系工具)[:：]?\s*[A-Za-zXx0-9_\-\.]+", r"\1: [MASKED]", text)
    return text


def build_prompt(sample_text, mode, pos_example=None, neg_example=None):
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


def deterministic_example(example_pool, sample_text, salt):
    if not example_pool:
        return ""
    digest = hashlib.md5((salt + sample_text).encode("utf-8")).hexdigest()
    return example_pool[int(digest, 16) % len(example_pool)]["text"]


def call_qwen(api_key, model, messages, base_url, temperature, max_tokens, disable_thinking, max_retries, retry_sleep):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if disable_thinking:
        # Qwen3-style models may enable thinking by default on some endpoints.
        # Keep both forms for compatibility with DashScope/OpenAI-compatible APIs.
        payload["enable_thinking"] = False
        payload["extra_body"] = {"enable_thinking": False}
    request = urllib.request.Request(
        url=base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_error = None
    for _ in range(max_retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {error_body}"
            time.sleep(retry_sleep)
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            last_error = repr(exc)
            time.sleep(retry_sleep)
    raise RuntimeError(f"Qwen API call failed after retries: {last_error}")


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


def evaluate(args):
    train_rows = load_tsv(args.train_path)
    eval_rows = load_tsv(args.eval_path)
    pos_pool = [row for row in train_rows if row["label"] == 1]
    neg_pool = [row for row in train_rows if row["label"] == 0]

    if args.mode == "icl" and (not pos_pool or not neg_pool):
        raise ValueError("ICL mode requires both positive and negative samples in train set.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{args.run_name}.predictions.jsonl"
    summary_path = output_dir / f"{args.run_name}.summary.json"

    y_true = []
    y_pred = []
    failures = []

    with pred_path.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(eval_rows, start=1):
            sample_text = sanitize_for_api(shorten_text(row["text"], args.max_text_chars))
            if args.mode == "icl":
                pos_example = sanitize_for_api(
                    shorten_text(deterministic_example(pos_pool, row["text"], "pos"), args.max_example_chars)
                )
                neg_example = sanitize_for_api(
                    shorten_text(deterministic_example(neg_pool, row["text"], "neg"), args.max_example_chars)
                )
                messages = build_prompt(sample_text, "icl", pos_example, neg_example)
            else:
                messages = build_prompt(sample_text, "zero-shot")

            try:
                raw_response = call_qwen(
                    api_key=args.api_key,
                    model=args.model,
                    messages=messages,
                    base_url=args.base_url,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    disable_thinking=args.disable_thinking,
                    max_retries=args.max_retries,
                    retry_sleep=args.retry_sleep,
                )
                pred_label, explanation = parse_label(raw_response)
                status = "ok"
                y_true.append(row["label"])
                y_pred.append(pred_label)
            except Exception as exc:
                raw_response = repr(exc)
                pred_label = None
                explanation = ""
                status = "failed"
                failures.append({
                    "index": idx,
                    "gold": row["label"],
                    "error": raw_response,
                    "text_preview": row["text"][:120],
                })

            fout.write(json.dumps({
                "index": idx,
                "status": status,
                "gold": row["label"],
                "pred": pred_label,
                "text": row["text"],
                "raw_response": raw_response,
                "explanation": explanation,
            }, ensure_ascii=False) + "\n")

            if idx % args.log_every == 0 or idx == len(eval_rows):
                print(f"[{idx}/{len(eval_rows)}] processed, valid={len(y_true)}, failed={len(failures)}")

    if y_true:
        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["normal", "fraud"],
            zero_division=0,
            output_dict=True,
        )
        accuracy = accuracy_score(y_true, y_pred)
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    else:
        report = {
            "normal": {"precision": 0, "recall": 0, "f1-score": 0},
            "fraud": {"precision": 0, "recall": 0, "f1-score": 0},
            "macro avg": {"f1-score": 0},
            "weighted avg": {"f1-score": 0},
        }
        accuracy = 0
        matrix = [[0, 0], [0, 0]]
    summary = {
        "mode": args.mode,
        "model": args.model,
        "eval_path": args.eval_path,
        "count_total": len(eval_rows),
        "count_valid": len(y_true),
        "count_failed": len(failures),
        "accuracy": accuracy,
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

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen API on ChiFraud-style binary detection.")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--output-dir", default="./result_qwen")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=["zero-shot", "icl"], default="icl")
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-text-chars", type=int, default=600)
    parser.add_argument("--max-example-chars", type=int, default=400)
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Please provide --api-key or set DASHSCOPE_API_KEY.")

    evaluate(args)


if __name__ == "__main__":
    main()
