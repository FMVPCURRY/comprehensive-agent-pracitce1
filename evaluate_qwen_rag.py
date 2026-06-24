# coding: utf-8
"""Evaluate Qwen API with lightweight RAG context.

This script retrieves similar fraud/normal examples from a local TSV/CSV corpus
using TF-IDF character n-grams, then asks Qwen API to classify the input text.
It is intentionally dependency-light and does not require a vector database.
"""

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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")


def clean_text(text):
    text = str(text).replace("\ufeff", "")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def shorten(text, max_chars):
    text = clean_text(text)
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + "...[TRUNCATED]"
    return text


def load_tsv(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0] in ("label", "label_id") or len(row) < 2:
                continue
            text = clean_text(row[1])
            if text:
                rows.append({"label": int(row[0]), "text": text, "source": Path(path).name})
    return rows


def load_csv_corpus(path, max_rows, seed):
    rows = []
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        text_field = "text" if "text" in fieldnames else fieldnames[0]
        for row in reader:
            label = row.get("label", "")
            if label not in ("0", "1"):
                continue
            text = clean_text(row.get(text_field, ""))
            if text:
                rows.append({"label": int(label), "text": text, "source": Path(path).name})
    if max_rows > 0 and len(rows) > max_rows:
        # Deterministic label-balanced sampling.
        import random
        rng = random.Random(seed)
        by_label = {0: [r for r in rows if r["label"] == 0], 1: [r for r in rows if r["label"] == 1]}
        half = max_rows // 2
        sampled = []
        for label, target in [(1, half), (0, max_rows - half)]:
            pool = by_label[label]
            sampled.extend(rng.sample(pool, min(target, len(pool))))
        rng.shuffle(sampled)
        rows = sampled
    return rows


def sanitize_for_api(text):
    text = shorten(text, 1200)
    text = re.sub(r"\d", "X", text)
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+", "[EMAIL]", text)
    text = re.sub(r"(微信|QQ|vx|VX|qq)[:：]?\s*[A-Za-z0-9_-]+", r"\1:[MASKED]", text)
    return text


class RagRetriever:
    def __init__(self, rows, max_doc_chars):
        self.rows = [dict(row, text=shorten(row["text"], max_doc_chars)) for row in rows if clean_text(row["text"])]
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=60000)
        self.matrix = self.vectorizer.fit_transform([row["text"] for row in self.rows])

    def search(self, query, top_k):
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        top_idx = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_idx:
            row = self.rows[int(idx)]
            results.append({
                "label": row["label"],
                "text": row["text"],
                "source": row.get("source", ""),
                "score": float(scores[int(idx)]),
            })
        return results


def has_strong_fraud_signal(text):
    """Keep high-risk obvious scam cases even when retrieval evidence is weak."""
    patterns = [
        "刷单", "返佣", "垫付", "保证金", "转账", "打款", "加群", "进群",
        "投资", "理财", "提现", "账户冻结", "客服", "退款", "中奖", "贷款",
        "验资", "银行卡", "私聊", "兼职", "高回报", "稳赚", "名额有限",
    ]
    return any(pattern in text for pattern in patterns)


def retrieval_stats(retrieved):
    fraud_scores = [item["score"] for item in retrieved if item["label"] == 1]
    normal_scores = [item["score"] for item in retrieved if item["label"] == 0]
    return {
        "fraud_count": len(fraud_scores),
        "normal_count": len(normal_scores),
        "max_fraud_score": max(fraud_scores) if fraud_scores else 0.0,
        "max_normal_score": max(normal_scores) if normal_scores else 0.0,
    }


def calibrate_prediction(pred_label, explanation, sample_text, retrieved, min_fraud_evidence):
    """Reduce false positives caused by LLM over-warning.

    In this dataset many normal samples contain contact numbers, ads, or noisy web
    snippets. If Qwen predicts fraud but retrieval finds no similar fraud case,
    the safer dataset-consistent decision is normal unless the text itself has
    explicit scam signals such as transfer inducement or advance payment.
    """
    stats = retrieval_stats(retrieved)
    calibrated = pred_label
    note = ""
    if (
        pred_label == 1
        and stats["max_fraud_score"] < min_fraud_evidence
        and not has_strong_fraud_signal(sample_text)
    ):
        calibrated = 0
        note = (
            "RAG校准：模型原判诈骗，但检索结果中缺少足够相似的诈骗案例，"
            "且文本没有明显转账、垫付、刷单、投资等强诈骗信号，因此按数据集标签标准校准为正常。"
        )
    if note:
        explanation = (explanation + " " + note).strip()
    return calibrated, explanation, stats, note


def build_prompt(sample_text, retrieved):
    system_prompt = (
        "你是中文网络聊天诈骗识别助手。请根据待检测文本以及检索到的相似案例，"
        "判断文本是否属于诈骗。标签1表示诈骗，标签0表示正常。只返回JSON。"
        "注意：不要仅因为出现电话、QQ、微信、广告语、网页版权信息或营销文本就判为诈骗；"
        "只有存在明确的诈骗意图、转账/垫付/保证金/刷单返佣/虚假投资/冒充客服/贷款引流等风险行为，"
        "或检索到高度相似的诈骗案例时，才判为诈骗。证据不足时应判为正常。"
    )
    cases = []
    for i, item in enumerate(retrieved, start=1):
        label_name = "诈骗" if item["label"] == 1 else "正常"
        cases.append(
            f"案例{i}: label={item['label']}({label_name}), similarity={item['score']:.4f}, text={sanitize_for_api(item['text'])}"
        )
    user_prompt = (
        "判定规则：\n"
        "1. 若相似案例主要为正常样本，且待检测文本没有明确诱导转账、垫付、投资、刷单、贷款或冒充身份行为，应判为0。\n"
        "2. 联系方式、广告推广、网页片段、普通咨询不能单独作为诈骗证据。\n"
        "3. 若判断为1，解释中必须指出具体诈骗行为证据，而不是泛泛说“可疑”。\n"
        "输出格式必须为："
        '{"label": 0或1, "conclusion": "normal或fraud", "explanation": "结合检索案例给出简短理由"}\n'
        "[检索到的相似案例]\n"
        + "\n".join(cases)
        + "\n[待检测文本]\n"
        + sanitize_for_api(sample_text)
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


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
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error = None
    for _ in range(max_retries):
        request = urllib.request.Request(
            url=base_url.rstrip("/") + "/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}"
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            last_error = repr(exc)
        time.sleep(retry_sleep)
    raise RuntimeError(last_error)


def parse_label(raw_text):
    raw_text = raw_text.strip()
    try:
        data = json.loads(raw_text)
        return int(data["label"]), str(data.get("explanation", ""))
    except Exception:
        lowered = raw_text.lower()
        if '"label": 1' in lowered or '"label":1' in lowered:
            return 1, raw_text
        if '"label": 0' in lowered or '"label":0' in lowered:
            return 0, raw_text
    raise ValueError(f"Unable to parse label from response: {raw_text}")


def stable_id(text):
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


def evaluate(args):
    eval_rows = load_tsv(args.eval_path)
    corpus_rows = []
    for path in args.rag_tsv:
        corpus_rows.extend(load_tsv(path))
    for path in args.rag_csv:
        corpus_rows.extend(load_csv_corpus(path, args.max_csv_rows, args.seed))

    if not corpus_rows:
        raise ValueError("RAG corpus is empty.")

    retriever = RagRetriever(corpus_rows, args.max_doc_chars)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{args.run_name}.predictions.jsonl"
    summary_path = output_dir / f"{args.run_name}.summary.json"

    y_true, y_pred, failures = [], [], []
    with pred_path.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(eval_rows, start=1):
            retrieved = retriever.search(shorten(row["text"], args.max_query_chars), args.top_k)
            try:
                raw_response = call_qwen(
                    args.api_key,
                    args.model,
                    build_prompt(row["text"], retrieved),
                    args.base_url,
                    args.temperature,
                    args.max_tokens,
                    args.disable_thinking,
                    args.max_retries,
                    args.retry_sleep,
                )
                raw_pred_label, explanation = parse_label(raw_response)
                pred_label = raw_pred_label
                stats = retrieval_stats(retrieved)
                calibration_note = ""
                if args.calibrate:
                    pred_label, explanation, stats, calibration_note = calibrate_prediction(
                        raw_pred_label,
                        explanation,
                        row["text"],
                        retrieved,
                        args.min_fraud_evidence,
                    )
                y_true.append(row["label"])
                y_pred.append(pred_label)
                status = "ok"
            except Exception as exc:
                raw_response = repr(exc)
                raw_pred_label = None
                pred_label = None
                explanation = ""
                stats = retrieval_stats(retrieved)
                calibration_note = ""
                status = "failed"
                failures.append({"index": idx, "error": raw_response, "id": stable_id(row["text"])})

            fout.write(json.dumps({
                "index": idx,
                "status": status,
                "gold": row["label"],
                "raw_pred": raw_pred_label,
                "pred": pred_label,
                "text": row["text"],
                "retrieved": retrieved,
                "retrieval_stats": stats,
                "calibration_note": calibration_note,
                "raw_response": raw_response,
                "explanation": explanation,
            }, ensure_ascii=False) + "\n")

            if idx % args.log_every == 0 or idx == len(eval_rows):
                print(f"[{idx}/{len(eval_rows)}] processed, valid={len(y_true)}, failed={len(failures)}")

    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["normal", "fraud"], zero_division=0, output_dict=True
    ) if y_true else {
        "normal": {"precision": 0, "recall": 0, "f1-score": 0},
        "fraud": {"precision": 0, "recall": 0, "f1-score": 0},
        "macro avg": {"f1-score": 0},
        "weighted avg": {"f1-score": 0},
    }
    summary = {
        "model": args.model,
        "mode": "rag",
        "eval_path": args.eval_path,
        "count_total": len(eval_rows),
        "count_valid": len(y_true),
        "count_failed": len(failures),
        "accuracy": accuracy_score(y_true, y_pred) if y_true else 0,
        "fraud_precision": report["fraud"]["precision"],
        "fraud_recall": report["fraud"]["recall"],
        "fraud_f1": report["fraud"]["f1-score"],
        "normal_precision": report["normal"]["precision"],
        "normal_recall": report["normal"]["recall"],
        "normal_f1": report["normal"]["f1-score"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist() if y_true else [[0, 0], [0, 0]],
        "top_k": args.top_k,
        "calibrate": args.calibrate,
        "min_fraud_evidence": args.min_fraud_evidence,
        "rag_corpus_size": len(corpus_rows),
        "failures": failures,
        "prediction_file": str(pred_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen API with local TF-IDF RAG.")
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--rag-tsv", action="append", default=[])
    parser.add_argument("--rag-csv", action="append", default=[])
    parser.add_argument("--output-dir", default="./result_qwen")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-csv-rows", type=int, default=3000)
    parser.add_argument("--max-doc-chars", type=int, default=400)
    parser.add_argument("--max-query-chars", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-fraud-evidence", type=float, default=0.01)
    args = parser.parse_args()
    if not args.api_key:
        raise ValueError("Please provide --api-key or set DASHSCOPE_API_KEY/QWEN_API_KEY.")
    evaluate(args)


if __name__ == "__main__":
    main()
