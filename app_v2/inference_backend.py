import argparse
import csv
import json
import os
import sys
import re
import urllib.error
import urllib.request
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent
ASSET_ROOTS = (PROJECT_ROOT, PROJECT_ROOT.parent)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from models import Bert as BertModule  # noqa: E402
from models import Chinese_Bert as ChineseBertModule  # noqa: E402
from utils_chinesebert import convert_sentence_to_pinyin_ids  # noqa: E402


LABEL_MAP = {0: "normal", 1: "fraud"}
DEFAULT_QWEN_API_KEY = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
QWEN_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
RAG_CORPUS_LIMIT = 3000


def resolve_path(*parts, required_file=None, fallback=None):
    checked = []
    for root in ASSET_ROOTS:
        path = root.joinpath(*parts)
        checked.append(str(path))
        if required_file:
            exists = (path / required_file).exists()
        else:
            exists = path.exists()
        if exists:
            return str(path)
    if fallback:
        return fallback
    raise FileNotFoundError(f"Path not found. Checked: {', '.join(checked)}")


def softmax_probs(logits):
    probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()
    return {LABEL_MAP[i]: float(probs[i]) for i in range(len(probs))}


def clean_text(text):
    text = str(text).replace("\ufeff", "")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def has_strong_fraud_signal(text):
    patterns = [
        "刷单", "返佣", "垫付", "保证金", "转账", "打款", "加群", "进群",
        "投资", "理财", "提现", "账户冻结", "客服", "退款", "中奖", "贷款",
        "验资", "银行卡", "私聊", "兼职", "高回报", "稳赚", "名额有限",
    ]
    return any(pattern in text for pattern in patterns)


def load_tsv_rows(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0] in ("label", "label_id") or len(row) < 2:
                continue
            text = clean_text(row[1])
            if text:
                rows.append({"label": int(row[0]), "text": text, "source": path.name})
    return rows


def load_csv_rows(path, limit=RAG_CORPUS_LIMIT):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        text_field = "text" if "text" in fieldnames else (fieldnames[0] if fieldnames else "text")
        for row in reader:
            label = row.get("label", "")
            if label not in ("0", "1"):
                continue
            text = clean_text(row.get(text_field, ""))
            if text:
                rows.append({"label": int(label), "text": text[:400], "source": path.name})
    if limit > 0 and len(rows) > limit:
        # Deterministic balanced sample without importing numpy/random state.
        fraud = [row for row in rows if row["label"] == 1][: limit // 2]
        normal = [row for row in rows if row["label"] == 0][: limit - len(fraud)]
        rows = fraud + normal
    return rows


class RagRetriever:
    def __init__(self):
        rows = []
        train_path = Path(resolve_path("dataset", "dialogue_binary_refined", "train.tsv"))
        crawler_path = Path(resolve_path("dataset", "fraud_text_label.csv"))
        rows.extend(load_tsv_rows(train_path))
        rows.extend(load_csv_rows(crawler_path))
        if not rows:
            self.rows = []
            self.vectorizer = None
            self.matrix = None
            return
        self.rows = rows
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=60000)
        self.matrix = self.vectorizer.fit_transform([row["text"] for row in rows])

    def search(self, query, top_k=4):
        if not self.rows or self.vectorizer is None:
            return []
        q = self.vectorizer.transform([clean_text(query)[:600]])
        scores = cosine_similarity(q, self.matrix).ravel()
        top_idx = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_idx:
            row = self.rows[int(idx)]
            results.append({
                "label": row["label"],
                "text": row["text"][:400],
                "source": row.get("source", ""),
                "score": float(scores[int(idx)]),
            })
        return results


_RAG_RETRIEVER = None


def get_rag_retriever():
    global _RAG_RETRIEVER
    if _RAG_RETRIEVER is None:
        _RAG_RETRIEVER = RagRetriever()
    return _RAG_RETRIEVER


def sanitize_for_qwen_api(text):
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\d", "X", text)
    replacements = {
        "微信": "联系方式",
        "QQ": "联系方式",
        "qq": "联系方式",
        "vx": "联系方式",
        "VX": "联系方式",
        "Telegram": "联系方式",
        "电报": "联系方式",
        "银行卡": "金融工具",
        "信用卡": "金融工具",
        "假证": "伪造材料",
        "毕业证": "证明材料",
        "成绩单": "证明材料",
        "迷药": "违禁物品",
        "毒品": "违禁物品",
        "赌博": "博彩活动",
        "嫖": "违规服务",
        "裸聊": "灰产服务",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text[:1200]


def parse_generated_label(raw_text):
    raw_text = raw_text.strip()
    try:
        data = json.loads(raw_text)
        label = int(data["label"])
        return LABEL_MAP[label]
    except Exception:
        pass

    lowered = raw_text.lower()
    if '"label": 1' in lowered or '"label":1' in lowered:
        return "fraud"
    if '"label": 0' in lowered or '"label":0' in lowered:
        return "normal"
    if "fraud" in lowered and "normal" not in lowered:
        return "fraud"
    if "normal" in lowered and "fraud" not in lowered:
        return "normal"
    raise ValueError(f"Unable to parse label from Qwen response: {raw_text}")


class BertPredictor:
    def __init__(self):
        self.config = BertModule.Config("ChiFraudDialogRefined", "random")
        self.config.class_list = ["0 正常", "1 诈骗"]
        self.config.num_classes = 2
        self.config.bert_path = resolve_path("pretrained", "bert-base-chinese", required_file="vocab.txt", fallback="bert-base-chinese")
        self.config.save_path = resolve_path("saved_dict", "ChiFraudDialogRefined", "Bert.ckpt")
        self.config.reload_tokenizer()
        self.model = BertModule.Model(self.config).to(self.config.device)
        state = torch.load(self.config.save_path, map_location=self.config.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, text):
        encoded = self.config.tokenizer(
            text,
            max_length=self.config.pad_size,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.config.device)
        attention_mask = encoded["attention_mask"].to(self.config.device)
        with torch.no_grad():
            logits = self.model((input_ids, None, attention_mask))
        probs = softmax_probs(logits)
        pred = max(probs, key=probs.get)
        return {"model": "bert", "label": pred, "probabilities": probs}


class ChineseBertPredictor:
    def __init__(self):
        self.config = ChineseBertModule.Config("ChiFraudDialogRefined", "random")
        self.config.class_list = ["0 正常", "1 诈骗"]
        self.config.num_classes = 2
        self.config.bert_path = resolve_path("pretrained", "ChineseBERT-base", required_file="vocab.txt")
        self.config.save_path = resolve_path("saved_dict", "ChiFraudDialogRefined", "Chinese_Bert.ckpt")
        self.config.reload_tokenizer()
        self.model = ChineseBertModule.Model(self.config).to(self.config.device)
        state = torch.load(self.config.save_path, map_location=self.config.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def _build_inputs(self, text):
        content = text[: self.config.pad_size - 2]
        tokenizer_output = self.config.tokenizer.encode(content)
        bert_tokens = tokenizer_output.ids
        pinyin_tokens = convert_sentence_to_pinyin_ids(self.config, content, tokenizer_output)
        if len(bert_tokens) > self.config.pad_size:
            bert_tokens = bert_tokens[0 : self.config.pad_size - 1] + [bert_tokens[-1]]
            pinyin_tokens = pinyin_tokens[0 : self.config.pad_size - 1] + [pinyin_tokens[-1]]
        else:
            bert_tokens += [0] * (self.config.pad_size - len(bert_tokens))
            pinyin_tokens += [[0] * 8] * (self.config.pad_size - len(pinyin_tokens))
        input_ids = torch.tensor([bert_tokens], dtype=torch.long).to(self.config.device)
        pinyin_ids = torch.tensor([pinyin_tokens], dtype=torch.long).to(self.config.device)
        return ((input_ids, pinyin_ids), None, None)

    def predict(self, text):
        with torch.no_grad():
            logits = self.model(self._build_inputs(text))
        probs = softmax_probs(logits)
        pred = max(probs, key=probs.get)
        return {"model": "chinesebert", "label": pred, "probabilities": probs}


class QwenPredictor:
    def __init__(self):
        self.base_dir = resolve_path("pretrained", "Qwen2.5-0.5B-Instruct", required_file="config.json")
        self.adapter_dir = resolve_path("saved_dict", "ChiFraudDialogMatched2x", "Qwen0.5B_LoRA_run2", required_file="adapter_config.json")
        self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_dir, local_files_only=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_dir,
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        self.model = PeftModel.from_pretrained(base_model, self.adapter_dir, local_files_only=True)
        # Keep Qwen on CPU for the web demo to avoid GPU memory collisions after BERT/ChineseBERT.
        self.device = "cpu"
        self.model = self.model.to(self.device).eval()

    def _messages(self, text):
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
        user_prompt = (
            "Please complete a binary classification task.\n"
            "Labels: 0=normal, 1=fraud.\n"
            'Output JSON in exactly this format: {"label": 0 or 1, "conclusion": "normal or fraud", "explanation": "short reason"}\n'
            f"[Text]\n{text[:600]}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def predict(self, text):
        prompt = self.tokenizer.apply_chat_template(self._messages(text), tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        raw = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        label = "unknown"
        try:
            data = json.loads(raw)
            label = LABEL_MAP[int(data["label"])]
        except Exception:
            lowered = raw.lower()
            if '"label": 1' in lowered or "fraud" in lowered:
                label = "fraud"
            elif '"label": 0' in lowered or "normal" in lowered:
                label = "normal"
        return {"model": "qwen_lora", "label": label, "raw_response": raw}


class QwenApiPredictor:
    def __init__(self):
        self.api_key = DEFAULT_QWEN_API_KEY
        if not self.api_key:
            raise RuntimeError("Qwen API key is not configured. Please set DASHSCOPE_API_KEY or QWEN_API_KEY.")
        self.model = "qwen3.7-plus"
        self.base_url = QWEN_API_BASE_URL
        self.retriever = get_rag_retriever()

    def _retrieval_stats(self, retrieved):
        fraud_scores = [item["score"] for item in retrieved if item["label"] == 1]
        normal_scores = [item["score"] for item in retrieved if item["label"] == 0]
        return {
            "fraud_count": len(fraud_scores),
            "normal_count": len(normal_scores),
            "max_fraud_score": max(fraud_scores) if fraud_scores else 0.0,
            "max_normal_score": max(normal_scores) if normal_scores else 0.0,
        }

    def _messages(self, text, retrieved):
        system_prompt = (
            "你是中文网络聊天诈骗识别助手。请结合待检测文本和检索到的相似案例判断是否诈骗。"
            "标签1表示诈骗，标签0表示正常。不要仅因为出现电话、QQ、微信、广告语、网页版权信息或营销文本就判为诈骗；"
            "只有存在明确的转账、垫付、保证金、刷单返佣、虚假投资、冒充客服、贷款引流等风险行为，"
            "或检索到高度相似的诈骗案例时，才判为诈骗。证据不足时应判为正常。只返回JSON。"
        )
        cases = []
        for i, item in enumerate(retrieved, start=1):
            label_name = "诈骗" if item["label"] == 1 else "正常"
            cases.append(
                f"案例{i}: label={item['label']}({label_name}), similarity={item['score']:.4f}, text={sanitize_for_qwen_api(item['text'])}"
            )
        user_prompt = (
            "判定规则：\n"
            "1. 若相似案例主要为正常样本，且待检测文本没有明确诱导转账、垫付、投资、刷单、贷款或冒充身份行为，应判为0。\n"
            "2. 联系方式、广告推广、网页片段、普通咨询不能单独作为诈骗证据。\n"
            "3. 若判断为1，解释中必须指出具体诈骗行为证据。\n"
            '输出JSON格式：{"label": 0或1, "conclusion": "normal或fraud", "explanation": "简短理由"}\n'
            "[检索到的相似案例]\n"
            + "\n".join(cases)
            + "\n[待检测文本]\n"
            f"[Text]\n{sanitize_for_qwen_api(text)}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _call_api(self, messages):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 128,
            "enable_thinking": False,
            "extra_body": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            url=self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Qwen API HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Qwen API request failed: {exc}") from exc

    def predict(self, text):
        retrieved = self.retriever.search(text, top_k=4)
        raw = self._call_api(self._messages(text, retrieved))
        label = parse_generated_label(raw)
        stats = self._retrieval_stats(retrieved)
        calibration_note = ""
        if (
            label == "fraud"
            and stats["max_fraud_score"] < 0.01
            and not has_strong_fraud_signal(text)
        ):
            label = "normal"
            calibration_note = "RAG校准：缺少相似诈骗案例且无强诈骗信号，校准为正常。"
        return {
            "model": "qwen_api",
            "label": label,
            "raw_response": raw,
            "retrieved": retrieved,
            "retrieval_stats": stats,
            "calibration_note": calibration_note,
        }


def build_predictor(model_name):
    if model_name == "bert":
        return BertPredictor()
    if model_name == "chinesebert":
        return ChineseBertPredictor()
    if model_name == "qwen_api":
        return QwenApiPredictor()
    if model_name in ("qwen", "qwen_lora"):
        return QwenPredictor()
    raise ValueError(f"Unsupported model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Local inference entry for web app.")
    parser.add_argument("--model", choices=["bert", "chinesebert", "qwen", "qwen_lora", "qwen_api"], required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    predictor = build_predictor(args.model)
    result = predictor.predict(args.text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

