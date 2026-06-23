import argparse
import json
import os
import sys
import re
import urllib.error
import urllib.request
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from models import Bert as BertModule  # noqa: E402
from models import Chinese_Bert as ChineseBertModule  # noqa: E402
from utils_chinesebert import convert_sentence_to_pinyin_ids  # noqa: E402


LABEL_MAP = {0: "normal", 1: "fraud"}
DEFAULT_QWEN_API_KEY = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
QWEN_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def resolve_path(*parts, required_file=None, fallback=None):
    path = PROJECT_ROOT.joinpath(*parts)
    if required_file:
        exists = (path / required_file).exists()
    else:
        exists = path.exists()
    if exists:
        return str(path)
    if fallback:
        return fallback
    raise FileNotFoundError(f"Path not found: {path}")


def softmax_probs(logits):
    probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()
    return {LABEL_MAP[i]: float(probs[i]) for i in range(len(probs))}


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
        self.config = BertModule.Config("ChiFraudDialogMatched2x", "random")
        self.config.class_list = ["0 正常", "1 诈骗"]
        self.config.num_classes = 2
        self.config.bert_path = resolve_path("pretrained", "bert-base-chinese", required_file="vocab.txt", fallback="bert-base-chinese")
        self.config.save_path = str(PROJECT_ROOT / "saved_dict" / "ChiFraudDialogMatched2x" / "Bert.ckpt")
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
        self.config = ChineseBertModule.Config("ChiFraudDialogMatched2x", "random")
        self.config.class_list = ["0 正常", "1 诈骗"]
        self.config.num_classes = 2
        self.config.bert_path = resolve_path("pretrained", "ChineseBERT-base", required_file="vocab.txt")
        self.config.save_path = str(PROJECT_ROOT / "saved_dict" / "ChiFraudDialogMatched2x" / "Chinese_Bert.ckpt")
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
        self.model = "qwen-plus"
        self.base_url = QWEN_API_BASE_URL

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
        raw = self._call_api(self._messages(text))
        label = parse_generated_label(raw)
        return {"model": "qwen_api", "label": label, "raw_response": raw}


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

