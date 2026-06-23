import argparse
import json
import mimetypes
import time
import traceback
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

import torch

from inference_backend import build_predictor


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"

MODEL_META = {
    "bert": {
        "name": "BERT",
        "description": "速度快，适合作为本地快速检测模型。",
        "best_result": "matched_2x 测试集 Acc 91.67%，诈骗类 F1 83.87%",
    },
    "chinesebert": {
        "name": "ChineseBERT",
        "description": "融合字形和拼音特征，目前诈骗类识别效果最好。",
        "best_result": "matched_2x 测试集 Acc 95.00%，诈骗类 F1 90.00%",
    },
    "qwen_api": {
        "name": "Qwen API",
        "description": "云端千问接口，不占用本机显存，适合网页演示和临时测试。",
        "best_result": "API 在线推理，结果受 prompt 和云端模型版本影响。",
    },
}

LABEL_TEXT = {
    0: "正常",
    1: "诈骗",
    "0": "正常",
    "1": "诈骗",
    "normal": "正常",
    "fraud": "诈骗",
}


class PredictorRegistry:
    def __init__(self):
        self._predictor = None
        self._model_key = None
        self._lock = Lock()

    def get(self, model_key):
        if model_key not in MODEL_META:
            raise ValueError(f"unsupported model: {model_key}")
        with self._lock:
            if self._model_key != model_key:
                if self._predictor is not None:
                    print(f"[model-load] releasing {self._model_key} ...", flush=True)
                    del self._predictor
                    self._predictor = None
                    self._model_key = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("[model-load] previous model released", flush=True)
                print(f"[model-load] loading {model_key} ...", flush=True)
                self._predictor = build_predictor(model_key)
                self._model_key = model_key
                print(f"[model-load] {model_key} loaded", flush=True)
            return self._predictor


REGISTRY = PredictorRegistry()


def json_bytes(payload):
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def normalize_messages(payload):
    if isinstance(payload.get("text"), str) and payload["text"].strip():
        return payload["text"].strip()

    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    lines = []
    for index, message in enumerate(messages, 1):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", f"消息{index}")).strip() or f"消息{index}"
        text = str(message.get("text", "")).strip()
        if text:
            lines.append(f"{role}: {text}")

    combined = "\n".join(lines).strip()
    if not combined:
        raise ValueError("请输入待检测的聊天内容")
    return combined


def normalize_prediction(model_key, text, raw_result, elapsed_ms):
    label = raw_result.get("label")
    if isinstance(label, str) and label.isdigit():
        label = int(label)

    probabilities = raw_result.get("probabilities") or {}
    fraud_score = probabilities.get("fraud") if probabilities else None
    normal_score = probabilities.get("normal") if probabilities else None

    is_generation_model = model_key == "qwen_api"
    if fraud_score is None:
        fraud_score = 1.0 if label in (1, "fraud", "诈骗") else 0.0
    if normal_score is None:
        normal_score = 1.0 - fraud_score

    risk_label = "fraud" if label in (1, "fraud", "诈骗") else "normal"
    return {
        "model": model_key,
        "model_name": MODEL_META[model_key]["name"],
        "input_chars": len(text),
        "label": risk_label,
        "label_text": LABEL_TEXT.get(label, "诈骗" if risk_label == "fraud" else "正常"),
        "risk_score": round(float(fraud_score), 6),
        "probabilities": {
            "normal": round(float(normal_score), 6),
            "fraud": round(float(fraud_score), 6),
        },
        "probability_note": "generation_label_only" if is_generation_model else "softmax_probability",
        "raw": raw_result,
        "elapsed_ms": elapsed_ms,
    }


class FraudHandler(SimpleHTTPRequestHandler):
    server_version = "SmartShieldFraud/1.0"

    def log_message(self, format, *args):
        print("[%s] %s" % (self.log_date_time_string(), format % args))

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            return self.send_json({"ok": True, "service": "Smart Shield fraud inference"})
        if parsed.path == "/api/models":
            return self.send_json({"ok": True, "models": MODEL_META})
        return self.serve_static(parsed.path)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            return self.send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            model_key = str(payload.get("model", "chinesebert")).strip().lower()
            text = normalize_messages(payload)

            start = time.perf_counter()
            predictor = REGISTRY.get(model_key)
            print(f"[predict] model={model_key}, chars={len(text)}", flush=True)
            raw_result = predictor.predict(text)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            print(f"[predict] model={model_key}, elapsed_ms={elapsed_ms}", flush=True)

            result = normalize_prediction(model_key, text, raw_result, elapsed_ms)
            self.send_json({"ok": True, "result": result})
        except Exception as exc:
            traceback.print_exc()
            self.send_json(
                {
                    "ok": False,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                    "model": payload.get("model") if "payload" in locals() else None,
                },
                HTTPStatus.BAD_REQUEST,
            )

    def serve_static(self, request_path):
        if request_path in ("", "/"):
            target = WEB_ROOT / "index.html"
        else:
            safe_parts = [part for part in request_path.split("/") if part and part not in ("..", ".")]
            target = WEB_ROOT.joinpath(*safe_parts)

        try:
            target = target.resolve()
            if not str(target).startswith(str(WEB_ROOT.resolve())) or not target.is_file():
                self.send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
                return

            mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            body = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{mime}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def send_json(self, payload, status=HTTPStatus.OK):
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="Smart Shield local web inference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7871, type=int)
    args = parser.parse_args()

    WEB_ROOT.mkdir(exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), FraudHandler)
    print(f"Smart Shield web app is running at http://{args.host}:{args.port}")
    print("Models are loaded lazily on first prediction.")
    server.serve_forever()


if __name__ == "__main__":
    main()
