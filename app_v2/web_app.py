import argparse
import base64
import binascii
import hashlib
import io
import json
import mimetypes
import os
import secrets
import sqlite3
import time
import traceback
import uuid
import html
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

import torch

from inference_backend import build_predictor


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
DB_DIR = ROOT / "data"
DB_PATH = DB_DIR / "app.db"
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60


def bytes_to_hex(value):
    return binascii.hexlify(value).decode("ascii")


def hex_to_bytes(value):
    return binascii.unhexlify(value.encode("ascii"))


def get_db_connection():
    DB_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_ts INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_ts INTEGER NOT NULL,
                expires_ts INTEGER NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_ts INTEGER NOT NULL,
                model TEXT NOT NULL,
                label TEXT NOT NULL,
                label_text TEXT NOT NULL,
                risk_score REAL NOT NULL,
                input_chars INTEGER NOT NULL,
                elapsed_ms INTEGER NOT NULL,
                preview TEXT,
                input_text TEXT,
                raw_json TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS captcha_challenges (
                token TEXT PRIMARY KEY,
                code TEXT NOT NULL,
                created_ts INTEGER NOT NULL,
                expires_ts INTEGER NOT NULL
            )
            """
        )
        existing_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()
        }
        if "input_text" not in existing_columns:
            conn.execute("ALTER TABLE predictions ADD COLUMN input_text TEXT")


def make_captcha_code(length=5):
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def captcha_svg(code):
    safe = html.escape(code)
    letters = "".join(
        f'<text x="{22 + i * 24}" y="36" transform="rotate({secrets.choice([-10, -6, 0, 7, 11])} {22 + i * 24} 36)" '
        f'font-size="24" font-family="Verdana" font-weight="800" fill="{secrets.choice(["#0f172a", "#1d4ed8", "#0f766e", "#b91c1c"])}">{html.escape(ch)}</text>'
        for i, ch in enumerate(safe)
    )
    noise = "".join(
        f'<line x1="{secrets.randbelow(150)}" y1="{secrets.randbelow(54)}" x2="{secrets.randbelow(150)}" y2="{secrets.randbelow(54)}" stroke="#94a3b8" stroke-width="1" opacity="0.45" />'
        for _ in range(7)
    )
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="150" height="52" viewBox="0 0 150 52"><rect width="150" height="52" rx="14" fill="#f8fafc"/><path d="M8 39 C38 12, 72 52, 142 17" fill="none" stroke="#bfdbfe" stroke-width="5" opacity=".8"/>{noise}{letters}</svg>'


def create_captcha():
    token = uuid.uuid4().hex
    code = make_captcha_code()
    now = int(time.time())
    with get_db_connection() as conn:
        conn.execute("DELETE FROM captcha_challenges WHERE expires_ts < ?", (now,))
        conn.execute(
            "INSERT INTO captcha_challenges (token, code, created_ts, expires_ts) VALUES (?, ?, ?, ?)",
            (token, code.lower(), now, now + 300),
        )
    return token, captcha_svg(code)


def verify_captcha(token, answer):
    token = str(token or "").strip()
    answer = str(answer or "").strip().lower()
    if not token or not answer:
        return False
    now = int(time.time())
    with get_db_connection() as conn:
        row = conn.execute("SELECT code, expires_ts FROM captcha_challenges WHERE token = ?", (token,)).fetchone()
        conn.execute("DELETE FROM captcha_challenges WHERE token = ? OR expires_ts < ?", (token, now))
        return bool(row and row["expires_ts"] >= now and secrets.compare_digest(row["code"], answer))

def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_bytes(16)
    else:
        salt = hex_to_bytes(salt)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200000)
    return bytes_to_hex(salt), bytes_to_hex(digest)


def verify_password(salt_hex, password_hash_hex, password):
    _, expected = hash_password(password, salt_hex)
    return secrets.compare_digest(expected, password_hash_hex)


def create_user(username, password):
    username = str(username or "").strip()
    if not username:
        raise ValueError("用户名不能为空")
    if not password:
        raise ValueError("密码不能为空")
    with get_db_connection() as conn:
        salt, password_hash = hash_password(password)
        created_ts = int(time.time())
        try:
            conn.execute(
                "INSERT INTO users (username, salt, password_hash, created_ts) VALUES (?, ?, ?, ?)",
                (username, salt, password_hash, created_ts),
            )
        except sqlite3.IntegrityError:
            raise ValueError("用户名已存在")
        row = conn.execute("SELECT id, username, created_ts FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


def find_user_by_username(username):
    if not username:
        return None
    with get_db_connection() as conn:
        row = conn.execute("SELECT id, username, salt, password_hash FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


def get_user_by_session(token):
    if not token:
        return None
    now = int(time.time())
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT s.token, s.user_id, s.expires_ts, u.username FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token = ?", 
            (token,),
        ).fetchone()
        if not row:
            return None
        if row["expires_ts"] < now:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            return None
        return {"id": row["user_id"], "username": row["username"]}


def create_session_token(user_id):
    token = uuid.uuid4().hex
    now = int(time.time())
    expires_ts = now + SESSION_TTL_SECONDS
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_ts, expires_ts) VALUES (?, ?, ?, ?)",
            (token, user_id, now, expires_ts),
        )
    return token


def clear_session_token(token):
    if not token:
        return
    with get_db_connection() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def change_user_password(user_id, old_password, new_password):
    if not old_password:
        raise ValueError("??????")
    if not new_password or len(new_password) < 6:
        raise ValueError("??????? 6 ?")
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT salt, password_hash FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if not row or not verify_password(row["salt"], row["password_hash"], old_password):
            raise ValueError("?????")
        salt, password_hash = hash_password(new_password)
        conn.execute(
            "UPDATE users SET salt = ?, password_hash = ? WHERE id = ?",
            (salt, password_hash, user_id),
        )


def extract_text_from_image_data(image_data):
    if not image_data:
        raise ValueError("??????")
    image_data = str(image_data)
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    try:
        raw_bytes = base64.b64decode(image_data, validate=False)
    except Exception:
        raise ValueError("?????????")

    try:
        from PIL import Image
        import pytesseract
    except Exception:
        raise RuntimeError("??????? OCR???? pillow?pytesseract???? Tesseract OCR ?????????")

    image = Image.open(io.BytesIO(raw_bytes))
    text = pytesseract.image_to_string(image, lang="chi_sim+eng")
    text = text.strip()
    if not text:
        raise ValueError("?????????????????????????????")
    return text


def save_prediction_record(user_id, model_key, result, preview_text, input_text):
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO predictions (user_id, created_ts, model, label, label_text, risk_score, input_chars, elapsed_ms, preview, input_text, raw_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                int(time.time()),
                model_key,
                result["label"],
                result["label_text"],
                float(result["risk_score"]),
                int(result["input_chars"]),
                int(result["elapsed_ms"]),
                preview_text,
                input_text,
                json.dumps(result.get("raw", {}), ensure_ascii=False),
            ),
        )


def get_user_history(user_id, limit=20):
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT id, created_ts, model, label, label_text, risk_score, input_chars, elapsed_ms, preview, input_text, raw_json FROM predictions WHERE user_id = ? ORDER BY created_ts DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]


def parse_cookies(header_value):
    cookies = {}
    if not header_value:
        return cookies
    for item in header_value.split(";"):
        if "=" in item:
            name, value = item.split("=", 1)
            cookies[name.strip()] = value.strip()
    return cookies


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

    def get_current_user(self):
        cookies = parse_cookies(self.headers.get("Cookie", ""))
        token = cookies.get("session")
        return get_user_by_session(token)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            return self.send_json({"ok": True, "service": "Smart Shield fraud inference"})
        if parsed.path == "/api/models":
            return self.send_json({"ok": True, "models": MODEL_META})
        if parsed.path == "/api/me":
            user = self.get_current_user()
            return self.send_json({"ok": True, "user": user})
        if parsed.path == "/api/history":
            user = self.get_current_user()
            if not user:
                return self.send_json({"ok": False, "error": "unauthenticated"}, HTTPStatus.UNAUTHORIZED)
            history = get_user_history(user["id"], limit=20)
            return self.send_json({"ok": True, "history": history})
        if parsed.path == "/api/captcha":
            token, svg = create_captcha()
            return self.send_json({"ok": True, "token": token, "svg": svg})
        if parsed.path == "/login":
            return self.serve_static("/login.html")
        if parsed.path == "/register":
            return self.serve_static("/register.html")
        return self.serve_static(parsed.path)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/history":
            return self.send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)

        user = self.get_current_user()
        if not user:
            return self.send_json({"ok": False, "error": "unauthenticated"}, HTTPStatus.UNAUTHORIZED)

        with get_db_connection() as conn:
            conn.execute("DELETE FROM predictions WHERE user_id = ?", (user["id"],))
        return self.send_json({"ok": True})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/login":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                username = str(payload.get("username", "")).strip()
                password = str(payload.get("password", ""))
                user = find_user_by_username(username)
                if not user or not verify_password(user["salt"], user["password_hash"], password):
                    raise ValueError("用户名或密码错误")
                token = create_session_token(user["id"])
                return self.send_json(
                    {"ok": True, "user": {"id": user["id"], "username": user["username"]}},
                    HTTPStatus.OK,
                    cookies=[f"session={token}; Path=/; HttpOnly; Max-Age={SESSION_TTL_SECONDS}"]
                )
            except Exception as exc:
                traceback.print_exc()
                return self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

        if parsed.path == "/api/register":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                username = str(payload.get("username", "")).strip()
                password = str(payload.get("password", ""))
                captcha = str(payload.get("captcha", ""))
                captcha_token = str(payload.get("captcha_token", ""))
                if not verify_captcha(captcha_token, captcha):
                    raise ValueError("验证码错误或已过期")
                user = create_user(username, password)
                token = create_session_token(user["id"])
                return self.send_json(
                    {"ok": True, "user": {"id": user["id"], "username": user["username"]}},
                    HTTPStatus.CREATED,
                    cookies=[f"session={token}; Path=/; HttpOnly; Max-Age={SESSION_TTL_SECONDS}"]
                )
            except Exception as exc:
                traceback.print_exc()
                return self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

        if parsed.path == "/api/logout":
            cookies = parse_cookies(self.headers.get("Cookie", ""))
            clear_session_token(cookies.get("session"))
            return self.send_json(
                {"ok": True},
                HTTPStatus.OK,
                cookies=["session=; Path=/; HttpOnly; Max-Age=0"]
            )

        if parsed.path == "/api/change-password":
            user = self.get_current_user()
            if not user:
                return self.send_json({"ok": False, "error": "unauthenticated"}, HTTPStatus.UNAUTHORIZED)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                change_user_password(user["id"], str(payload.get("old_password", "")), str(payload.get("new_password", "")))
                return self.send_json({"ok": True})
            except Exception as exc:
                traceback.print_exc()
                return self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

        if parsed.path == "/api/ocr":
            user = self.get_current_user()
            if not user:
                return self.send_json({"ok": False, "error": "unauthenticated"}, HTTPStatus.UNAUTHORIZED)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                text = extract_text_from_image_data(payload.get("image"))
                return self.send_json({"ok": True, "text": text})
            except Exception as exc:
                traceback.print_exc()
                return self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

        if parsed.path != "/api/predict":
            return self.send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)

        user = self.get_current_user()
        if not user:
            return self.send_json({"ok": False, "error": "unauthenticated"}, HTTPStatus.UNAUTHORIZED)

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
            save_prediction_record(user["id"], model_key, result, text[:120], text)
            return self.send_json({"ok": True, "result": result})
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

    def send_json(self, payload, status=HTTPStatus.OK, cookies=None):
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if cookies:
            for cookie in cookies:
                self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="Smart Shield local web inference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7871, type=int)
    args = parser.parse_args()

    WEB_ROOT.mkdir(exist_ok=True)
    init_db()
    server = ThreadingHTTPServer((args.host, args.port), FraudHandler)
    print(f"Smart Shield web app is running at http://{args.host}:{args.port}")
    print("Models are loaded lazily on first prediction.")
    server.serve_forever()


if __name__ == "__main__":
    main()

