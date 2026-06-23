import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
BASE_DIR = ROOT / "pretrained" / "Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = ROOT / "saved_dict" / "ChiFraudDialogMatched2x" / "Qwen0.5B_LoRA_run2"


def log(message):
    print(message, flush=True)


def main():
    log(f"base_dir={BASE_DIR}")
    log(f"adapter_dir={ADAPTER_DIR}")
    log(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")

    log("[1/5] loading tokenizer from adapter...")
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR), local_files_only=True)
    log(f"[ok] tokenizer loaded, vocab_size={len(tokenizer)}")

    log("[2/5] loading base model on CPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(BASE_DIR),
        torch_dtype=torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=False,
    )
    log("[ok] base model loaded")

    log("[3/5] resizing token embeddings if needed...")
    old_size = base_model.get_input_embeddings().weight.shape[0]
    if old_size != len(tokenizer):
        log(f"[info] resize embeddings: {old_size} -> {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))
    else:
        log("[info] no resize needed")

    log("[4/5] loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR), local_files_only=True)
    model = model.to("cpu").eval()
    log("[ok] adapter loaded")

    log("[5/5] generating one token...")
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "Labels: 0=normal, 1=fraud.\n[Text]\n办理假证加微信"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([prompt], return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    log(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    log("[done]")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"[python-error] {exc.__class__.__name__}: {exc}")
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
