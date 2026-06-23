import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


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


def build_messages(sample_text, label):
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
        f"[Text]\n{sample_text}"
    )
    assistant = {
        "label": label,
        "conclusion": "fraud" if label == 1 else "normal",
        "explanation": "The text contains fraud or illegal trading signals." if label == 1
        else "The text is ordinary or benign content."
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)},
    ]


class FraudInstructionDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length, max_text_chars):
        self.examples = []
        for row in rows:
            sample_text = shorten_text(row["text"], max_text_chars)
            messages = build_messages(sample_text, row["label"])
            prompt_messages = messages[:-1]

            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

            full_enc = tokenizer(full_text, truncation=True, max_length=max_length, padding="max_length")
            prompt_enc = tokenizer(prompt_text, truncation=True, max_length=max_length, padding="max_length")

            input_ids = full_enc["input_ids"]
            attention_mask = full_enc["attention_mask"]
            prompt_len = sum(prompt_enc["attention_mask"])

            labels = input_ids.copy()
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100
            labels = [token if mask == 1 else -100 for token, mask in zip(labels, attention_mask)]

            self.examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


@dataclass
class SimpleCollator:
    def __call__(self, features):
        return {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for local Qwen fraud detection.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--dev-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-text-chars", type=int, default=600)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--fp16-train", action="store_true")
    parser.add_argument("--disable-intermediate-save", action="store_true")
    parser.add_argument("--disable-eval-during-train", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16 if (torch.cuda.is_available() and args.fp16_train) else torch.float32,
        local_files_only=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_rows = load_tsv(args.train_path)
    dev_rows = load_tsv(args.dev_path)
    train_dataset = FraudInstructionDataset(train_rows, tokenizer, args.max_length, args.max_text_chars)
    dev_dataset = FraudInstructionDataset(dev_rows, tokenizer, args.max_length, args.max_text_chars)

    evaluation_strategy = "no" if args.disable_eval_during_train else "steps"
    save_strategy = "no" if args.disable_intermediate_save else "steps"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        save_total_limit=2,
        fp16=torch.cuda.is_available() and args.fp16_train,
        bf16=False,
        report_to="none",
        dataloader_num_workers=0,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=SimpleCollator(),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
