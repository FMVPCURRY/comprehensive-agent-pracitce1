import os
import subprocess
import sys
from pathlib import Path


def run_and_log(command, stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        stdout.write(f"$ {' '.join(command)}\n")
        stdout.flush()
        process = subprocess.run(command, stdout=stdout, stderr=stderr, cwd=ROOT, check=True)
        stdout.write(f"exit_code={process.returncode}\n")
        stdout.flush()


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
LOG_DIR = ROOT / "logs"


def main() -> None:
    run_and_log(
        [PYTHON, "prepare_dialogue_dataset.py"],
        LOG_DIR / "prepare.log",
        LOG_DIR / "prepare.err.log",
    )
    run_and_log(
        [
            PYTHON,
            "run.py",
            "--model",
            "Bert",
            "--dataset-name",
            "ChiFraudDialog",
            "--num-epochs",
            "3",
            "--batch-size",
            "16",
            "--pad-size",
            "256",
            "--learning-rate",
            "5e-5",
        ],
        LOG_DIR / "bert_train.log",
        LOG_DIR / "bert_train.err.log",
    )
    run_and_log(
        [
            PYTHON,
            "run.py",
            "--model",
            "Chinese_Bert",
            "--dataset-name",
            "ChiFraudDialog",
            "--bert-path",
            "./pretrained/ChineseBERT-base",
            "--num-epochs",
            "3",
            "--batch-size",
            "8",
            "--pad-size",
            "256",
            "--learning-rate",
            "5e-5",
        ],
        LOG_DIR / "chinesebert_train.log",
        LOG_DIR / "chinesebert_train.err.log",
    )


if __name__ == "__main__":
    main()
