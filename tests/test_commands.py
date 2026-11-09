"""Run the documented ProteinTalks commands on a small data subset."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "output"


def _run(args: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["uv", "run", "ProteinTalks", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    log_path.write_text(completed.stdout + "\n" + completed.stderr)
    return completed


def test_help() -> None:
    completed = _run(["--help"], OUTPUT / "help.txt")
    assert completed.returncode == 0
    assert "--dataset_file_dir" in completed.stdout


def test_predict_subset(predict_input: Path, file_prefix: str) -> None:
    output_dir = OUTPUT / "predict"
    completed = _run(
        [
            "--train_from_scratch",
            "predict",
            "--cp_save_dir_best",
            "ckpts/best_checkpoint.pth",
            "--dataset_file_dir",
            str(predict_input),
            "--test_file_prefix",
            file_prefix,
            "--hidden_size",
            "64",
            "--dropout_rate",
            "0",
            "--time_stamp_predict_drug",
            "6_24_48",
            "--dir_save",
            str(output_dir),
        ],
        OUTPUT / "predict.log",
    )
    assert completed.returncode == 0, completed.stderr
    assert list(output_dir.glob("**/predictions.csv"))


def test_train_subset(train_input: Path, file_prefix: str) -> None:
    output_dir = OUTPUT / "train"
    completed = _run(
        [
            "--dataset_file_dir",
            str(train_input),
            "--trainval_file_prefix",
            file_prefix,
            "--time_stamp_predict_drug",
            "6_24_48",
            "--train_percent",
            "0.7",
            "--val_percent",
            "0.2",
            "--test_percent",
            "0.1",
            "--batch_size",
            "4",
            "--patience",
            "1",
            "--total_epoch",
            "1",
            "--learning_rate",
            "0.0005",
            "--lambda_pheno",
            "0.8",
            "--dir_save",
            str(output_dir),
        ],
        OUTPUT / "train.log",
    )
    assert completed.returncode == 0, completed.stderr
    assert list(output_dir.glob("**/*checkpoint*"))
