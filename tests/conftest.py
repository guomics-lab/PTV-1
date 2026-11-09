"""Fixtures that prepare a small slice of the downloaded dataset."""

import csv
import re
import shutil
from pathlib import Path

import pytest

PREFIX = "allcelltype_drugpair_crossdrug_"
ROW_FILES = (
    "expr.csv",
    "drug_fp_phychem_A.csv",
    "drug_fp_phychem_B.csv",
    "loo_label.csv",
    "pert.csv",
    "pheno.csv",
)
INDEX_FILE = "node_Index.csv"
REQUIRED_TIMES = {6, 24, 48}
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def _read_rows(path: Path) -> list[list[str]]:
    with path.open(newline="") as handle:
        return list(csv.reader(handle))


def _write_rows(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)


def _complete_experiments(labels: list[list[str]]) -> list[str]:
    times_by_id: dict[str, set[int]] = {}
    for row in labels:
        experiment_id = row[0]
        if "#" not in experiment_id:
            continue
        times_by_id.setdefault(experiment_id, set()).add(int(float(row[1])))
    return [
        experiment_id
        for experiment_id, times in times_by_id.items()
        if REQUIRED_TIMES <= times
    ]


def _row_indexes(labels: list[list[str]], experiment_ids: list[str]) -> list[int]:
    chosen = set(experiment_ids)
    patterns = [re.escape(experiment_id.split("_")[0]) for experiment_id in chosen]
    indexes: list[int] = []
    for index, row in enumerate(labels):
        experiment_id = row[0]
        time_point = int(float(row[1]))
        matches_baseline = time_point == 0 and any(
            re.search(pattern, experiment_id) for pattern in patterns
        )
        if experiment_id in chosen or matches_baseline:
            indexes.append(index)
    return indexes


def _write_subset(output_dir: Path, experiment_count: int) -> Path:
    label_path = RAW / f"{PREFIX}loo_label.csv"
    if not label_path.exists():
        pytest.skip("Download the dataset with scripts/download_data.py first")
    labels = _read_rows(label_path)
    experiment_ids = _complete_experiments(labels)[:experiment_count]
    if len(experiment_ids) < experiment_count:
        raise ValueError(
            f"Need {experiment_count} complete experiments, found {len(experiment_ids)}"
        )
    indexes = _row_indexes(labels, experiment_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ROW_FILES:
        rows = _read_rows(RAW / f"{PREFIX}{name}")
        _write_rows(output_dir / f"{PREFIX}{name}", [rows[index] for index in indexes])
    shutil.copyfile(RAW / f"{PREFIX}{INDEX_FILE}", output_dir / f"{PREFIX}{INDEX_FILE}")
    return output_dir


@pytest.fixture(scope="session")
def file_prefix() -> str:
    return PREFIX


@pytest.fixture(scope="session")
def predict_input() -> Path:
    return _write_subset(ROOT / "tests" / "input" / "predict", experiment_count=2)


@pytest.fixture(scope="session")
def train_input() -> Path:
    return _write_subset(ROOT / "tests" / "input" / "train", experiment_count=8)
