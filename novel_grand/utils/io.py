from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(path: str | Path, rows: Iterable[Mapping], gzip_compress: bool = False) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    opener = gzip.open if gzip_compress else open
    mode = "wt"
    with opener(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, rows: Iterable[Mapping], gzip_compress: bool = False) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    opener = gzip.open if gzip_compress else open
    mode = "at"
    with opener(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_csv(path: str | Path, rows: Sequence[Mapping]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def dataframe_to_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
