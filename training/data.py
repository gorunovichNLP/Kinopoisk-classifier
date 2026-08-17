"""Load, clean, snapshot, and split the Kinopoisk review corpus."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from sklearn.model_selection import train_test_split

from kinopoisk_classifier.shared.contracts import LABEL_MAP


def _read_one(args):
    fp, cls = args
    text = fp.read_text(encoding="utf-8", errors="replace").strip()
    return {"text": text, "label": cls, "label_id": LABEL_MAP[cls]}


def _save_snapshot(df: pd.DataFrame, out_path: str) -> None:
    """Persist a reproducible dataset snapshot."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"[snapshot] {len(df)} rows -> {out}")


def load_reviews(dataset_dir: str, max_workers: int = 16) -> pd.DataFrame:
    """Load all review files into a dataframe."""

    base = Path(dataset_dir)
    out = Path("data/raw_dataset.csv")
    if out.exists():
        print(f"[load] cache hit -> {out}")
        return pd.read_csv(out)

    tasks = [(fp, cls) for cls in LABEL_MAP for fp in (base / cls).glob("*.txt")]
    if not tasks:
        raise FileNotFoundError(
            f"No .txt files found in {base}/[neg,neu,pos]. Check the path."
        )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        rows = list(ex.map(_read_one, tasks))
    df = pd.DataFrame(rows)

    _save_snapshot(df, out)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove blank and duplicate review texts."""
    before = len(df)
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(
        f"[clean] removed {before - len(df)} blank/duplicate rows, "
        f"{len(df)} remain"
    )

    _save_snapshot(df, "data/clean_dataset.csv")
    return df


def split(df: pd.DataFrame, seed: int = 42):
    """Create stratified train, validation, and test splits."""
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label_id"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=seed
    )
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = part["label"].value_counts(normalize=True).round(3).to_dict()
        print(f"[split] {name}: {len(part):>6} | {dist}")
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def load_clean_split(dataset_dir: str, seed: int = 42):
    """Load, clean, and split the review corpus."""
    return split(clean(load_reviews(dataset_dir)), seed=seed)
