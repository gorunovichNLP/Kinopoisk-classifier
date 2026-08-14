"""
Загрузка корпуса Кинопоиска, чистка, стратифицированный сплит.

Структура датасета: dataset/{neg,neu,pos}/*.txt, один .txt = один отзыв,
имя папки = метка.
"""

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
    """Снапшот датасета (raw или clean) для воспроизводимости и EDA."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8") 
    print(f"[snapshot] {len(df)} строк -> {out}")


def load_reviews(dataset_dir: str, max_workers: int = 16) -> pd.DataFrame:
    """
    Читает все .txt из neg/neu/pos в один DataFrame.
    """

    base = Path(dataset_dir)
    out = Path("data/raw_dataset.csv")
    if out.exists():
        print(f"[load] cache hit -> {out}")
        return pd.read_csv(out)
    
    tasks = [(fp, cls) for cls in LABEL_MAP for fp in (base / cls).glob("*.txt")]
    if not tasks:
        raise FileNotFoundError(f"Нет .txt в {base}/[neg,neu,pos]. Проверь путь.")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        rows = list(ex.map(_read_one, tasks))
    df = pd.DataFrame(rows)

    _save_snapshot(df, out)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Чистка ДО сплита. Дубли обязательно убрать здесь: если один отзыв
    попадёт и в train, и в test — лик, метрика на тесте завысится.
    """
    before = len(df)
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"[clean] убрано {before - len(df)} пустых/дублей, осталось {len(df)}")

    _save_snapshot(df, "data/clean_dataset.csv")
    return df


def split(df: pd.DataFrame, seed: int = 42):
    """
    Стратифицированный сплит 70/15/15. train_test_split бинарный -> два шага.
    Стратификация по label_id: доли классов одинаковы во всех трёх частях
    (важно для малочисленного нейтрального — иначе val-оценка по нему шумная).
    """
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
    """Полный путь: загрузка -> чистка -> сплит. Возвращает (train, val, test)."""
    return split(clean(load_reviews(dataset_dir)), seed=seed)
