"""Train the headline sentiment classifier.

THREE-CLASS as of v0.2. The previous binary version dropped every `neutral`
row before training:

    df = df[df["label"] != "neutral"]

That discarded 1,391 of 2,264 rows (61% of the corpus) and left a model with
nowhere to put a factual statement, so every headline was forced to positive or
negative. In production that showed up as items like "Cloudera Teams With NVIDIA
to Lower Cloud Compute Spend" being scored negative, and it made downstream
sentiment unusable at the item level.

Selection (see ml/eval/model_selection.json for the full sweep): AllAgree with
class_weight="balanced" and C=10, chosen on 5-fold x4 repeated CV macro-F1.
A single 80/20 split preferred 75Agree, but that reversed under CV — the
cleaner labels win despite 34% less data. class_weight matters: negative is
only 13% of the corpus, and without it the model under-predicts exactly the
class the old model already got wrong.

Usage:
    python scripts/train_baseline.py                  # 3-class (default)
    python scripts/train_baseline.py --binary         # reproduce the v0.1 baseline
"""

from pathlib import Path
import argparse
import datetime
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = "data/raw/Sentences_AllAgree.txt"
LABELS = ["negative", "neutral", "positive"]
MODEL_VERSION = "baseline-0.2-3class"
MODEL_DIR = Path("ml/models/baseline")
EVAL_DIR = Path("ml/eval")


def load_data(path: str = DATA_PATH, drop_neutral: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["text", "label"], encoding="latin1", sep="@")
    df = df.dropna()
    df["text"] = df["text"].str.strip()
    df["label"] = df["label"].str.strip()
    if drop_neutral:
        df = df[df["label"] != "neutral"]
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def make_pipeline(binary: bool = False) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), lowercase=True, min_df=2, max_df=0.95,
            strip_accents="unicode", stop_words="english")),
        # class_weight/C are the v0.2 change; the binary path keeps v0.1's defaults
        # so `--binary` still reproduces the committed baseline metrics exactly.
        ("clf", LogisticRegression(max_iter=2000, random_state=42,
                                   class_weight=None if binary else "balanced",
                                   C=1.0 if binary else 10.0)),
    ])


def train_and_eval(df: pd.DataFrame, pipeline: Pipeline, binary: bool) -> tuple[Pipeline, dict]:
    X, y = df["text"], (df["label"].map({"positive": 1, "negative": 0}) if binary else df["label"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    metrics = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_version": "baseline-0.1" if binary else MODEL_VERSION,
        "classes": ["negative", "positive"] if binary else LABELS,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "accuracy": accuracy_score(y_test, pred),
    }

    if binary:
        proba = pipeline.predict_proba(X_test)[:, 1]
        metrics.update(
            precision=precision_score(y_test, pred), recall=recall_score(y_test, pred),
            f1=f1_score(y_test, pred), roc_auc=roc_auc_score(y_test, proba),
            confusion_matrix=confusion_matrix(y_test, pred).tolist())
    else:
        metrics.update(
            macro_f1=f1_score(y_test, pred, average="macro"),
            weighted_f1=f1_score(y_test, pred, average="weighted"),
            per_class=classification_report(y_test, pred, labels=LABELS,
                                            output_dict=True, zero_division=0),
            confusion_matrix=confusion_matrix(y_test, pred, labels=LABELS).tolist(),
            confusion_matrix_labels=LABELS,
            class_distribution={k: int(v) for k, v in df["label"].value_counts().items()},
        )
        # A single split on ~2.2k rows is noisy; report CV alongside it.
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=42)
        scores = cross_val_score(make_pipeline(), X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        metrics["cv_macro_f1_mean"] = float(scores.mean())
        metrics["cv_macro_f1_std"] = float(scores.std())
        metrics["cv_folds"] = int(len(scores))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if not binary:
        joblib.dump(pipeline, MODEL_DIR / "tfidf_logreg.joblib")
        (MODEL_DIR / "MODEL_VERSION").write_text(MODEL_VERSION)
        (EVAL_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    return pipeline, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", action="store_true",
                    help="Reproduce the v0.1 binary baseline (does not overwrite artifacts).")
    args = ap.parse_args()

    df = load_data(drop_neutral=args.binary)
    _, metrics = train_and_eval(df, make_pipeline(args.binary), args.binary)
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
