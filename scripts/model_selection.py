"""Reproduce the v0.2 model-selection sweep and the neutral-forcing comparison.

Writes ml/eval/model_selection.json. Run after changing anything in
train_baseline.make_pipeline so the recorded justification stays true.

    python scripts/model_selection.py
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.feature_extraction.text import TfidfVectorizer          # noqa: E402
from sklearn.linear_model import LogisticRegression                  # noqa: E402
from sklearn.model_selection import (RepeatedStratifiedKFold,        # noqa: E402
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline                                # noqa: E402

from train_baseline import load_data, make_pipeline                  # noqa: E402

DATASETS = ["Sentences_AllAgree.txt", "Sentences_75Agree.txt"]


def pipe(cw, C):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=2,
                                  max_df=0.95, strip_accents="unicode", stop_words="english")),
        ("clf", LogisticRegression(max_iter=2000, random_state=42, class_weight=cw, C=C))])


def main() -> None:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=42)
    sweep = []
    for name in DATASETS:
        df = load_data(f"data/raw/{name}")
        for cw in ["balanced", None]:
            for C in [4.0, 10.0, 20.0]:
                s = cross_val_score(pipe(cw, C), df["text"], df["label"],
                                    cv=cv, scoring="f1_macro", n_jobs=-1)
                sweep.append({"dataset": name, "class_weight": cw, "C": C,
                              "cv_macro_f1_mean": float(s.mean()),
                              "cv_macro_f1_std": float(s.std()), "n": int(len(df))})
    sweep.sort(key=lambda r: r["cv_macro_f1_mean"], reverse=True)

    # What the discarded neutral rows were actually being labelled as, under the
    # binary model this replaces. This is the defect in one number.
    b = load_data(drop_neutral=True)
    y = b["label"].map({"positive": 1, "negative": 0})
    Xtr, _, ytr, _ = train_test_split(b["text"], y, stratify=y, test_size=0.2, random_state=42)
    old = make_pipeline(binary=True).fit(Xtr, ytr)
    neu = load_data().query("label == 'neutral'")["text"]
    p = old.predict_proba(neu)[:, 1]
    forced = {f"threshold_{t}": {"positive": int((p >= t).sum()),
                                 "negative": int((p < t).sum()),
                                 "positive_pct": round(float((p >= t).mean()) * 100, 1)}
              for t in (0.50, 0.70)}

    out = {
        "selected": {"dataset": "Sentences_AllAgree.txt", "class_weight": "balanced", "C": 10.0},
        "selection_metric": "macro F1, 5-fold x4 repeated stratified CV",
        "note": ("A single 80/20 split ranked 75Agree above AllAgree; that reversed under "
                 "repeated CV, so the split result was noise. Cleaner labels beat 34% more data."),
        "sweep": sweep,
        "binary_model_forced_neutral": {
            "n_neutral_sentences": int(len(neu)),
            "description": ("The v0.1 binary model has no neutral class, so these genuinely "
                            "neutral sentences were each forced to a side. At the production "
                            "threshold of 0.70 the split is close to a coin flip, and neutral "
                            "is 61% of the corpus."),
            "results": forced,
        },
    }
    Path("ml/eval").mkdir(parents=True, exist_ok=True)
    Path("ml/eval/model_selection.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"selected": out["selected"], "top3": sweep[:3],
                      "forced_neutral": forced}, indent=2))


if __name__ == "__main__":
    main()
