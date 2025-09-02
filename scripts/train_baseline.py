import numpy
import pandas as pd
import csv

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DATA_PATH = "../../data/raw/Sentences_AllAgree.txt"

def load_data() -> "pd.DataFrame":
    """
    - Drop NaNs and empty rows
    - Trim whitespace
    - Return dataframe with columns exactly ['text', 'label']
    - Drop 'neutral' rows for now (we're doing binary classification).
    """
    df = pd.read_csv(DATA_PATH, header=None, names=["text", "label"], encoding="latin1", sep="@")
    df = df.dropna()
    df["text"] = df["text"].str.strip()
    df["label"] = df["label"].str.strip()
    df = df[df["label"] != "neutral"]
    return df


def to_binary_labels(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    - Map: positive -> 1, negative -> 0
    - Shuffle with a fixed random_state for reproducibility
    """
    label_map = {
        "positive": 1,
        "negative": 0
    }

    df["label"] = df["label"].map(label_map)

    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def make_pipeline() -> "Pipeline":
    """
    Creates a pipeliene with TfidfVectorizer and Logistic Regression
    """
    return Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1,2),
        lowercase=True,
        min_df=2,
        max_df=0.95,
        strip_accents='unicode',
        stop_words='english'
        )),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])


def train_and_eval(df_bin: "pd.DataFrame", pipeline: "Pipeline"):
    """
    TODO:
    - X = df_bin['text'], y = df_bin['label'] (0/1)
    - train_test_split(stratify=y, test_size=0.2, random_state=42)
    - fit vectorizer on X_train; transform X_train/X_test
    - fit model; predict on X_test; proba = predict_proba
    - compute metrics: accuracy, precision, recall, f1 (binary), ROC-AUC
    - return: fitted vectorizer, model, metrics dict
    """
    X = df_bin["text"]
    y = df_bin["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)

    return pipeline
