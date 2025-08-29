import numpy
import pandas as pd
import csv

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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

