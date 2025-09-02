import joblib
from pathlib import Path
import sys

texts = sys.argv[1:] or ["Default headline if none passed"]
MODEL_NAME = "tfidf_logreg.joblib"
label_map = {
        1: "positive",
        0: "negative"
    }

model_dir = Path("ml/models/baseline")

loaded_pipeline = joblib.load(model_dir / MODEL_NAME)

# Now you can use it directly
texts = ["Nvidia still hasn't finalized deal to kick some China chip sales back to the US government",
         "Bitcoin price recovers as Nvidia earnings fuel risk appetite",
         "Wall Street ends lower as Dell and Nvidia drop",
         "Nvidia CEO Jensen Huang Just Delivered Spectacular News for Palantir Stock Investors",
         "Snowflake (SNOW) Expands AI Cloud Reach With Siemens And AWS In South Africa"]

for t in texts:
    pred = loaded_pipeline.predict([t])[0]
    print(pred)
    proba = loaded_pipeline.predict_proba([t])[0][1]
    print(f"{t}\n -> {label_map[pred]} (p={proba:.3f})\n")
