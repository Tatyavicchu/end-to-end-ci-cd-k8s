import os
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from logging_config import get_logger

# ==== Paths ====
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

os.makedirs(LOG_DIR, exist_ok=True)

logger = get_logger('evaluate', level=20, to_file=os.path.join(LOG_DIR, "evaluate.log"))


def evaluate():
    model_path = os.path.join(MODEL_DIR, "random_forest.pkl")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    metrics_path = os.path.join(ROOT_DIR, "metrics.json")

    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return
    if not os.path.exists(test_path):
        logger.error(f"Test set not found at {test_path}")
        return

    logger.info("Loading model and test set...")
    model = joblib.load(model_path)
    df = pd.read_csv(test_path)
    X = df.drop(columns=["Species"])
    y = df["Species"]

    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="weighted")

    metrics = {
        "accuracy": round(float(acc), 4),
        "f1_weighted": round(float(f1), 4)
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Metrics saved to {metrics_path}")
    logger.info(metrics)


if __name__ == "__main__":
    evaluate()
