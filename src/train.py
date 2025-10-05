import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from logging_config import get_logger
import yaml

# ==== Paths ====
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
DATA_PATH = os.path.join(ROOT_DIR, "data", "preprocessed.csv")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

logger = get_logger('train', level=20, to_file=os.path.join(LOG_DIR, "train.log"))


def load_params():
    with open(os.path.join(ROOT_DIR, "params.yaml")) as f:
        return yaml.safe_load(f)


def train():
    params = load_params()
    logger.info("Loading data for training...")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Species"])
    y = df["Species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"]
    )

    clf = RandomForestClassifier(
        n_estimators=params["model"]["n_estimators"],
        max_depth=params["model"]["max_depths"],
        random_state=params["model"]["random_state"]
    )

    logger.info("Training RandomForest model...")
    clf.fit(X_train, y_train)

    model_path = os.path.join(MODEL_DIR, "random_forest.pkl")
    joblib.dump(clf, model_path)
    logger.info(f"Model saved to {model_path}")

    test_out = os.path.join(PROCESSED_DIR, "test.csv")
    pd.concat([X_test, y_test.rename("Species")], axis=1).to_csv(test_out, index=False)
    logger.info(f"Saved test set to {test_out}")

    logger.info("Training complete")


if __name__ == "__main__":
    train()
