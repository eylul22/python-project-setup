"""Train the iris species classifier and persist it to disk."""

from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "iris_classifier.joblib"

RANDOM_STATE = 42


def train() -> float:
    """Train the classifier, save it with its metadata, and return test accuracy."""
    dataset = load_iris()

    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=dataset.target,
    )

    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(x_train, y_train)

    accuracy = float(accuracy_score(y_test, model.predict(x_test)))

    bundle = {
        "model": model,
        "target_names": list(dataset.target_names),
        "feature_names": list(dataset.feature_names),
        "accuracy": accuracy,
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    return accuracy


if __name__ == "__main__":
    score = train()
    print(f"Test accuracy: {score:.3f}")
    print(f"Model saved to: {MODEL_PATH}")
