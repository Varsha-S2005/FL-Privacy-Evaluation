
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from model import MLP


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = Path(r"C:\Users\91991\Downloads\model_ready.parquet")

VALIDATION_METRICS_PATH = (
    PROJECT_DIR / "results" / "validation" / "validation_metrics.json"
)

RESULTS_DIR = PROJECT_DIR / "results" / "test"
METRICS_PATH = RESULTS_DIR / "test_metrics.json"

MODELS = {
    "baseline": PROJECT_DIR / "checkpoints" / "baseline_model.pth",
    "class_weighted": PROJECT_DIR / "checkpoints" / "class_weighted_model.pth",
    "dp_fedavg_weighted": PROJECT_DIR / "checkpoints" / "dp_fedavg_model.pth",
    "dp_fedavg_unweighted": PROJECT_DIR / "checkpoints" / "dp_fedavg_unweighted_model.pth",
    "dp_secagg": PROJECT_DIR / "checkpoints" / "dp_secagg_model.pth",
}

FEATURE_COLUMNS = [
    "pos_x", "pos_y", "spd_x", "spd_y",
    "RSSI", "dt", "dist", "spd_mag",
    "implied_spd", "spd_err", "delay", "rx_dist",
]

BATCH_SIZE = 8192
DEVICE = torch.device("cpu")


# --------------------------------------------------
# Load held-out test data
# --------------------------------------------------

def load_test_data():

    print("Loading held-out test partition...")

    df = pd.read_parquet(
        DATA_PATH,
        columns=FEATURE_COLUMNS + ["y_bin", "client", "is_train"],
    )

    test_df = df[df["is_train"] == False].copy()

    del df
    gc.collect()

    if test_df.empty:
        raise ValueError("Test dataset is empty.")

    X = test_df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    y = test_df["y_bin"].to_numpy(dtype=np.int64)

    test_clients = test_df["client"].nunique()

    print("\nTest records:", len(y))
    print("Test clients:", test_clients)
    print("Class distribution:", np.bincount(y))

    del test_df
    gc.collect()

    return X, y


# --------------------------------------------------
# Generate predictions
# --------------------------------------------------

def get_probabilities(model_path, X):

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = MLP(input_features=len(FEATURE_COLUMNS)).to(DEVICE)

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    # Handle DP + SecAgg checkpoint with numeric keys
    if list(state_dict.keys()) == ["0", "1", "2", "3", "4", "5"]:
        state_dict = {
            "network.0.weight": state_dict["0"],
            "network.0.bias": state_dict["1"],
            "network.2.weight": state_dict["2"],
            "network.2.bias": state_dict["3"],
            "network.4.weight": state_dict["4"],
            "network.4.bias": state_dict["5"],
        }

    elif list(state_dict.keys()) == [
        "network.0",
        "network.1",
        "network.2",
        "network.3",
        "network.4",
        "network.5",
    ]:
        state_dict = {
            "network.0.weight": state_dict["network.0"],
            "network.0.bias": state_dict["network.1"],
            "network.2.weight": state_dict["network.2"],
            "network.2.bias": state_dict["network.3"],
            "network.4.weight": state_dict["network.4"],
            "network.4.bias": state_dict["network.5"],
        }

    expected_keys = set(model.state_dict().keys())
    actual_keys = set(state_dict.keys())

    if actual_keys != expected_keys:
        raise ValueError(
            f"Incompatible checkpoint: {model_path.name}\n"
            f"Missing keys: {expected_keys - actual_keys}\n"
            f"Unexpected keys: {actual_keys - expected_keys}"
        )

    model.load_state_dict(state_dict)
    model.eval()

    probabilities = np.empty(len(X), dtype=np.float32)

    print("\nLoading model:", model_path.name)

    with torch.no_grad():

        for start in range(0, len(X), BATCH_SIZE):

            end = min(start + BATCH_SIZE, len(X))

            batch_X = torch.from_numpy(X[start:end]).to(DEVICE)

            logits = model(batch_X).squeeze(1)

            batch_probabilities = torch.sigmoid(logits)

            probabilities[start:end] = batch_probabilities.numpy()

            if start % (BATCH_SIZE * 10) == 0:
                print(f"Processed {end}/{len(X)} records")

    return probabilities


# --------------------------------------------------
# Calculate test metrics
# --------------------------------------------------

def calculate_metrics(y_true, probabilities, threshold):

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


# --------------------------------------------------
# Evaluate each experiment
# --------------------------------------------------

def evaluate_model(name, model_path, threshold, X, y):

    probabilities = get_probabilities(model_path, X)

    metrics = calculate_metrics(
        y,
        probabilities,
        threshold,
    )

    auc = float(roc_auc_score(y, probabilities))

    results = {
        "experiment": name,
        "roc_auc": auc,
        "test_metrics_at_validation_threshold": metrics,
        "probability_distribution": {
            "min": float(probabilities.min()),
            "max": float(probabilities.max()),
            "mean": float(probabilities.mean()),
            "percentiles": np.percentile(
                probabilities,
                [25, 50, 75, 90, 95, 99],
            ).tolist(),
        },
    }

    print(f"\n========== {name.upper()} TEST RESULTS ==========")
    print("ROC-AUC:", auc)
    print("Validation-selected threshold:", threshold)
    print("Test metrics:", metrics)

    return results


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not VALIDATION_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Validation metrics not found: {VALIDATION_METRICS_PATH}"
        )

    with open(VALIDATION_METRICS_PATH) as file:
        validation_results = json.load(file)

    X, y = load_test_data()

    all_results = {}

    for name, model_path in MODELS.items():

        if name not in validation_results:
            raise ValueError(
                f"Validation threshold missing for experiment: {name}"
            )

        threshold = validation_results[name]["best_f1_threshold"]

        all_results[name] = evaluate_model(
            name,
            model_path,
            threshold,
            X,
            y,
        )

    with open(METRICS_PATH, "w") as file:
        json.dump(all_results, file, indent=4)

    print("\n========== TEST EVALUATION COMPLETED ==========")
    print("Results saved to:", METRICS_PATH)


if __name__ == "__main__":
    main()
