
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
    precision_recall_curve,
)

from model import MLP


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(r"C:\Users\91991\Downloads\model_ready.parquet")

SELECTED_CLIENTS_PATH = (
    PROJECT_DIR / "results" / "selected_clients_seed42.json"
)

RESULTS_DIR = PROJECT_DIR / "results" / "validation"
METRICS_PATH = RESULTS_DIR / "validation_metrics.json"

MODELS = {
    "baseline": PROJECT_DIR / "checkpoints" / "baseline_model.pth",

    "class_weighted": (
        PROJECT_DIR / "checkpoints" / "class_weighted_model.pth"
    ),

    "dp_fedavg_weighted": (
        PROJECT_DIR / "checkpoints" / "dp_fedavg_model.pth"
    ),

    "dp_fedavg_unweighted": (
        PROJECT_DIR / "checkpoints" / "dp_fedavg_unweighted_model.pth"
    ),

    "dp_secagg": (
        PROJECT_DIR / "checkpoints" / "dp_secagg_model.pth"
    ),

    "distributed_dp_secagg": (
        PROJECT_DIR / "checkpoints" / "distributed_dp_secagg_model.pth"
    ),
        "distributed_dp_secagg_no_noise": (
        PROJECT_DIR / "checkpoints" / "distributed_dp_secagg_no_noise_model.pth"
    ),
    "distributed_dp_secagg_classweight1": (
    PROJECT_DIR
    / "checkpoints"
    / "distributed_dp_secagg_classweight1_model.pth"
    ),
}

FEATURE_COLUMNS = [
    "pos_x",
    "pos_y",
    "spd_x",
    "spd_y",
    "RSSI",
    "dt",
    "dist",
    "spd_mag",
    "implied_spd",
    "spd_err",
    "delay",
    "rx_dist",
]

BATCH_SIZE = 8192
DEVICE = torch.device("cpu")


# --------------------------------------------------
# Load validation data
# --------------------------------------------------

def load_validation_data():

    print("Loading training partition...")

    df = pd.read_parquet(
        DATA_PATH,
        columns=FEATURE_COLUMNS + ["y_bin", "client", "is_train"],
    )

    with open(SELECTED_CLIENTS_PATH, "r") as file:
        selected_clients = set(json.load(file))

    validation_df = df[
        (df["is_train"] == True)
        & (~df["client"].isin(selected_clients))
    ].copy()

    del df
    gc.collect()

    if validation_df.empty:
        raise ValueError("Validation dataset is empty.")

    validation_clients = set(validation_df["client"].unique())

    overlap = validation_clients.intersection(selected_clients)

    assert len(overlap) == 0, (
        "Validation/training client overlap detected!"
    )

    X = validation_df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    y = validation_df["y_bin"].to_numpy(dtype=np.int64)

    print("\nValidation records:", len(y))
    print("Validation clients:", len(validation_clients))
    print("Training/validation client overlap:", len(overlap))
    print("Class distribution:", np.bincount(y))

    del validation_df
    gc.collect()

    return X, y


# --------------------------------------------------
# Normalize checkpoint key formats
# --------------------------------------------------

def normalize_state_dict(state_dict, model_path):

    # Handle checkpoints saved inside a wrapper dictionary.
    if isinstance(state_dict, dict):

        for wrapper_key in (
            "state_dict",
            "model_state_dict",
            "model",
        ):
            if (
                wrapper_key in state_dict
                and isinstance(state_dict[wrapper_key], dict)
            ):
                state_dict = state_dict[wrapper_key]
                break

    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Invalid checkpoint format: {model_path.name}"
        )

    # Remove a possible DataParallel prefix.
    if state_dict and all(
        key.startswith("module.") for key in state_dict
    ):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    expected_keys = {
        "network.0.weight",
        "network.0.bias",
        "network.2.weight",
        "network.2.bias",
        "network.4.weight",
        "network.4.bias",
    }

    actual_keys = set(state_dict.keys())

    if actual_keys == expected_keys:
        return state_dict

    # Handle checkpoints saved with numeric Sequential keys:
    # 0, 1, 2, 3, 4, 5.
    numeric_mapping = {
        "0": "network.0.weight",
        "1": "network.0.bias",
        "2": "network.2.weight",
        "3": "network.2.bias",
        "4": "network.4.weight",
        "5": "network.4.bias",
    }

    if actual_keys == set(numeric_mapping.keys()):
        return {
            numeric_mapping[key]: value
            for key, value in state_dict.items()
        }

    # Handle Flower-style keys:
    # network.0, network.1, ..., network.5.
    flower_mapping = {
        "network.0": "network.0.weight",
        "network.1": "network.0.bias",
        "network.2": "network.2.weight",
        "network.3": "network.2.bias",
        "network.4": "network.4.weight",
        "network.5": "network.4.bias",
    }

    if actual_keys == set(flower_mapping.keys()):
        return {
            flower_mapping[key]: value
            for key, value in state_dict.items()
        }

    return state_dict


# --------------------------------------------------
# Generate predictions
# --------------------------------------------------

def get_probabilities(model_path, X):

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = MLP(
        input_features=len(FEATURE_COLUMNS)
    ).to(DEVICE)

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    state_dict = normalize_state_dict(
        state_dict,
        model_path,
    )

    expected_keys = set(model.state_dict().keys())
    actual_keys = set(state_dict.keys())

    if actual_keys != expected_keys:

        raise ValueError(
            f"\nIncompatible checkpoint: {model_path.name}\n"
            f"Missing keys: {expected_keys - actual_keys}\n"
            f"Unexpected keys: {actual_keys - expected_keys}"
        )

    model.load_state_dict(state_dict)
    model.eval()

    probabilities = np.empty(
        len(X),
        dtype=np.float32,
    )

    print("\nLoading model:", model_path.name)

    with torch.no_grad():

        for start in range(0, len(X), BATCH_SIZE):

            end = min(
                start + BATCH_SIZE,
                len(X),
            )

            batch_X = torch.from_numpy(
                X[start:end]
            ).to(DEVICE)

            logits = model(batch_X).squeeze(1)

            batch_probabilities = torch.sigmoid(logits)

            probabilities[start:end] = (
                batch_probabilities.cpu().numpy()
            )

            if start % (BATCH_SIZE * 10) == 0:
                print(
                    f"Processed {end}/{len(X)} records"
                )

    del model
    gc.collect()

    return probabilities


# --------------------------------------------------
# Evaluate at a given threshold
# --------------------------------------------------

def calculate_metrics(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),

        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


# --------------------------------------------------
# Evaluate experiment
# --------------------------------------------------

def evaluate_model(name, model_path, X, y):

    probabilities = get_probabilities(
        model_path,
        X,
    )

    # Default threshold evaluation.
    default_metrics = calculate_metrics(
        y,
        probabilities,
        0.5,
    )

    # Find the threshold that maximizes validation F1.
    precision, recall, thresholds = precision_recall_curve(
        y,
        probabilities,
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    best_index = int(np.argmax(f1_scores))

    best_threshold = float(
        thresholds[best_index]
    )

    best_metrics = calculate_metrics(
        y,
        probabilities,
        best_threshold,
    )

    auc = float(
        roc_auc_score(y, probabilities)
    )

    results = {
        "experiment": name,

        "roc_auc": auc,

        "default_threshold_metrics": default_metrics,

        "best_f1_threshold": best_threshold,

        "best_f1_metrics": best_metrics,

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

    print(
        f"\n========== {name.upper()} =========="
    )

    print("ROC-AUC:", auc)

    print(
        "Metrics at threshold 0.5:",
        default_metrics,
    )

    print(
        "Best F1 threshold:",
        best_threshold,
    )

    print(
        "Metrics at best threshold:",
        best_metrics,
    )

    del probabilities
    gc.collect()

    return results


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X, y = load_validation_data()

    all_results = {}

    for name, model_path in MODELS.items():

        print(
            f"\n\nEvaluating experiment: {name}"
        )

        all_results[name] = evaluate_model(
            name,
            model_path,
            X,
            y,
        )

    with open(METRICS_PATH, "w") as file:

        json.dump(
            all_results,
            file,
            indent=4,
        )

    print(
        "\n========== VALIDATION COMPLETED =========="
    )

    print(
        "Results saved to:",
        METRICS_PATH,
    )

    print("\n========== MODEL COMPARISON ==========")

    for name, result in all_results.items():

        print(
            f"{name:25s} "
            f"ROC-AUC: {result['roc_auc']:.6f} | "
            f"Best F1: "
            f"{result['best_f1_metrics']['f1_score']:.6f} | "
            f"Threshold: "
            f"{result['best_f1_threshold']:.4f}"
        )

    del X, y
    gc.collect()


if __name__ == "__main__":
    main()
