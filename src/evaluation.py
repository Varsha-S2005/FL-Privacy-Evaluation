
import json
import sys
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

# Usage:
# python src/evaluation.py baseline
# python src/evaluation.py class_weighted

EXPERIMENT = sys.argv[1] if len(sys.argv) > 1 else "baseline"

if EXPERIMENT == "baseline":
    MODEL_PATH = PROJECT_DIR / "checkpoints" / "baseline_model.pth"
    RESULTS_DIR = PROJECT_DIR / "results" / "baseline"

elif EXPERIMENT == "class_weighted":
    MODEL_PATH = PROJECT_DIR / "checkpoints" / "class_weighted_model.pth"
    RESULTS_DIR = PROJECT_DIR / "results" / "class_weighted"

else:
    raise ValueError(
        "Unknown experiment. Choose 'baseline' or 'class_weighted'."
    )

METRICS_PATH = RESULTS_DIR / "evaluation_metrics.json"

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

# Thresholds selected using the validation set.
# These values must remain fixed during test evaluation.
THRESHOLDS = {
    "baseline": 0.3347703218460083,
    "class_weighted": 0.5757317543029785,
}

THRESHOLD = THRESHOLDS[EXPERIMENT]

DEVICE = torch.device("cpu")


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print(f"\n========== {EXPERIMENT.upper()} EVALUATION ==========")

    print("Model:", MODEL_PATH)
    print("Evaluation threshold:", THRESHOLD)

    # --------------------------------------------------
    # Load held-out test data
    # --------------------------------------------------

    print("\nLoading held-out test data...")

    df = pd.read_parquet(
        DATA_PATH,
        columns=FEATURE_COLUMNS + ["y_bin", "client", "is_train"],
    )

    test_df = df[df["is_train"] == False].copy()

    del df

    if test_df.empty:
        raise ValueError("No held-out test records found.")

    # --------------------------------------------------
    # Verify client-level separation
    # --------------------------------------------------

    with open(
        PROJECT_DIR / "results" / "selected_clients_seed42.json",
        "r",
    ) as file:
        selected_clients = set(json.load(file))

    test_clients = set(test_df["client"].unique())

    overlap = selected_clients.intersection(test_clients)

    assert len(overlap) == 0, (
        "Data leakage detected: training/test client overlap!"
    )

    print("Test records:", len(test_df))
    print("Held-out clients:", len(test_clients))
    print("Training/test client overlap:", len(overlap))

    # --------------------------------------------------
    # Prepare tensors
    # --------------------------------------------------

    X = torch.tensor(
        test_df[FEATURE_COLUMNS].to_numpy(
            dtype=np.float32,
            copy=True,
        ),
        dtype=torch.float32,
        device=DEVICE,
    )

    y = torch.tensor(
        test_df["y_bin"].to_numpy(
            dtype=np.float32,
            copy=True,
        ),
        dtype=torch.float32,
        device=DEVICE,
    )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    model = MLP(input_features=len(FEATURE_COLUMNS)).to(DEVICE)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")

    all_probabilities = []
    total_loss = 0.0

    # --------------------------------------------------
    # Batch evaluation
    # --------------------------------------------------

    print("\nEvaluating model...")

    with torch.no_grad():

        for start in range(0, len(X), BATCH_SIZE):

            end = min(start + BATCH_SIZE, len(X))

            batch_X = X[start:end]
            batch_y = y[start:end]

            logits = model(batch_X).squeeze(1)

            loss = criterion(logits, batch_y)

            total_loss += loss.item()

            probabilities = torch.sigmoid(logits)

            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )

            if start % (BATCH_SIZE * 10) == 0:
                print(f"Processed {end}/{len(X)} records")

    probabilities = np.array(all_probabilities)

    y_true = y.cpu().numpy().astype(int)

    # --------------------------------------------------
    # Probability distribution
    # --------------------------------------------------

    print("\nPrediction probability distribution:")

    print("Min:", probabilities.min())
    print("Max:", probabilities.max())
    print("Mean:", probabilities.mean())

    print(
        "Percentiles:",
        np.percentile(
            probabilities,
            [25, 50, 75, 90, 95, 99],
        ),
    )

    print("\nActual class distribution:")

    print(
        pd.Series(y_true)
        .value_counts(normalize=True)
        .sort_index()
    )

    # --------------------------------------------------
    # Predictions using validation-selected threshold
    # --------------------------------------------------

    y_pred = (probabilities >= THRESHOLD).astype(int)

    # --------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    auc = roc_auc_score(y_true, probabilities)

    average_loss = total_loss / len(y_true)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    results = {
        "experiment": EXPERIMENT,
        "test_records": len(test_df),
        "test_clients": len(test_clients),
        "training_test_client_overlap": len(overlap),
        "threshold": THRESHOLD,
        "test_loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(results, file, indent=4)

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print("\n========== EVALUATION RESULTS ==========")

    for key, value in results.items():
        print(f"{key}: {value}")

    print("\nEvaluation metrics saved to:", METRICS_PATH)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
