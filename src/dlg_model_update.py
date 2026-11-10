import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from client import train_local_model
from model import MLP


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "results" / "selected_50_clients_seed42.parquet"

METRICS_PATH = (
    ROOT
    / "results"
    / "dlg_comparison"
    / "dlg_comparison_metrics.json"
)

INITIAL_CHECKPOINT = ROOT / "checkpoints" / "class_weighted_model.pth"

OUTPUT_DIR = ROOT / "results" / "dlg_model_update"

DEVICE = torch.device("cpu")

LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.01

DELTA = 1e-5

SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def load_initial_state():

    checkpoint = torch.load(
        INITIAL_CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    model = MLP().to(DEVICE)

    model.load_state_dict(checkpoint)

    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def calculate_delta(initial_state, updated_state):

    return {
        key: (
            updated_state[key] - initial_state[key]
        ).detach().cpu().clone()
        for key in initial_state
    }


def calculate_norm(delta):

    total = 0.0

    for value in delta.values():

        total += torch.sum(value ** 2).item()

    return float(np.sqrt(total))


def aggregate_deltas(client_deltas):

    aggregate = {}

    for key in client_deltas[0]:

        aggregate[key] = torch.stack(
            [
                delta[key]
                for delta in client_deltas
            ],
            dim=0
        ).mean(dim=0)

    return aggregate


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

def main():

    print("=" * 55)
    print("ACTUAL DP MODEL-UPDATE EXPERIMENT")
    print("=" * 55)

    print("\nLoading dataset...")

    df = pd.read_parquet(DATA_PATH)

    with open(METRICS_PATH, "r") as file:

        previous_metrics = json.load(file)

    selected_clients = previous_metrics["selected_clients"]

    print("Selected clients:", len(selected_clients))

    print("\nLoading common initial checkpoint...")

    initial_state = load_initial_state()

    print("Checkpoint:", INITIAL_CHECKPOINT)

    client_deltas = []
    client_results = []

    # --------------------------------------------------
    # Train each client
    # --------------------------------------------------

    for index, client_id in enumerate(selected_clients):

        print("\n" + "=" * 55)
        print(f"CLIENT {index + 1}/{len(selected_clients)}")
        print("=" * 55)

        client_data = df[
            df["client"] == client_id
        ].copy()

        print("Client:", client_id)
        print("Records:", len(client_data))

        print(
            "Class distribution:",
            client_data["y_bin"].value_counts().to_dict()
        )

        # Ensure reproducible initialization of DP randomness
        # and local DataLoader shuffling.
        torch.manual_seed(SEED + index)

        (
            updated_state,
            record_count,
            epoch_losses,
            accountant_state,
            epsilon
        ) = train_local_model(
            initial_state=initial_state,
            client_data=client_data,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            accountant_state=None
        )

        delta = calculate_delta(
            initial_state,
            updated_state
        )

        delta_norm = calculate_norm(delta)

        client_deltas.append(delta)

        client_results.append({

            "client": client_id,
            "records": record_count,
            "class_distribution": {
                str(key): int(value)
                for key, value in
                client_data["y_bin"].value_counts().items()
            },
            "epoch_losses": epoch_losses,
            "epsilon": epsilon,
            "delta_norm": delta_norm

        })

        print("\nClient training completed.")
        print("Epoch losses:", epoch_losses)
        print("Epsilon:", epsilon)
        print("Model update norm:", delta_norm)

    # --------------------------------------------------
    # Aggregate client model updates
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("FIVE-CLIENT MODEL-UPDATE AGGREGATION")
    print("=" * 55)

    aggregate_delta = aggregate_deltas(client_deltas)

    aggregate_norm = calculate_norm(aggregate_delta)

    print("Aggregation: equal-weight average")
    print("Number of clients:", len(client_deltas))
    print("Aggregate update norm:", aggregate_norm)

    # --------------------------------------------------
    # Save actual model updates
    # --------------------------------------------------

    update_file = OUTPUT_DIR / "client_model_updates.npz"

    arrays = {}

    for client_index, delta in enumerate(client_deltas):

        for key, value in delta.items():

            arrays[
                f"client_{client_index}_{key}"
            ] = value.numpy()

    for key, value in aggregate_delta.items():

        arrays[
            f"aggregate_{key}"
        ] = value.numpy()

    np.savez_compressed(
        update_file,
        **arrays
    )

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------

    results = {

        "experiment": "Actual local DP-SGD model-update capture",

        "initial_checkpoint": str(INITIAL_CHECKPOINT),

        "num_clients": len(selected_clients),

        "local_epochs": LOCAL_EPOCHS,

        "batch_size": BATCH_SIZE,

        "learning_rate": LEARNING_RATE,

        "dp_enabled": True,

        "noise_multiplier": 1.0,

        "max_grad_norm": 1.0,

        "delta": DELTA,

        "aggregation": "Equal-weight average of client model deltas",

        "clients": client_results,

        "aggregate_update_norm": aggregate_norm,

        "client_update_file": str(update_file),

        "limitations": [

            "This experiment uses actual local DP-SGD training.",

            "Each client starts from the same global checkpoint.",

            "The aggregation is an equal-weight mathematical simulation.",

            "This is not a cryptographic SecAgg+ execution.",

            "The experiment captures final model-parameter deltas, not raw gradients.",

            "The accountant is reset for each independent client experiment."

        ]

    }

    metrics_file = OUTPUT_DIR / "model_update_metrics.json"

    with open(metrics_file, "w") as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\n" + "=" * 55)
    print("EXPERIMENT COMPLETED")
    print("=" * 55)

    print("Metrics:", metrics_file)
    print("Model updates:", update_file)


if __name__ == "__main__":

    main()