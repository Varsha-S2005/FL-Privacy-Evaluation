
import copy
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

try:
    from .model import MLP
except ImportError:
    from model import MLP


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "results"
    / "selected_50_clients_seed42.parquet"
)

FEATURE_COLUMNS = [
    "pos_x", "pos_y",
    "spd_x", "spd_y",
    "RSSI", "dt",
    "dist", "spd_mag",
    "implied_spd", "spd_err",
    "delay", "rx_dist",
]

BATCH_SIZE = 32
LOCAL_EPOCHS = 2
LEARNING_RATE = 0.01

POSITIVE_CLASS_WEIGHT = 1.0

DEVICE = torch.device("cpu")


# --------------------------------------------------
# Differential Privacy Configuration
# --------------------------------------------------

DP_ENABLED = True

MAX_GRAD_NORM = 1.0
NOISE_MULTIPLIER = 1.0
DELTA = 1e-5


# --------------------------------------------------
# Local training function
# --------------------------------------------------

def train_local_model(
    initial_state,
    client_data,
    epochs=LOCAL_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    accountant_state=None
):

    # Create a fresh copy of the global model.
    base_model = MLP().to(DEVICE)
    base_model.load_state_dict(initial_state)

    # Convert local data into tensors.
    X = client_data[FEATURE_COLUMNS].to_numpy(
        dtype="float32",
        copy=True
    )

    y = client_data["y_bin"].to_numpy(
        dtype="float32",
        copy=True
    ).reshape(-1, 1)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # Weighted binary classification loss.
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [POSITIVE_CLASS_WEIGHT],
            dtype=torch.float32,
            device=DEVICE
        )
    )

    optimizer = torch.optim.SGD(
        base_model.parameters(),
        lr=learning_rate
    )

    privacy_engine = None

    # --------------------------------------------------
    # Enable DP-SGD
    # --------------------------------------------------

    if DP_ENABLED:

        privacy_engine = PrivacyEngine()

        # Restore this client's previous privacy history.
        if accountant_state is not None:

            # JSON converts tuples into lists.
            # Convert them back before loading into Opacus.
            restored_state = dict(accountant_state)

            restored_state["history"] = [
                tuple(entry)
                for entry in restored_state["history"]
            ]

            privacy_engine.accountant.load_state_dict(
                restored_state
            )

            print("\nPrevious privacy history restored.")

        private_model, optimizer, loader = (
            privacy_engine.make_private(
                module=base_model,
                optimizer=optimizer,
                data_loader=loader,
                noise_multiplier=NOISE_MULTIPLIER,
                max_grad_norm=MAX_GRAD_NORM
            )
        )

        model = private_model

        print("\nDP-SGD enabled.")
        print("Maximum gradient norm:", MAX_GRAD_NORM)
        print("Noise multiplier:", NOISE_MULTIPLIER)

    else:

        model = base_model
        print("\nDP-SGD disabled.")

    # --------------------------------------------------
    # Local training
    # --------------------------------------------------

    model.train()

    epoch_losses = []

    for epoch in range(epochs):

        total_loss = 0.0
        total_records = 0

        for features, labels in loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            predictions = model(features)

            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(features)
            total_records += len(features)

        average_loss = (
            total_loss / total_records
            if total_records > 0
            else 0.0
        )

        epoch_losses.append(average_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {average_loss:.4f}"
        )

    # --------------------------------------------------
    # Cumulative privacy accounting
    # --------------------------------------------------

    updated_accountant_state = None
    epsilon = None

    if privacy_engine is not None:

        epsilon = privacy_engine.get_epsilon(
            delta=DELTA
        )

        # Save the complete accountant history.
        updated_accountant_state = (
            privacy_engine.accountant.state_dict()
        )

        print("\nCumulative local privacy accounting:")
        print(f"Epsilon: {epsilon:.6f}")
        print(f"Delta: {DELTA}")
        print(f"Noise multiplier: {NOISE_MULTIPLIER}")
        print(f"Max gradient norm: {MAX_GRAD_NORM}")

    # --------------------------------------------------
    # Return updated model and privacy state
    # --------------------------------------------------

    updated_state = {
        key: value.detach().cpu().clone()
        for key, value in base_model.state_dict().items()
    }

    return (
        updated_state,
        len(dataset),
        epoch_losses,
        updated_accountant_state,
        epsilon
    )


# --------------------------------------------------
# Test local training with one client
# --------------------------------------------------

def main():

    print("Loading selected client data...")

    df = pd.read_parquet(DATA_PATH)

    summary = (
        df.groupby("client")["y_bin"]
        .agg(
            total_records="size",
            unique_classes="nunique"
        )
    )

    eligible_clients = summary[
        summary["unique_classes"] == 2
    ]

    client_id = eligible_clients.index[0]

    client_data = df[
        df["client"] == client_id
    ].copy()

    print("\nSelected client:", client_id)
    print("Local records:", len(client_data))

    print("Label distribution:")
    print(
        client_data["y_bin"]
        .value_counts()
        .sort_index()
    )

    print(
        "Positive class weight:",
        POSITIVE_CLASS_WEIGHT
    )

    global_model = MLP().to(DEVICE)

    initial_state = copy.deepcopy(
        global_model.state_dict()
    )

    print("\nStarting DP local training...\n")

    (
        updated_state,
        record_count,
        losses,
        accountant_state,
        epsilon
    ) = train_local_model(
        initial_state,
        client_data
    )

    parameter_changed = any(
        not torch.equal(initial_state[key], updated_state[key])
        for key in initial_state
    )

    print("\nLocal training completed.")
    print("Records used:", record_count)
    print("Initial loss:", round(losses[0], 4))
    print("Final loss:", round(losses[-1], 4))
    print("Model parameters changed:", parameter_changed)
    print("Accountant state returned:", accountant_state is not None)
    print("Epsilon:", epsilon)


if __name__ == "__main__":
    main()
