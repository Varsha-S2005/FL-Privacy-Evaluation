
import copy
import math
from pathlib import Path
from random import SystemRandom

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

DEVICE = torch.device("cpu")


# --------------------------------------------------
# Original Opacus DP-SGD configuration
# Keep this for the existing DP-FedAvg experiment.
# --------------------------------------------------

POSITIVE_CLASS_WEIGHT = 1.0

DP_ENABLED = True

MAX_GRAD_NORM = 1.0
NOISE_MULTIPLIER = 1.0
DELTA = 1e-5


# --------------------------------------------------
# Distributed DP configuration
# Used only by the new distributed-DP experiment.
# --------------------------------------------------

DISTRIBUTED_POSITIVE_CLASS_WEIGHT = 4.0

CLIENT_UPDATE_CLIP_NORM = 0.1
DISTRIBUTED_NOISE_MULTIPLIER = 1.0

SELECTED_CLIENTS_PER_ROUND = 5
MIN_HONEST_NOISE_CONTRIBUTORS = 3


# --------------------------------------------------
# Original local training function: Opacus DP-SGD
# --------------------------------------------------

def train_local_model(
    initial_state,
    client_data,
    epochs=LOCAL_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    accountant_state=None
):

    base_model = MLP().to(DEVICE)
    base_model.load_state_dict(initial_state)

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

    if DP_ENABLED:

        privacy_engine = PrivacyEngine()

        if accountant_state is not None:

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

        print("\nOpacus DP-SGD enabled.")
        print("Maximum gradient norm:", MAX_GRAD_NORM)
        print("Noise multiplier:", NOISE_MULTIPLIER)

    else:

        model = base_model
        print("\nDP-SGD disabled.")

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

    updated_accountant_state = None
    epsilon = None

    if privacy_engine is not None:

        epsilon = privacy_engine.get_epsilon(
            delta=DELTA
        )

        updated_accountant_state = (
            privacy_engine.accountant.state_dict()
        )

        print("\nCumulative local privacy accounting:")
        print(f"Epsilon: {epsilon:.6f}")
        print(f"Delta: {DELTA}")
        print(f"Noise multiplier: {NOISE_MULTIPLIER}")
        print(f"Max gradient norm: {MAX_GRAD_NORM}")

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
# Cryptographically seeded Gaussian noise generation
# --------------------------------------------------

def generate_gaussian_noise(reference, standard_deviation, rng):

    if standard_deviation <= 0:
        return torch.zeros_like(reference)

    values = [
        rng.gauss(0.0, standard_deviation)
        for _ in range(reference.numel())
    ]

    return torch.tensor(
        values,
        dtype=reference.dtype,
        device=reference.device
    ).reshape(reference.shape)


# --------------------------------------------------
# Distributed DP local training
#
# This is separate from Opacus DP-SGD.
# It performs:
# 1. Ordinary local training
# 2. Client model-update calculation
# 3. Global L2 clipping of the update
# 4. Independent Gaussian noise-share addition
# 5. Return of the noisy model parameters
# --------------------------------------------------

def train_local_model_distributed_dp(
    initial_state,
    client_data,
    epochs=LOCAL_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    clip_norm=CLIENT_UPDATE_CLIP_NORM,
    noise_multiplier=DISTRIBUTED_NOISE_MULTIPLIER,
    selected_clients=SELECTED_CLIENTS_PER_ROUND,
    min_honest_clients=MIN_HONEST_NOISE_CONTRIBUTORS,
    positive_class_weight=DISTRIBUTED_POSITIVE_CLASS_WEIGHT
):

    if client_data.empty:
        raise ValueError("Client dataset is empty.")

    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive.")

    if noise_multiplier < 0:
        raise ValueError("noise_multiplier cannot be negative.")

    if not 1 <= min_honest_clients <= selected_clients:
        raise ValueError("Invalid honest-client configuration.")

    # --------------------------------------------------
    # 1. Initialize local model from global parameters
    # --------------------------------------------------

    original_state = {
        key: value.detach().to(DEVICE).clone()
        for key, value in initial_state.items()
    }

    base_model = MLP().to(DEVICE)
    base_model.load_state_dict(original_state)

    # --------------------------------------------------
    # 2. Prepare local dataset
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 3. Class-weighted local training
    # --------------------------------------------------

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [positive_class_weight],
            dtype=torch.float32,
            device=DEVICE
        )
    )

    optimizer = torch.optim.SGD(
        base_model.parameters(),
        lr=learning_rate
    )

    base_model.train()

    epoch_losses = []

    for epoch in range(epochs):

        total_loss = 0.0
        total_records = 0

        for features, labels in loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            predictions = base_model(features)
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
    # 4. Calculate client model update
    # --------------------------------------------------

    local_state = base_model.state_dict()

    delta_state = {}

    squared_norm = 0.0

    for key, local_tensor in local_state.items():

        delta = (
            local_tensor.detach()
            - original_state[key]
        )

        delta_state[key] = delta

        squared_norm += torch.sum(
            delta.to(torch.float64) ** 2
        ).item()

    update_norm = math.sqrt(squared_norm)

    # --------------------------------------------------
    # 5. Global L2 clipping across all parameters
    # --------------------------------------------------

    clip_factor = min(
        1.0,
        clip_norm / (update_norm + 1e-12)
    )

    clipped_delta_state = {
        key: delta * clip_factor
        for key, delta in delta_state.items()
    }

    clipped_norm = min(update_norm, clip_norm)

    # --------------------------------------------------
    # 6. Calculate independent Gaussian noise share
    #
    # Assumption:
    # At least min_honest_clients participants contribute
    # independent noise and the aggregation weights are equal.
    # --------------------------------------------------

    noise_share_std = (
        2.0
        * noise_multiplier
        * clip_norm
        / math.sqrt(min_honest_clients)
    )

    rng = SystemRandom()

    # --------------------------------------------------
    # 7. Add noise to the clipped client update
    # --------------------------------------------------

    updated_state = {}

    for key, clipped_delta in clipped_delta_state.items():

        noise = generate_gaussian_noise(
            reference=clipped_delta,
            standard_deviation=noise_share_std,
            rng=rng
        )

        noisy_parameter = (
            original_state[key]
            + clipped_delta
            + noise
        )

        updated_state[key] = (
            noisy_parameter.detach().cpu().clone()
        )

    # --------------------------------------------------
    # 8. Return updated parameters and experiment metrics
    # --------------------------------------------------

    print("\nDistributed DP training summary:")
    print(f"Local records: {len(dataset)}")
    print(f"Original update norm: {update_norm:.6f}")
    print(f"Clipped update norm: {clipped_norm:.6f}")
    print(f"Clip factor: {clip_factor:.6f}")
    print(f"Noise multiplier: {noise_multiplier}")
    print(f"Noise-share standard deviation: {noise_share_std:.6f}")

    return (
        updated_state,
        len(dataset),
        epoch_losses,
        update_norm,
        clipped_norm,
        clip_factor,
        noise_share_std
    )


# --------------------------------------------------
# Test original Opacus local training
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

    if eligible_clients.empty:
        raise ValueError(
            "No client with both classes was found."
        )

    client_id = eligible_clients.index[0]

    client_data = df[
        df["client"] == client_id
    ].copy()

    print("\nSelected client:", client_id)
    print("Local records:", len(client_data))

    print("\nLabel distribution:")
    print(
        client_data["y_bin"]
        .value_counts()
        .sort_index()
    )

    global_model = MLP().to(DEVICE)

    initial_state = copy.deepcopy(
        global_model.state_dict()
    )

    print("\nStarting original Opacus DP-SGD test...\n")

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
