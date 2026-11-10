
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import MLP


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "results" / "selected_50_clients_seed42.parquet"
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "class_weighted_model.pth"

OUTPUT_DIR = PROJECT_DIR / "results" / "dlg_dp"

FEATURE_COLUMNS = [
    "pos_x", "pos_y", "spd_x", "spd_y",
    "RSSI", "dt", "dist", "spd_mag",
    "implied_spd", "spd_err", "delay", "rx_dist"
]

LABEL_COLUMN = "y_bin"
CLIENT_COLUMN = "client"

DEVICE = torch.device("cpu")

SEED = 42

# DLG settings
DLG_STEPS = 3000
DLG_LEARNING_RATE = 0.03
DLG_RESTARTS = 3

# DP settings
MAX_GRAD_NORM = 1.0
NOISE_MULTIPLIER = 1.0

POS_WEIGHT = 17193 / 4027

torch.manual_seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD ONE RECORD
# ============================================================

print("Loading selected client dataset...")

df = pd.read_parquet(DATA_PATH)

required_columns = (
    FEATURE_COLUMNS + [LABEL_COLUMN, CLIENT_COLUMN]
)

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=required_columns)

# Select a client with valid records.
selected_client = None
selected_row = None

for client_id, client_df in df.groupby(CLIENT_COLUMN, sort=True):
    if len(client_df) > 0:
        selected_client = client_id
        selected_row = client_df.iloc[0]
        break

if selected_row is None:
    raise ValueError("No valid client records found.")

X_original = torch.tensor(
    selected_row[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
    dtype=torch.float32,
    device=DEVICE
).unsqueeze(0)

y_original = torch.tensor(
    [[float(selected_row[LABEL_COLUMN])]],
    dtype=torch.float32,
    device=DEVICE
)

print(f"Selected client: {selected_client}")
print("Batch size: 1")
print(f"Label: {y_original.item()}")
print(f"Number of input features: {X_original.shape[1]}")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = MLP(input_features=len(FEATURE_COLUMNS)).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

print(f"Loaded model: {CHECKPOINT_PATH}")


# ============================================================
# LOSS AND GRADIENT FUNCTIONS
# ============================================================

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [POS_WEIGHT],
        dtype=torch.float32,
        device=DEVICE
    )
)

parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]


def calculate_gradients(inputs, labels):
    model.zero_grad(set_to_none=True)

    logits = model(inputs)
    loss = criterion(logits, labels)

    gradients = torch.autograd.grad(
        loss,
        parameters,
        create_graph=False
    )

    return [g.detach().clone() for g in gradients]


def flatten_gradients(gradients):
    return torch.cat([
        gradient.reshape(-1)
        for gradient in gradients
    ])


# ============================================================
# DP GRADIENT GENERATION
# ============================================================

def generate_dp_gradients(inputs, labels):

    print("\nCalculating per-record gradients...")

    # One record: calculate its individual gradient.
    individual_gradients = calculate_gradients(inputs, labels)

    # Calculate global L2 norm.
    total_norm = torch.sqrt(
        sum(torch.sum(g ** 2) for g in individual_gradients)
    )

    clipping_factor = torch.clamp(
        MAX_GRAD_NORM / (total_norm + 1e-12),
        max=1.0
    )

    clipped_gradients = [
        g * clipping_factor
        for g in individual_gradients
    ]

    # Add Gaussian noise to the clipped gradient.
    noisy_gradients = []

    for gradient in clipped_gradients:

        noise = torch.randn_like(gradient) * (
            NOISE_MULTIPLIER * MAX_GRAD_NORM
        )

        noisy_gradients.append(gradient + noise)

    print(f"Original gradient norm: {total_norm.item():.6f}")
    print(f"Clipping factor: {clipping_factor.item():.6f}")
    print("Gaussian noise added.")

    return [g.detach() for g in noisy_gradients]


# ============================================================
# DLG ATTACK
# ============================================================

def run_dlg(target_gradients, labels, attack_name):

    print(f"\nStarting DLG attack: {attack_name}")

    target_vector = flatten_gradients(target_gradients)

    best_reconstruction = None
    best_matching_loss = float("inf")

    for restart in range(DLG_RESTARTS):

        torch.manual_seed(SEED + 100 + restart)

        dummy_input = torch.randn_like(X_original)
        dummy_input.requires_grad_(True)

        optimizer = torch.optim.Adam(
            [dummy_input],
            lr=DLG_LEARNING_RATE
        )

        for step in range(DLG_STEPS):

            optimizer.zero_grad()

            dummy_logits = model(dummy_input)
            dummy_loss = criterion(dummy_logits, labels)

            dummy_gradients = torch.autograd.grad(
                dummy_loss,
                parameters,
                create_graph=True
            )

            dummy_vector = torch.cat([
                g.reshape(-1)
                for g in dummy_gradients
            ])

            # Relative gradient matching error.
            matching_loss = (
                torch.sum((dummy_vector - target_vector) ** 2)
                / (torch.sum(target_vector ** 2) + 1e-12)
            )

            matching_loss.backward()
            optimizer.step()

            if (step + 1) % 500 == 0:
                print(
                    f"Restart {restart + 1}/{DLG_RESTARTS} "
                    f"- Step {step + 1}/{DLG_STEPS} "
                    f"- Loss: {matching_loss.item():.8f}"
                )

        final_loss = matching_loss.item()

        if final_loss < best_matching_loss:
            best_matching_loss = final_loss
            best_reconstruction = dummy_input.detach().clone()

    return best_reconstruction, best_matching_loss


# ============================================================
# RECONSTRUCTION METRICS
# ============================================================

def calculate_metrics(original, reconstructed, target_gradients):

    feature_difference = reconstructed - original

    mse = torch.mean(feature_difference ** 2).item()
    mae = torch.mean(torch.abs(feature_difference)).item()

    # Recalculate gradients from reconstructed input.
    reconstructed_gradients = calculate_gradients(
        reconstructed,
        y_original
    )

    target_vector = flatten_gradients(target_gradients)
    reconstructed_vector = flatten_gradients(reconstructed_gradients)

    cosine = torch.nn.functional.cosine_similarity(
        target_vector.unsqueeze(0),
        reconstructed_vector.unsqueeze(0)
    ).item()

    relative_error = (
        torch.norm(reconstructed_vector - target_vector)
        / (torch.norm(target_vector) + 1e-12)
    ).item()

    return {
        "reconstruction_mse": mse,
        "reconstruction_mae": mae,
        "gradient_cosine_similarity": cosine,
        "gradient_relative_error": relative_error
    }


# ============================================================
# ORIGINAL GRADIENTS: DLG WITHOUT DP
# ============================================================

print("\n========== DLG WITHOUT DP ==========")

original_gradients = calculate_gradients(
    X_original,
    y_original
)

reconstructed_without_dp, loss_without_dp = run_dlg(
    original_gradients,
    y_original,
    "WITHOUT DP"
)

metrics_without_dp = calculate_metrics(
    X_original,
    reconstructed_without_dp,
    original_gradients
)


# ============================================================
# DP GRADIENTS: DLG WITH DP
# ============================================================

print("\n========== DLG WITH DP ==========")

dp_gradients = generate_dp_gradients(
    X_original,
    y_original
)

reconstructed_with_dp, loss_with_dp = run_dlg(
    dp_gradients,
    y_original,
    "WITH DP"
)

metrics_with_dp = calculate_metrics(
    X_original,
    reconstructed_with_dp,
    dp_gradients
)


# ============================================================
# DP GRADIENT PERTURBATION
# ============================================================

original_vector = flatten_gradients(original_gradients)
dp_vector = flatten_gradients(dp_gradients)

dp_cosine = torch.nn.functional.cosine_similarity(
    original_vector.unsqueeze(0),
    dp_vector.unsqueeze(0)
).item()

dp_relative_error = (
    torch.norm(dp_vector - original_vector)
    / (torch.norm(original_vector) + 1e-12)
).item()


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========== FINAL RESULTS ==========")

print("\nWITHOUT DP:")

for key, value in metrics_without_dp.items():
    print(f"{key}: {value:.6f}")

print("\nWITH DP:")

for key, value in metrics_with_dp.items():
    print(f"{key}: {value:.6f}")

print("\nDP GRADIENT PERTURBATION:")
print(f"Gradient cosine similarity: {dp_cosine:.6f}")
print(f"Gradient relative error: {dp_relative_error:.6f}")


# ============================================================
# SAVE METRICS
# ============================================================

results = {
    "experiment": "Single-record DLG and DP",
    "client": str(selected_client),
    "batch_size": 1,
    "label": float(y_original.item()),
    "max_grad_norm": MAX_GRAD_NORM,
    "noise_multiplier": NOISE_MULTIPLIER,
    "dlg_steps": DLG_STEPS,
    "dlg_restarts": DLG_RESTARTS,
    "matching_loss_without_dp": loss_without_dp,
    "matching_loss_with_dp": loss_with_dp,
    "without_dp": metrics_without_dp,
    "with_dp": metrics_with_dp,
    "dp_gradient_perturbation": {
        "gradient_cosine_similarity": dp_cosine,
        "gradient_relative_error": dp_relative_error
    }
}

metrics_path = OUTPUT_DIR / "dlg_dp_metrics.json"

with open(metrics_path, "w") as f:
    json.dump(results, f, indent=4)


# ============================================================
# SAVE FEATURE COMPARISON
# ============================================================

original_array = X_original.detach().cpu().numpy().flatten()
without_dp_array = reconstructed_without_dp.cpu().numpy().flatten()
with_dp_array = reconstructed_with_dp.cpu().numpy().flatten()

comparison = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "original": original_array,
    "reconstructed_without_dp": without_dp_array,
    "reconstructed_with_dp": with_dp_array,
    "absolute_error_without_dp": np.abs(
        original_array - without_dp_array
    ),
    "absolute_error_with_dp": np.abs(
        original_array - with_dp_array
    )
})

comparison.to_csv(
    OUTPUT_DIR / "reconstructed_features.csv",
    index=False
)


# ============================================================
# GRAPH 1: ORIGINAL VS RECONSTRUCTED FEATURES
# ============================================================

positions = np.arange(len(FEATURE_COLUMNS))
width = 0.25

fig, axes = plt.subplots(2, 1, figsize=(15, 10))

axes[0].bar(
    positions - width,
    original_array,
    width,
    label="Original"
)

axes[0].bar(
    positions,
    without_dp_array,
    width,
    label="Reconstructed without DP"
)

axes[0].bar(
    positions + width,
    with_dp_array,
    width,
    label="Reconstructed with DP"
)

axes[0].set_title("DLG: Original vs Reconstructed Input Features")
axes[0].set_ylabel("Standardized feature value")
axes[0].set_xticks(positions)
axes[0].set_xticklabels(FEATURE_COLUMNS, rotation=45, ha="right")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(
    positions - width / 2,
    np.abs(original_array - without_dp_array),
    width,
    label="Without DP"
)

axes[1].bar(
    positions + width / 2,
    np.abs(original_array - with_dp_array),
    width,
    label="With DP"
)

axes[1].set_title("Absolute Reconstruction Error per Feature")
axes[1].set_ylabel("Absolute error")
axes[1].set_xticks(positions)
axes[1].set_xticklabels(FEATURE_COLUMNS, rotation=45, ha="right")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()

graph_path = OUTPUT_DIR / "dlg_single_record_comparison.png"

plt.savefig(graph_path, dpi=200)
plt.close()


# ============================================================
# GRAPH 2: OVERALL RECONSTRUCTION ERROR
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].bar(
    ["Without DP", "With DP"],
    [
        metrics_without_dp["reconstruction_mse"],
        metrics_with_dp["reconstruction_mse"]
    ]
)

axes[0].set_title("Reconstruction MSE")
axes[0].set_ylabel("MSE (lower is better)")
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(
    ["Without DP", "With DP"],
    [
        metrics_without_dp["reconstruction_mae"],
        metrics_with_dp["reconstruction_mae"]
    ]
)

axes[1].set_title("Reconstruction MAE")
axes[1].set_ylabel("MAE (lower is better)")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()

error_graph_path = OUTPUT_DIR / "dlg_reconstruction_error.png"

plt.savefig(error_graph_path, dpi=200)
plt.close()


print("\n========== FILES SAVED ==========")
print(f"Metrics: {metrics_path}")
print(f"Feature comparison: {OUTPUT_DIR / 'reconstructed_features.csv'}")
print(f"Feature graph: {graph_path}")
print(f"Error graph: {error_graph_path}")

print("\nExperiment completed.")
