
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

try:
    from .model import MLP
except ImportError:
    from model import MLP


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "results" / "selected_50_clients_seed42.parquet"
CLIENT_LIST_PATH = PROJECT_DIR / "results" / "selected_clients_seed42.json"
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "class_weighted_model.pth"

OUTPUT_DIR = PROJECT_DIR / "results" / "dlg_comparison"

FEATURE_COLUMNS = [
    "pos_x", "pos_y", "spd_x", "spd_y",
    "RSSI", "dt", "dist", "spd_mag",
    "implied_spd", "spd_err", "delay", "rx_dist"
]

LABEL_COLUMN = "y_bin"
CLIENT_COLUMN = "client"

DEVICE = torch.device("cpu")
SEED = 42

# Attack configuration
DLG_STEPS = 1500
DLG_LEARNING_RATE = 0.03
DLG_RESTARTS = 2

# DP configuration
MAX_GRAD_NORM = 1.0
NOISE_MULTIPLIER = 1.0
DELTA = 1e-5

# Number of clients in the simulated aggregate
SECAGG_CLIENT_COUNT = 5

# Unweighted loss
POSITIVE_CLASS_WEIGHT = 1.0

torch.manual_seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n========== DLG COMPARISON EXPERIMENT ==========")
print("Device:", DEVICE)
print("DP noise multiplier:", NOISE_MULTIPLIER)
print("Maximum gradient norm:", MAX_GRAD_NORM)
print("DLG steps:", DLG_STEPS)


# ============================================================
# 2. LOAD DATA AND SELECT FIVE CLIENTS
# ============================================================

print("\nLoading client data...")

df = pd.read_parquet(DATA_PATH)

required_columns = (
    FEATURE_COLUMNS + [LABEL_COLUMN, CLIENT_COLUMN]
)

missing = [
    column for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=required_columns)

with open(CLIENT_LIST_PATH, "r") as file:
    selected_clients = json.load(file)

# Select one valid record per client.
client_records = {}

for client_id in selected_clients:

    client_df = df[df[CLIENT_COLUMN] == client_id]

    if not client_df.empty:
        client_records[client_id] = client_df.iloc[0]

if len(client_records) < SECAGG_CLIENT_COUNT:
    raise ValueError("Not enough clients with valid records.")

# Select five clients having the same label.
# This allows the aggregate attack to use known, consistent labels.
label_groups = {}

for client_id, row in client_records.items():

    label = float(row[LABEL_COLUMN])

    if label not in label_groups:
        label_groups[label] = []

    label_groups[label].append((client_id, row))

eligible_groups = [
    (label, records)
    for label, records in label_groups.items()
    if len(records) >= SECAGG_CLIENT_COUNT
]

if not eligible_groups:
    raise ValueError(
        "Could not find five clients with the same label. "
        "Inspect the selected client class distribution."
    )

# Use the largest eligible group.
selected_label, matching_records = max(
    eligible_groups,
    key=lambda item: len(item[1])
)

matching_records = matching_records[:SECAGG_CLIENT_COUNT]

selected_client_ids = [
    client_id for client_id, row in matching_records
]

selected_records = {}

for client_id, row in matching_records:

    features = torch.tensor(
        row[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    label = torch.tensor(
        [[float(row[LABEL_COLUMN])]],
        dtype=torch.float32,
        device=DEVICE
    )

    selected_records[client_id] = {
        "features": features,
        "label": label
    }

print("\nSelected clients:")

for client_id in selected_client_ids:
    print(client_id)

print("Common label:", selected_label)
print("Number of clients:", len(selected_client_ids))


# ============================================================
# 3. LOAD COMMON INITIAL MODEL
# ============================================================

print("\nLoading common initial model...")

initial_model = MLP(
    input_features=len(FEATURE_COLUMNS)
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

initial_model.load_state_dict(checkpoint)

INITIAL_STATE = {
    key: value.detach().clone()
    for key, value in initial_model.state_dict().items()
}

print("Checkpoint loaded:", CHECKPOINT_PATH)


# ============================================================
# 4. LOSS AND GRADIENT UTILITIES
# ============================================================

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [POSITIVE_CLASS_WEIGHT],
        dtype=torch.float32,
        device=DEVICE
    )
)


def flatten_gradients(gradients):

    return torch.cat([
        gradient.reshape(-1)
        for gradient in gradients
    ])


def calculate_gradients(inputs, labels, model):

    model.zero_grad(set_to_none=True)

    predictions = model(inputs)
    loss = criterion(predictions, labels)

    gradients = torch.autograd.grad(
        loss,
        list(model.parameters()),
        create_graph=False
    )

    return [
        gradient.detach().clone()
        for gradient in gradients
    ]


# ============================================================
# 5. CAPTURE ACTUAL OPACUS DP GRADIENTS
# ============================================================

def capture_one_step_gradient(client_id, use_dp):

    record = selected_records[client_id]

    features = record["features"]
    labels = record["label"]

    model = MLP(
        input_features=len(FEATURE_COLUMNS)
    ).to(DEVICE)

    model.load_state_dict(INITIAL_STATE)
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01
    )

    if not use_dp:

        gradients = calculate_gradients(
            features,
            labels,
            model
        )

        optimizer.zero_grad(set_to_none=True)

        return gradients

    # One-record controlled DP experiment.
    dataset = TensorDataset(features, labels)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    privacy_engine = PrivacyEngine()

    private_model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=NOISE_MULTIPLIER,
        max_grad_norm=MAX_GRAD_NORM
    )

    captured_gradients = []

    # Opacus calls this hook after:
    # 1. Per-sample gradient clipping
    # 2. Gaussian noise addition
    # 3. Gradient scaling
    def capture_processed_gradients(dp_optimizer):

        captured_gradients.clear()

        for parameter in model.parameters():

            if parameter.grad is None:
                raise RuntimeError(
                    "A DP gradient was not available."
                )

            captured_gradients.append(
                parameter.grad.detach().clone()
            )

    optimizer.step_hook = capture_processed_gradients

    for batch_features, batch_labels in loader:

        optimizer.zero_grad(set_to_none=True)

        predictions = private_model(batch_features)

        loss = criterion(
            predictions,
            batch_labels
        )

        loss.backward()
        optimizer.step()

        break

    if not captured_gradients:
        raise RuntimeError(
            f"DP gradient capture failed for {client_id}"
        )

    return captured_gradients


# ============================================================
# 6. CAPTURE BASELINE AND DP GRADIENTS
# ============================================================

print("\n========== BASELINE GRADIENT ==========")

first_client = selected_client_ids[0]

baseline_gradients = capture_one_step_gradient(
    first_client,
    use_dp=False
)

print("Baseline gradient captured.")


print("\n========== DP CLIENT GRADIENTS ==========")

dp_client_gradients = {}

for index, client_id in enumerate(selected_client_ids):

    torch.manual_seed(SEED + 100 + index)

    print("Capturing DP gradient:", client_id)

    dp_client_gradients[client_id] = capture_one_step_gradient(
        client_id,
        use_dp=True
    )

print("All DP gradients captured.")


# ============================================================
# 7. CONSTRUCT FIVE-CLIENT AGGREGATE
# ============================================================

print("\n========== SIMULATED SECAGG AGGREGATE ==========")

aggregate_gradients = []

for parameter_index in range(len(baseline_gradients)):

    client_gradients = [
        dp_client_gradients[client_id][parameter_index]
        for client_id in selected_client_ids
    ]

    aggregated_parameter = torch.stack(
        client_gradients,
        dim=0
    ).mean(dim=0)

    aggregate_gradients.append(
        aggregated_parameter.detach().clone()
    )

print("Five-client aggregate constructed.")
print("Aggregation: equal-weight average.")

attack_targets = {
    "baseline_individual": baseline_gradients,
    "dp_individual": dp_client_gradients[first_client],
    "dp_secagg_aggregate": aggregate_gradients
}


# ============================================================
# 8. DLG ATTACK
# ============================================================

attack_model = MLP(
    input_features=len(FEATURE_COLUMNS)
).to(DEVICE)

attack_model.load_state_dict(INITIAL_STATE)
attack_model.eval()


def run_dlg(target_gradients, known_labels, attack_name, reference_inputs):

    print("\nStarting DLG:", attack_name)

    target_vector = flatten_gradients(target_gradients)

    best_reconstruction = None
    best_matching_loss = float("inf")

    for restart in range(DLG_RESTARTS):

        torch.manual_seed(SEED + 500 + restart)

        dummy_input = torch.randn_like(
            reference_inputs,
            requires_grad=True
        )

        optimizer = torch.optim.Adam(
            [dummy_input],
            lr=DLG_LEARNING_RATE
        )

        for step in range(DLG_STEPS):

            optimizer.zero_grad(set_to_none=True)

            dummy_logits = attack_model(dummy_input)

            dummy_loss = criterion(
                dummy_logits,
                known_labels
            )

            dummy_gradients = torch.autograd.grad(
                dummy_loss,
                list(attack_model.parameters()),
                create_graph=True
            )

            dummy_vector = torch.cat([
                gradient.reshape(-1)
                for gradient in dummy_gradients
            ])

            matching_loss = (
                torch.sum(
                    (dummy_vector - target_vector) ** 2
                )
                / (
                    torch.sum(target_vector ** 2) + 1e-12
                )
            )

            matching_loss.backward()
            optimizer.step()

            if (step + 1) % 500 == 0:

                print(
                    f"{attack_name} | "
                    f"Restart {restart + 1}/{DLG_RESTARTS} | "
                    f"Step {step + 1}/{DLG_STEPS} | "
                    f"Loss: {matching_loss.item():.8f}"
                )

        # Recalculate after the final optimizer update.
        optimizer.zero_grad(set_to_none=True)

        final_logits = attack_model(dummy_input)

        final_loss = criterion(
            final_logits,
            known_labels
        )

        final_gradients = torch.autograd.grad(
            final_loss,
            list(attack_model.parameters()),
            create_graph=False
        )

        final_vector = torch.cat([
            gradient.detach().reshape(-1)
            for gradient in final_gradients
        ])

        final_matching_loss = (
            torch.sum(
                (final_vector - target_vector) ** 2
            )
            / (
                torch.sum(target_vector ** 2) + 1e-12
            )
        ).item()

        if final_matching_loss < best_matching_loss:

            best_matching_loss = final_matching_loss

            best_reconstruction = (
                dummy_input.detach().clone()
            )

    return best_reconstruction, best_matching_loss


# ============================================================
# 9. RUN THREE ATTACK SCENARIOS
# ============================================================

original_inputs = torch.cat([
    selected_records[client_id]["features"]
    for client_id in selected_client_ids
], dim=0)

original_labels = torch.cat([
    selected_records[client_id]["label"]
    for client_id in selected_client_ids
], dim=0)

first_input = original_inputs[:1]
first_label = original_labels[:1]

print("\n========== ATTACK 1: BASELINE ==========")

baseline_reconstruction, baseline_loss = run_dlg(
    target_gradients=baseline_gradients,
    known_labels=first_label,
    attack_name="Baseline",
    reference_inputs=first_input
)

print("\n========== ATTACK 2: DP-FEDAVG ==========")

dp_reconstruction, dp_loss = run_dlg(
    target_gradients=dp_client_gradients[first_client],
    known_labels=first_label,
    attack_name="DP-FedAvg",
    reference_inputs=first_input
)

print("\n========== ATTACK 3: DP + SECAGG ==========")

aggregate_labels = original_labels

aggregate_reconstruction, aggregate_loss = run_dlg(
    target_gradients=aggregate_gradients,
    known_labels=aggregate_labels,
    attack_name="DP + simulated SecAgg",
    reference_inputs=original_inputs
)


# ============================================================
# 10. RECONSTRUCTION METRICS
# ============================================================

def single_record_metrics(original, reconstructed):

    difference = reconstructed - original

    return {
        "reconstruction_mse": torch.mean(
            difference ** 2
        ).item(),

        "reconstruction_mae": torch.mean(
            torch.abs(difference)
        ).item()
    }


def aggregate_metrics(original, reconstructed):

    # DLG reconstruction has permutation ambiguity.
    # Find the best one-to-one assignment between reconstructed
    # and original records.

    count = original.shape[0]

    cost_matrix = torch.zeros(
        (count, count),
        dtype=torch.float64
    )

    for i in range(count):
        for j in range(count):

            cost_matrix[i, j] = torch.mean(
                (reconstructed[i] - original[j]) ** 2
            ).double()

    best_cost = float("inf")
    best_permutation = None

    for permutation in itertools.permutations(range(count)):

        total_cost = sum(
            cost_matrix[i, permutation[i]].item()
            for i in range(count)
        )

        if total_cost < best_cost:
            best_cost = total_cost
            best_permutation = permutation

    aligned_original = torch.stack([
        original[best_permutation[i]]
        for i in range(count)
    ])

    difference = reconstructed - aligned_original

    return {
        "reconstruction_mse": torch.mean(
            difference ** 2
        ).item(),

        "reconstruction_mae": torch.mean(
            torch.abs(difference)
        ).item(),

        "assignment": list(best_permutation)
    }


def gradient_similarity(
    reconstruction,
    labels,
    target_gradients
):

    reconstructed_gradients = calculate_gradients(
        reconstruction,
        labels,
        attack_model
    )

    target_vector = flatten_gradients(target_gradients)

    reconstructed_vector = flatten_gradients(
        reconstructed_gradients
    )

    cosine = torch.nn.functional.cosine_similarity(
        target_vector.unsqueeze(0),
        reconstructed_vector.unsqueeze(0)
    ).item()

    relative_error = (
        torch.norm(reconstructed_vector - target_vector)
        / (torch.norm(target_vector) + 1e-12)
    ).item()

    return {
        "gradient_cosine_similarity": cosine,
        "gradient_relative_error": relative_error
    }


baseline_metrics = single_record_metrics(
    first_input,
    baseline_reconstruction
)

dp_metrics = single_record_metrics(
    first_input,
    dp_reconstruction
)

aggregate_result_metrics = aggregate_metrics(
    original_inputs,
    aggregate_reconstruction
)

baseline_metrics.update(
    gradient_similarity(
        baseline_reconstruction,
        first_label,
        baseline_gradients
    )
)

dp_metrics.update(
    gradient_similarity(
        dp_reconstruction,
        first_label,
        dp_client_gradients[first_client]
    )
)

aggregate_result_metrics.update(
    gradient_similarity(
        aggregate_reconstruction,
        aggregate_labels,
        aggregate_gradients
    )
)


# ============================================================
# 11. GRADIENT PERTURBATION
# ============================================================

baseline_vector = flatten_gradients(baseline_gradients)

dp_vector = flatten_gradients(
    dp_client_gradients[first_client]
)

dp_cosine = torch.nn.functional.cosine_similarity(
    baseline_vector.unsqueeze(0),
    dp_vector.unsqueeze(0)
).item()

dp_relative_error = (
    torch.norm(dp_vector - baseline_vector)
    / (torch.norm(baseline_vector) + 1e-12)
).item()


# ============================================================
# 12. PRINT RESULTS
# ============================================================

print("\n========== FINAL RESULTS ==========")

results_by_scenario = {
    "baseline_individual": {
        "matching_loss": baseline_loss,
        **baseline_metrics
    },
    "dp_individual": {
        "matching_loss": dp_loss,
        **dp_metrics
    },
    "dp_secagg_aggregate": {
        "matching_loss": aggregate_loss,
        **aggregate_result_metrics
    }
}

for scenario, metrics in results_by_scenario.items():

    print("\n", scenario)

    for key, value in metrics.items():
        print(f"{key}: {value}")


print("\nDP gradient perturbation:")
print("Cosine similarity:", dp_cosine)
print("Relative error:", dp_relative_error)


# ============================================================
# 13. SAVE METRICS
# ============================================================

results = {
    "experiment": "Controlled DLG gradient inversion comparison",
    "method": "One-record-per-client Opacus DP-SGD demonstration",
    "initial_checkpoint": str(CHECKPOINT_PATH),
    "selected_clients": [str(x) for x in selected_client_ids],
    "common_label": selected_label,
    "num_aggregate_clients": SECAGG_CLIENT_COUNT,
    "aggregation": "Equal-weight average of five DP gradients",
    "max_grad_norm": MAX_GRAD_NORM,
    "noise_multiplier": NOISE_MULTIPLIER,
    "delta": DELTA,
    "dlg_steps": DLG_STEPS,
    "dlg_restarts": DLG_RESTARTS,
    "scenarios": results_by_scenario,
    "dp_gradient_perturbation": {
        "gradient_cosine_similarity": dp_cosine,
        "gradient_relative_error": dp_relative_error
    },
    "limitations": [
        "This is a controlled one-minibatch attack demonstration.",
        "One-record client datasets imply a sampling rate of 1.0.",
        "This does not reproduce the full Flower local training process.",
        "The secure aggregation result is a mathematical aggregate simulation, not a cryptographic SecAgg+ execution.",
        "Known labels are supplied to the DLG attacker.",
        "The five-client aggregate reconstruction uses permutation-invariant matching."
    ]
}

metrics_path = OUTPUT_DIR / "dlg_comparison_metrics.json"

with open(metrics_path, "w") as file:
    json.dump(results, file, indent=4)


# ============================================================
# 14. SAVE RECONSTRUCTED FEATURES
# ============================================================

comparison_rows = []

for index, feature in enumerate(FEATURE_COLUMNS):

    comparison_rows.append({
        "feature": feature,
        "original_first_client": first_input[0, index].item(),
        "baseline_reconstruction": baseline_reconstruction[0, index].item(),
        "dp_reconstruction": dp_reconstruction[0, index].item(),
        "aggregate_reconstruction_mean": aggregate_reconstruction[:, index].mean().item()
    })

comparison_df = pd.DataFrame(comparison_rows)

comparison_path = OUTPUT_DIR / "dlg_feature_comparison.csv"

comparison_df.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# 15. PLOT RECONSTRUCTION ERRORS
# ============================================================

scenario_names = [
    "Baseline",
    "DP-FedAvg",
    "DP + simulated SecAgg"
]

mse_values = [
    baseline_metrics["reconstruction_mse"],
    dp_metrics["reconstruction_mse"],
    aggregate_result_metrics["reconstruction_mse"]
]

mae_values = [
    baseline_metrics["reconstruction_mae"],
    dp_metrics["reconstruction_mae"],
    aggregate_result_metrics["reconstruction_mae"]
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].bar(scenario_names, mse_values)
axes[0].set_title("DLG Reconstruction MSE")
axes[0].set_ylabel("MSE (lower is better)")
axes[0].tick_params(axis="x", rotation=15)
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(scenario_names, mae_values)
axes[1].set_title("DLG Reconstruction MAE")
axes[1].set_ylabel("MAE (lower is better)")
axes[1].tick_params(axis="x", rotation=15)
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()

plot_path = OUTPUT_DIR / "dlg_comparison_errors.png"

plt.savefig(plot_path, dpi=200)
plt.close()


# ============================================================
# 16. SAVE RECONSTRUCTION ARRAYS
# ============================================================

np.savez(
    OUTPUT_DIR / "dlg_reconstructions.npz",
    original=original_inputs.cpu().numpy(),
    baseline=baseline_reconstruction.cpu().numpy(),
    dp=dp_reconstruction.cpu().numpy(),
    aggregate=aggregate_reconstruction.cpu().numpy()
)


print("\n========== EXPERIMENT COMPLETED ==========")
print("Metrics:", metrics_path)
print("Feature comparison:", comparison_path)
print("Error plot:", plot_path)
print("Reconstruction arrays:", OUTPUT_DIR / "dlg_reconstructions.npz")
