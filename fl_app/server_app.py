
import json
from pathlib import Path

import torch

from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig
from flwr.server.compat import LegacyContext
from flwr.server.strategy import FedAvg as LegacyFedAvg
from flwr.server.workflow import (
    DefaultWorkflow,
    SecAggPlusWorkflow,
)
from flwr.server.workflow.default_workflows import MAIN_PARAMS_RECORD
from flwr.serverapp import ServerApp

from src.model import MLP

from src.client import (
    CLIENT_UPDATE_CLIP_NORM,
    DISTRIBUTED_NOISE_MULTIPLIER,
    SELECTED_CLIENTS_PER_ROUND,
    MIN_HONEST_NOISE_CONTRIBUTORS,
    DISTRIBUTED_POSITIVE_CLASS_WEIGHT,
)




# --------------------------------------------------
# Paths
# Keep all previous experiments untouched.
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "distributed_dp_secagg"
)

# Class weight = 1.0, noise multiplier = 0.0
MODEL_PATH = (
    CHECKPOINT_DIR
    / "distributed_dp_secagg_classweight1_model.pth"
)

METRICS_PATH = (
    RESULTS_DIR
    / "training_metrics_classweight1.json"
)


# --------------------------------------------------
# Secure Aggregation configuration
# --------------------------------------------------

SECAGG_NUM_SHARES = 3
SECAGG_RECONSTRUCTION_THRESHOLD = 2

# ClientApp returns num_examples=1 for every participant.
SECAGG_MAX_WEIGHT = 1.0

# Flower applies element-wise clipping before quantization.
SECAGG_CLIPPING_RANGE = 8.0

SECAGG_QUANTIZATION_RANGE = 4194304
SECAGG_MODULUS_RANGE = 4294967296


# --------------------------------------------------
# Server application
# --------------------------------------------------

app = ServerApp()


@app.main()
def main(grid, context):

    # --------------------------------------------------
    # 1. Read experiment configuration
    # --------------------------------------------------

    num_rounds = int(
        context.run_config["num-server-rounds"]
    )

    fraction_train = float(
        context.run_config["fraction-train"]
    )

    local_epochs = int(
        context.run_config["local-epochs"]
    )

    batch_size = int(
        context.run_config["batch-size"]
    )

    learning_rate = float(
        context.run_config["learning-rate"]
    )

    num_clients = 50

    expected_clients = max(
        1,
        int(num_clients * fraction_train)
    )

    # The distributed noise calibration assumes five
    # selected clients per round.
    if expected_clients != SELECTED_CLIENTS_PER_ROUND:
        raise ValueError(
            "Client participation does not match the "
            "distributed-noise configuration. "
            f"Expected {SELECTED_CLIENTS_PER_ROUND} clients, "
            f"but configuration selects {expected_clients}."
        )

    if not (
        1
        <= MIN_HONEST_NOISE_CONTRIBUTORS
        <= expected_clients
    ):
        raise ValueError(
            "Invalid minimum honest noise contributor count."
        )

    # --------------------------------------------------
    # 2. Calculate client noise-share standard deviation
    # --------------------------------------------------

    noise_share_std = (
        2.0
        * DISTRIBUTED_NOISE_MULTIPLIER
        * CLIENT_UPDATE_CLIP_NORM
        / (MIN_HONEST_NOISE_CONTRIBUTORS ** 0.5)
    )

    print("\n========== DISTRIBUTED DP + SECAGG ==========")

    print("Total virtual clients:", num_clients)
    print("Clients selected per round:", expected_clients)
    print("Communication rounds:", num_rounds)

    print("\nLocal training:")
    print("Local epochs:", local_epochs)
    print("Batch size:", batch_size)
    print("Learning rate:", learning_rate)
    print(
        "Positive class weight:",
        DISTRIBUTED_POSITIVE_CLASS_WEIGHT
    )

    print("\nDistributed DP:")
    print("Client update clipping norm:", CLIENT_UPDATE_CLIP_NORM)
    print("Noise multiplier:", DISTRIBUTED_NOISE_MULTIPLIER)
    print(
        "Minimum assumed honest noise contributors:",
        MIN_HONEST_NOISE_CONTRIBUTORS
    )
    print(
        "Gaussian noise-share standard deviation:",
        round(noise_share_std, 6)
    )

    print("\nSecure Aggregation:")
    print("Number of shares:", SECAGG_NUM_SHARES)
    print(
        "Reconstruction threshold:",
        SECAGG_RECONSTRUCTION_THRESHOLD
    )
    print("Maximum client weight:", SECAGG_MAX_WEIGHT)
    print("Element-wise clipping range:", SECAGG_CLIPPING_RANGE)
    print(
        "Quantization range:",
        SECAGG_QUANTIZATION_RANGE
    )

    # --------------------------------------------------
    # 3. Initialize global model
    # --------------------------------------------------

    global_model = MLP()

    initial_parameters = ndarrays_to_parameters(
        [
            tensor.detach().cpu().numpy()
            for tensor in global_model.state_dict().values()
        ]
    )

    # --------------------------------------------------
    # 4. Initialize equal-client FedAvg
    # --------------------------------------------------

    strategy = LegacyFedAvg(
        fraction_fit=fraction_train,
        fraction_evaluate=0.0,
        min_fit_clients=expected_clients,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
    )

    # Each client returns num_examples=1.
    # Therefore FedAvg assigns equal weight to each client.
    print(
        "\nAggregation method: equal-client weighted FedAvg"
    )

    # --------------------------------------------------
    # 5. Configure Secure Aggregation+
    # --------------------------------------------------

    secure_aggregation = SecAggPlusWorkflow(
        num_shares=SECAGG_NUM_SHARES,
        reconstruction_threshold=SECAGG_RECONSTRUCTION_THRESHOLD,

        # Must match num_examples=1 in client_app.py.
        max_weight=SECAGG_MAX_WEIGHT,

        clipping_range=SECAGG_CLIPPING_RANGE,
        quantization_range=SECAGG_QUANTIZATION_RANGE,
        modulus_range=SECAGG_MODULUS_RANGE,
    )

    # --------------------------------------------------
    # 6. Create Flower compatibility context
    # --------------------------------------------------

    legacy_context = LegacyContext(
        context,
        config=ServerConfig(
            num_rounds=num_rounds
        ),
        strategy=strategy,
    )

    workflow = DefaultWorkflow(
        fit_workflow=secure_aggregation
    )

    # --------------------------------------------------
    # 7. Run federated training
    # --------------------------------------------------

    print(
        "\nStarting distributed DP + Secure Aggregation..."
    )

    workflow(
        grid,
        legacy_context
    )

    print(
        "\nFederated training workflow completed."
    )

    # --------------------------------------------------
    # 8. Retrieve final global model
    # --------------------------------------------------

    final_arrays = (
        legacy_context
        .state
        .array_records[MAIN_PARAMS_RECORD]
    )

    final_state = final_arrays.to_torch_state_dict()

    # --------------------------------------------------
    # 9. Save checkpoint
    # --------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    final_state = {
        key: value.detach().cpu()
        for key, value in final_state.items()
    }

    torch.save(
        final_state,
        MODEL_PATH
    )

    print("\nFinal global model saved:")
    print(MODEL_PATH)

    # --------------------------------------------------
    # 10. Save experiment metadata
    # --------------------------------------------------

    experiment_metrics = {

        "experiment": "distributed_dp_secagg",

        "num_clients": num_clients,
        "participation_fraction": fraction_train,
        "clients_per_round": expected_clients,
        "num_rounds": num_rounds,

        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,

        "aggregation_method": "equal_client_weighted_fedavg",
        "num_examples_per_client_for_aggregation": 1,

        "distributed_dp_enabled": True,
        "secure_aggregation_enabled": True,

        "privacy_mechanism": (
            "Client-update L2 clipping with independent "
            "Gaussian noise shares before SecAgg+"
        ),

        "client_update_clip_norm": CLIENT_UPDATE_CLIP_NORM,
        "noise_multiplier": DISTRIBUTED_NOISE_MULTIPLIER,
        "noise_share_std": noise_share_std,

        "minimum_assumed_honest_noise_contributors": (
            MIN_HONEST_NOISE_CONTRIBUTORS
        ),

        "positive_class_weight": (
            DISTRIBUTED_POSITIVE_CLASS_WEIGHT
        ),

        "secagg_num_shares": SECAGG_NUM_SHARES,
        "secagg_reconstruction_threshold": (
            SECAGG_RECONSTRUCTION_THRESHOLD
        ),
        "secagg_max_weight": SECAGG_MAX_WEIGHT,
        "secagg_clipping_range": SECAGG_CLIPPING_RANGE,
        "secagg_quantization_range": (
            SECAGG_QUANTIZATION_RANGE
        ),
        "secagg_modulus_range": SECAGG_MODULUS_RANGE,

        "formal_epsilon_accounting_available": False,

        "privacy_accounting_note": (
            "This experiment does not use the Opacus per-example "
            "DP-SGD accountant. A formal client-level privacy "
            "guarantee requires analysis of the actual aggregation, "
            "noise contributors, dropout behavior, and quantization."
        ),

        "honest_contributor_note": (
            "The minimum honest noise contributor count is an "
            "assumption for calibration. SecAgg+ reconstruction "
            "threshold does not enforce this assumption."
        ),

        "checkpoint": str(MODEL_PATH),
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            experiment_metrics,
            file,
            indent=4
        )

    print("\nExperiment metadata saved:")
    print(METRICS_PATH)

    print(
        "\n========== DISTRIBUTED DP + SECAGG COMPLETED =========="
    )
