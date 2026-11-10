
import json
from pathlib import Path
from collections import defaultdict

import torch

from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig
from flwr.server.compat import LegacyContext
from flwr.server.strategy import FedAvg as LegacyFedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from flwr.server.workflow.default_workflows import MAIN_PARAMS_RECORD
from flwr.serverapp import ServerApp

from src.model import MLP


PROJECT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
RESULTS_DIR = PROJECT_DIR / "results" / "dp_secagg"

MODEL_PATH = CHECKPOINT_DIR / "dp_secagg_model.pth"
METRICS_PATH = RESULTS_DIR / "training_metrics.json"

# --------------------------------------------------
# Privacy tracking
# --------------------------------------------------

client_privacy = {}
round_privacy = defaultdict(list)


class PrivacyTrackingFedAvg(LegacyFedAvg):

    def aggregate_fit(self, server_round, results, failures):

        reports = []

        for _, fit_res in results:

            metrics = fit_res.metrics

            if "local-epsilon" not in metrics:
                continue

            client_id = str(metrics["client-id"])
            epsilon = float(metrics["local-epsilon"])

            client_privacy[client_id] = epsilon

            reports.append({
                "client_id": client_id,
                "epsilon": epsilon
            })

        round_privacy[server_round] = reports

        print(f"\nPrivacy reports for round {server_round}:")

        for report in reports:
            print(
                f"Client {report['client_id']}: "
                f"epsilon={report['epsilon']:.6f}"
            )

        if reports:
            values = [report["epsilon"] for report in reports]

            print(f"Round maximum epsilon: {max(values):.6f}")
            print(
                f"Round mean epsilon: "
                f"{sum(values) / len(values):.6f}"
            )

        return super().aggregate_fit(
            server_round,
            results,
            failures
        )


app = ServerApp()


@app.main()
def main(grid, context):

    # --------------------------------------------------
    # Read experiment configuration
    # --------------------------------------------------

    num_rounds = int(context.run_config["num-server-rounds"])
    fraction_train = float(context.run_config["fraction-train"])
    local_epochs = int(context.run_config["local-epochs"])
    batch_size = int(context.run_config["batch-size"])
    learning_rate = float(context.run_config["learning-rate"])

    num_clients = 50
    expected_clients = max(1, int(num_clients * fraction_train))

    print("\n========== DP + SECURE AGGREGATION ==========")
    print("Total virtual clients:", num_clients)
    print("Expected clients per round:", expected_clients)
    print("Communication rounds:", num_rounds)
    print("Local epochs:", local_epochs)
    print("Batch size:", batch_size)
    print("Learning rate:", learning_rate)

    # Reset tracking for this experiment.
    client_privacy.clear()
    round_privacy.clear()

    # --------------------------------------------------
    # Initialize global model
    # --------------------------------------------------

    global_model = MLP()

    initial_parameters = ndarrays_to_parameters(
        [
            tensor.detach().cpu().numpy()
            for tensor in global_model.state_dict().values()
        ]
    )

    # --------------------------------------------------
    # Initialize privacy-tracking FedAvg
    # --------------------------------------------------

    strategy = PrivacyTrackingFedAvg(
        fraction_fit=fraction_train,
        fraction_evaluate=0.0,
        min_fit_clients=expected_clients,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
    )

    # --------------------------------------------------
    # Secure Aggregation configuration
    # --------------------------------------------------

    secure_aggregation = SecAggPlusWorkflow(
        num_shares=3,
        reconstruction_threshold=2,
        max_weight=10000.0,
        clipping_range=8.0,
        quantization_range=4194304,
        modulus_range=4294967296,
    )

    # --------------------------------------------------
    # Run Flower compatibility workflow
    # --------------------------------------------------

    legacy_context = LegacyContext(
        context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    workflow = DefaultWorkflow(
        fit_workflow=secure_aggregation,
    )

    print("\nStarting DP + Secure Aggregation workflow...")

    workflow(grid, legacy_context)

    # --------------------------------------------------
    # Save final global model
    # --------------------------------------------------

    final_arrays = legacy_context.state.array_records[
        MAIN_PARAMS_RECORD
    ]

    final_state = final_arrays.to_torch_state_dict()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(final_state, MODEL_PATH)

    print("\nFinal global model saved:", MODEL_PATH)

    # --------------------------------------------------
    # Calculate privacy summaries
    # --------------------------------------------------

    all_epsilons = list(client_privacy.values())

    privacy_summary = {
        "clients_with_privacy_reports": len(all_epsilons),
        "maximum_client_epsilon": (
            max(all_epsilons) if all_epsilons else None
        ),
        "mean_client_epsilon": (
            sum(all_epsilons) / len(all_epsilons)
            if all_epsilons else None
        ),
        "delta": 1e-5,
        "client_epsilon": client_privacy,
        "round_privacy": {
            str(round_number): reports
            for round_number, reports in round_privacy.items()
        },
    }

    # --------------------------------------------------
    # Save experiment metadata and privacy reports
    # --------------------------------------------------

    experiment_metrics = {
        "experiment": "dp_secagg",
        "num_clients": num_clients,
        "participation_fraction": fraction_train,
        "expected_clients_per_round": expected_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dp_enabled": True,
        "secure_aggregation_enabled": True,
        "noise_multiplier": 1.0,
        "max_grad_norm": 1.0,
        "delta": 1e-5,
        "secagg_num_shares": 3,
        "secagg_reconstruction_threshold": 2,
        "secagg_max_weight": 10000.0,
        "secagg_clipping_range": 8.0,
        "privacy_summary": privacy_summary,
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(experiment_metrics, file, indent=4)

    print("\n========== PRIVACY SUMMARY ==========")
    print(
        "Clients with reports:",
        privacy_summary["clients_with_privacy_reports"]
    )
    print(
        "Maximum client epsilon:",
        privacy_summary["maximum_client_epsilon"]
    )
    print(
        "Mean client epsilon:",
        privacy_summary["mean_client_epsilon"]
    )

    print("\nExperiment metadata saved:", METRICS_PATH)
    print("\n========== DP + SECURE AGGREGATION COMPLETED ==========")
