
import json
from pathlib import Path

import pandas as pd

from flwr.client import ClientApp
from flwr.client.mod import secaggplus_mod
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.common import Code

from src.client import (
    train_local_model_distributed_dp,
    CLIENT_UPDATE_CLIP_NORM,
    DISTRIBUTED_NOISE_MULTIPLIER,
    SELECTED_CLIENTS_PER_ROUND,
    MIN_HONEST_NOISE_CONTRIBUTORS,
    DISTRIBUTED_POSITIVE_CLASS_WEIGHT,
)
from src.model import MLP


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "results"
    / "selected_50_clients_seed42.parquet"
)

CLIENT_LIST_PATH = (
    PROJECT_DIR
    / "results"
    / "selected_clients_seed42.json"
)


# --------------------------------------------------
# Flower ClientApp
# --------------------------------------------------

app = ClientApp(mods=[secaggplus_mod])


@app.train()
def train(msg: Message, context):

    # --------------------------------------------------
    # 1. Identify the dataset client
    # --------------------------------------------------

    partition_id = int(
        context.node_config["partition-id"]
    )

    with open(CLIENT_LIST_PATH, "r") as file:
        selected_clients = json.load(file)

    if not 0 <= partition_id < len(selected_clients):
        raise ValueError(
            f"Invalid partition ID: {partition_id}"
        )

    client_id = selected_clients[partition_id]

    print(
        f"\nFlower partition {partition_id} "
        f"assigned to dataset client {client_id}"
    )

    # --------------------------------------------------
    # 2. Load this client's local data
    # --------------------------------------------------

    df = pd.read_parquet(
        DATA_PATH,
        filters=[("client", "==", client_id)],
    )

    if df.empty:
        raise ValueError(
            f"No records found for client {client_id}"
        )

    print("Actual local records:", len(df))

    # --------------------------------------------------
    # 3. Read global parameters from Flower
    # --------------------------------------------------

    if "fitins.parameters" not in msg.content:
        raise KeyError(
            "Missing fitins.parameters. "
            f"Available records: {list(msg.content.keys())}"
        )

    if "fitins.config" not in msg.content:
        raise KeyError(
            "Missing fitins.config. "
            f"Available records: {list(msg.content.keys())}"
        )

    global_parameters = msg.content["fitins.parameters"]
    train_config = msg.content["fitins.config"]

    if not isinstance(global_parameters, ArrayRecord):
        raise TypeError(
            "fitins.parameters is not an ArrayRecord"
        )

    if not isinstance(train_config, ConfigRecord):
        raise TypeError(
            "fitins.config is not a ConfigRecord"
        )

    received_state = (
        global_parameters.to_torch_state_dict()
    )

    model_keys = list(MLP().state_dict().keys())

    if len(received_state) != len(model_keys):
        raise ValueError(
            "Model parameter count mismatch: "
            f"received {len(received_state)}, "
            f"expected {len(model_keys)}"
        )

    global_state = {
        key: value.detach().cpu().clone()
        for key, value in zip(
            model_keys,
            received_state.values()
        )
    }

    # --------------------------------------------------
    # 4. Read local training configuration
    # --------------------------------------------------

    epochs = int(
        context.run_config["local-epochs"]
    )

    batch_size = int(
        context.run_config["batch-size"]
    )

    learning_rate = float(
        context.run_config["learning-rate"]
    )

    print(
        "\nTraining configuration:"
        f"\nEpochs: {epochs}"
        f"\nBatch size: {batch_size}"
        f"\nLearning rate: {learning_rate}"
        f"\nUpdate clipping norm: {CLIENT_UPDATE_CLIP_NORM}"
        f"\nNoise multiplier: {DISTRIBUTED_NOISE_MULTIPLIER}"
        f"\nSelected clients: {SELECTED_CLIENTS_PER_ROUND}"
        f"\nMinimum honest contributors: "
        f"{MIN_HONEST_NOISE_CONTRIBUTORS}"
        f"\nPositive class weight: "
        f"{DISTRIBUTED_POSITIVE_CLASS_WEIGHT}"
    )

    # --------------------------------------------------
    # 5. Distributed-DP local training
    # --------------------------------------------------

    (
        updated_state,
        num_records,
        losses,
        update_norm,
        clipped_norm,
        clip_factor,
        noise_share_std,
    ) = train_local_model_distributed_dp(
        initial_state=global_state,
        client_data=df,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        clip_norm=CLIENT_UPDATE_CLIP_NORM,
        noise_multiplier=DISTRIBUTED_NOISE_MULTIPLIER,
        selected_clients=SELECTED_CLIENTS_PER_ROUND,
        min_honest_clients=MIN_HONEST_NOISE_CONTRIBUTORS,
        positive_class_weight=DISTRIBUTED_POSITIVE_CLASS_WEIGHT,
    )

    # --------------------------------------------------
    # 6. Prepare updated model parameters
    # --------------------------------------------------

    updated_arrays = ArrayRecord(
        torch_state_dict=updated_state
    )

    # --------------------------------------------------
    # 7. Prepare training metrics
    # --------------------------------------------------

    metrics = ConfigRecord(
        {
            "actual-records": int(num_records),
            "train-loss": float(losses[-1]),
            "client-id": str(client_id),
            "client-update-norm": float(update_norm),
            "clipped-update-norm": float(clipped_norm),
            "update-clip-factor": float(clip_factor),
            "noise-share-std": float(noise_share_std),
        }
    )

    # --------------------------------------------------
    # 8. Equal client weighting for SecAgg+
    #
    # Each client contributes weight 1.
    # Actual dataset size is recorded separately.
    # --------------------------------------------------

    num_examples_record = MetricRecord(
        {
            "num_examples": 1
        }
    )

    # --------------------------------------------------
    # 9. Flower response status
    # --------------------------------------------------

    status_record = ConfigRecord(
        {
            "code": Code.OK.value,
            "message": "",
        }
    )

    # --------------------------------------------------
    # 10. Construct Flower response
    # --------------------------------------------------

    reply_content = RecordDict(
        {
            "fitres.parameters": updated_arrays,
            "fitres.metrics": metrics,
            "fitres.num_examples": num_examples_record,
            "fitres.status": status_record,
        }
    )

    print(
        f"\nClient {client_id} training complete."
        f"\nActual records: {num_records}"
        f"\nTraining loss: {losses[-1]:.6f}"
        f"\nOriginal update norm: {update_norm:.6f}"
        f"\nClipped update norm: {clipped_norm:.6f}"
        f"\nNoise share std: {noise_share_std:.6f}"
        f"\nFlower aggregation weight: 1"
    )

    return Message(
        content=reply_content,
        reply_to=msg,
    )
