
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

from src.client import train_local_model
from src.model import MLP


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "results" / "selected_50_clients_seed42.parquet"
CLIENT_LIST_PATH = PROJECT_DIR / "results" / "selected_clients_seed42.json"

app = ClientApp(mods=[secaggplus_mod])


@app.train()
def train(msg: Message, context):
    partition_id = int(context.node_config["partition-id"])

    with open(CLIENT_LIST_PATH, "r") as file:
        selected_clients = json.load(file)

    if not 0 <= partition_id < len(selected_clients):
        raise ValueError(f"Invalid partition ID: {partition_id}")

    client_id = selected_clients[partition_id]

    print(
        f"\nFlower partition {partition_id} "
        f"assigned to dataset client {client_id}"
    )

    df = pd.read_parquet(
        DATA_PATH,
        filters=[("client", "==", client_id)],
    )

    if df.empty:
        raise ValueError(f"No records found for client {client_id}")

    print("Local records:", len(df))

    if "fitins.parameters" not in msg.content:
        raise KeyError(
            f"Missing fitins.parameters. Available: {list(msg.content.keys())}"
        )

    if "fitins.config" not in msg.content:
        raise KeyError(
            f"Missing fitins.config. Available: {list(msg.content.keys())}"
        )

    global_parameters = msg.content["fitins.parameters"]
    train_config = msg.content["fitins.config"]

    if not isinstance(global_parameters, ArrayRecord):
        raise TypeError("fitins.parameters is not an ArrayRecord")

    if not isinstance(train_config, ConfigRecord):
        raise TypeError("fitins.config is not a ConfigRecord")

    received_state = global_parameters.to_torch_state_dict()
    model_keys = list(MLP().state_dict().keys())

    if len(received_state) != len(model_keys):
        raise ValueError(
            f"Model parameter count mismatch: received "
            f"{len(received_state)}, expected {len(model_keys)}"
        )

    global_state = dict(zip(model_keys, received_state.values()))

    epochs = int(context.run_config["local-epochs"])
    batch_size = int(context.run_config["batch-size"])
    learning_rate = float(context.run_config["learning-rate"])

    print(
        f"Training configuration: epochs={epochs}, "
        f"batch_size={batch_size}, lr={learning_rate}"
    )

    accountant_state = None

    if "privacy" in context.state:
        privacy_record = context.state["privacy"]
        accountant_json = privacy_record["accountant_state"]
        accountant_state = json.loads(accountant_json)
        print("Previous privacy accountant state found.")
    else:
        print("No previous privacy history. Starting fresh.")

    (
        updated_state,
        num_records,
        losses,
        updated_accountant_state,
        epsilon,
    ) = train_local_model(
        initial_state=global_state,
        client_data=df,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        accountant_state=accountant_state,
    )

    if updated_accountant_state is not None:
        accountant_json = json.dumps(updated_accountant_state)
        context.state["privacy"] = ConfigRecord(
            {"accountant_state": accountant_json}
        )
        print(f"Saved cumulative epsilon: {epsilon:.6f}")

    updated_arrays = ArrayRecord(torch_state_dict=updated_state)

    metrics_data = {
        "num-examples": int(num_records),
        "train-loss": float(losses[-1]),
    }

    if epsilon is not None:
        metrics_data["local-epsilon"] = float(epsilon)

    # Flower compatibility records required by recorddict_to_fitres
    metrics = ConfigRecord(metrics_data)

    num_examples_record = MetricRecord(
        {"num_examples": int(num_records)}
    )

    status_record = ConfigRecord(
    {
        "code": Code.OK.value,
        "message": "",
    }
    )

    reply_content = RecordDict(
        {
            "fitres.parameters": updated_arrays,
            "fitres.metrics": metrics,
            "fitres.num_examples": num_examples_record,
            "fitres.status": status_record,
        }
    )

    print(
        f"Client {client_id} training complete. "
        f"Examples={num_records}, loss={losses[-1]:.6f}"
    )

    return Message(content=reply_content, reply_to=msg)
