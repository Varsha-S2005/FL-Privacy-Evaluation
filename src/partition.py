
from pathlib import Path
import json

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = Path(
    r"C:\Users\91991\Downloads\model_ready.parquet"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

SEED = 42
NUM_CLIENTS = 50

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

REQUIRED_COLUMNS = (
    FEATURE_COLUMNS + ["y_bin", "client", "is_train"]
)


# --------------------------------------------------
# Load the required dataset columns
# --------------------------------------------------

def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")

    df = pd.read_parquet(
        DATA_PATH,
        columns=REQUIRED_COLUMNS
    )

    print("Total records loaded:", len(df))

    # --------------------------------------------------
    # Separate training clients from test clients
    # --------------------------------------------------

    train_df = df[df["is_train"] == True].copy()
    test_df = df[df["is_train"] == False]

    train_clients = train_df["client"].unique()
    test_clients = test_df["client"].unique()

    print("\nTraining clients:", len(train_clients))
    print("Test clients:", len(test_clients))

    # --------------------------------------------------
    # Randomly select 50 training clients
    # --------------------------------------------------

    rng = np.random.default_rng(SEED)

    selected_clients = rng.choice(
        train_clients,
        size=NUM_CLIENTS,
        replace=False
    )

    selected_clients = sorted(selected_clients.tolist())

    print("\nSelected clients:", len(selected_clients))

    # --------------------------------------------------
    # Keep ALL records belonging to selected clients
    # --------------------------------------------------

    selected_df = train_df[
        train_df["client"].isin(selected_clients)
    ].copy()

    # --------------------------------------------------
    # Verify the partition
    # --------------------------------------------------

    actual_client_count = selected_df["client"].nunique()

    assert actual_client_count == NUM_CLIENTS
    assert selected_df["is_train"].all()

    overlap = set(selected_clients).intersection(
        set(test_clients)
    )

    assert len(overlap) == 0

    print("Records belonging to selected clients:", len(selected_df))
    print("Verified unique clients:", actual_client_count)
    print("Overlap with test clients:", len(overlap))

    # --------------------------------------------------
    # Generate per-client distribution report
    # --------------------------------------------------

    summary = (
        selected_df.groupby("client")
        .agg(
            total_records=("y_bin", "size"),
            genuine_records=("y_bin", lambda x: (x == 0).sum()),
            attack_records=("y_bin", lambda x: (x == 1).sum()),
            attack_percentage=("y_bin", lambda x: 100 * (x == 1).mean())
        )
        .reset_index()
    )

    summary["attack_percentage"] = (
        summary["attack_percentage"].round(2)
    )

    # --------------------------------------------------
    # Save partition outputs
    # --------------------------------------------------

    selected_df = selected_df.drop(columns=["is_train"])

    selected_df.to_parquet(
        RESULTS_DIR / "selected_50_clients_seed42.parquet",
        index=False
    )

    summary.to_csv(
        RESULTS_DIR / "client_distribution_seed42.csv",
        index=False
    )

    with open(
        RESULTS_DIR / "selected_clients_seed42.json",
        "w"
    ) as file:
        json.dump(selected_clients, file, indent=4)

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print("\nClient distribution:")
    print(summary.to_string(index=False))

    print("\nOverall selected cohort distribution:")
    print(selected_df["y_bin"].value_counts().sort_index())

    print("\nPartition completed successfully.")

    print("\nSaved files:")
    print("1. results/selected_50_clients_seed42.parquet")
    print("2. results/client_distribution_seed42.csv")
    print("3. results/selected_clients_seed42.json")


if __name__ == "__main__":
    main()
