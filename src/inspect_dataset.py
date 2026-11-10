import pandas as pd

DATA_PATH = r"C:\Users\91991\Downloads\model_ready.parquet"

columns = [
    "client", "receiver", "run",
    "y_bin", "y_multi",
    "attackerType", "attack_type", "is_train"
]

df = pd.read_parquet(DATA_PATH, columns=columns)

print("\n1. LABEL DISTRIBUTION")
print("\nBinary labels:")
print(df["y_bin"].value_counts(dropna=False).sort_index())

print("\nMulticlass labels:")
print(df["y_multi"].value_counts(dropna=False).sort_index())

print("\n2. CLIENT INFORMATION")
print("Unique client IDs:", df["client"].nunique())
print("Unique receivers:", df["receiver"].nunique())
print("Unique simulation runs:", df["run"].nunique())

print("\nRows per client:")
print(df.groupby("client").size().describe())

print("\nDoes each client map to one receiver?")
print(df.groupby("client")["receiver"].nunique().value_counts())

print("\nDoes each client map to one run?")
print(df.groupby("client")["run"].nunique().value_counts())

print("\n3. TRAIN / TEST SPLIT")
print(df["is_train"].value_counts(dropna=False))

train = df[df["is_train"] == True]
test = df[df["is_train"] == False]

print("Training rows:", len(train))
print("Testing rows:", len(test))

train_clients = set(train["client"].unique())
test_clients = set(test["client"].unique())

print("Training clients:", len(train_clients))
print("Testing clients:", len(test_clients))
print("Overlapping clients:", len(train_clients & test_clients))

train_receivers = set(train["receiver"].unique())
test_receivers = set(test["receiver"].unique())

print("Overlapping receiver IDs:", len(train_receivers & test_receivers))

print("\n4. ATTACK LABELS")
print("attackerType:")
print(df["attackerType"].value_counts(dropna=False).sort_index())

print("\nattack_type:")
print(df["attack_type"].value_counts(dropna=False).sort_index())

print("\n5. CLIENT DATA DISTRIBUTION")
print(
    df.groupby("client")
      .size()
      .describe(percentiles=[0.25, 0.50, 0.75, 0.90])
)