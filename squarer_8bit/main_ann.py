import os
import subprocess
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from joblib import dump, load

# ──────────────────────────────────────────────
#            Configurable Parameters
# ──────────────────────────────────────────────
csv_path      = "verification_data/sim_data.csv"
model_path    = "verification_data/ann_model.joblib"
verilog_paths = [               # <<< Verilog source files
    "squarer.v"
]   

top_module    = "squarer"       # Top-level HDL module name
test_module   = "run_sims"      # Python file with test case (omit .py extension)
sim_tool      = "verilator"     # Simulation tool (e.g., "verilator", "icarus")
extra_args    = ""

random_runs = 1000
max_cycles = 50

# \/ \/ \/ Coverage goal features (must match column names in CSV)
coverage_features = ["result"]
input_features = ["a"]
input_ranges = {
    "a": 255
}

# \/ \/ \/ Coverage goals for each feature
coverage_goals = {
    f"prod_{n**2}": [n**2]
    for n in range(0, 255, 4)
}

test_random = "simulate_random"
test_model = "simulate_from_model"

# ──────────────────────────────────────────────
#               Helper Functions
# ──────────────────────────────────────────────

def train_ann(df, coverage_goals):
    from sklearn.metrics import mean_squared_error
    from itertools import product

    if df.empty:
        raise ValueError("Empty training data.")

    # Filter to training-relevant data
    rows = []
    for label, target_vals in coverage_goals.items():
        target_val = target_vals[0]
        matching = df[df["result"] == target_val]
        for _, row in matching.iterrows():
            rows.append({
                "target_result": target_val,
                **{k: row[k] for k in input_features}
            })

    if not rows:
        raise ValueError("No data to train on after filtering for coverage goals.")

    df_train = pd.DataFrame(rows)

    # Normalize inputs/outputs
    MAX_A = input_ranges["a"]
    MAX_RESULT = MAX_A ** 2

    def normalize_result(x):
        return np.log10(x + 1) / np.log10(MAX_RESULT + 1)

    def normalize_a(x):
        return x / MAX_A

    X = normalize_result(df_train["target_result"].values.reshape(-1, 1))
    y = normalize_a(df_train[input_features].values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter grid
    hidden_layer_opts = [(64, 64), (128, 64), (64, 64, 32)]
    activations = ["relu", "tanh"]
    learning_rates = [0.001, 0.005, 0.01]

    best_model = None
    best_mse = float("inf")
    best_config = None

    print("\n🔍 Searching for best ANN model...")

    for layers, act, lr in product(hidden_layer_opts, activations, learning_rates):
        try:
            model = MLPRegressor(
                hidden_layer_sizes=layers,
                activation=act,
                learning_rate_init=lr,
                max_iter=2000,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            print(f"  ✔️ Config: {layers}, act={act}, lr={lr} → MSE={mse:.6f}")

            if mse < best_mse:
                best_mse = mse
                best_model = model
                best_config = (layers, act, lr)
        except Exception as e:
            print(f"  ❌ Failed config {layers}, {act}, {lr}: {e}")

    if best_model is None:
        raise RuntimeError("Failed to train any ANN model")

    dump(best_model, model_path)
    print("\n✅ Best model saved.")
    print(f"   Layers: {best_config[0]}")
    print(f"   Activation: {best_config[1]}")
    print(f"   Learning rate: {best_config[2]}")
    print(f"   MSE: {best_mse:.6f}")

    return best_model

def get_uncovered_bins(df, coverage_goals):
    df = add_bin_vector_columns(df, coverage_goals)

    uncovered = []
    for label in coverage_goals:
        if df[label].sum() == 0:
            uncovered.append(label)
    return uncovered

def add_bin_vector_columns(df, coverage_goals):
    for label, values in coverage_goals.items():
        df[label] = df["result"].isin(values).astype(int)
    return df

def generate_ann_inputs(model, uncovered_labels):
    input_batch = []
    MAX_A = input_ranges["a"]
    MAX_RESULT = MAX_A ** 2

    def normalize_result(x):
        return np.log10(x + 1) / np.log10(MAX_RESULT + 1)

    for label in uncovered_labels:
        target_val = coverage_goals[label][0]
        normalized_target = np.log10(target_val + 1) / np.log10((input_ranges["a"] ** 2) + 1)

        # Predict normalized a
        base_prediction = model.predict([[normalized_target]])[0]
        predicted_a = int(round(np.clip(base_prediction, 0.0, 1.0) * MAX_A))
        predicted_result = predicted_a ** 2

        print(f"🔢 Predicting for {label}")
        print(f"   Target result:    {target_val}")
        print(f"   Predicted a:      {predicted_a}")
        print(f"   Predicted a²:     {predicted_result}")
        print(f"   Error:            {abs(predicted_result - target_val)}\n")

        input_batch.append([predicted_a])

    return input_batch

def clean_sim_data(csv_path, coverage_goals):
    df = pd.read_csv(csv_path)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Filter rows that match any coverage bin result
    goal_values = set(val[0] for val in coverage_goals.values())
    df = df[df["result"].isin(goal_values)]

    # Save cleaned data back
    df.to_csv(csv_path, index=False)
    print(f"🧹 Cleaned sim_data.csv: {len(df)} rows remain")

def run_cocotb_test(testcase, extra_env=None):
    env = os.environ.copy()
    env["TESTCASE"]        = testcase
    env["CSV_PATH"]        = csv_path
    env["VERILOG_SOURCES"] = " ".join(os.path.abspath(f) for f in verilog_paths)
    env["TOPLEVEL"]        = top_module
    env["MODULE"]          = test_module
    env["SIM"]             = sim_tool
    env["EXTRA_ARGS"]      = extra_args
    if extra_env:
        env.update(extra_env)
    subprocess.run(["make"], env=env)

# ──────────────────────────────────────────────
#                Verification Loop
# ──────────────────────────────────────────────

print(f"🔹 Running {random_runs} initial random simulations...")
run_cocotb_test(test_random, extra_env={"INIT_RUNS": str(random_runs)})
clean_sim_data(csv_path, coverage_goals)

df = pd.read_csv(csv_path)

model = train_ann(df, coverage_goals)
for cycle in range(max_cycles):
    df = pd.read_csv(csv_path)

    model = train_ann(df, coverage_goals)
    uncovered = get_uncovered_bins(df, coverage_goals)
    
    print(f"\nCycle {cycle + 1}: uncovered bins → {uncovered}")
    if not uncovered:
        print("🎉 All coverage goals met!")
        break

    input_batch = [
        [int(x) for x in row] for row in generate_ann_inputs(model, uncovered)
    ]


    print(f"Simulating {len(input_batch)} ANN-predicted inputs...")
    run_cocotb_test(test_model, extra_env={"MODEL_INPUTS": json.dumps(input_batch)})
    clean_sim_data(csv_path, coverage_goals)

print("✅ Verification loop complete.")


print("\n🧪 Final model test on each individual coverage bin...")
df = pd.read_csv(csv_path)
model = load(model_path)

for label in coverage_goals:
    target_val = coverage_goals[label][0]
    normalized_target = np.log10(target_val + 1) / np.log10((input_ranges["a"] ** 2) + 1)
    prediction = model.predict([[normalized_target]])[0]  # scalar float in [0, 1]

    max_val = input_ranges["a"]
    value = int(round(prediction * max_val))
    value = np.clip(value, 0, max_val)

    print(f"📦 Bin {label}: predicted input = {{'a': {value}}}")

# Plot predictions vs true values
print("\n📈 Plotting predictions vs coverage goals...")

MAX_A = input_ranges["a"]
MAX_RESULT = MAX_A ** 2

def normalize_result(x):
    return np.log10(x + 1) / np.log10(MAX_RESULT + 1)

predicted_a = []
true_results = []

for label in coverage_goals:
    target_val = coverage_goals[label][0]
    normalized_target = normalize_result(target_val)
    prediction = model.predict([[normalized_target]])[0]
    a_pred = int(round(np.clip(prediction, 0.0, 1.0) * MAX_A))

    predicted_a.append(a_pred)
    true_results.append(target_val)

# Convert to arrays
predicted_a = np.array(predicted_a)
true_results = np.array(true_results)
expected_results = predicted_a ** 2

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(predicted_a, true_results, label="Model Predictions", color="blue", s=50)
plt.plot(predicted_a, expected_results, label="Expected a²", linestyle="--", linewidth=2, color="orange")

plt.xlabel("Predicted a")
plt.ylabel("Coverage Goal Result (a²)")
plt.title("Model Prediction vs Coverage Goal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
