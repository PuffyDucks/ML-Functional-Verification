import os
import subprocess
import json
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from joblib import dump, load

# ──────────────────────────────────────────────
#            Configurable Parameters
# ──────────────────────────────────────────────
csv_path      = "verification_data/sim_data.csv"
model_path    = "verification_data/ann_model.joblib"
verilog_paths = [               # <<< Verilog source files
    "multiplier.v"
] 

top_module    = "multiplier"    # Top-level HDL module name
test_module   = "run_sims"      # Python file with test case (omit .py extension)
sim_tool      = "verilator"     # Simulation tool (e.g., "verilator", "icarus")
extra_args    = ""

initial_random_runs = 10
max_cycles = 50

# \/ \/ \/ Coverage goal features (must match column names in CSV)
coverage_features = ["product"]
input_features = ["a", "b"]
input_ranges = {
    "a": 31,
    "b": 31
}

# \/ \/ \/ Coverage goals for each feature
coverage_goals = {
    "gt": lambda row: row["gt"] == 1,
    "lt": lambda row: row["lt"] == 1,
    "eq": lambda row: row["eq"] == 1,
}

test_random = "simulate_random"
test_model = "simulate_from_model"

# ──────────────────────────────────────────────
#               Helper Functions
# ──────────────────────────────────────────────

def train_ann(df, coverage_goals):
    df = add_bin_vector_columns(df, coverage_goals)

    # Remove rows where no coverage bins were hit
    bin_cols = list(coverage_goals.keys())
    df["bin_sum"] = df[bin_cols].sum(axis=1)
    df = df[df["bin_sum"] > 0].drop(columns=["bin_sum"])

    X = df[bin_cols].values  # multi-hot bin vector
    y = df[input_features].copy()

    # Normalize target inputs
    for col in input_features:
        y[col] = y[col] / input_ranges[col]

    X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=0.2, random_state=42)

    model = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    dump(model, model_path)
    print(f"✅ Model trained and saved to {model_path}")
    return model


def get_uncovered_bins(df, coverage_goals):
    df = add_bin_vector_columns(df, coverage_goals)

    uncovered = []
    for label in coverage_goals:
        if df[label].sum() == 0:
            uncovered.append(label)
    return uncovered

def add_bin_vector_columns(df, coverage_goals):
    for label, condition_fn in coverage_goals.items():
        df[label] = df.apply(condition_fn, axis=1).astype(int)
    return df

def generate_ann_inputs(model, uncovered_labels, coverage_goals, input_features, input_ranges, noise_std=0.05, num_variants=5):
    """
    Generate multiple ANN-predicted inputs per uncovered coverage bin,
    with Gaussian noise added in normalized space.

    noise_std: standard deviation of Gaussian noise to add
    num_variants: number of input variants to generate per bin
    """
    input_batch = []

    for label in uncovered_labels:
        goal_vector = np.array([[1 if l == label else 0 for l in coverage_goals]])
        base_prediction = model.predict(goal_vector)[0]

        for _ in range(num_variants):
            # Add Gaussian noise
            noisy = base_prediction + np.random.normal(0, noise_std, size=len(base_prediction))
            noisy = np.clip(noisy, 0.0, 1.0)

            # De-normalize
            input_values = []
            for i, feature in enumerate(input_features):
                max_val = input_ranges[feature]
                value = int(round(noisy[i] * max_val))
                value = np.clip(value, 0, max_val)
                input_values.append(value)

            input_batch.append(input_values)

    return input_batch

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

print(f"🔹 Running {initial_random_runs} initial random simulations...")
run_cocotb_test(test_random, extra_env={"INIT_RUNS": str(initial_random_runs)})

for cycle in range(max_cycles):
    df = pd.read_csv(csv_path)

    model = train_ann(df, coverage_goals)
    uncovered = get_uncovered_bins(df, coverage_goals)
    
    print(f"\nCycle {cycle + 1}: uncovered bins → {uncovered}")
    if not uncovered:
        print("🎉 All coverage goals met!")
        break

    input_batch = [
        [int(x) for x in row] for row in generate_ann_inputs(
                                            model,
                                            uncovered,
                                            coverage_goals,
                                            input_features,
                                            input_ranges,
                                            noise_std=0.0,
                                            num_variants=1
                                        )
    ]


    print(f"Simulating {len(input_batch)} ANN-predicted inputs...")
    run_cocotb_test(test_model, extra_env={"MODEL_INPUTS": json.dumps(input_batch)})

print("✅ Verification loop complete.")
