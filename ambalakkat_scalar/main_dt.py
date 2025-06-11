import os
import subprocess
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load

# ──────────────────────────────────────────────
#            Configurable Parameters
# ──────────────────────────────────────────────
csv_path      = "verification_data/sim_data.csv"
model_path    = "verification_data/dt_model.joblib"
verilog_paths = [
    "comparator.v"
] 

top_module    = "comparator"
test_module   = "run_sims"
sim_tool      = "verilator"
extra_args    = ""

initial_random_runs = 20
max_cycles = 50

coverage_features = ["match"]
input_features = ["a", "b"]
input_ranges = {
    "a": 15,
    "b": 15
}

coverage_goals = {
    "prod_1":   lambda row: row["match"] == 1,
    "prod_4":   lambda row: row["match"] == 4,
    "prod_9":   lambda row: row["match"] == 9,
    "prod_16":  lambda row: row["match"] == 16,
    "prod_25":  lambda row: row["match"] == 25,
    "prod_36":  lambda row: row["match"] == 36,
    "prod_49":  lambda row: row["match"] == 49,
    "prod_64":  lambda row: row["match"] == 64,
    "prod_81":  lambda row: row["match"] == 81,
    "prod_100": lambda row: row["match"] == 100,
    "prod_121": lambda row: row["match"] == 121,
    "prod_144": lambda row: row["match"] == 144,
    "prod_169": lambda row: row["match"] == 169,
    "prod_196": lambda row: row["match"] == 196,
    "prod_225": lambda row: row["match"] == 225,
}

test_random = "simulate_random"
test_model = "simulate_from_model"

# ──────────────────────────────────────────────
#               Helper Functions
# ──────────────────────────────────────────────

def train_dt(df, coverage_goals):
    df = add_bin_vector_columns(df, coverage_goals)
    X = df[list(coverage_goals.keys())].values  # multi-hot bin vector
    y = df[input_features].copy()

    for col in input_features:
        y[col] = y[col] / input_ranges[col]

    X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=0.2, random_state=42)

    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    dump(model, model_path)
    print(f"✅ Decision Tree model trained and saved to {model_path}")
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

def generate_ann_inputs(model, uncovered_labels, coverage_goals, input_features, input_ranges):
    input_batch = []

    for label in uncovered_labels:
        goal_vector = np.array([[1 if l == label else 0 for l in coverage_goals]])

        prediction = model.predict(goal_vector)[0]

        input_values = []
        for i, feature in enumerate(input_features):
            max_val = input_ranges[feature]
            value = int(round(prediction[i] * max_val))
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

    model = train_dt(df, coverage_goals)
    uncovered = get_uncovered_bins(df, coverage_goals)
    
    print(f"\nCycle {cycle + 1}: uncovered bins → {uncovered}")
    if not uncovered:
        print("🎉 All coverage goals met!")
        break

    input_batch = [
        [int(x) for x in row] for row in generate_ann_inputs(model, uncovered, coverage_goals, input_features, input_ranges)
    ]

    print(f"Simulating {len(input_batch)} DT-predicted inputs...")
    run_cocotb_test(test_model, extra_env={"MODEL_INPUTS": json.dumps(input_batch)})

print("✅ Verification loop complete.")
