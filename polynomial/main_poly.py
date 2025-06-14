import os
import subprocess
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from joblib import dump, load

# ──────────────────────────────────────────────
#            Configurable Parameters
# ──────────────────────────────────────────────
csv_path = "verification_data/sim_data.csv"
model_path = "verification_data/poly_model.joblib"
verilog_paths = ["polynomial.v"]

top_module = "polynomial"
test_module = "run_sims"
sim_tool = "verilator"
extra_args = ""

random_runs = 100

input_features = ["a"]
input_ranges = {"a": 1023}
MAX_A = input_ranges["a"]
MAX_RESULT = max(((a * (a - 600) * (a - 940)) >> 15) + 700 for a in range(1024))

coverage_goals = {
    f"prod_{((a * (a - 600) * (a - 940)) >> 15) + 700}": [((a * (a - 600) * (a - 940)) >> 15) + 700]
    for a in range(0, 1023, 4)
}

test_random = "simulate_random"
test_model = "simulate_from_model"

# ──────────────────────────────────────────────
#                  Helpers
# ──────────────────────────────────────────────

def normalize_result(x):
    return np.log10(x + 1) / np.log10(MAX_RESULT + 1)

def train_polynomial_model(df, coverage_goals):
    if df.empty:
        raise ValueError("Empty training data.")

    rows = []
    for label, target_vals in coverage_goals.items():
        target_val = target_vals[0]
        matching = df[df["result"] == target_val]
        for _, row in matching.iterrows():
            rows.append({
                "target_result": target_val,
                **{k: row[k] for k in input_features}
            })

    if len(rows) < 5:
        raise ValueError(f"Insufficient data to train on after filtering for coverage goals (found {len(rows)} rows, need at least 5).")

    df_train = pd.DataFrame(rows)

    X = normalize_result(df_train["target_result"].values.reshape(-1, 1))
    y = df_train["a"].values.reshape(-1, 1) / MAX_A  # normalized output

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model = None
    best_mse = float("inf")
    best_degree = None

    print("\n🔍 Tuning polynomial regression...")

    for degree in range(2, 6):  # try degrees 2 to 5
        try:
            model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"  ✔️ Degree {degree} → MSE = {mse:.6f}")

            if mse < best_mse:
                best_mse = mse
                best_model = model
                best_degree = degree
        except Exception as e:
            print(f"  ❌ Degree {degree} failed: {e}")

    if best_model is None:
        raise RuntimeError("Polynomial training failed")

    dump(best_model, model_path)
    print("\n✅ Best polynomial model saved.")
    print(f"   Degree: {best_degree}")
    print(f"   MSE:    {best_mse:.6f}")

    return best_model, best_mse

def get_uncovered_bins(df, coverage_goals):
    df = add_bin_vector_columns(df, coverage_goals)
    return [label for label in coverage_goals if df[label].sum() == 0]

def add_bin_vector_columns(df, coverage_goals):
    for label, values in coverage_goals.items():
        df[label] = df["result"].isin(values).astype(int)
    return df

def generate_poly_inputs(model, uncovered_labels):
    input_batch = []

    for label in uncovered_labels:
        target_val = coverage_goals[label][0]
        normalized_target = normalize_result(target_val)
        predicted = model.predict([[normalized_target]])[0][0]
        predicted_a = int(round(np.clip(predicted, 0.0, 1.0) * MAX_A))
        predicted_result = predicted_a ** 2
        input_batch.append(predicted_a)

    return input_batch

def clean_sim_data(csv_path, coverage_goals):
    df = pd.read_csv(csv_path).drop_duplicates()
    goal_values = set(val[0] for val in coverage_goals.values())
    df = df[df["result"].isin(goal_values)]
    df.to_csv(csv_path, index=False)
    print(f"🧹 Cleaned sim_data.csv: {len(df)} rows remain")

def run_cocotb_test(testcase, extra_env=None):
    if not hasattr(run_cocotb_test, "_has_built"):
        run_cocotb_test._has_built = False  # static var

    env = os.environ.copy()
    env.update({
        "TESTCASE": testcase,
        "CSV_PATH": csv_path,
        "VERILOG_SOURCES": " ".join(os.path.abspath(f) for f in verilog_paths),
        "TOPLEVEL": top_module,
        "MODULE": test_module,
        "SIM": sim_tool,
        "EXTRA_ARGS": extra_args,
        "PYTHONPATH": os.getcwd(),
    })
    if extra_env:
        env.update(extra_env)

    sim_binary = os.path.join("sim_build", f"Vtop")

    if not run_cocotb_test._has_built or not os.path.exists(sim_binary):
        print("🛠️  Running make (first time or binary missing)...")
        subprocess.run(["make"], env=env, check=True)
        run_cocotb_test._has_built = True
    else:
        print("⚡ Reusing simulator binary...")
        subprocess.run([sim_binary], env=env, check=True)

# ──────────────────────────────────────────────
#              Verification Loop
# ──────────────────────────────────────────────

# Clear CSV
if os.path.exists(csv_path):
    os.remove(csv_path)

sim_counts = []
coverage_progress = []
total_bins = len(coverage_goals)
mse_progress = []
total_sims = 0
cycle = 0

while True:
    # Random simulation phase
    print(f"🔹 Running {random_runs} random simulations...")
    # Python subprocess call in run_cocotb_test
    seed_offset = cycle * random_runs  # for example
    run_cocotb_test(test_random, extra_env={"RUNS": str(random_runs), "SEED_OFFSET": str(seed_offset)})
    clean_sim_data(csv_path, coverage_goals)

    total_sims += random_runs
    df = pd.read_csv(csv_path)
    uncovered = get_uncovered_bins(df, coverage_goals)
    covered = total_bins - len(uncovered)

    sim_counts.append(total_sims)
    coverage_progress.append(covered)

    print(f"\n[Random] Simulations run: {total_sims}, Covered bins: {covered}/{total_bins}")

    # Model training phase
    print("\n📊 Training polynomial regression model...")
    df = pd.read_csv(csv_path)
    if len(df) < 5:
        print(f"Insufficient data in CSV for training (found {len(df)} rows, need at least 5).")
        continue
    model, current_mse = train_polynomial_model(df, coverage_goals)
    mse_progress.append(current_mse)

    uncovered = get_uncovered_bins(df, coverage_goals)
    print(f"❗ Uncovered bins → {uncovered}")

    if not uncovered:
        print("🎉 All coverage goals met!")
        break

    # Simulate uncovered bins phase
    input_batch = generate_poly_inputs(model, uncovered)
    print(f"Simulating {len(input_batch)} predicted inputs...")
    run_cocotb_test(test_model, extra_env={"MODEL_INPUTS": json.dumps(input_batch)})
    clean_sim_data(csv_path, coverage_goals)
    

    total_sims += len(input_batch)

    df = pd.read_csv(csv_path)
    uncovered = get_uncovered_bins(df, coverage_goals)
    covered = total_bins - len(uncovered)

    sim_counts.append(total_sims)
    coverage_progress.append(covered)

    print(f"[Model] Simulations run: {total_sims}, Covered bins: {covered}/{total_bins}")
    cycle += 1

print("✅ Verification loop complete.")

# ──────────────────────────────────────────────
#          Final Prediction + Plotting
# ──────────────────────────────────────────────

print("\n🧪 Final model test on each individual coverage bin...")

model = load(model_path)
predicted_a = []
true_results = []

for label in coverage_goals:
    target_val = coverage_goals[label][0]
    normalized_target = normalize_result(target_val)
    predicted = model.predict([[normalized_target]])[0][0]
    a_pred = int(round(np.clip(predicted, 0.0, 1.0) * MAX_A))
    predicted_a.append(a_pred)
    true_results.append(target_val)

# Plotting
predicted_a = np.array(predicted_a)
true_results = np.array(true_results)
expected_results = ((predicted_a * (predicted_a - 600) * (predicted_a - 940)) >> 15) + 700

plt.figure(figsize=(10, 6))
plt.scatter(predicted_a, true_results, label="Model Predictions", color="blue", s=50)
plt.plot(predicted_a, expected_results, label="Expected a²", linestyle="--", linewidth=2, color="orange")

plt.xlabel("Predicted a")
plt.ylabel("Coverage Goal Result")
plt.title("Polynomial Regression: Prediction vs Coverage Goal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────
#           Coverage vs Simulation Count
# ──────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.plot(sim_counts, coverage_progress, marker='o')
plt.xlabel("Total Simulations Run")
plt.ylabel("Covered Bins")
plt.title("Coverage Progress vs Simulations")
plt.grid(True)
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────
#              Loss Plotting (MSE)
# ──────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.plot(sim_counts[:len(mse_progress)], mse_progress, marker='o')
plt.xlabel("Total Simulations Run")
plt.ylabel("Model MSE (Loss)")
plt.title("Model Loss (MSE) vs Simulations")
plt.grid(True)
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────
#         Save metrics for future reference
# ──────────────────────────────────────────────
print("📊 All metrics saved to verification_data/poly_metrics.npz")
np.savez("verification_data/poly_metrics.npz",
         sim_counts=np.array(sim_counts),
         coverage=np.array(coverage_progress),
         mse=np.array(mse_progress),
         predicted_a=predicted_a,
         true_results=true_results,
         expected_results=expected_results)

# Compute final evaluation metrics
mse_final = mean_squared_error(true_results, expected_results)
rmse_final = np.sqrt(mse_final)
mae_final = mean_absolute_error(true_results, expected_results)
r2_final = r2_score(true_results, expected_results)

# Extract model name from path
model_name = os.path.splitext(os.path.basename(model_path))[0]

# Create a DataFrame to store the predictions
final_df = pd.DataFrame({
    "coverage_label": list(coverage_goals.keys()),
    "predicted_a": predicted_a,
    "expected_result": expected_results,
    "target_result": true_results,
    "model_name": model_name
})

# Save prediction inputs
metrics_summary = {
    "model_name": model_name,
    "Total Simulations": total_sims,
    "MSE": mse_final,
    "RMSE": rmse_final,
    "MAE": mae_final,
    "R2_Score": r2_final
}

# Save metrics summary to a master CSV
summary_csv_path = "verification_data/poly_final_metrics.csv"

# Load existing data if available
if os.path.exists(summary_csv_path):
    summary_df = pd.read_csv(summary_csv_path)
    # Remove existing row for this model name
    summary_df = summary_df[summary_df["model_name"] != model_name]
else:
    summary_df = pd.DataFrame()

# Append new summary row
summary_df = pd.concat([summary_df, pd.DataFrame([metrics_summary])], ignore_index=True)
summary_df.to_csv(summary_csv_path, index=False)

# Also save the detailed prediction inputs to a separate file
predictions_csv_path = f"verification_data/{model_name}_predicted_inputs.csv"
final_df.to_csv(predictions_csv_path, index=False)

# Print final summary
print("\n📈 Final Model Evaluation Metrics:")
for k, v in metrics_summary.items():
    if k != "model_name":
        print(f"  {k}: {v:.6f}")
print(f"\n📝 Summary saved to: {summary_csv_path}")
print(f"🧾 Detailed predictions saved to: {predictions_csv_path}")