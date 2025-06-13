import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────
#            Configurable Parameters
# ──────────────────────────────────────────────
csv_path = "verification_data/sim_data.csv"
verilog_paths = ["polynomial.v"]

top_module = "polynomial"
test_module = "run_sims"
sim_tool = "verilator"
extra_args = ""

random_runs = 100

input_features = ["a"]
input_ranges = {"a": 1023}
MAX_A = input_ranges["a"]

coverage_goals = {
    f"prod_{((a * (a - 600) * (a - 940)) >> 15) + 700}": [((a * (a - 600) * (a - 940)) >> 15) + 700]
    for a in range(0, 1023, 4)
}
total_bins = len(coverage_goals)

# ──────────────────────────────────────────────
#                     Helpers
# ──────────────────────────────────────────────
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

def add_bin_vector_columns(df, coverage_goals):
    for label, values in coverage_goals.items():
        df[label] = df["result"].isin(values).astype(int)
    return df

def get_uncovered_bins(df, coverage_goals):
    df = add_bin_vector_columns(df, coverage_goals)
    return [label for label in coverage_goals if df[label].sum() == 0]

def clean_sim_data(csv_path, coverage_goals):
    df = pd.read_csv(csv_path).drop_duplicates()
    goal_values = set(val[0] for val in coverage_goals.values())
    df = df[df["result"].isin(goal_values)]
    df.to_csv(csv_path, index=False)
    print(f"🧹 Cleaned sim_data.csv: {len(df)} rows remain")

# ──────────────────────────────────────────────
#              Random-Only Loop
# ──────────────────────────────────────────────
if os.path.exists(csv_path):
    os.remove(csv_path)

sim_counts = []
coverage_progress = []
total_sims = 0
cycle = 0

while True:
    print(f"\n🔹 Running {random_runs} random simulations...")
    seed_offset = cycle * random_runs
    run_cocotb_test("simulate_random", extra_env={"RUNS": str(random_runs), "SEED_OFFSET": str(seed_offset)})
    clean_sim_data(csv_path, coverage_goals)

    total_sims += random_runs
    df = pd.read_csv(csv_path)
    uncovered = get_uncovered_bins(df, coverage_goals)
    covered = total_bins - len(uncovered)

    sim_counts.append(total_sims)
    coverage_progress.append(covered)

    print(f"[Random] Simulations run: {total_sims}, Covered bins: {covered}/{total_bins}")
    if not uncovered:
        print("🎉 All coverage goals met!")
        break

    cycle += 1

print("✅ Random-only verification complete.")

# ──────────────────────────────────────────────
#         Plot Coverage vs Simulation Count
# ──────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.plot(sim_counts, coverage_progress, marker='o')
plt.xlabel("Total Simulations Run")
plt.ylabel("Covered Bins")
plt.title("Coverage Progress vs Simulations (Random Only)")
plt.grid(True)
plt.tight_layout()
plt.savefig("verification_data/random_only_coverage.png")
plt.show()

# Save data
np.savez("verification_data/random_only_metrics.npz",
         sim_counts=np.array(sim_counts),
         coverage=np.array(coverage_progress))

print("📊 Metrics saved to verification_data/random_only_metrics.npz")
