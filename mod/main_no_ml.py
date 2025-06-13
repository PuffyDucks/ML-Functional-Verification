import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
#            Configurable Parameters
# ──────────────────────────────────────────────
csv_path = "verification_data/sim_data.csv"
verilog_paths = ["squarer.v"]

top_module = "squarer"
test_module = "run_sims"
sim_tool = "verilator"
extra_args = ""

random_runs = 25
max_cycles = 100

input_ranges = {"a": 255}
MAX_A = input_ranges["a"]

coverage_goals = {
    f"prod_{n**2}": [n**2]
    for n in range(0, 255, 4)
}

test_random = "simulate_random"

# ──────────────────────────────────────────────
#                  Helpers
# ──────────────────────────────────────────────

def run_cocotb_test(testcase, extra_env=None):
    env = os.environ.copy()
    env.update({
        "TESTCASE": testcase,
        "CSV_PATH": csv_path,
        "VERILOG_SOURCES": " ".join(os.path.abspath(f) for f in verilog_paths),
        "TOPLEVEL": top_module,
        "MODULE": test_module,
        "SIM": sim_tool,
        "EXTRA_ARGS": extra_args
    })
    if extra_env:
        env.update(extra_env)
    subprocess.run(["make"], env=env)

def clean_sim_data(csv_path, coverage_goals):
    df = pd.read_csv(csv_path).drop_duplicates()
    goal_values = set(val[0] for val in coverage_goals.values())
    df = df[df["result"].isin(goal_values)]
    df.to_csv(csv_path, index=False)
    print(f"🧹 Cleaned sim_data.csv: {len(df)} rows remain")
    return df

def get_uncovered_bins(df, coverage_goals):
    for label, values in coverage_goals.items():
        df[label] = df["result"].isin(values).astype(int)
    uncovered = [label for label in coverage_goals if df[label].sum() == 0]
    return uncovered

# ──────────────────────────────────────────────
#              Verification Loop
# ──────────────────────────────────────────────

if os.path.exists(csv_path):
    os.remove(csv_path)

sim_counts = []
coverage_progress = []
total_bins = len(coverage_goals)
total_sims = 0

for cycle in range(max_cycles):
    print(f"\n🔹 Random batch {cycle+1}: {random_runs} simulations")
    seed_offset = cycle * random_runs
    run_cocotb_test(test_random, extra_env={"RUNS": str(random_runs), "SEED_OFFSET": str(seed_offset)})
    df = clean_sim_data(csv_path, coverage_goals)

    total_sims += random_runs
    uncovered = get_uncovered_bins(df, coverage_goals)
    covered = total_bins - len(uncovered)

    sim_counts.append(total_sims)
    coverage_progress.append(covered)

    print(f"[Cycle {cycle+1}] Total Sims: {total_sims}, Covered: {covered}/{total_bins}")

    if not uncovered:
        print("🎉 All coverage goals met!")
        break

print("\n✅ Random-only verification complete.")

# ──────────────────────────────────────────────
#              Coverage Plot
# ──────────────────────────────────────────────

plt.figure(figsize=(10, 6))
plt.plot(sim_counts, coverage_progress, marker='o')
plt.xlabel("Total Simulations Run")
plt.ylabel("Covered Bins")
plt.title("Coverage Progress (Random Only)")
plt.grid(True)
plt.tight_layout()
plt.savefig("verification_data/coverage_random_only.png")
plt.show()

# ──────────────────────────────────────────────
#              Save Metrics
# ──────────────────────────────────────────────

np.savez("verification_data/random_metrics.npz",
         sim_counts=np.array(sim_counts),
         coverage=np.array(coverage_progress))

print("📊 Metrics saved to verification_data/random_metrics.npz")
