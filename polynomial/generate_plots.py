import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# Use style and explicitly disable LaTeX
plt.style.use(['science', 'ieee'])
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

# ──────────────────────────────────────────────
#               Load Saved Metrics
# ──────────────────────────────────────────────
ann = np.load("verification_data/ann_metrics.npz")
poly = np.load("verification_data/poly_metrics.npz")
dt = np.load("verification_data/dt_metrics.npz")
random = np.load("verification_data/random_metrics.npz")

models = {
    "ANN": ann,
    "Polynomial": poly,
    "Decision Tree": dt,
    "Random Only": random
}

colors = {
    "ANN": "blue",
    "Polynomial": "orange",
    "Decision Tree": "green",
    "Random Only": "gray"
}

markers = {
    "ANN": "o",
    "Polynomial": "s",
    "Decision Tree": "^",
    "Random Only": "x"
}

# ──────────────────────────────────────────────
#        Coverage Progress vs Simulations
# ──────────────────────────────────────────────
plt.figure(figsize=(10, 6))
for name, data in models.items():
    if "coverage" in data and "sim_counts" in data:
        plt.plot(data["sim_counts"], data["coverage"], marker=markers[name], label=name)
plt.xlabel("Total Simulations Run")
plt.ylabel("Covered Bins")
plt.title("Polynomial DUT Coverage Progress vs Simulations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("verification_data/polynomial_compare_coverage.png", dpi=300, bbox_inches='tight')
plt.show()

# ──────────────────────────────────────────────
#        MSE (Loss) vs Simulations (Trimmed)
# ──────────────────────────────────────────────
plt.figure(figsize=(10, 6))
for name, data in models.items():
    if "mse" in data and "sim_counts" in data:
        mse = data["mse"]
        sim = data["sim_counts"]
        idx = np.argmax(mse > 0)  # skip pre-training 0s
        plt.plot(sim[idx:idx + len(mse[idx:])], mse[idx:], marker=markers[name], label=name)
plt.xlabel("Total Simulations Run")
plt.ylabel("Model MSE (Loss)")
plt.title("Polynomial DUT Model Loss (MSE) vs Simulations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("verification_data/polynomial_compare_mse.png", dpi=300, bbox_inches='tight')
plt.show()

# ──────────────────────────────────────────────
#        Final Prediction vs Coverage Goal
# ──────────────────────────────────────────────

def coverage_fn(a):
    return ((a * (a - 600) * (a - 940)) >> 15) + 700

a_vals = list(range(0, 1024))
f_a_vals = [coverage_fn(a) for a in a_vals]

for name, data in models.items():
    if "predicted_a" in data and "true_results" in data:
        pred_a = data["predicted_a"]
        true_y = data["true_results"]

        plt.figure(figsize=(10, 6))
        plt.scatter(true_y, pred_a, s=40, label=f"{name} Predictions", alpha=0.8)
        plt.plot(f_a_vals, a_vals, '-', color='red', label="Valid input $a$", linewidth=1.5)

        plt.xlabel("Coverage Goal ($f(a)$)")
        plt.ylabel("Predicted $a$")
        plt.title(f"Polynomial DUT Final Prediction vs Coverage Goal - {name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"verification_data/polynomial_{name}_predictions.png", dpi=300, bbox_inches='tight')
        plt.show()
