import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
#               Load Saved Metrics
# ──────────────────────────────────────────────
ann = np.load("verification_data/ann_metrics.npz")
poly = np.load("verification_data/poly_metrics.npz")
dt = np.load("verification_data/dt_metrics.npz")
random = np.load("verification_data/random_metrics.npz")

# Extract data
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
plt.title("Coverage Progress vs Simulations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("verification_data/compare_coverage.png")
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
plt.title("Model Loss (MSE) vs Simulations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("verification_data/compare_mse.png")
plt.show()

# ──────────────────────────────────────────────
#        Final Prediction vs Coverage Goal
# ──────────────────────────────────────────────

# Determine the number of models
num_models = sum(1 for name, data in models.items() if "predicted_a" in data and "true_results" in data)

# Create separate plots for each model
for i, (name, data) in enumerate(models.items()):
    if "predicted_a" in data and "true_results" in data:
        pred_a = data["predicted_a"]
        true_y = data["true_results"]
        expected_y = data["expected_results"]

        plt.figure(figsize=(8, 5))
        plt.scatter(true_y, pred_a, s=40, label=f"{name} Predictions", alpha=0.8)
        plt.plot(expected_y, pred_a, linestyle="-", label=f"{name} Expected a²", linewidth=1.5)

        plt.ylabel("Predicted a")
        plt.xlabel("Coverage Goal (a²)")
        plt.title(f"Final Prediction vs Coverage Goal - {name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"verification_data/{name}_final_predictions.png")
        plt.show()