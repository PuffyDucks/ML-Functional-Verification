import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

N = 5.0
max_input = N**2-1

# ─── Load CSV ───────────────────────────────────────────────────────
df = pd.read_csv("sim_data.csv")

# Input: coverage goal (gt, lt, eq)
X = df[["gt", "lt", "eq"]].values

# Output: inputs (a, b) — scale to [0, 1] for better training
y = df[["a", "b"]].values / max_input

# ─── Train-Test Split ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─── Train Model ────────────────────────────────────────────────────
model = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# ─── Evaluate Model ─────────────────────────────────────────────────
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")

# ─── Predict for All 3 Coverage Goals ───────────────────────────────
goals = {
    "gt=1": [1, 0, 0],
    "lt=1": [0, 1, 0],
    "eq=1": [0, 0, 1],
}

for label, vec in goals.items():
    target = np.array([vec])
    pred_scaled = model.predict(target)[0]
    pred = np.round(pred_scaled * max_input).astype(int)
    a_pred, b_pred = pred
    print(f"{label}: predicted a={a_pred}, b={b_pred}")
