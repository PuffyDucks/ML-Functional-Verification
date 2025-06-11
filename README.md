### Prerequisites

* Python 3.8+
* `verilator` (or change `sim_tool` to `icarus`)
* Required Python packages:

```bash
pip install numpy pandas scikit-learn joblib cocotb
```

### Define the DUT

Make sure your Verilog design (e.g., `multiplier.v`) is placed in the repo. The top module should match `top_module` in the script (`multiplier` by default).


## Configuration Options

Edit these in `ann_verification.py`:

| Parameter             | Purpose                                |
| --------------------- | -------------------------------------- |
| `csv_path`            | Where simulation results are stored    |
| `verilog_paths`       | List of Verilog files for the DUT      |
| `top_module`          | DUT's top-level module name            |
| `initial_random_runs` | Number of initial random tests         |
| `max_cycles`          | Max number of ANN feedback iterations  |
| `input_ranges`        | Maximum values for input normalization |

---