import cocotb
from cocotb.triggers import Timer
import random
import csv
import os
import json

# ───────────────────────────────────────────────────────────────
def ensure_csv(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["a", "b", "product", "passed"]) # <<< FEATURES AND TARGET HEDAERS

async def run_simulation_batch(dut, csv_path, input_batch):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)

        for a_val, b_val in input_batch: # \/ \/ \/ SIMULATION INPUTS AND OUTPUTS \/ \/ \/
            dut.a.value = a_val
            dut.b.value = b_val

            await Timer(1, units='ns')

            product = int(dut.product.value)

            passed = int(product == a_val * b_val)

            writer.writerow([a_val, b_val, product, passed])

# ───────────────────────────────────────────────────────────────

@cocotb.test()
async def simulate_random(dut):
    '''
    Initial constrained random verification test for the DUT.
    '''
    csv_path = os.getenv("CSV_PATH", "verification_data/sim_data.csv")
    ensure_csv(csv_path)
    
    num_runs = int(os.getenv("INIT_RUNS", "1"))
    input_batch = []

    for seed in range(num_runs): # \/ \/ \/ RANDOMIZED INPUTS \/ \/ \/
        random.seed(seed)
        a = random.randint(0, 255)
        b = random.randint(0, 255)
        input_batch.append([a, b])

    await run_simulation_batch(dut, csv_path, input_batch)

# ───────────────────────────────────────────────────────────────
@cocotb.test()
async def simulate_from_model(dut):
    '''
    Run batch of ANN-predicted inputs from environment
    '''
    csv_path = os.getenv("CSV_PATH", "verification_data/sim_data.csv")
    input_batch = json.loads(os.getenv("MODEL_INPUTS", "[]"))

    await run_simulation_batch(dut, csv_path, input_batch)
