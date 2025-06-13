import cocotb
from cocotb.triggers import Timer
import random
import csv
import os
import json

# ───────────────────────────────────────────────────────────────
def ensure_csv(csv_path):
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["a", "result"])  # <<< FEATURES AND TARGET HEADERS
            
async def run_simulation_batch(dut, csv_path, input_batch):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)

        for a_val in input_batch: # \/ \/ \/ SIMULATION INPUTS AND OUTPUTS \/ \/ \/
            dut.a.value = a_val

            await Timer(1, units='ns')

            result = int(dut.result.value)

            writer.writerow([a_val, result])

# ───────────────────────────────────────────────────────────────
@cocotb.test()
async def simulate_random(dut):
    '''
    Initial constrained random verification test for the DUT.
    '''
    csv_path = os.getenv("CSV_PATH", "verification_data/sim_data.csv")
    ensure_csv(csv_path)

    num_runs = int(os.getenv("RUNS", "1"))
    base_seed = int(os.getenv("SEED_OFFSET", "0"))  # new: deterministic shift

    input_batch = []

    for i in range(num_runs):
        seed = base_seed + i
        random.seed(seed)
        a = random.randint(0, 1023)
        input_batch.append(a)

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
