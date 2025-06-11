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
        writer.writerow(["a", "b", "gt", "lt", "eq", "pass"])

async def run_simulation_batch(dut, csv_path, input_batch):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)

        for a_val, b_val in input_batch:
            dut.a.value = a_val
            dut.b.value = b_val

            await Timer(1, units='ns')

            gt = int(dut.gt.value)
            lt = int(dut.lt.value)
            eq = int(dut.eq.value)

            expected_gt = int(a_val > b_val)
            expected_lt = int(a_val < b_val)
            expected_eq = int(a_val == b_val)
            passed = int(gt == expected_gt and lt == expected_lt and eq == expected_eq)

            writer.writerow([a_val, b_val, gt, lt, eq, passed])

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

    for seed in range(num_runs):
        random.seed(seed)
        a = random.randint(0, 31)
        b = random.randint(0, 31)
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
