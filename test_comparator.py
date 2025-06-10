import cocotb
from cocotb.triggers import Timer
import random
import os
import csv

DATA_FILE = "sim_data.csv"

def log_data(a, b, gt, lt, eq, passed):
    write_header = not os.path.exists(DATA_FILE)
    with open(DATA_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['a', 'b', 'gt', 'lt', 'eq', 'pass'])
        writer.writerow([a, b, gt, lt, eq, passed])

@cocotb.test()
async def run_batch(dut):
    for seed in range(400):  # number of iterations
        random.seed(seed)
        a_val = random.randint(0, 31)
        b_val = random.randint(0, 31)

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

        log_data(a_val, b_val, gt, lt, eq, passed)

