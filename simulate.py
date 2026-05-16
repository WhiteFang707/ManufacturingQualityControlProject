"""
Factory Floor Data Simulator
=============================
Generates a synthetic dataset representing one full year (2025) of quality-control
records from a factory with:
  - 3 production lines  (A, B, C)
  - 5 suppliers
  - 4 quality inspectors
  - 3 shifts per day   (Morning, Afternoon, Night)
and a dataset with each error and the type of it.

Output: 'Data.csv'.
"""

import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path
from distributions import sample_production_units, sample_defect_rate

# ---------------------------------------------------------------------------
# 1. CONSTANTS & CONFIGURATION
# ---------------------------------------------------------------------------

# Column headers for the main production/quality table
GENERAL_TABLE_HEADER = (
    'date', 'shift', 'line', 'supplier',
    'inspector', 'units_produced', 'defects_found'
)

# Production lines and shift names
LINES  = ('A', 'B', 'C')
SHIFTS = ('Morning', 'Afternoon', 'Night')

# --- Scheduling maps ---
# Each line has a fixed supplier and inspector per shift.
# Changing a single entry here automatically propagates throughout the dataset.

SUPPLIER_MAP = {
    'A': {'Morning': 'Supplier 1', 'Afternoon': 'Supplier 1', 'Night': 'Supplier 2'},
    'B': {'Morning': 'Supplier 2', 'Afternoon': 'Supplier 4', 'Night': 'Supplier 4'},
    'C': {'Morning': 'Supplier 3', 'Afternoon': 'Supplier 3', 'Night': 'Supplier 5'},
}

INSPECTOR_MAP = {
    'A': {'Morning': 'Inspector_01', 'Afternoon': 'Inspector_02', 'Night': 'Inspector_03'},
    'B': {'Morning': 'Inspector_01', 'Afternoon': 'Inspector_03', 'Night': 'Inspector_04'},
    'C': {'Morning': 'Inspector_02', 'Afternoon': 'Inspector_04', 'Night': 'Inspector_03'},
}

# Line B has a higher baseline defect rate (4 %) than lines A and C (default rate).
LINE_B_DEFECT_RATE = 0.04

# ---------------------------------------------------------------------------
# 2. DATE RANGE
# ---------------------------------------------------------------------------

START_DATE = dt.date(2025, 1, 1)
END_DATE   = dt.date(2025, 12, 31)

# Build a list of every calendar date in 2025, formatted as DD/MM/YYYY
dates: list[str] = [
    (START_DATE + dt.timedelta(days=x)).strftime("%d/%m/%Y")
    for x in range((END_DATE - START_DATE).days + 1)
]

# ---------------------------------------------------------------------------
# 3. PRE-SAMPLED UNIT PRODUCTION POOL
# ---------------------------------------------------------------------------
# Drawing 1 000 values once from a log-normal distribution (mean ≈ 1 000,
# bounded roughly between 800 and 1 200)
UNIT_POOL: np.ndarray = sample_production_units(1_000, 800, 1_200, sigma=1)

# ---------------------------------------------------------------------------
# 4. DATA GENERATION
# ---------------------------------------------------------------------------

def generate_records() -> list[tuple]:
    """
    Iterate over every (date, line, shift) combination and produce one
    production record per combination.

    Returns
    -------
    list[tuple]
        Each tuple matches GENERAL_TABLE_HEADER:
        (date, shift, line, supplier, inspector, units_produced, defects_found)
    """

    records = []

    for date in dates:
        for line in LINES:
            for shift in SHIFTS:

                # --- Units produced ---
                # Pick a random value from the pre-sampled pool
                units = int(np.random.choice(UNIT_POOL))

                # --- Defects found ---
                # Line B historically shows a higher defect rate; all others use
                # the lognormal default rate from sample_error_rate_().
                if line == 'B':
                    defect_rate = sample_defect_rate(mean=LINE_B_DEFECT_RATE)
                else:
                    defect_rate = sample_defect_rate()

                defects = int(units * defect_rate)

                # --- Look up the scheduled supplier and inspector ---
                supplier  = SUPPLIER_MAP[line][shift]
                inspector = INSPECTOR_MAP[line][shift]

                records.append((date, shift, line, supplier, inspector, units, defects))

    return records


def build_dataframe(records: list[tuple]) -> pd.DataFrame:
    """
    Convert the raw list of records into a tidy pandas DataFrame.

    Parameters
    ----------
    records : list[tuple]
        Output of generate_records().

    Returns
    -------
    pd.DataFrame
    """

    return pd.DataFrame(records, columns=GENERAL_TABLE_HEADER)


# ---------------------------------------------------------------------------
# 5. ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate data generation and write the result to disk as a CSV file.

    Steps
    -----
    1. Generate all production records.
    2. Wrap them in a DataFrame.
    3. Export to 'Data.csv' (no row index).
    """

    records = generate_records()
    df = build_dataframe(records)

    output_path = Path("data") / 'data.csv'
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()