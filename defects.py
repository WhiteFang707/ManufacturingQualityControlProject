"""
Factory Floor Defect Data
=============================
Generates a synthetic dataset representing one full year (2025) of every defect found.

Output: 'defect.csv'.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Defect profile per line: which defect types occur and how often.
DEFECT_WEIGHTS = {
    'A': {
        'types':   ['Dimensional', 'Assembly', 'Surface', 'Material'],
        'weights': [0.55, 0.3, 0.10, 0.05],
    },
    'B': {
        'types':   ['Surface', 'Dimensional', 'Assembly', 'Material'],
        'weights': [0.60, 0.25, 0.10, 0.05],
    },
    'C': {
        'types':   ['Material', 'Assembly', 'Dimensional', 'Surface'],
        'weights': [0.50, 0.30, 0.15, 0.05],
    },
}

def assign_defect_type(line: str) -> str:
   return np.random.choice(DEFECT_WEIGHTS[line]['types'], p=DEFECT_WEIGHTS[line]['weights'])

def generate_defect_records(df: pd.DataFrame) -> pd.DataFrame:
    defects = df.loc[df.index.repeat(df['defects_found'])].copy()
    defects = defects[['date', 'shift', 'line', 'supplier', 'inspector']].reset_index(drop=True)
    defects['defect_type'] = defects['line'].map(assign_defect_type)

    return defects

def main():
    input_path = Path('data') / 'data.csv'
    df = pd.read_csv(input_path)

    defects = generate_defect_records(df)

    output_path = Path('data') / 'defect.csv'
    defects.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()

