"""
Corruption of data from a factory floor
=============================
Corrupt certain data inside the data.csv.
The types are missing values, wrong types, out-of-range values, duplicates rows.
Which each will have its own function.

Input: 'data.csv'
Output: 'corrupt_data.csv'
"""

import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
from distributions import sample_defect_rate as sp

def inject_whitespace_errors(df: pd.DataFrame, frac: float) -> None:
    """
    The function change a certain percentage of data in the inspector column
    by injecting whitespace.

    Parameters
    ----------
    df : pd.DataFrame
        Contains the inspector data
    frac : float
        fraction of data to be corrupter
    """

    #row to corrupt
    target_index = df.sample(frac=frac).index

    #For each targeted row, pick a random whitespace variant of its value
    whitespace_variants = [
        lambda x : f' {x} ',
        lambda x : f' {x}',
        lambda x : f'{x} ',
    ]

    df.loc[target_index, 'inspector'] = df.loc[target_index, 'inspector'].map(lambda x : np.random.choice(whitespace_variants)(x))

    return None

def injecting_values_errors(df: pd.DataFrame, frac: float) -> None:
    """
    The function change a fraction of the values of unit produced and defects found in the data.

    Parameters
    ----------
    df : pd.DataFrame
        The data
    frac : float
        fraction of data to be corrupter
    """

    #rows to be targeted
    target_index = df.sample(frac=frac).index

    # For each targeted row, pick a random lamda function or value.
    function_map = [
        lambda x : -x,
        lambda x : 1000 * x,
        lambda x : -1000 * x,
        lambda x : np.nan
    ]

    df.loc[target_index, 'units_produced'] = df.loc[target_index, 'units_produced'].map(lambda x : np.random.choice(function_map)(x))

def injecting_date_formating_errors(df: pd.DataFrame, frac: float) -> None:
    """
    The function changes a certain percentage of dates into another format.

    Parameters
    ----------
    df : pd.DataFrame
        Data
    frac : float
        fraction of data to be changed
    """

    target_index = df.sample(frac=frac).index

    date_format_map = [ lambda x : dt.datetime.strptime(str(x), '%d/%m/%Y').strftime('%Y-%m-%d'),
                        lambda x : dt.datetime.strptime(str(x), '%d/%m/%Y').strftime('%B %d %Y'),
                        lambda x : dt.datetime.strptime(str(x), '%d/%m/%Y').strftime('%d %b %Y'),

    ]

    df.loc[target_index, 'date'] = df.loc[target_index, 'date'].map(lambda x : np.random.choice(date_format_map)(x))

def duplicate_row(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """
    The function duplicate a certain percentage of rows.

    Parameters
    ----------
    df : pd.DataFrame
        Data
    frac : float
        fraction of rows to be duplicated

    Returns
    ---------
    pd.DataFrame
    The dataframe with some duplicates rows
    """

    return pd.concat([df, df.sample(frac=frac)]).reset_index(drop=True)


# noinspection PyArgumentList
def main():
    #Opening the csv.
    input_path: str = Path("data") / 'data.csv'
    data: pd.DataFrame = pd.read_csv(Path(input_path))

    inject_whitespace_errors(data, frac=sp(mean=0.03, std=0.001))
    injecting_values_errors(data, frac=sp(mean=0.01, std=0.001))
    injecting_date_formating_errors(data, frac=sp(mean=0.01, std=0.001))
    data = duplicate_row(data, frac=sp(mean=0.0005, std=0.001))

    output_path = Path("data") / "corrupt_data.csv"
    data.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()




