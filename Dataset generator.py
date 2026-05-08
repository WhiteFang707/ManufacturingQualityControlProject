import numpy as np
import pandas as pd
import datetime as dt
from lognormal import sample_, sample_error_rate_




def dates_between(start_date: dt.date, end_date: dt.date) -> list[str]:
    return [((start_date + dt.timedelta(days=x)).strftime("%d/%m/%Y")) for x in range((end_date - start_date).days + 1)]

GeneralTableHeader = ('date', 'shift', 'line', 'supplier', 'unit_produced', 'defect_found')
DefectTableHeader = ('date', 'shift', 'line', 'defect_found', 'defect_type')
line_map = ('A', 'B', 'C')
shift_map = ('Morning', 'Afternoon', 'Night')
supplier_map = {
    'A' : {'Morning': 'Supplier 1', 'Afternoon' : 'Supplier 1', 'Night' : 'Supplier 2'},
    'B' : {'Morning': 'Supplier 2', 'Afternoon' : 'Supplier 4', 'Night' : 'Supplier 4'},
    'C' : {'Morning': 'Supplier 3', 'Afternoon' : 'Supplier 3', 'Night' : 'Supplier 5'},
}
inspector_map = {
    'A': {'Morning': 'Inspector_01', 'Afternoon': 'Inspector_02', 'Night': 'Inspector_03'},
    'B': {'Morning': 'Inspector_01', 'Afternoon': 'Inspector_03', 'Night': 'Inspector_04'},
    'C': {'Morning': 'Inspector_02', 'Afternoon': 'Inspector_04', 'Night': 'Inspector_03'},
}
defect_weights = {
    'A': {'types': ['Dimensional', 'Assembly', 'Surface', 'Material'],
          'weights': [0.55, 0.30, 0.10, 0.05]},
    'B': {'types': ['Surface', 'Dimensional', 'Assembly', 'Material'],
          'weights': [0.60, 0.25, 0.10, 0.05]},
    'C': {'types': ['Material', 'Assembly', 'Dimensional', 'Surface'],
          'weights': [0.50, 0.30, 0.15, 0.05]},
}
StartDate = dt.date(2025, 1, 1)
EndDate = dt.date(2025, 12, 31)

dates = dates_between(StartDate, EndDate)
data = pd.DataFrame(None, columns=GeneralTableHeader)
unit_produced = sample_(1000, 800, 1200, sigma=1)

def raw_data():
    for date in dates:
        for line in line_map:
            for shift in shift_map:
                unit = np.random.choice(unit_produced)
                unit_error = int(unit * sample_error_rate_()) if line != 'B' else int(unit * sample_error_rate_(mean=0.04))
                data.loc[len(data)] = [date, shift, line, supplier_map[line][shift], unit, unit_error]

def error_addition():
    #random numbers been negative instead of positive for unit produced
    mask = np.random.uniform(0, 1, size=len(data)) <= 0.98
    data['unit_produced'] = data['unit_produced'].where(mask, -data['unit_produced'])
    # random numbers been negative instead of positive for defect found
    mask = np.random.uniform(0, 1, size=len(data)) <= 0.99
    data['defect_found'] = data['defect_found'].where(mask, -data['defect_found'])
    mask = np.random.uniform(0, 1, size=len(data)) <= 0.999
    data['defect_found'] = data['defect_found'].where(mask, data['defect_found']*100)
    # inspector 3 has lower rate
    

def __main__():
    raw_data()
    error_addition()
    data.to_csv('Data.csv', index=False)
__main__()