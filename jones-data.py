import pandas as pd
import numpy as np
import os

df = pd.read_csv('/Users/noahliguori-bills/Downloads/pyc/Jones-PyC-Production-Factors.csv')

numbers = {}
pubs = set()
finals = []

def raw2float(raw):
    return float(row['PyC_BioOC_Affected'].replace('%', ''))

for index, row in df.iterrows():
    Comps_detail = row['Comps_detail']
    #if type(Comps_detail) == type('') and 'Leaves' in Comps_detail:
    if Comps_detail == 'Leaves / needles on trees / crown fuels':
        raw = row['PyC_BioOC_Affected']
        if raw != '#DIV/0!':
            if row['study'] in numbers:
                numbers[row['study']] = (numbers[row['study']][0] + raw2float(raw), numbers[row['study']][1] + 1)
            else:
                numbers[row['study']] = (raw2float(raw), 1)
for key in numbers.keys():
    print(key)
    finals.append(numbers[key][0] / numbers[key][1])
print(np.mean(finals) / 100)