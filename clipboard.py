#import pyperclip
import numpy as np
import os
#import tools
#T = tools.ToolBox()

def filename2modelname(filename):
    prefix = 'AERmon_'
    start_year = filename[filename.rfind("_") + 1:filename.rfind("-") - 2]
    end_year = filename[filename.rfind("-") + 1:filename.rfind(".") - 2]
    model_name = filename[filename.index(prefix) + len(prefix): filename.index('_historical')]
    return int(start_year), int(end_year), model_name

def valid_range(years):
    s_year, e_year = years
    def within_one(a, b):
        return np.abs(a - b) <= 1
    target_years = (1850, 1980)
    for year in target_years:
        if s_year <= year <= e_year:
            return True
    return False

def get_years(filename):
    years = filename[filename.rfind('_') + 1:len(filename) - 3].split('-')
    years = [int(year[0:4]) for year in years]
    return years

'''print("rm ")
m = {}
for name in pyperclip.paste().split('\n'):
    if name:
        s_year, e_year = filename2modelname(name)
        if not valid_range(s_year, e_year):
            print(name, end=" ")'''

'''for filename in pyperclip.paste().split('\n'):
    if not ('wget' in filename or 'wetbc' in filename or 'drybc' in filename):
        print(filename, end=' ')'''

#print(', '.join(pyperclip.paste().split()))

'''year_year_map = {1850: 1875, 1980: 1955}


def year_in_range(filename):
    b_year = get_year(t_file, t_file, 'earlier')
    e_year = get_year(t_file, t_file, 'later')
    if type(b_year) == type(e_year) == type(2):
        if b_year <= 1850 <= e_year and b_year <= 1980 <= e_year:
            return 'both'
        elif b_year <= 1850 <= e_year:
            return 1850
        elif b_year <= 1980 <= e_year:
            return 1980
    else:
        return "None!"

def get_year(filename, filematch, end):
    token = 'gn_'
    match = filematch[0:filematch.index(token) + len(token)]
    if not match in filename:
        return 'BAD!'
    end_index = 0 if end == 'earlier' else 1
    year = int(filename.split(token)[1].replace('.nc', '').split('-')[end_index][0:4])
    return year

for t_file, key in targets.items():
    yir = year_in_range(t_file)
    if yir == 'both' or yir == 'None!':
        continue
    end = 'earlier' if yir == 1980 else 'later'
    t_year = get_year(t_file, t_file, end)
    files = []
    for file in os.listdir(dir):
        e_year = get_year(file, t_file, 'earlier')
        l_year = get_year(file, t_file, 'later')
        if not (type(e_year) == type(l_year) == type(2)):
            continue
        if e_year:
            pass'''

input = [
    'loadbc_Eday_CESM2-FV2_historical_r1i1p1f1_gn_19500101-19591231.nc',
    'loadbc_Eday_CESM2-FV2_historical_r1i1p1f1_gn_19600101-19691231.nc',
    'loadbc_Eday_CESM2-FV2_historical_r1i1p1f1_gn_19700101-19791231.nc',
    'loadbc_Eday_CESM2-FV2_historical_r1i1p1f1_gn_19800101-19891231.nc'
    ]

print('ncrcat -O ' + ' '.join(input) + ' ' + 'loadbc_Eday_CESM2-FV2_historical_r1i1p1f1_gn_19500101-19891231.nc')
print('\n')
print('rm ' + ' '.join(input))