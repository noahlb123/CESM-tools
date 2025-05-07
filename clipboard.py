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


def year_in_range(filename):
    b_year = get_year(filename, filename, 'earlier')
    e_year = get_year(filename, filename, 'later')
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
    last_under = filematch.rfind('_')
    match = filematch[0:last_under]
    if not match in filename:
        return 'BAD!'
    end_index = 0 if end == 'earlier' else 1
    year = int(filename[last_under+1:len(filename)].replace('.nc', '').split('-')[end_index][0:4])
    return year

def replace_year(filename, year):
    last_under = filename.rfind('_')
    prefix = filename[0:last_under]
    years = filename[last_under+1:len(filename)].replace('.nc', '').split('-')
    years = [int(year[0:4]) for year in years]
    years = np.add([10, 10], years)
    return prefix + '_' + str(years[0]) + '01' + '-' + str(years[1]) + '12.nc'

def squares(lat, lon):
    meta_a = []
    for d in [5, 10, 15]:
        a = [lat - d, lat + d, lon - d, lon + d]
        a = [str(i) for i in a]
        meta_a.append('[' + ', '.join(a) + ']')
    print(',\n'.join(meta_a))

squares(-16.65, -67.8)