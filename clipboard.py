import pyperclip
import numpy as np
import tools

T = tools.ToolBox()

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

wets = set()
drys = set()
for filename in pyperclip.paste().split('\n'):
    if '.nc' in filename and 'AERmon_' in filename and ('wetbc' in filename or 'drybc' in filename):
        if 'wetbc' in filename:
            wets.add(filename2modelname(filename)[2])
        elif 'drybc' in filename:
            drys.add(filename2modelname(filename)[2])
print(wets)
print(drys)
print(wets.intersection(drys))