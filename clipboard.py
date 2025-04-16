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

root = '/glade/campaign/cesm/collections/cesmLE/restarts'
for sub_dir in os.listdir(root):
    if 'B20TRC5CNBDRD' in sub_dir:
        i = int(sub_dir.split('.')[-1])
        for i in range(1, 36):
            for year in [1980, 1975, 1970, 1965, 1960, 1955]:
                path = os.path.join(root, sub_dir, 'b.e11.B20TRC5CNBDRD.f09_g16.0' + "{:02d}".format(i) + '.rest.' + str(year) + '-01-01-00000.tar')
                if not os.path.isfile(path):
                    print(path)