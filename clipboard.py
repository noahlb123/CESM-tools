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

year_year_map = {1850: 1875, 1980: 1955}


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

'''for t_file, key in targets.items():
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
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_195501-195512.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_195601-195612.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_195701-195712.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_195801-195812.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_195901-195912.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_196001-196012.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_196101-196112.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_196201-196212.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_196301-196312.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_196401-196412.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_196501-196512.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_196601-196612.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_196701-196712.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_196801-196812.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_196901-196912.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_197001-197012.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_197101-197112.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_197201-197212.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_197301-197312.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_197401-197412.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_197501-197512.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_197601-197612.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_197701-197712.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_197801-197812.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_197901-197912.nc',
    'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_198001-198012.nc'
    ]


print('ncrcat -O ' + ' '.join(input) + ' ' + 'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_195501-198012.nc')
print('\n')
print('rm ' + ' '.join(input))

mmrbc = {'mmrbc_AERmon_EC-Earth3-AerChem_historical_r1i1p4f1_gn_198001-198012.nc': 1980}