import sys
from netCDF4 import Dataset
import numpy as np
import os
import tools

#get filename and year from arguments
if len(sys.argv) < 3:
    raise Exception('format command as:\npython3 slice-netcdf.py <FILENAME> <YEAR>')
filename = sys.argv[1]
year = int(sys.argv[2])

#setup global vars
T = tools.ToolBox()
day = year * 365
path = os.path.join(os.getcwd(), filename)
target_v = filename[0:filename.find('_')]
date_l = 4 if target_v == 'loadbc' else 2

def get_years(filename):
    start_year = filename[filename.rfind("_") + 1:filename.rfind("-") - date_l]
    end_year = filename[filename.rfind("-") + 1:filename.rfind(".") - date_l]
    return int(start_year), int(end_year)

def date(day):
    date = ''
    units = [365, 12, 1] if date_l == 4 else [365, 12]
    for unit in units:
        section = int(day // unit)
        if unit == 365:
            date += "{:02d}".format(section)
        else: 
            date += "{:02d}".format(section + 1)
        day -= section * unit
    return date

#slice and dice
s_y, e_y = get_years(filename)
f = Dataset(path)
time_var = f.variables['time']
times = f['time'][:]
assert 'days since 1850' in time_var.units
time_index = T.nearest_search(times, day - (1850 * 365))
file_day = times[time_index] + 1850 * 365
if np.abs(file_day / 365 - year) > 2 or (not (s_y < year < e_y)):
    raise Exception('year ' + str(year) + ' is not in file')
f.close()
new_filename = filename[0:filename.rfind('_')+1] + date(file_day) + '-' + date(file_day + 365)
to_eval = 'ncks -d time,' + str(time_index) + ',' + str(time_index + 365) + ' ' + filename + ' ' + new_filename + ' -O && '
to_eval += 'echo "success!"'
print(to_eval)
print('evaluating...')
os.system(to_eval)