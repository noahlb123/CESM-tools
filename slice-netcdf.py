import sys
from netCDF4 import Dataset
import numpy as np
import os
import tools

#get filename and year from arguments
if len(sys.argv) < 3:
    raise Exception('format command as:\npython3 slice-netcdf.py <PATH> <YEAR>')
path = sys.argv[1]
year = int(sys.argv[2])

#setup global vars
T = tools.ToolBox()
day = year * 365
filename = path[path.rfind('/') + 1:len(path)]
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
unit_year = int(time_var.units[11:15])
year_modifier = 1850
if unit_year != s_y and unit_year != 1:
    year_modifier = unit_year
elif unit_year == 1:
    year_modifier = s_y
time_index = T.nearest_search(times, day - (year_modifier * 365))
file_day = times[time_index] + year_modifier * 365
if np.abs(file_day / 365 - year) > 2 or (not (s_y < year < e_y)):
    print('file year, year:', file_day / 365, year)
    print('raw file min, max:', np.min(times) / 365, np.max(times) / 365)
    raise Exception('year ' + str(year) + ' is not in file')
f.close()
new_filename = filename[0:filename.rfind('_')+1] + date(file_day) + '-' + date(file_day + 365)
new_path = path[0:path.rfind('/') + 1] + new_filename
to_eval = 'ncks -d time,' + str(time_index) + ',' + str(time_index + 365) + ' ' + path + ' ' + new_path + ' -O && '
to_eval += 'echo "success!"'
print(to_eval)
print('evaluating...')
os.system(to_eval)