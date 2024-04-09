import numpy as np
from netCDF4 import Dataset
from tools import ToolBox
T = ToolBox()

#setup data
f = Dataset("/Users/noahliguori-bills/Downloads/CESM-tools/data/douglas-course-emissions.nc", "r")
'''var_wrap = {
    'bc_a4DDF': None,
    'bc_a4SFWET': None,
    'bc_a1DDF': None,
    'bc_a1SFWET': None,
    'bc_c4DDF': None,
    'bc_c4SFWET': None,
    'bc_c1DDF': None,
    'bc_c1SFWET': None
    }
#for key in var_wrap.keys():
    #var_wrap[key] = f[key][:]'''
lats =  f["lat"][:]
lons =  f["lon"][:]
lat = T.nearest_search(lats, 71.4)
lon = T.nearest_search(lons, 180-44)
var = f['BC'][:]
f.close()

#extract total 2023 NEEM BC flux (Lat: 71.4, Lon: -44)
#457 ("record": time, daily, 15 months, starting: 2009-09-01), 1 (time var), 192 (lat), 288 (lon)
#print(len(test), len(test[0]), len(test[0][0]), len(test[0][0][0]))
'''
for m in range(len(emissions)):
    print(m)
    adj_e.append([])
    for x in range(len(emissions[m])):
        adj_e[m].append([])
        for y in range(len(emissions[m][x])):
            e = emissions[m][x][y]
            b = bias[m][x][y]
            adj_e[m][x].append(e * b)'''