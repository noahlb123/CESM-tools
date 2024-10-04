import numpy as np
from netCDF4 import Dataset
import pandas as pd
from tools import ToolBox
from matplotlib.lines import Line2D
import datetime
import csv
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
T = ToolBox()

'''#setup data
f = Dataset("/Users/noahliguori-bills/Downloads/CESM-tools/data/climo-mean.nc", "r")
var_wrap = {
    'bc_a4DDF': [None, 1],
    'bc_a4SFWET': [None, -1],
    'bc_a1DDF': [None, 1],
    'bc_a1SFWET': [None, -1],
    'bc_c4DDF': [None, 1],
    'bc_c4SFWET': [None, -1],
    'bc_c1DDF': [None, 1],
    'bc_c1SFWET': [None, -1]
    }
for key in var_wrap.keys():
    var_wrap[key][0] = f[key][:]

lats =  f["lat"][:]
lons =  f["lon"][:]
lat = T.nearest_search(lats, 71.4)
lon = T.nearest_search(lons, 180-44)
main = var_wrap['bc_a4DDF'][0]
start_date = datetime.datetime(2022, 9, 1)
min_date = datetime.datetime(2022, 10, 25)
max_date = datetime.datetime(2023, 10, 25)
f.close()

#notes on .nc file
#extract total 2023 NEEM BC flux (Lat: 71.4, Lon: -44)
#457 ("record": time, daily, 15 months, starting: 2009-09-01), 1 (time var), 192 (lat), 288 (lon)
#bc flux units = kg/m2/s
#kg/m2/s *(10^9 ug/kg)(1 m^2/100 dm^2)(3.154*10^7 s/year) = ug/dm^2/year
#ug/dm^2/year / 2.2 dm/year = ug/l = ng/n
#in total kg/m2/s *(3.154*10^14/2.2) = ng/g
#want ug/L

def avg_box(list, i1, i2, radius):
    x = 0
    difs = [
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [1, -1],
        [1, 0],
        [1, 1],
        [0, -1],
        [0, 0],
        [0, 1]
    ]
    for pair in difs:
        x += list[i1 + pair[0]][i2 + pair[1]]
    return x/len(difs)

#get time series
data = []
mx = []
my = []
for day in range(len(main)):
    date = start_date + datetime.timedelta(days=day)
    x = 0
    var, sign = var_wrap[key]
    for key in var_wrap.keys():
        flux = var[day][0][lat][lon]
        if math.isnan(flux):
            print(date)
        x += flux * sign * 3.154/2.2 * np.power(10, 14)
    data.append({'date': date, 'bc': x})
    if min_date <= date < max_date:
        mx.append(date)
        my.append(x)

#save file
with open("/Users/noahliguori-bills/Downloads/CESM-tools/data/canadian-NEEM-timeseries.csv", 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['date', 'bc'])
    writer.writeheader()
    writer.writerows(data)

plt.scatter(mx, my)
plt.show()'''
#date is from 10/25/22 to 10/25/23
#output: 63.05360592510554 ng/g BC

p = pd.read_csv('data/mcconnell-2007-1-figure.csv')
colors = ['#982724' for i in range(len(p['Yr']))]
colors[0] = ('#249598')
fig, ax = plt.subplots()
#plt.figure(figsize=(959, 480))
ax.set_yscale('log')
ax.set_ylim(1, 100)
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.minorticks_off()
ax.set_yticks([1, 10, 100])
ax.set_xticks([1800, 1850, 1900, 1950, 2000, 2023])
plt.title("NEEM Ice Core vs Modeled 2023 Fire Season BC")
plt.xlabel("Year (CE)")
plt.ylabel("BC Concentration (ng/g)")
plt.scatter(p['Yr'], p['BC'], c=colors)
legend_elements = [
            Line2D([0], [0], marker='o', color="#982724", label='NEEM', markersize=5),
            Line2D([0], [0], marker='o', color="#249598", label='Modeled 2023 Value', markersize=5),
            ]
plt.legend(handles=legend_elements)
#plt.show()
fig.set_size_inches(3.9*2, 3.9)
plt.savefig('matplot-out.png', dpi=72*3)