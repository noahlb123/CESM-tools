#i need to use a geopandas GeoDataFrame to better organize my polygons, now it is in a GeoSeries
import sys
import os
import numpy as np
from netCDF4 import Dataset
from shapely.geometry import box
import geopandas as gpd
from math import radians, sin
import pandas as pd

#month, year to GFED5 .nc filename
def d2fname(m, y=0):
    return "BA" + str(year) + str(month).zfill(2) + ".nc"

#grid resolution from year
def d2res(month, year):
    return 0.25 if year > 2000 or (year == 2000 and month > 11) else 1

#get total BA from each GFED5 file
if len(sys.argv) < 3:
    #python3 total-ba.py Total /Users/noahliguori-bills/Downloads/CESM-tools/data/
    raise Exception('2 command line arguments required: python3 total-ba.py <.nc varaible name> <root dir>')
v = sys.argv[1]
root = sys.argv[2]
v2name = {'Total': 'BA'}
save_name = v2name[v] if v in v2name.keys() else v
df = pd.DataFrame(columns=['year', 'total-' + v])
year_tot = 0
for year in range(1997, 2021):
    print(year)
    for month in range(1, 13):
        f = Dataset(os.path.join(root, save_name, d2fname(month, year)), "r")
        x = f[v][:][0]
        year_tot += np.sum(x)
        if year == 1997 and month == 1:
            if 'units' in f[v]:
                units = f[v].units
            else:
                units = 'no units in file, gfed5 BA has units=km^2'
        f.close()
    df.loc[year] = [year, year_tot]
    year_tot = 0

#save
df.to_csv(os.path.join(root, 'GFED5-total-global-yearly-' + save_name + ".csv"))
print('2001 to 2020 avg:', df['total-' + v].loc[2001:2020].mean())
print(units)

#How I am doing mass number calculations:
#Min: 0.12 (jones percent annual emissions) * 2160 * 10^14 g (gfed4 emissions)          = 26 pg C
#Max: 0.15 (Bowring percent annual emissions) * 1.61 * 2160 * 10^14 g (gfed4 emissions) = 52 pg C
#numbers listed in papers: 89- Tg
