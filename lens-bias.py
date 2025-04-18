from netCDF4 import Dataset
import pandas as pd
import numpy as np
import platform
import tools
import csv
import sys
import os
from tools import ToolBox
T = ToolBox()

index_path = 'data/standardized-ice-cores/index.csv'
dupe_path = 'data/standardized-ice-cores/index-dup-cores.csv'
ice_coords = T.get_ice_coords(index_path, dupe_path)
df = pd.DataFrame(columns=['model'] + list(ice_coords.keys()))

lens_data = pd.read_csv('/glade/derecho/scratch/nlbills/all-ice-core-data/lens.csv')
f = Dataset('/glade/derecho/scratch/nlbills/all-ice-core-data/bias.nc')
lats, lons = T.adjust_lat_lon_format(f['lat'][:], f['lon'][:])
times = f['time']
v = f['loadbc'][:]
for core_name in ice_coords.keys():
    y, x = ice_coords[core_name]
    lat = T.nearest_search(lats, y)
    lon = T.nearest_search(lons, x)
    lens_data[core_name] = lens_data[core_name] * v[0, lat, lon]
f.close()

lens_data.to_csv(os.path.join(os.getcwd(), 'lens-bias.csv'))