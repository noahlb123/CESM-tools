from netCDF4 import Dataset
import pandas as pd
import numpy as np
import sys
import os
from tools import ToolBox
T = ToolBox()

if len(sys.argv) < 2:
    #python lens-bias.py filname1 filename2...
    raise Exception('filename(s) required: (python lens-bias.py bias1.nc bias2.nc...)')
files = sys.argv[1:]

index_path = 'data/standardized-ice-cores/index.csv'
dupe_path = 'data/standardized-ice-cores/index-dup-cores.csv'
ice_coords = T.get_ice_coords(index_path, dupe_path)
df = pd.DataFrame(index=files, columns=list(ice_coords.keys()))
lens_data = pd.read_csv('data/model-ice-depo/lens/lens.csv')
lens_data = lens_data.set_index('Unnamed: 0').drop(['pi'], axis=0).mean(axis=0)

for file in files:
    f = Dataset(file)
    lats, lons = T.adjust_lat_lon_format(f['lat'][:], f['lon'][:])
    times = f['time']
    v = f['X'][0,0,:,:] if 'mmrbc' in file else f['X'][0,:,:]
    for core_name in ice_coords.keys():
        y, x = ice_coords[core_name]
        lat = T.nearest_search(lats, y)
        lon = T.nearest_search(lons, x)
        df.loc[file, core_name] = lens_data[core_name] * v[lat, lon]
    f.close()

df.to_csv(os.path.join(os.getcwd(), 'lens-bias.csv'))