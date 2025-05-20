import cartopy
from matplotlib import colormaps
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from tools import ToolBox
import os
import sys
T = ToolBox()

if len(sys.argv) < 5:
    raise Exception('4 command line arguments required: <file path> <lat name> <lon name> <target variable name>')
path = sys.argv[1]
lat_name = sys.argv[2]
lon_name = sys.argv[3]
var_name = sys.argv[4]

#setup cartopy
plt.clf()
fig, ax = plt.subplots(dpi=200, subplot_kw={'projection': cartopy.crs.Robinson()})
ax.add_feature(cartopy.feature.COASTLINE, edgecolor='grey')

#get data
f = Dataset(path)
lats = f[lat_name][:]
lons = f[lon_name][:]
x = f[var_name][:]

#color
cmap = colormaps['viridis']
try:
    vmin, vmax = (np.nanmin(np.ma.masked_invalid(x)), np.nanmax(np.ma.masked_invalid(x)))
    if vmin == 0:
        vmin = vmax / 100
except:
    vmin, vmax = (0.1, 1000)
c_norm = LogNorm(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=cmap, norm=c_norm)

#plot
plt.pcolormesh(lons, lats, f[var_name][:][0], cmap=cmap, norm=c_norm, transform=cartopy.crs.PlateCarree())
plt.colorbar(mappable=sm, label=var_name, orientation="horizontal", ax=ax, extend='both')
plt.savefig(os.path.join(os.getcwd(), 'robinson-fig.png'), dpi=200)
print('saved to ' + os.path.join(os.getcwd(), 'robinson-fig.png'))