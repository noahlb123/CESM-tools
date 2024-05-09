import cartopy.crs as ccrs
from cartopy import feature
from netCDF4 import Dataset
import matplotlib.pyplot as plt

#get data
f = Dataset('data/matt.nc')
lats = f['lat'][:]
lons = f['lon'][:]
x = f['aot_869'][:]

#plot
ax = plt.axes(projection=ccrs.Robinson())
ax.add_feature(feature.COASTLINE, edgecolor='grey')
plt.pcolormesh(lons, lats, x, transform=ccrs.PlateCarree())
plt.show()