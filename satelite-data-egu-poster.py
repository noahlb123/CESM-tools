import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
import cartopy
import numpy as np

ax = plt.axes(projection=cartopy.crs.Robinson())
ax.set_extent([-180, 180, -90, 90], crs=cartopy.crs.PlateCarree())
ax.add_feature(cartopy.feature.COASTLINE, edgecolor='grey')

#read shapefile
shape = cartopy.io.shapereader.Reader("/Users/noahliguori-bills/Downloads/CESM-tools/data/FIRMS-data/fire_nrt_LS_455286.shp").geometries()
#shape_feature = cartopy.feature.ShapelyFeature(shape, cartopy.crs.PlateCarree(), facecolor='green')
ax.add_geometries(shape, cartopy.crs.PlateCarree(), facecolor='white', hatch='xxxx')
#ax.add_feature(shape_feature)

#setup color scale
cmap = colormaps['BuGn']
bounds = [round(x, 1) for x in np.linspace(0, 2, 10)]
norm = BoundaryNorm(bounds, cmap.N)
sm = ScalarMappable(cmap=cmap, norm=norm)

plt.plot(-44, 71.4, c='red', markeredgecolor='black', marker='.', markersize=9, transform=cartopy.crs.PlateCarree())
#plt.plot(lon, lat, c=cmap(norm(math.log(obj['ratio'], 10))), markeredgecolor='black', marker='.', markersize=6, transform=cartopy.crs.PlateCarree())
plt.colorbar(mappable=sm, label="PD/PI BC Conc.", orientation="horizontal")

#plt.savefig('figures/ice-cores/rotated-pole.png', dpi=300)
plt.show()