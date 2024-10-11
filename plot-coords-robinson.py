import matplotlib.pyplot as plt
import pandas as pd
import cartopy
import sys

#read lat lons
if len(sys.argv) < 4:
    #python3 plot-coords-robinson.py "/Users/noahliguori-bills/Downloads/pyc/Jones-PyC-Production-Factors.csv" "lat" "lon"
    raise Exception('3 command line arguments required: python3 plot-coords-robinson.py <lat-lon-csv path> <lat name> <lon name>')
data_path, lat_name, lon_name = sys.argv[1:4]
df = pd.read_csv(data_path)
df = df[[lat_name, lon_name]].dropna()
print('number of coords:', len(df))

#plot
ax = plt.axes(projection=cartopy.crs.Robinson())
ax.set_extent([-180, 180, -90, 90], crs=cartopy.crs.PlateCarree())
ax.add_feature(cartopy.feature.COASTLINE, edgecolor='grey')
plt.scatter(df[lon_name], df[lat_name], c='#E93223', edgecolors='black', linewidth=1, marker='.', s=120, transform=cartopy.crs.PlateCarree(), zorder=2.5)
save_path = data_path[0:data_path.rfind('/') + 1] + 'robinson-lat-lons.png'
plt.savefig(save_path, bbox_inches='tight', pad_inches=0.01, dpi=300)

print('saved to ' + save_path + '!')