import requests
from netCDF4 import Dataset
import tools
from global_land_mask import globe

T = tools.ToolBox()

#open old dataset
f = Dataset('data/elevation-copy.nc')
z = f["z"][:] #772 1543
lons = f["x"][:]
lats = f["y"][:]
n = len(lons)

#zero or land elevation
def mask(x, y):
    if y == 0 and x % 20 == 0:
        print(x, '/', n)
    #if x % 100 == 0:
    #print(y, x, "/", len(lons))
    #wrapper = {'latlon': [str(lats[x]),str(lons[y])]}
    #is_ocean = requests.post('http://127.0.0.1:4000/latlon', json=wrapper).json()['result']
    return z[y][x] if globe.is_land(lats[y], lons[x]) else 0

#create new dataset
new_f = Dataset('data/test.nc', "w", format="NETCDF4_CLASSIC")
new_f.createDimension("lat", len(lats))
new_f.createDimension("lon", len(lons))
lat_var = new_f.createVariable("lat","f4",("lat"))
lon_var = new_f.createVariable("lon","f4",("lon"))
land_elev = new_f.createVariable("land_elev","f4",("lon","lat"))
land_elev.units = "m"
lat_var[:] = lats
lon_var[:] = lons
land_elev[:] = [[mask(x, y) for y in range(len(lats))] for x in range(len(lons))]

new_f.close()
f.close()