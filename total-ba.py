#i need to use a geopandas GeoDataFrame to better organize my polygons, now it is in a GeoSeries
import numpy as np
from netCDF4 import Dataset
from shapely.geometry import box
import geopandas as gpd
from math import radians, sin

#month, year to GFED5 .nc filename
def d2fname(m, y=0):
    return "BA" + str(year) + str(month).zfill(2) + ".nc"

#grid resolution from year
def d2res(month, year):
    return 0.25 if year > 2000 or (year == 2000 and month > 11) else 1

#get area from lat lon grid cells
def coords2area(lat, lon, res):
    AVG_EARTH_RADIUS_KM = 6371.0088
    west = radians(lon - res / 2)
    east = radians(lon + res / 2)
    south = radians(lat - res / 2)
    north = radians(lat + res / 2)
    area = (east - west) * (sin(north) - sin(south)) * (AVG_EARTH_RADIUS_KM**2)
    return area

#get total BA from each GFED5 file
data = [] #["Year", "GFED 5 Global Total BA"]
year_tot = 0
for year in range(1997, 2021):
    print(year)
    for month in range(1, 13):
        f = Dataset("/Users/noahliguori-bills/Downloads/CESM-tools/data/BA/" + d2fname(month, year), "r")
        BA = f["Total"][:][0]
        #lat = f["lat"][:]
        #lon = f["lon"][:]
        #res = d2res(month, year)
        #lon_num = 360 // res
        #lat_num = 180 // res
        for x in range(len(BA)):
            for y in range(len(BA[x])):
                #area = coords2area(lat[x], lon[y], res)
                year_tot += BA[x][y] #* area
        f.close()
    data.append([year, year_tot])
    year_tot = 0


#save
np.savetxt("/Users/noahliguori-bills/Downloads/CESM-tools/test.csv", data, delimiter=",")
print('done.')