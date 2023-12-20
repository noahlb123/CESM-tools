import numpy as np
from netCDF4 import Dataset
import interpolate

n_months = 24

#month to GFED5 .nc filename
def m2fname(m):
    first_yr = 2010
    if (m <= 12):
        y = first_yr
        month = m
    else:
        y = first_yr + 1
        month = m - 12
    return "BA" + str(y) + str(month).zfill(2) + ".nc"

#get total BA from each GFED5 file
'''g5_BA = []
for m in range(n_months):
    f5 = Dataset("/Users/noahliguori-bills/Downloads/gfed-interpolation/BA/" + m2fname(m + 1), "r")
    g5_BA.append(f5["Total"][:][0])
    f5.close()'''
f5 = Dataset("/Users/noahliguori-bills/Downloads/gfed-interpolation/g5x2.nc", "r")
g5_BA = f5["BA"][:]

#calculate bias using GFED4 file
f4 = Dataset("/Users/noahliguori-bills/Downloads/gfed-interpolation/g4x2.nc", "r")
g4_BA = []
lats =  f4["lat"][:]
lons =  f4["lon"][:]
for year in f4["BurntArea"][:]:
    for month in year:
        g4_BA.append(month)
f4.close()
if (n_months != 24):
    g4_BA = g4_BA[12*9:12*9+1]
else:
    g4_BA = g4_BA[12*9:12*11] #check this
bias = []
c = 0
for m in range(len(g4_BA)):
    print(m)
    bias.append([])
    for x in range(len(g4_BA[m])):
        bias[m].append([])
        for y in range(len(g4_BA[m][x])):
            g4 = g4_BA[m][x][y]
            g5 = 100 * g5_BA[m][x][y]
            if (g4 == 0 and g5 != 0):
                bias[m][x].append(0)
                c += 1
            elif (g4 == 0 and g5 == 0):
                bias[m][x].append(0)
            else:
                b = g5/g4 if g5/g4 <= 10 else 10
                bias[m][x].append(b)

#placeholder bias for testing
'''bias = []
for i in range(24):
    bias.append([])
    for x in range(720):
        bias[i].append([])
        for y in range(1440):
            bias[i][x].append(0)'''

f = Dataset("/Users/noahliguori-bills/Downloads/gfed-interpolation/g4x2.nc", "r")
out = Dataset("/Users/noahliguori-bills/Downloads/gfed-interpolation/test.nc", "w")
month = out.createDimension("month", 24)
lat = out.createDimension("lat", 192)
lon = out.createDimension("lon", 288)
months = out.createVariable("month","i1",("month"))
BAs = out.createVariable("BA","f8",("month", "lat", "lon"))
latitudes = out.createVariable("lat","f8",("lat"))
longitudes = out.createVariable("lon","f8",("lon"))
#lats =  np.arange(-90,90,0.25)
#lons =  np.arange(-180,180,0.25)
#apply interpolation
latitudes[:] = f["lat"][:]
longitudes[:] = f["lon"][:]
BAs[:] = bias
out.close()
print('done.')