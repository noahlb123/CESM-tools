import numpy as np
from netCDF4 import Dataset
import interpolate

n_time = 24

#get total BA from each GFED5 file
'''g5_BA = []
for m in range(n_time):
    f5 = Dataset("/Users/noahliguori-bills/Downloads/gfed-interpolation/BA/" + m2fname(m + 1), "r")
    g5_BA.append(f5["Total"][:][0])
    f5.close()'''
f = Dataset("/Users/noahliguori-bills/Downloads/gfed-interpolation/emissions-cmip6_SOAGx1.5_bb_surface_2010climo_0.9x1.25_c20180918.nc", "r")
emissions = f["emiss_bb"][:]

#calculate bias using GFED4 file
f4 = Dataset("/Users/noahliguori-bills/Downloads/gfed-interpolation/bias.nc", "r")
bias = f4["BA"][:]
lats =  f4["lat"][:]
lons =  f4["lon"][:]
f4.close()
adj_e = []
c = 0
for m in range(len(emissions)):
    print(m)
    adj_e.append([])
    for x in range(len(emissions[m])):
        adj_e[m].append([])
        for y in range(len(emissions[m][x])):
            e = emissions[m][x][y]
            b = bias[m][x][y]
            adj_e[m][x].append(e * b)

#placeholder bias for testing
'''bias = []
for i in range(24):
    bias.append([])
    for x in range(720):
        bias[i].append([])
        for y in range(1440):
            bias[i][x].append(0)'''

#oldemissions = Dataset("/Users/noahliguori-bills/Downloads/gfed-interpolation/oldemissions.nc", "r")
out = Dataset("/Users/noahliguori-bills/Downloads/gfed-interpolation/test.nc", "w")
month = out.createDimension("time", 12)
lat = out.createDimension("lat", 192)
lon = out.createDimension("lon", 288)
time = out.createVariable("time","f8",("time"))
BAs = out.createVariable("emiss_bb","f8",("time", "lat", "lon"))
latitudes = out.createVariable("lat","f8",("lat"))
longitudes = out.createVariable("lon","f8",("lon"))
dates = out.createVariable("date","f8",("time"))
dates.units = "YYYYMMDD"
dates.cell_methods = "time: mean"
dates.long_name = "Date"
latitudes.long_name = "Latitude"
longitudes.long_name = "Longitude"
time.long_name = "Time"
BAs.long_name = "bc_a4 CEDS biomass burning emissions"
BAs.units = "bc_a4 CEDS biomass burning emissions"
BAs.history = "CEDS species: BC*1.0"
BAs.molecular_weight = 12
BAs.cell_methods = "time: mean"
latitudes.units = "degrees_north"
longitudes.units = "degrees_east"
time.units = "days since 1750-01-01 00:00:00"
time.calendar = "Gregorian"
time.cell_methods = "time: mean"
BAs.units = "molecules/cm2/s"
#apply interpolation
latitudes[:] = f["lat"][:]
longitudes[:] = f["lon"][:]
time[:] = f["time"][:]
dates[:] = f["date"][:]
f.close()
BAs[:] = adj_e
#oldemissions.close()
out.close()
print('done.')