import numpy as np
from netCDF4 import Dataset
from tools import ToolBox
T = ToolBox()

#setup data
f = Dataset("/Users/noahliguori-bills/Downloads/CESM-tools/data/douglas-course-emissions.nc", "r")
lats =  f["lat"][:]
lons =  f["lon"][:]

f.close()

'''test = [
    [1, 2],
    [3, 4],
    [5, 6]
]

print(np.mean(test, axis=(1)))'''