from netCDF4 import Dataset
import sys
import os

if len(sys.argv) < 2:
    raise Exception('1 command line argument required: <root directory>')
dir = sys.argv[1]
min_lat = 999999999
min_lon = 999999999
min_name = ''

for filename in os.listdir(dir):
    if '.nc' in filename:
        f = Dataset(os.path.join(dir, filename))
        lat = f['lat'].shape[0]
        lon = f['lon'].shape[0]
        if lat < min_lat or lon < min_lon:
            min_lat = lat
            min_lon = lon
            min_name = filename

print('min lat, lon (', lat, lon, '):')
print(min_name)
