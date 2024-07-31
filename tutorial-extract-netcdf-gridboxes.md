# Extract NetCDF vars at specific grid boxes tutorial:

To download this repository:
```
git clone https://github.com/noahlb123/CESM-tools
cd CESM-tools
```
To use these tools with your code
1. Make a new python script
```
touch index.py
```
1. Or move an existing python script into the CESM-tools directory
```
mv [path]/index.py [path]/CESM-tools/index.py
```
2. In your python file, add lines to import tools.py and make a tools object
```python
import tools
T = tools.ToolBox()
```
3. Import your NetCDF file any way you want. I use the netCDF4 package.
```python
from netCDF4 import Dataset
f = Dataset(path)
lats = f['lats'][:]
lons = f['lons'][:]
x_var = f['x'][:]
```
4. Use nearest_search to get the indexes of the lat and lon you are interested in. The nearest_search function takes two input parameters (l [array like object], n [number]) and returns the index of the value in l closest to n.
```python
my_lat = 20
my_lon = -20
lat_index = T.nearest_search(lats, my_lat)
lon_index = T.nearest_search(lons, my_lon)
```
5. Access the variable "x" at that lat and lon
```python
x = x_var[lat_index][lon_index]
```
6. Each NetCDF file is formatted differently, the lat and lon variable names used in step 3 can vary, the order and number of the dimensions can vary, and the units of everything can vary. You must check these things using Panoply or `ncdump -h filename.nc`. For example many NetCDF files have a time dimension, which would change step 5 to be...
```python
x = x_var[time][lat_index][lon_index]
```
7. All together this gives:
```python
import tools
from netCDF4 import Dataset

#setup toolbox object
T = tools.ToolBox()

#setup NetCDF variables
f = Dataset(path)
lats = f['lats'][:]
lons = f['lons'][:]
x_var = f['x'][:]

#get lat and lon indexes
my_lat = 20
my_lon = -20
lat_index = T.nearest_search(lats, my_lat)
lon_index = T.nearest_search(lons, my_lon)

#extract the value at my_lat, my_lon
x = x_var[lat_index][lon_index]
print(x)
```
8. For more exmaples of using nearest_search, see [lens.py](https://github.com/noahlb123/CESM-tools/blob/main/lens.py) and [cmip.py](https://github.com/noahlb123/CESM-tools/blob/main/cmip.py).
