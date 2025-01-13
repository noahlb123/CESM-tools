import os
import sys
import platform
from tools import ToolBox
from netCDF4 import Dataset
import numpy as np
T = ToolBox()

def evaluate(s):
    l = len(s)
    if s[l - 4:l] == ' && ':
        s = s[0:l - 4]
    os.system(s)
    return 'cd ' + root + ' && '

#python3 lev_p.py /glade/derecho/scratch/nlbills/all-ice-core-data/mmrbc output.nc mmrbc
if len(sys.argv) < 4:
    raise Exception('3 command line arguments required: <root path> <file name> <target variable>')
root = sys.argv[1]
file_name = sys.argv[2]
target_v = sys.argv[3]

'''to_eval = 'cd ' + root + ' && '
to_eval += "ncap2 -O -s 'p=double(a*p0+b*ps);' " + file_name + ' ' + file_name + ' && '
evaluate(to_eval)'''

f = Dataset(os.path.join(root, file_name))
lats, lons = T.adjust_lat_lon_format(f['lat'][:], f['lon'][:])
lev = f['lev'][:]
p0 = f['p0'][:]
b = f['b'][:]
ps = f['ps'][:][0]
a = f['a'][:]
f.close()


ps = np.stack([ps for i in range(len(lev))], 0)
b = np.stack([np.stack([b for i in range(len(lons))], -1) for j in range(len(lats))], -2)
print(np.shape(ps), np.shape(b))