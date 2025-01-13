import os
import sys
import platform
from tools import ToolBox
from netCDF4 import Dataset
import numpy as np
import pandas as pd
T = ToolBox()

def evaluate(s):
    l = len(s)
    if s[l - 4:l] == ' && ':
        s = s[0:l - 4]
    os.system(s)
    return 'cd ' + root + ' && '

if platform.system() == 'Linux':
    #python3 lev_p.py /glade/derecho/scratch/nlbills/all-ice-core-data/mmrbc output.nc mmrbc
    if len(sys.argv) < 4:
        raise Exception('3 command line arguments required: <root path> <file name> <target variable>')
    root = sys.argv[1]
    file_name = sys.argv[2]
    target_v = sys.argv[3]

    '''to_eval = 'cd ' + root + ' && '
    to_eval += "ncap2 -O -s 'p=double(a*p0+b*ps);' " + file_name + ' ' + file_name + ' && '
    evaluate(to_eval)'''

    #read variables
    f = Dataset(os.path.join(root, file_name))
    lats, lons = T.adjust_lat_lon_format(f['lat'][:], f['lon'][:])
    lev = f['lev'][:]
    p0 = f['p0'][:]
    b = f['b'][:]
    ps = f['ps'][:][0]
    a = f['a'][:]
    x = f[target_v][:]
    f.close()

    #extend var dimensions for elementwise multiplication
    ps = np.stack([ps for i in range(len(lev))], 0)
    b = np.stack([np.stack([b for i in range(len(lons))], -1) for j in range(len(lats))], -2)
    a = np.stack([np.stack([a for i in range(len(lons))], -1) for j in range(len(lats))], -2)

    #calculate pressure levels
    p = a * p0 + b * ps

    #average northern hemisphere
    lat_30_i = T.nearest_search(lats, 30)
    nh_p = p[:,lat_30_i:,:]
    nh_x = x[:,lat_30_i:,:]
    print(x, nh_x)
    mean_p_nh = np.mean(nh_p, axis=(1,2))
    mean_x_nh = np.mean(nh_x, axis=(1,2))

    #save
    print(np.shape(mean_p_nh), np.shape(mean_x_nh))
    exit()
    df = pd.DataFrame(columns=['pressure', 'var'], data=[mean_p_nh, mean_x_nh])
    print('saved to ' + os.path.join(root, 'lev_p.csv'))
    df.to_csv(os.path.join(root, 'lev_p.csv'))
elif platform.system() == 'Darwin':
    pass