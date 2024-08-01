from netCDF4 import Dataset
import pandas as pd
import numpy as np
import tools
import os
T = tools.ToolBox()

#get files from each model
lens_files = pd.read_csv(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'lens', 'fileuse-index.csv'))
cmip_files = pd.read_csv(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'cmip6', 'fileuse-index.csv'))
sootsn_files = pd.read_csv(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'cesm-sootsn', 'fileuse-index.csv'))

#combine files into one dataframe
files = pd.DataFrame(columns=['files'])
files['files'] = lens_files['files']
i = len(files['files']) - 1
for file in cmip_files['wet file']: #for some ungodly reason I have to do this for all files
    files.loc[i] = [file]
    i += 1
for file in cmip_files['dry file']:
    files.loc[i] = [file]
    i += 1
for file in sootsn_files['wet file']:
    files.loc[i] = [file]
    i += 1

#add var each file uses
vars = pd.Series(['bc_a1_SRF'] * (len(lens_files['files']) - 1) + ['wetbc'] * len(cmip_files['wet file']) + ['drybc'] * len(cmip_files['dry file']) + ['sootsn'] * len(sootsn_files['wet file']))
files['var'] = vars

#use location of southernmost ice core (#23), mcconnell-2021-4.csv, lat, lon = (-82.1, 54.9)
s_lat, s_lon = T.get_ice_coords('data/standardized-ice-cores/index.csv', 'data/standardized-ice-cores/index-dup-cores.csv')['mcconnell-2021-4.csv']

#get timeseries
x = [i + 0.5 for i in range(1850, 1981)]
timeseries = pd.DataFrame(columns=['year'], data=x)
c = 0
for index, row in files.iterrows():
    file = row['files']
    v = row['var']
    if v == 'bc_a1_SRF': #skip lens files bec they only have 1 time pt
        continue
    f = Dataset(file)
    lat = T.nearest_search(f['lat'], s_lat)
    lon = T.nearest_search(f['lon'], s_lon)
    yr = f['time'][:]
    bc = f[v][:,lat,lon]
    print(np.shape(x))
    print(yr)
    print(np.shape(bc))
    timeseries[file] = np.interp(x, yr, bc)
    f.close()
    if c > 10:
        break

print(timeseries)