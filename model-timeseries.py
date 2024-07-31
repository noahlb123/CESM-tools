from netCDF4 import Dataset
import pandas as pd
import numpy as np
import os

#get files from each model
lens_files = pd.read_csv(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'lens', 'fileuse-index.csv'))
cmip_files = pd.read_csv(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'cmip6', 'fileuse-index.csv'))
sootsn_files = pd.read_csv(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'cesm-sootsn', 'fileuse-index.csv'))

#combine files into one dataframe
files = pd.DataFrame(columns=['files'])
files['files'] = lens_files['files']
print(files.isnull().values.any())
files['files'] = lens_files['files'] + cmip_files['wet file']
print(files.isnull().values.any())
files['files'] = lens_files['files'] + cmip_files['wet file'] + cmip_files['dry file']
print(files.isnull().values.any())
i = len(files['files']) - 1
for file in sootsn_files['wet file']: #for some ungodly reason I have to do this for sootsn files
    files.loc[i] = [file]
    i += 1
print(files.isnull().values.any())

for file in files['files']:
    #print(file)
    pass
    #print(os.path.isfile(file), file)

#add var each file uses
vars = pd.Series(['bc_a1_SRF'] * len(lens_files['files']) + ['wetbc'] * len(cmip_files['wet file']) + ['drybc'] * len(cmip_files['dry file']) + ['sootsn'] * len(sootsn_files['wet file']))
files['var'] = vars
x = [i + 0.5 for i in range(1850, 1981)]
timeseries = pd.DataFrame(columns=['year'], data=x)
c = 0
for index, row in files.iterrows():
    file = row['files']
    v = row['var']
    f = Dataset(file)
    yr = f['time']
    bc = f[v]
    timeseries[file] = np.interp(x, yr, bc)
    if c >= 10:
        break

print(timeseries)