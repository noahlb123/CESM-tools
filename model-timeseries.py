from netCDF4 import Dataset
import pandas as pd
import os

#get files from each model
lens_files = pd.read_csv(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'lens', 'fileuse-index.csv'))
cmip_files = pd.read_csv(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'cmip6', 'fileuse-index.csv'))
sootsn_files = pd.read_csv(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'cesm-sootsn', 'fileuse-index.csv'))

#combine files into one dataframe
files = pd.DataFrame(columns=['files'])
files['files'] = lens_files['files'] + cmip_files['wet file'] + cmip_files['dry file'] + sootsn_files['wet file']

print(files)
print(lens_files['files'] + sootsn_files['wet file'])
#print(sootsn_files)