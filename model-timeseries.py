from netCDF4 import Dataset
import pandas as pd
import os

lens_files = pd.DataFrame(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'lens', 'fileuse-index.csv'))
cmip_files = pd.DataFrame(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'cmip6', 'fileuse-index.csv'))
sootsn_files = pd.DataFrame(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'cesm-sootsn', 'fileuse-index.csv'))