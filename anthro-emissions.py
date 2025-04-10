from netCDF4 import Dataset
import numpy as np

f = Dataset('/glade/derecho/scratch/nlbills/ceds-anthro-emissions/cropped.nc')
print(np.shape(f['BC_em_anthro']))