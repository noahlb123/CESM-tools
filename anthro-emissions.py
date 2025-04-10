from netCDF4 import Dataset
import numpy as np

f = Dataset('/glade/derecho/scratch/nlbills/ceds-anthro-emissions/cropped.nc')
print(np.shape(np.mean(np.mean(f['BC_em_anthro'][:], axis=3), axis=2)))
#print(np.shape(np.mean(np.mean(f['BC_em_anthro'][:]), axis=3), axis=2))