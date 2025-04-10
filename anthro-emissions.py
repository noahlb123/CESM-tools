from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import os

f = Dataset('/glade/derecho/scratch/nlbills/ceds-anthro-emissions/cropped.nc')
regional_mean = np.reshape(np.mean(np.mean(f['BC_em_anthro'][:], axis=3), axis=2), (8, 1200))

sector_dict = {0: 'Agriculture', 1: 'Energy', 2: 'Industrial', 3: 'Transportation', 4: 'Residential, Commercial, Other', 5: 'Solvents production and application', 6: 'Waste', 7: 'International Shipping'}

plt.stackplot(f['time'], regional_mean, labels=list(sector_dict.values()))
plt.savefig(os.path.join(os.getcwd(), 'stacked-anthro-emissions.png'), dpi=200)