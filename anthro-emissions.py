from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import os

f = Dataset('/glade/derecho/scratch/nlbills/ceds-anthro-emissions/cropped.nc')
regional_mean = np.mean(np.mean(f['BC_em_anthro'][:], axis=3), axis=2)

sector_dict = {0: 'Agriculture', 1: 'Energy', 2: 'Industrial', 3: 'Transportation', 4: 'Residential, Commercial, Other', 5: 'Solvents production and application', 6: 'Waste', 7: 'International Shipping'}

for i in range(8):
    plt.plot(f['time'][:] / 365 + 1750, regional_mean[:,i], label=sector_dict[i])
plt.legend()
plt.title('North American CEDS BC Emissions')
plt.xlabel('Year (CE)')
plt.ylabel('BC (kg m-2 s-1)')
plt.savefig(os.path.join(os.getcwd(), 'stacked-anthro-emissions.png'), dpi=200)