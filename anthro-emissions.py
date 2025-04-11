from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

#crop to north america
#ncks -d lat,7,90 -d lon,-179,-20 out.nc cropped-na.nc
#and north hemisphere
#ncks -d lat,0,90 -d lon,-180,180 out.nc cropped-nh.nc

f_na = Dataset('/glade/derecho/scratch/nlbills/ceds-anthro-emissions/cropped-na.nc')
f_nh = Dataset('/glade/derecho/scratch/nlbills/ceds-anthro-emissions/cropped-nh.nc')

sector_dict = {0: 'Agriculture', 1: 'Energy', 2: 'Industrial', 3: 'Transportation', 4: 'Residential, Commercial, Other', 5: 'Solvents production and application', 6: 'Waste', 7: 'International Shipping'}

#compentnet
na_regional_mean = np.mean(np.mean(f_na['BC_em_anthro'][:], axis=3), axis=2)

for i in range(8):
    plt.plot(f_na['time'][:] / 365 + 1750, na_regional_mean[:,i], label=sector_dict[i])
plt.legend()
plt.title('North American CEDS BC Emissions')
plt.xlabel('Year (CE)')
plt.ylabel('BC (kg m-2 s-1)')
plt.ylim()
plt.savefig(os.path.join(os.getcwd(), 'component-anthro-emissions.png'), dpi=200)
plt.close()

#combined
na_sum = np.sum(np.mean(np.mean(f_na['BC_em_anthro'][:], axis=3), axis=2), axis=1)
nh_sum = np.sum(np.mean(np.mean(f_nh['BC_em_anthro'][:], axis=3), axis=2), axis=1)
plt.plot(f_na['time'][:] / 365 + 1750, na_sum, label='North America')
plt.plot(f_nh['time'][:] / 365 + 1750, nh_sum, label='Northern Hemisphere')
plt.legend()
plt.title('CEDS BC Emissions')
plt.xlabel('Year (CE)')
plt.ylabel('BC (kg m-2 s-1)')
plt.ylim()
plt.savefig(os.path.join(os.getcwd(), 'combined-anthro-emissions.png'), dpi=200)

#save pandas csv
df = pd.DataFrame(columns=['nh time', 'na time', 'nh', 'na'], data=np.transpose([f_nh['time'][:] / 365 + 1750, f_na['time'][:] / 365 + 1750, nh_sum, na_sum]))
df.to_csv(os.path.join(os.getcwd(), 'antrho-emissions.csv'))