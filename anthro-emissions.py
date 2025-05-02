from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tools
import os
T = tools.ToolBox()

#combine files
'''
ncks -O --mk_rec_dmn time BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_185101-189912.nc BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_185101-189912.nc
ncks -O --mk_rec_dmn time BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_190001-194912.nc BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_190001-194912.nc
ncrcat BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_185101-189912.nc BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_190001-194912.nc out.nc
'''
#crop to north america
#ncks -d lat,7,90 -d lon,-179,-20 out.nc cropped-na.nc
#and north hemisphere
#ncks -d lat,0,90 -d lon,-180,180 out.nc cropped-nh.nc

f_na = Dataset('/glade/derecho/scratch/nlbills/ceds-anthro-emissions/cropped-na.nc')
f_nh = Dataset('/glade/derecho/scratch/nlbills/ceds-anthro-emissions/cropped-nh.nc')

sector_dict = {0: 'Agriculture', 1: 'Energy', 2: 'Industrial', 3: 'Transportation', 4: 'Residential, Commercial, Other', 5: 'Solvents production and application', 6: 'Waste', 7: 'International Shipping'}

#compentnet
na_regional_mean = np.sum(np.sum(f_na['BC_em_anthro'][:], axis=3), axis=2)

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
na_sum = np.sum(np.sum(np.sum(f_na['BC_em_anthro'][:], axis=3), axis=2), axis=1)
nh_sum = np.sum(np.sum(np.sum(f_nh['BC_em_anthro'][:], axis=3), axis=2), axis=1)
na_times = f_na['time'][:] / 365 + 1750
nh_times = f_nh['time'][:] / 365 + 1750
plt.plot(na_times, na_sum, label='North America')
plt.plot(nh_times, nh_sum, label='Northern Hemisphere')
plt.legend()
plt.title('CEDS BC Emissions')
plt.xlabel('Year (CE)')
plt.ylabel('BC (kg m-2 s-1)')
plt.ylim()
plt.savefig(os.path.join(os.getcwd(), 'combined-anthro-emissions.png'), dpi=200)
plt.close()

#bar chart
dates = {'pi': [1850, 1875], 'pd': [1955, 1980]}
means = {}
for key, pair in dates.items():
    indexes = [T.nearest_search(nh_times, pair[i]) for i in range(len(pair))]
    means[key] = np.mean(f_nh['BC_em_anthro'][indexes[0]: indexes[1] + 1])
plt.bar(list(means.keys()), list(means.values()))
plt.savefig(os.path.join(os.getcwd(), 'bar-anthro-emissions.png'), dpi=200)
plt.close()

#save pandas csv
df = pd.DataFrame(columns=['nh time', 'na time', 'nh', 'na'], data=np.transpose([f_nh['time'][:] / 365 + 1750, f_na['time'][:] / 365 + 1750, nh_sum, na_sum]))
df.to_csv(os.path.join(os.getcwd(), 'antrho-emissions.csv'))
print('saved to ' + os.path.join(os.getcwd(), 'combined-anthro-emissions.png'))