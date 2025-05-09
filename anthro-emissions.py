from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tools
import json
import sys
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

if len(sys.argv) < 2:
    raise Exception('1 command line arguments required: <mode (t/r)>')
mode = sys.argv[1]
root = '/glade/derecho/scratch/nlbills/ceds-anthro-emissions'

if mode == 't':
    f_na = Dataset(os.path.join(root, 'cropped-na.nc'))
    f_nh = Dataset(os.path.join(root, 'cropped-nh.nc'))

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
    print('pd/pi ratio: ' + str(means['pd'] / means['pi']))
    plt.bar(list(means.keys()), list(means.values()))
    plt.savefig(os.path.join(os.getcwd(), 'bar-anthro-emissions.png'), dpi=200)
    plt.close()

    #save pandas csv
    df = pd.DataFrame(columns=['nh time', 'na time', 'nh', 'na'], data=np.transpose([f_nh['time'][:] / 365 + 1750, f_na['time'][:] / 365 + 1750, nh_sum, na_sum]))
    df.to_csv(os.path.join(os.getcwd(), 'antrho-emissions.csv'))
    print('saved to ' + os.path.join(os.getcwd(), '*'))
elif mode == 'r':
    ncdf_dict = {
        'pi': {
            'filename': '185001-189912.nc',
            'start': 1850,
            'end': 1875
            },
        'pd': {
            'filename': 'BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_195001-199912.nc',
            'start': 1955,
            'end': 1980
            }
        }
    #ice_coords = T.get_ice_coords('data/standardized-ice-cores/index.csv', 'data/standardized-ice-cores/index-dup-cores.csv')
    anthro_boxes = json.load(open('data/emission-boxes.json'))
    df = pd.DataFrame(index=[0, 1, 2], columns=list(anthro_boxes.keys()))
    for era in ncdf_dict.keys():
        print(era + ' data setup...')
        d = ncdf_dict[era]
        f = Dataset(os.path.join(root, d['filename']))
        d['times'] = f['time'][:]
        d['lats'] = f['lat'][:]
        d['lons'] = f['lon'][:]
        start_i = T.nearest_search(d['times'], 365 * (d['start'] - 1750))
        end_i = T.nearest_search(d['times'], 365 * (d['end'] - 1750))
        #dim order: time, sector, lat, lon
        arr = f['BC_em_anthro'][start_i:end_i,:,:,:]
        d['arr'] = np.sum(np.mean(arr, axis=0), axis=0)
        d['arr'][d['arr'] == 0] = 1
        f.close()
    print('extracting ratios...')
    main_arr = np.divide(ncdf_dict['pd']['arr'], ncdf_dict['pi']['arr'])
    for region, boxes in anthro_boxes.items():
        for i in range(3):
            box = boxes[i]
            lat_min = T.nearest_search(ncdf_dict['pd']['lats'], box[0])
            lat_max = T.nearest_search(ncdf_dict['pd']['lats'], box[1])
            lon_min = T.nearest_search(ncdf_dict['pd']['lons'], box[2])
            lon_max = T.nearest_search(ncdf_dict['pd']['lons'], box[3])
            df.loc[i, region] = np.mean(main_arr[lat_min:lat_max, lon_min:lon_max])
    df.to_csv(os.path.join(os.getcwd(), 'anthro-ratios.csv'))
    print('saved to ' + os.path.join(os.getcwd(), 'anthro-ratios.csv'))
    print('plotting...')
    import cartopy
    from matplotlib import colormaps
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm
    #color
    cmap = colormaps['BrBG_r']
    c_norm = LogNorm(vmin=0.00001, vmax=100000)
    sm = ScalarMappable(cmap=cmap, norm=c_norm)
    #plot
    fig, ax = plt.subplots(dpi=200, subplot_kw={'projection': cartopy.crs.Robinson()})
    ax.add_feature(cartopy.feature.COASTLINE, edgecolor='grey')
    plt.pcolormesh(ncdf_dict['pd']['lons'], ncdf_dict['pd']['lats'], main_arr, cmap=cmap, norm=c_norm, transform=cartopy.crs.PlateCarree())
    plt.colorbar(mappable=sm, label='Anthro BC Emission Ratio', orientation="horizontal", ax=ax, extend='both')
    plt.savefig(os.path.join(os.getcwd(), 'robinson-fig.png'), dpi=200)
    print('saved to ' + os.path.join(os.getcwd(), 'robinson-fig.png'))