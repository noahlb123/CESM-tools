import json
import sys
import requests
import math

#set analsis mode
if len(sys.argv) < 2 or sys.argv[1] not in ["Seasonal PM2.5", "2024 LA Wildfires"]:
    #example: python epa-aqi-api.py '2024 LA Wildfires' /glade/derecho/scratch/nlbills/la-pm2.5/la-pm2.5/temp.nc
    raise Exception('1 command line argument required: <anaylysis type ("Seasonal PM2.5" or "2024 LA Wildfires")> ex. python3 epa-aqi-api.py "2024 LA Wildfires"')
analysis = sys.argv[1]
#file = sys.argv[2] if analysis == '2024 LA Wildfires' else None


if analysis == '2024 LA Wildfires':
    #crop by lat, lon
    #ncks -O -d lat_0,31.,37. -d lon_0,238.,244. copy.nc copy.nc
    #get data from specific lat,lon
    #ncks --no_nm_prn -H -C -v AEROT_P0_L101_GLL0 -d lat_0,34.0549 -d lon_0,241.7574 copy.nc
    #average Jan 10 files
    #ncra
    #remap
    #cdo remapbil,mygrid.txt epa-final.nc epa-regridded.nc
    #import la specific packages
    print('loading libraries...')
    import cartopy
    from matplotlib import colormaps
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm
    from matplotlib.colors import Normalize
    from matplotlib.colors import BoundaryNorm
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    import pandas as pd
    from netCDF4 import Dataset
    from tools import ToolBox
    import os
    T = ToolBox()

    #return BP_lo, BP_hi, I_lo, I_hi
    def epa_t6_map(pm25):
        if pm25 > 225.4:
            return (225.5, 325.4, 301, 500)
        elif 125.4 < pm25 <= 225.4:
            return (125.5, 225.4, 201, 300)
        elif 55.4 < pm25 <= 125.4:
            return (55.5, 125.4, 151, 200)
        elif 35.4 < pm25 <= 55.4:
            return (35.5, 55.4, 101, 150)
        elif 9 < pm25 <= 35.4:
            return (9.1, 35.4, 51, 100)
        else:
            return (0, 9, 0, 50)
    
    #return BP_lo, BP_hi, I_lo, I_hi
    def epa_t6_aqi_map(AQI):
        if AQI > 300:
            return (225.5, 325.4, 301, 500)
        elif 200 < AQI <= 300:
            return (125.5, 225.4, 201, 300)
        elif 150 < AQI <= 200:
            return (55.5, 125.4, 151, 200)
        elif 100 < AQI <= 150:
            return (35.5, 55.4, 101, 150)
        elif 50 < AQI <= 100:
            return (9.1, 35.4, 51, 100)
        else:
            return (0, 9, 0, 50)
    
    #return f, I_lo, BP_lo
    def calc_f(d):
        (BP_lo, BP_hi, I_lo, I_hi) = d
        return ((BP_hi - BP_lo)/(I_hi - I_lo), I_lo, BP_lo)
    
    #return concentration in units ug/m3
    def conc(AQI):
        BP_lo, BP_hi, I_lo, I_hi = epa_t6_aqi_map(AQI)
        return (AQI - I_lo) * (BP_hi - BP_lo) / (I_hi - I_lo) + BP_lo
    
    #get data
    '''f = Dataset(os.path.join(file))
    lats = f['lat_0'][:]
    lons = f['lon_0'][:]
    x = f['AEROT_P0_L101_GLL0'][:]

    #convert units
    print('convering...')
    x = np.vectorize(conc)(x)

    #save new netcdf
    print('saving...')
    out = Dataset(file.replace('temp', 'epa-final'), "w")
    time_d = out.createDimension("time", 744)
    lat = out.createDimension("lat", 267)
    lon = out.createDimension("lon", 267)
    time = out.createVariable("time","f8",("time"))
    main_v = out.createVariable("pm2.5","f8",("time", "lat", "lon"))
    latitudes = out.createVariable("lat","f8",("lat"))
    longitudes = out.createVariable("lon","f8",("lon"))
    #dates = out.createVariable("date","f8",("time"))
    #dates.units = "YYYYMMDD"
    #dates.cell_methods = "time: mean"
    #dates.long_name = "Date"
    latitudes.long_name = "Latitude"
    longitudes.long_name = "Longitude"
    time.long_name = "Time"
    main_v.long_name = "pm2.5"
    main_v.units = "ug/m^3"
    #BAs.history = "CEDS species: BC*1.0"
    #BAs.molecular_weight = 12
    #BAs.cell_methods = "time: mean"
    latitudes.units = "degrees_north"
    longitudes.units = "degrees_east"
    #time.units = "days since 1750-01-01 00:00:00"
    #time.calendar = "Gregorian"
    #time.cell_methods = "time: mean"
    latitudes[:] = lats
    longitudes[:] = lons
    time[:] = [i for i in range(744)]
    main_v[:] = x
    #oldemissions.close()
    out.close()
    f.close()'''

    print('ploting...')
    root = '/glade/derecho/scratch/nlbills/la-pm2.5/la-pm2.5'
    mask_root = '/glade/derecho/scratch/nlbills/ocean-land-masks'
    name_var_map = {'pm25_exp_sub.nc': 'var73', 'aqi-regrid.nc': 'AEROT_P0_L101_GLL0'}
    files = ('aqi-regrid.nc', 'pm25_exp_sub.nc')
    fig, ax = plt.subplots(1, len(files), dpi=300, subplot_kw={'projection': cartopy.crs.NearsidePerspective(central_latitude=34, central_longitude=-119)})

    for i in range(len(files)):
        #setup cartopy
        ax[i].set_extent((238.2, 243.8, 31.5, 36.9), cartopy.crs.PlateCarree())
        ax[i].add_feature(cartopy.feature.COASTLINE, edgecolor='grey')

        #get data
        f = Dataset(os.path.join(root, files[i]))
        x = f[name_var_map[files[i]]][:]
        f_mask = Dataset(os.path.join(mask_root, files[i].replace('.nc', '-mask.nc')))
        mask = f_mask['landseamask'][:]
        f_mask.close()

        if files[i] == 'aqi-regrid.nc':
            #Jan 8 0:00 to Jan 13 0:00
            start_t = 168
            end_t = 697
            lats = f['lat_0'][:]
            lons = f['lon_0'][:]
            vmax = 80
        elif files[i] == 'pm25_exp_sub.nc':
            #Jan 8 0:00 to Jan 13 0:00
            times = f['time'][:]
            start_t = T.nearest_search(times, (8-6) * 24 - 24)
            end_t = T.nearest_search(times, (13-6) * 24 - 24)
            lats = f['lat'][:]
            lons = f['lon'][:]
            vmax = 2
        to_plot = np.multiply(np.mean(x[start_t:end_t,:,:], axis=0), mask)
        to_plot *= np.power(10, 8) if files[i] == 'pm25_exp_sub.nc' else 1

        #setup color scale
        cmap = colormaps['viridis']
        # extract all colors from the map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
        # define the bins and normalize
        if files[i] == 'aqi-regrid.nc':
            bounds = [i for i in range(0, 100, 20)]
        elif files[i] == 'pm25_exp_sub.nc':
            bounds = [0, 0.5, 1.0, 1.5, 2.0]
        c_norm = BoundaryNorm(bounds, cmap.N)
        sm = ScalarMappable(cmap=cmap, norm=c_norm)

        #plot
        label_map = {'pm25_exp_sub.nc': 'PM2.5 * 10^8', 'aqi-regrid.nc': 'Air Quality Index (AQI)'}
        title_map = {'pm25_exp_sub.nc': 'Modeled PM2.5', 'aqi-regrid.nc': 'EPA Observed AQI'}
        ax[i].pcolormesh(lons, lats, to_plot, cmap=cmap, norm=c_norm, transform=cartopy.crs.PlateCarree())
        ax[i].set_title(title_map[files[i]])
        plt.colorbar(mappable=sm, label=label_map[files[i]], orientation="horizontal", ax=ax[i], extend='both')
    #plt.title('Mean Conditions from January 8th to January 13th')
    plt.savefig(os.path.join(os.getcwd(), 'epa-fig.png'), dpi=200)
    print('saved to ' + os.path.join(os.getcwd(), 'epa-fig.png'))

elif analysis == 'Seasonal PM2.5':
    #Get api credentials
    f = open('secrets.json')
    data = json.load(f)
    email = data['email']
    api_key = data['epa-key']
    f.close()

    #define scope
    east = [
        ["12", "073", "125", "0012", "0001", "Florida, Tallahassee"],
        ["13", "121", "033", "0032", "0001", "Georgia, Atlanta"],
        ["45", "079", "045", "0001", "0001", "South Carolina, Columbia"],
        ["37", "183", "029", "0017", "0001", "North Carolina, Raleigh"],
        ["51", "087", "159", "0014", "0001", "Virginia, Richmond"],
        ["24", "003", "001", "0014", "0001", "Maryland, Annapolis"],
        ["10", "001", "003", "0002", "0001", "Delaware, Dover"],
        ["34", "021", "033", "0005", "0001", "New Jersey, Trenton"],
        ["42", "043", "005", "0102", "0001", "Pennsylvania, Harrisburg"],
        ["36", "001", "003", "0005", "0001", "New York, Albany"],
        ["44", "007", "009", "0012", "0001", "Rhode Island, Providence"],
        ["09", "003", "015", "0013", "0001", "Connecticut, Hartford"],
        ["25", "025", "001", "0002", "0001", "Massachusetts, Boston"]
        ]

    west = [
        ["53", "067", "023", "0013", "0001", "Washington, Olympia"],
        ["41", "047", "037", "0040", "0001", "Oregon, Salem"],
        ["06", "067", "049", "0006", "0001", "California, Sacramento"]
    ]

    east_data = {"1995-2005": [0, 0], "2005-2015": [0, 0], "2015-2024": [0, 0]}
    west_data = {"1995-2005": [0, 0], "2005-2015": [0, 0], "2015-2024": [0, 0]}
    data_wrapper = ((west, west_data), (east, east_data))

    #request data
    def get_dates(year):
        year = str(year)
        return (year + "0601", year + "0831")
        #return (year + "0101", year + "0531")

    def construct_url(state, county, year, site):
        by, ey = get_dates(year)
        return "https://aqs.epa.gov/data/api/sampleData/bySite?email=" + email + "&key=" + api_key + "&param=88101&bdate=" + by + "&edate=" + ey + "&state=" + state + "&county=" + county + "&site=" + site

    for obj in data_wrapper:
        locations, data = obj
        for location in locations:
            print(location[-1])
            for year in range(1995, 2025):
                state, urban, rural, u_site, r_site, meta = location
                url = construct_url(state, urban, year, u_site)
                response = requests.get(url).json()
                d = response["Data"]
                for datum in d:
                    pm = datum["sample_measurement"]
                    if pm is not None and not math.isnan(pm) and (type(pm) == type(3) or type(pm) == type(1.1)):
                        if year >= 2015:
                            key = "2015-2024"
                        elif year < 2005:
                            key = "1995-2005"
                        else:
                            key = "2005-2015"
                        data[key][0] += pm
                        data[key][1] += 1
        for k in data.keys():
            data[k].append(data[k][0]/data[k][1])

    print(east_data)
    print(west_data)

    #units Micrograms PM2.5/cubic meter (LC)
    #output:
    #east {'1995-2005': [27245.200000000063, 1562, 17.442509603073024], '2005-2015': [84590.89999999997, 8255, 10.247231980617803], '2015-2024': [602319.7000000002, 68211, 8.830242922695756]}
    #west {'1995-2005': [2986.3999999999987, 463, 6.450107991360689], '2005-2015': [4089.2000000000025, 507, 8.065483234714009], '2015-2024': [56648.00000000004, 7366, 7.690469725767042]}
    #east change: 17.443, 10.247231980617803, 8.830242922695756
    #west change: 6.450107991360689, 8.065483234714009, 7.690469725767042
    #standards:   15.0, 12.0, 9.0 (1.8598, 1.4878, 1.1159)
    #2.163, 1.2705, 1.09481
    #0.7997, 1, 0.9535
    # 1 fire emoji = 8.0655 ug PM2.5/cubic meter/day (decadal average)