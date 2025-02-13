import json
import sys
import requests
import math

#set analsis mode
if len(sys.argv) < 2 or sys.argv[1] not in ["Seasonal PM2.5", "2024 LA Wildfires"]:
    raise Exception('1 command line argument required: <anaylysis type ("Seasonal PM2.5" or "2024 LA Wildfires")> ex. python3 epa-aqi-api.py "2024 LA Wildfires"')
analysis = sys.argv[1]


if analysis == '2024 LA Wildfires':
    #crop by lat, lon
    #ncks -O -d lat_0,31.,37. -d lon_0,238.,244. copy.nc copy.nc
    #get data from specific lat,lon
    #ncks --no_nm_prn -H -C -v AEROT_P0_L101_GLL0 -d lat_0,34.0549 -d lon_0,241.7574 copy.nc
    #import la specific packages
    import numpy as np
    import pandas as pd
    from netCDF4 import Dataset
    from tools import ToolBox
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

    #return f, I_lo, BP_lo
    def epa_t6_factors(pm25):
        if pm25 > 225.4:
            return (0.18, 0, 0)
        elif 125.4 < pm25 <= 225.4:
            return (0.536734693877551, 51, 9.1)
        elif 55.4 < pm25 <= 125.4:
            return (0.4061224489795918, 101, 35.5)
        elif 35.4 < pm25 <= 55.4:
            return (1.426530612244898, 151, 55.5)
        elif 9 < pm25 <= 35.4:
            return (1.009090909090909, 201, 125.5)
        else:
            return (0.5020100502512561, 301, 225.5)
    
    #return f, I_lo, BP_lo
    def calc_f(d):
        (BP_lo, BP_hi, I_lo, I_hi) = d
        return ((BP_hi - BP_lo)/(I_hi - I_lo), I_lo, BP_lo)
    
    for i in [1, 10, 50, 100, 200, 500]:
        print(calc_f(epa_t6_map(i)))


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