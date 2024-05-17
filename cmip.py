from netCDF4 import Dataset
import platform
import numpy as np
import tools
import csv
import os

system = platform.system() #differentiate local and derecho env by sys platform
T = tools.ToolBox()

#setup some vars
model_data_map = {'pi': {}, 'pd': {}}
coord_min_maxes = (90.0, 358.75, 0.0, -90.0)
models = [1, 35] if system == "Darwin" else range(1, 36)
sheets = {'pi': 1850.5, 'pd': 1950.5}

#file name to model name
def filename2modelname(filename):
    model_name = filename.split('_historical')[0].replace('sootsn_LImon_', '')
    start_year = filename[filename.rfind("_") + 1:filename.rfind("-") - 2]
    end_year = filename[filename.rfind("-") + 1:filename.rfind(".") - 2]
    return model_name, int(start_year), int(end_year)

#check lat lons are all in same format
def check_format(lats, lons, time):
    new_coords = (np.max(lats), np.max(lons), np.min(lons), np.min(lats))
    for i in range(len(coord_min_maxes)):
        assert np.abs(new_coords[i] - coord_min_maxes[i]) < 5
    assert time.units == 'days since 0001-01-01 00:00:00'

#get paths
index_path = 'data/standardized-ice-cores/index.csv' if system == "Darwin" else "index.csv"
dupe_path = 'data/standardized-ice-cores/index-dup-cores.csv' if system == "Darwin" else "index-dup-cores.csv"
ice_coords = T.get_ice_coords(index_path, dupe_path)
first_core = list(ice_coords.keys())[0]
data_path = os.path.join(os.getcwd(), 'data', 'cmip6') if system == "Darwin" else os.getcwd()

#get data from cmip6 files
for filename in os.listdir(data_path):
    if filename not in ('tools.py', 'index-dup-cores.csv', '__pycache__', 'cmip.py', 'index.csv', 'ip.py', 'pi.csv', 'pd.csv', '.DS_Store') and 'wget' not in filename:
        file_path = os.path.join(data_path, filename)
        model_name, start_year, end_year = filename2modelname(filename)
        k = None
        if np.abs(start_year - sheets['pi']) < 10:
            if model_name not in model_data_map['pi']:
                model_data_map['pi'][model_name] = [(file_path, start_year, end_year)]
            else: 
                model_data_map['pi'][model_name].append((file_path, start_year, end_year))
        if np.abs(end_year - sheets['pd']) < 10:
            if model_name not in model_data_map['pd']:
                model_data_map['pd'][model_name] = [(file_path, start_year, end_year)]
            else: 
                model_data_map['pd'][model_name].append((file_path, start_year, end_year))

#get sootsn
for era, year in sheets.items():
    csv_dict = []
    #print(era)
    for model_name, pairs in model_data_map[era].items():
        print(model_name)
        row = {"model": model_name}
        try:
            for core_name in ice_coords.keys():#[first_core]:
                #print(core_name)
                y, x = ice_coords[core_name]
                total_sootsn = 0
                for model_path, start_year, end_year in pairs:
                    #formatted as time 1980, lat 192, lon 288
                    f = Dataset(model_path)
                    sootsn = f['sootsn'][:]
                    lats = f["lat"][:]
                    lons = f["lon"][:]
                    times = f["time"]
                    lat = T.nearest_search(lats, y)
                    lon = T.nearest_search(lons, x + 180)
                    window = 5 * 365
                    a, y_out = T.get_avgs(times[:], sootsn[:,lat,lon], year * 365, [window])
                    if core_name == first_core:
                        check_format(lats, lons, times)
                        #print(y_out/365)
                    total_sootsn += a[window]
                    f.close()
                row['n ensemble members'] = len(pairs)
                row['window'] = window
                row[core_name] = total_sootsn / len(pairs)
            csv_dict.append(row)
        except:
            continue

    #save to csv
    fields = ["model", 'n ensemble members', 'window']
    [fields.append(name) for name in ice_coords.keys()]
    with open(era + ".csv", 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_dict)

print("done.")