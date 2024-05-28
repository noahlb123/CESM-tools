from netCDF4 import Dataset
import platform
import numpy as np
import tools
import csv
import os

system = platform.system() #differentiate local and derecho env by sys platform
T = tools.ToolBox()

#setup some vars
model_data_map = {'pi': [], 'pd': []}
path_name_map = {}
name_path_map = {}
coord_min_maxes = (90.0, 358.75, 0.0, -90.0)
models = [1, 35] if system == "Darwin" else range(1, 36)
sheets = {'pi': 1850.5, 'pd': 1980.5}
nc_vars = ['wetbc', 'drybc']

#file name to model name
def filename2modelname(filename):
    model_name = filename[filename.index('AERmon_') + 7: filename.index('_historical')]
    start_year = filename[filename.rfind("_") + 1:filename.rfind("-") - 2]
    end_year = filename[filename.rfind("-") + 1:filename.rfind(".") - 2]
    return model_name, int(start_year), int(end_year)

#check lat lons are all in same format
def check_format(lats, lons, time):
    new_coords = (np.max(lats), np.max(lons), np.min(lons), np.min(lats))
    for i in range(len(coord_min_maxes)):
        assert np.abs(new_coords[i] - coord_min_maxes[i]) < 5
    assert time.units == 'days since 0001-01-01 00:00:00'

#check if pi/pd year is within model time bounds
def is_within(s_year, e_year, y):
    return s_year - 10 <= y <= e_year + 10

def mass_mix2conc(bc_mixing_ratio, TREHT, p):
    return bc_mixing_ratio * p / 287.053 / TREHT

#get paths
index_path = 'data/standardized-ice-cores/index.csv' if system == "Darwin" else "index.csv"
dupe_path = 'data/standardized-ice-cores/index-dup-cores.csv' if system == "Darwin" else "index-dup-cores.csv"
ice_coords = T.get_ice_coords(index_path, dupe_path)
first_core = list(ice_coords.keys())[0]
data_path = os.path.join(os.getcwd(), 'data', 'cmip6') if system == "Darwin" else os.getcwd()

#plan:
#take only files with wetbc and drybc

#get data from cmip6 files
for filename in os.listdir(data_path): #from https://esgf-node.ipsl.upmc.fr/search/cmip6-ipsl/
    if filename not in ('tools.py', 'index-dup-cores.csv', 'cesm', 'sootsn', '__pycache__', 'cmip.py', 'index.csv', 'ip.py', 'pi.csv', 'pd.csv', 'pi-cmip.csv', 'pd-cmip.csv', '.DS_Store') and 'wget' not in filename and ".csv" not in filename:
        file_path = os.path.join(data_path, filename)
        model_name, start_year, end_year = filename2modelname(filename)
        if model_name not in name_path_map:
            name_path_map[model_name] = [filename]
        else:
            name_path_map[model_name].append(filename)
        path_name_map[file_path] = model_name
        for era in sheets.keys():
            if is_within(start_year, end_year, sheets[era]):
                model_data_map[era].append(file_path)

#get sootsn
'''for era, year in sheets.items():
    csv_dict = []
    #print(era)
    for model_name, pairs in model_data_map[era].items():
        print(model_name)
        row = {"model": model_name}
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

    #save to csv
    fields = ["model", 'n ensemble members', 'window']
    [fields.append(name) for name in ice_coords.keys()]
    with open(era + ".csv", 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_dict)'''

#get bc
csv_dict = []
for model_name, model_list in name_path_map.items():
    for model_path in model_list:
        w = Dataset(os.path.join(data_path, model_path))['wetbc'][:]
        print(len(w), len(w[0]), len(w[0][0]))
        '''row = {"model name": model_name}
        f = Dataset(model_path)
        #1 192 288 (t, lat, lon)
        wetbc = f['wetbc'][:]
        drybc = f['drybc'][:]
        hyai = f["hyai"][:]
        hybi = f["hybi"][:]
        P0 = f["P0"][:]
        PSL = f["PSL"][:]
        TREFHT = f["TREFHT"][:]
        lats = f["lat"][:]
        lons = f["lon"][:]
        for name in ice_coords.keys():
            y, x = ice_coords[name]
            lat = T.nearest_search(lats, y)
            lon = T.nearest_search(lons, x + 180)
            total_bc = 0
            #black carbon
            total_bc += bc[0][lat][lon]
            #pressure
            pressure = 0
            for level in range(lv):
                delta_p = PSL[0][lat][lon] * (hybi[level + 1] - hybi[level]) + P0 * (hyai[level + 1] - hyai[level])
                pressure += delta_p / 9.81
            row[name] = mass_mix2conc(total_bc, TREFHT[0][lat][lon], pressure)
        csv_dict.append(row)
        f.close()

#save to csv
fields = ["model name"]
[fields.append(name) for name in ice_coords.keys()]
with open(mode + "-lv" + str(lv) + ".csv", 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    writer.writerows(csv_dict)'''

print("done.")