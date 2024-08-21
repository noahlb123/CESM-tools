from netCDF4 import Dataset
import platform
import numpy as np
import tools
import csv
import os
import pandas as pd
import sys

system = platform.system() #differentiate local and derecho env by sys platform
target_model = sys.argv[1] if sys.argv[1] == 'CMIP6' or 'CESM' else 'CMIP6'
target_v = sys.argv[2] if sys.argv[2] in ['wetbc', 'sootsn', 'loadbc'] else 'wetbc'
if not target_v in ['wetbc', 'sootsn', 'loadbc']:
    raise Exception('format command as: python3 cmip.py <MODEL> <VAR>')
T = tools.ToolBox()

#setup some vars
model_data_map = {'pi': {}, 'pd': {}}
coord_min_maxes = (90.0, -90.0, 358.75, 0.0) #(np.max(lats), np.min(lats), np.max(lons), np.min(lons))
models = [1, 35] if system == "Darwin" else range(1, 36)
sheets = {'pi': 1850.5, 'pd': 1980.5}
wet_models = {}
dry_models = {}
lat_ant_inds = {}
lon_ant_inds = {}
prefix_map = {'sootsn': 'LImon_', 'wetbc': 'AERmon_', 'loadbc': 'Eday_'}
prefix = prefix_map[target_v]
year_mods = pd.DataFrame(columns=['pi', 'pd'])
fileuse_index = pd.DataFrame(columns=['wet file', 'dry file'])

def in_antartica(lat, lon):
    return T.within_patch(lat, lon, (-180, -60, 360, -30), 'Antartica')

def add_ant_ind(lat_i, lon_i, lat, lon):
    if in_antartica(lat, lon):
        if lat_i not in lat_ant_inds:
            lat_ant_inds[lat_i] = 1
        else:
            lat_ant_inds[lat_i] += 1
        if lon_i not in lon_ant_inds:
            lon_ant_inds[lon_i] = 1
        else:
            lon_ant_inds[lon_i] += 1
        return 1
    return 0

#file name to model name
def filename2modelname(filename):
    dist = 4 if target_v == 'loadbc' else 2
    model_name = filename[filename.index(prefix) + len(prefix): filename.index('_historical')]
    start_year = filename[filename.rfind("_") + 1:filename.rfind("-") - dist]
    end_year = filename[filename.rfind("-") + 1:filename.rfind(".") - dist]
    return model_name, int(start_year), int(end_year)

#check lat lons are all in same format
def fix_format(lats, lons):
    changes = [0, 0]
    coords = (lats, lons)
    for i in range(2):
        max_diff = coord_min_maxes[0 + i * 2] - np.max(coords[i])
        min_diff = coord_min_maxes[1 + i * 2] - np.min(coords[i])
        if np.abs(max_diff) > 5 or np.abs(min_diff) > 5:
            changes[i] = np.mean((max_diff, min_diff))
    return changes

def is_within(s_year, e_year, y):
    return s_year - 10 <= y <= e_year + 10

#get paths
data_path_map = {'sootsn': 'cmip6-snow-dep', 'wetbc': os.path.join('cmip6-snow-dep', 'all'), 'loadbc': 'cmip-atmos'}
data_path = os.path.join('/', 'glade', 'derecho', 'scratch', 'nlbills', data_path_map[target_v])
index_path = 'data/standardized-ice-cores/index.csv'
dupe_path = 'data/standardized-ice-cores/index-dup-cores.csv'
ice_coords = T.get_ice_coords(index_path, dupe_path)
first_core = list(ice_coords.keys())[0]

#divide files into wet and dry
for filename in os.listdir(data_path): #from https://esgf-node.ipsl.upmc.fr/search/cmip6-ipsl/
    if '.nc' in filename and prefix in filename:
        file_path = os.path.join(data_path, filename)
        model_name, start_year, end_year = filename2modelname(filename)
        if (target_model == 'CESM' and 'CESM' in model_name) or target_model != 'CESM':
            if 'wetbc' in filename or target_v != 'wetbc':
                wet_models[filename] = (model_name, file_path, start_year, end_year)
            elif 'drybc' in filename:
                dry_models[filename] = (model_name, file_path, start_year, end_year)

#divide into pi and pd
for wet_name, obj in wet_models.items():
    dry_name = wet_name.replace('wetbc', 'drybc')
    model_name, file_path, start_year, end_year = obj
    if dry_name in dry_models.keys() or target_v != 'wetbc':
        model_dict = {'wet': wet_models[wet_name][1:4]} if target_v != 'wetbc' else {'wet': wet_models[wet_name][1:4], 'dry': dry_models[dry_name][1:4]}
        if np.abs(start_year - sheets['pi']) < 10 or start_year < sheets['pi']:
            if model_name not in model_data_map['pi']:
                model_data_map['pi'][model_name] = [model_dict]
            else: 
                model_data_map['pi'][model_name].append(model_dict)
        if np.abs(end_year - sheets['pd']) < 10 or end_year > sheets['pd']:
            if model_name not in model_data_map['pd']:
                model_data_map['pd'][model_name] = [model_dict]
            else: 
                model_data_map['pd'][model_name].append(model_dict)

#get output vars
window = 10 * 365
pandas_i = 0
fileuse_i = 0
for era, year in sheets.items():
    csv_dict = []
    csv_coords = []
    csv_years = []
    print(era)
    length = len(model_data_map[era].items())
    i = 1
    for model_name, pairs in model_data_map[era].items():
        if 'MIROC6' in model_name:
            continue
        print(model_name, i, '/', length)
        print('dict setup')
        row = {"model": model_name}
        row_year = {"model": model_name}
        row_coord = {"model": model_name}
        for wet_dry in pairs:
            wet_pair = wet_dry['wet']
            model_path, start_year, end_year = wet_pair
            #formatted as time 1980, lat 192, lon 288
            print('load netcdf')
            #load netcdf file
            f_wet = Dataset(model_path)
            print('format arrays')
            #format arrays
            lats = f_wet["lat"][:]
            lons = f_wet["lon"][:]
            changes = fix_format(lats, lons)
            print('small funcs')
            lats = lats + changes[0]
            lons = lons + changes[1]
            times = f_wet["time"]
            total_v = 0
            unit_year = int(times.units[11:15])
            year_modifier = 1850
            if unit_year != start_year and unit_year != 1:
                year_modifier = unit_year
            elif unit_year == 1:
                year_modifier = start_year
            year_mods.at[model_name, era] = (year - year_modifier) * 365
            pandas_i += 1
            if target_v == 'wetbc':
                dry_pair = wet_dry['dry']
                f_dry = Dataset(dry_pair[0])
                drybc = f_dry['drybc'][:]
            print('format wetbc')
            wetbc = f_wet[target_v][:]
            print('get avereges')
            #get vars
            for core_name in ice_coords.keys():#[first_core]:
                y, x = ice_coords[core_name]
                lat = T.nearest_search(lats, y)
                lon = T.nearest_search(lons, x + 180)
                wet_a, wet_y_out = T.get_avgs(times[:], wetbc[:,lat,lon], (year - year_modifier) * 365, [window])
                if target_v == 'wetbc':
                    dry_a, dry_y_out = T.get_avgs(times[:], drybc[:,lat,lon], (year - year_modifier) * 365, [window])
                    total_v += np.abs(wet_a[window]) + np.abs(dry_a[window])
                else:
                    total_v += wet_a[window]
                if core_name in row:
                    row[core_name] += total_v
                else:
                    row[core_name] = total_v
                row_year[core_name] = wet_y_out / 365
                row_coord[core_name] = str(lat) + ',' + str(lon)
            #memory management
            print('#memory management')
            f_wet.close()
            del wetbc, lats, lons
            fileuse_index.loc[fileuse_i] = [wet_dry['wet'][0], wet_dry['dry'][0]] if target_v == 'wetbc' else [wet_dry['wet'][0], '']
            fileuse_i += 1
            if target_v == 'wetbc':
                f_dry.close()
        row['n ensemble members'] = len(pairs)
        row['window'] = window
        for core_name, value in row.items():
            if '.csv' in core_name:
                row[core_name] = value / len(pairs)
                if core_name in row_year:
                    #row_year[core_name] = value / len(pairs)
                    pass
        csv_dict.append(row)
        csv_years.append(row_year)
        csv_coords.append(row_coord)
        i += 1

    #save to csv
    for data_type, csv_inst in {'main': csv_dict, 'year': csv_years, 'coords': csv_coords}.items():
        fields = ["model", 'n ensemble members', 'window'] if data_type == 'main' else ["model"]
        [fields.append(name) for name in ice_coords.keys()]
        filename = data_type + '-' + era + '.csv' if data_type != 'main' else era + '.csv'
        subfolder = target_model if target_model != 'CESM' else target_model + '-' + target_v.replace('wetbc', 'wetdry')
        subfolder = subfolder.lower()
        write_path = os.path.join(os.getcwd(), 'data', 'model-ice-depo', subfolder, filename)
        with open(write_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            writer.writerows(csv_inst)
year_mods.to_csv(os.path.join(os.getcwd(), 'data', 'model-ice-depo', subfolder, 'year-mods.csv'))
fileuse_index.to_csv(os.path.join(os.getcwd(), 'data', 'model-ice-depo', subfolder, 'fileuse-index.csv'))

#loadbc

print("done.")