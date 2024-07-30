from netCDF4 import Dataset
import pandas as pd
import numpy as np
import platform
import tools
import csv
import os

mode = "pd"

system = platform.system() #differentiate local and derecho env by sys platform
T = tools.ToolBox()

#arrays to get adjacent gridcells
cross_cells = [
    [0, 1],
    [0, -1],
    [1, 0],
    [-1, 0]
]
corner_cells = [
    [1, 1],
    [-1, -1],
    [1, -1],
    [-1, 1]
]
adj_8 = cross_cells + corner_cells

def mass_mix2conc(bc_mixing_ratio, TREHT, p):
    return bc_mixing_ratio * p / 287.053 / TREHT

#path of lens files
def modelN2fnames(model_n, year, h):
    n = str(model_n).zfill(2)
    h0year = str(int(year[0:4]) - 1) + "-12"
    def get_path(y, h):
        return os.path.join(
            "b.e11.B20TRC5CNBDRD.f09_g16.0" + n + ".rest." + year[0:-1],
            year,
            "b.e11.B20TRC5CNBDRD.f09_g16.0" + n + ".cam.h" + str(h) + "." + y + ".nc")
    return [get_path(h0year, h)] if h == 0 else [get_path(year, h)]

#setup some vars
lvls = [30]
h = 1 if mode == "pd" else 0
models = [1, 35] if system == "Darwin" else range(1, 36)
model_var_map = {}
model_path_map = {}
model_year_map = {}
viable_models = []
#the only present var in any run is bc_a1_SRF
vars = ['bc_a4_SRF', 'bc_a1_SRF'] #these are suspended bc
vars = ['bc_a4_SRF', 'bc_a1_SRF', 'SFbc_a4',
    'SFbc_a1',
    'bc_a4DDF',
    'bc_a4SFWET',
    'bc_a1DDF',
    'bc_a1SFWET',
    'bc_c4DDF',
    'bc_c4SFWET',
    'bc_c1DDF',
    'bc_c1SFWET',
    'bc_c1']#these are depositions

#get viable models
for model in models:
    if mode == 'pd':
        years = [1980 + 5 * (x - 2) for x in range(5)] if model != 34 else (1977, 1982, 1987, 1972, 1970)
    else:
        years = (1865, 1855, 1860)
    for year in years:
        #get path
        year_s = str(year) + "-01-01-00000"
        working_path = os.path.dirname(os.path.abspath(__file__))
        file_paths = modelN2fnames(model, year_s, h)
        for file_path in file_paths:
            if system == "Darwin":
                path = os.path.join(working_path, "data", "model-ice-depo", "lens", file_path)
            else:
                path = os.path.join(working_path, file_path)
            if os.path.exists(path):
                #check if has bc vars
                f = Dataset(path)
                all_vars = f.variables
                model_index = str(model) + "-h" + file_path[file_path.index("cam.") + len("cam.") + 1] + year_s
                for v in vars:
                    if v in all_vars:
                        if model_index not in viable_models:
                            viable_models.append(model_index)
                        if model_index not in model_path_map.keys():
                            model_path_map[model_index] = path
                        if model_index not in model_year_map.keys():
                            model_year_map[model_index] = year
                        if model_index in model_var_map:
                            model_var_map[model_index].append(v)
                        else:
                            model_var_map[model_index] = [v]
                f.close()

#get ice core lat, lon
index_path = 'data/standardized-ice-cores/index.csv' if system == "Darwin" else "index.csv"
dupe_path = 'data/standardized-ice-cores/index-dup-cores.csv' if system == "Darwin" else "index-dup-cores.csv"
ice_coords = T.get_ice_coords(index_path, dupe_path)

#get bc depo
for lv in lvls:
    print(lv)
    for shape in [[[0, 0]]]:
        print('shape len:', len(shape))
        csv_dict = []
        for model in viable_models:
            print(model)
            row = {"model number": model}
            f = Dataset(model_path_map[model])
            #1 192 288 (t, lat, lon)
            bc = f[model_var_map[model][0]][:]
            hyai = f["hyai"][:]
            hybi = f["hybi"][:]
            P0 = f["P0"][:]
            PSL = f["PSL"][:]
            TREFHT = f["TREFHT"][:]
            lats = f["lat"][:]
            lons = f["lon"][:]
            row["BC_vars"] = ",".join(model_var_map[model])
            row["year"] = model_year_map[model]
            for name in ice_coords.keys():#[list(ice_coords.keys())[0]]
                y, x = ice_coords[name]
                lat = T.nearest_search(lats, y)
                lon = T.nearest_search(lons, x + 180)
                total_conc = 0
                for offsets in shape:
                    lat_o, lon_o = np.add((lat, lon), offsets)
                    #black carbon
                    total_bc = 0
                    for v in model_var_map[model]:
                        total_bc += bc[0][lat_o][lon_o]
                    #pressure
                    pressure = 0
                    for level in range(lv):
                        delta_p = PSL[0][lat_o][lon_o] * (hybi[level + 1] - hybi[level]) + P0 * (hyai[level + 1] - hyai[level])
                        pressure += delta_p / 9.81
                    total_conc += mass_mix2conc(total_bc, TREFHT[0][lat_o][lon_o], pressure)
                row[name] = total_conc / len(shape)
            csv_dict.append(row)
            f.close()

        #save to csv
        fields = ["model number", "BC_vars", "year"]
        [fields.append(name) for name in ice_coords.keys()]
        with open(mode + "-lv" + str(lv) + '-s' + str(len(shape)) + ".csv", 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            writer.writerows(csv_dict)

print("done.")