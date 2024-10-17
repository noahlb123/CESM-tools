import os
import sys
import platform
from tools import ToolBox
from netCDF4 import Dataset
import numpy as np
T = ToolBox()

#DRY MUST COME FIRST IF USING WET DRY PAIRS
bad_boy_mode = True #should the output be written in the cwd
main_dict = {}
if len(sys.argv) < 2:
    raise Exception('2 command line arguments required: <varaible name common in all desired files> <root directory> <OPTINOAL: cesm mode (cesm or *)> <OPTINOAL: do nco (True, False)>')
target_v = sys.argv[1]
root = sys.argv[2]
cesm_mode = sys.argv[3].lower() == 'cesm' if len(sys.argv) >= 4 else False
do_nco = sys.argv[4].lower() == 'true' if len(sys.argv) >= 5 else True
smallest_grid = T.smallest_grid(root, lambda s, p: ('.nc' in s) and (p in s), target_v)
smallest = Dataset(root + '/' + smallest_grid)
smallest_lat_lon_shape = [smallest.variables['lat'].shape[0], smallest.variables['lon'].shape[0]]
smallest.close()
prefix_map = {'sootsn': 'LImon_', 'drybc': 'AERmon_', 'loadbc': 'Eday_'}
prefix = prefix_map[target_v]
system = platform.system() #differentiate local and derecho env by sys platform
partners = {}
files = os.listdir(root)
bads = set([])
to_eval = 'cd ' + root + ' && '

def get_years(filename):
    dist = 4 if target_v == 'loadbc' else 2
    years = filename[filename.rfind('_') + 1:filename.rfind(".")].split('-')
    years = [int(year[0:4]) for year in years]
    return years

def contains(years, target_year):
    s_year, e_year = years
    if s_year <= target_year <= e_year:
        return True
    return False

def get_model_name(filename):
    return filename[filename.index(prefix) + len(prefix): filename.index('_historical')]

def base_model(model):
        model = model.replace('.nc', '')
        if dir != 'loadbc':
            if 'MIROC6' in model:
                base_model = 'MIROC'
            if '-' in model:
                base_model = model[0:model.index('-')]
            elif '_' in model:
                base_model = model[0:model.index('_')]
            else:
                base_model = model
        else:
            base_model = model
        return base_model

def evaluate(s):
    l = len(s)
    if s[l - 4:l] == ' && ':
        s = s[0:l - 4]
    os.system(s)
    return 'cd ' + root + ' && '

if do_nco:
    #find start and end files
    for filename in files:
        if target_v in filename and (not cesm_mode or T.any_substrings_in_string(['CanESM', 'CESM'], filename)):
            if (target_v == 'drybc'):
                partner_name = filename.replace('wetbc', 'drybc') if 'wetbc' in filename else filename.replace('drybc', 'wetbc')
            if target_v != 'drybc' or os.path.isfile(os.path.join(root, partner_name)):
                partners = [filename, partner_name] if target_v == 'drybc' else [filename]
                for f_name in partners:
                    model_name = get_model_name(f_name)
                    years = get_years(f_name)
                    if (target_v == 'drybc'):
                        model_name += '_b' if target_v == 'drybc' and f_name == partner_name else '_a'
                    if contains(years, 1850) and contains(years, 1980):
                        if model_name not in main_dict:
                            main_dict[model_name] = {'file': f_name}
                        else:
                            main_dict[model_name]['file'] = f_name
    
    #comands to wetbc files with drybc (subtraction)
    if target_v == 'drybc':
        to_eval += 'echo "combining files with partners..." && '
        valid_models = list(set(main_dict.keys()).difference(bads))
        valid_er_models = set()
        for model_name in valid_models:
            if '_b' in model_name:
                continue
            partner = model_name.replace('_a', '_b') + '.nc'
            new_name = model_name.replace('_a', '').replace('.nc', '')
            #get sign of wetbc
            f = Dataset(root + '/' + partner)
            wet_arr = f['wetbc'][:]
            f.close()
            if np.min(wet_arr) >= 0 and not np.max(wet_arr) <= 0:
                operation = 'add'
            elif np.max(wet_arr) <= 0 and not np.min(wet_arr) >= 0:
                operation = 'sub'
            if np.max(wet_arr) >= 0 and np.min(wet_arr) <= 0:
                raise Exception('this wetbc file contains both negative (' + np.min(wet_arr) + ') and positive values (' + np.max(wet_arr) + '): ' + partner)
            to_eval += 'ncrename -h -O -v wetbc,drybc ' + partner + ' && '
            to_eval += 'ncbo --op_typ=' + operation + ' ' + model_name + ' ' + partner + ' ' + new_name + '.nc -O && '
            valid_er_models.add(new_name.replace('_pi', '').replace('_pd', ''))
        to_eval = evaluate(to_eval)
    else:
        valid_er_models = list(set(main_dict.keys()).difference(bads))

    #filter out bad models
    to_eval += 'echo "extracting timeslices..." && '
    for model_name, d in main_dict.items():
        if d['file'] != None:
            filename = d['file']
            try:
                f = Dataset(root + '/' + filename)
            except OSError:
                #print('wrong format:', model_name)
                bads.add(model_name)
                continue
            time_var = f.variables['time']
            assert 'days since' in time_var.units
            f.close()
            to_eval += 'cp -f ' + filename + ' ' + model_name + '.nc && '
        else:
            #print('doesnt have start and end:', model_name)
            bads.add(model_name)

    valid_er_models = list(set(main_dict.keys()).difference(bads))
    filenames = [model_name + '.nc' for model_name in valid_er_models]
    to_eval = evaluate(to_eval)

    #commands to remove time_bnds variable
    to_eval += 'echo "removing time_bnds variable..." && '
    for file_name in filenames:
        #testing if this line breaks everything, if these comments are here it does
        #to_eval += 'ncks -C -O -x -v time_bnds ' + file_name + ' ' + file_name + ' -O && '
        pass

    to_eval = evaluate(to_eval)

    #commands to regrid all models
    to_eval += 'echo "regriding..." && '
    for i in range(len(filenames)):
        file_name = filenames[i]
        f = Dataset(root + '/' + file_name)
        to_eval += "ncap2 -O -s '" + target_v + "=double(" + target_v + ");' " + file_name + ' ' + file_name + ' && '
        if f.variables['lat'].shape[0] > smallest_lat_lon_shape[0] or f.variables['lon'].shape[0] > smallest_lat_lon_shape[1]:
            to_eval += 'ncremap -d ' + smallest_grid + ' ' + file_name + ' ' + file_name.replace('.nc', '_re.nc') + ' && '
            filenames[i] = file_name.replace('.nc', '_re.nc')
        f.close()

    to_eval = evaluate(to_eval)

    #bin models
    bins = {}
    for file in filenames:
        base = base_model(file)
        if base in bins:
            bins[base].append(file)
        else:
            bins[base] = [file]

    #average bases
    to_eval += 'echo "binning..." && '
    bases = []
    for base, files in bins.items():
        to_eval += 'cdo -O ensmean ' + ' '.join(files) + ' ' + base + '.nc && '
        bases.append(base + '.nc')

    to_eval = evaluate(to_eval)

    #comand to average files
    to_eval += 'echo "averaging..." && '
    to_eval += 'cdo -O ensmean ' + ' '.join(bases) + ' output.nc && '
    to_eval += 'echo "nco workflow done!"'
    evaluate(to_eval)

#python
print('begin python workflow...')
import pandas as pd #this has to be here otherwise var "pd" is overwritten
import matplotlib.pyplot as plt
index_path = 'data/standardized-ice-cores/index.csv'
dupe_path = 'data/standardized-ice-cores/index-dup-cores.csv'
ice_coords = T.get_ice_coords(index_path, dupe_path)
s_lat, s_lon = ice_coords['mcconnell-2007-1.csv']
x = [365 * (i + 0.5) for i in range(1850, 1981)]
f = Dataset(os.path.join(root, 'output.nc'))
years = f['time'][:]
lats, lons = T.adjust_lat_lon_format(f['lat'][:], f['lon'][:])
lat = T.nearest_search(lats, s_lat)
lon = T.nearest_search(lons, s_lon)
assert T.within(lats[lat], s_lat, 5) and T.within(lons[lon], s_lon, 5)
variable = f[target_v][:,lat,lon]
timeseries = np.interp(x, years, variable)

#plot
plt.plot(x, timeseries)

#save
var2subfolder = {'drybc': 'cmip6', 'loadbc': 'loadbc', 'sootsn': 'cesm-sootsn'}
subfolder = var2subfolder[target_v]
if subfolder == 'cmip' and cesm_mode:
    subfolder = 'cesm-wetdry'
output_path = os.path.join(os.getcwd(), 'data', 'model-ice-depo', subfolder, 'nco.png') if not bad_boy_mode else os.path.join(os.getcwd(), target_v + '.png')
plt.savefig(output_path)
print('saved to', output_path)