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
    raise Exception('3 command line arguments required: <varaible name common in all desired files> <root directory> <OPTINOAL: cesm mode (cesm or *)> <OPTINOAL: [1850 avg window],[1980 avg window]>')
target_v = sys.argv[1]
root = sys.argv[2]
cesm_mode = sys.argv[3].lower() == 'cesm' if len(sys.argv) >= 4 else False
avg_window = 25 #sys.argv[4] if len(sys.argv) >= 5 else 10
#avg_window = [int(x) for x in avg_window.split(',')] if ',' in avg_window else [int(avg_window), int(avg_window)]
smallest_grid = T.smallest_grid(root, lambda s, p: ('.nc' in s) and (p in s), target_v)
prefix_map = {'sootsn': 'LImon_', 'drybc': 'AERmon_', 'loadbc': 'Eday_', 'mmrbc': 'AERmon_'}
prefix = prefix_map[target_v]
system = platform.system() #differentiate local and derecho env by sys platform
partners = {}
og_new_name_map = {}
files = os.listdir(root)
bads = set([])
to_eval = 'cd ' + root + ' && '

def get_years(filename):
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
        model = model.replace('_re', '')
        if dir != 'loadbc':
            if 'MIROC6' in model:
                base_model = 'MIROC'
            if '-' in model:
                base_model = model[0:model.index('-')]
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

#find start and end files
for filename in files:
    if target_v in filename and (not cesm_mode or T.any_substrings_in_string(['CanESM', 'CESM'], filename)) and filename != target_v + '.nc':
        if (target_v == 'drybc'):
            partner_name = filename.replace('wetbc', 'drybc') if 'wetbc' in filename else filename.replace('drybc', 'wetbc')
        if target_v != 'drybc' or os.path.isfile(os.path.join(root, partner_name)):
            partners = [filename, partner_name] if target_v == 'drybc' else [filename]
            for f_name in partners:
                model_name = get_model_name(f_name)
                years = get_years(f_name)
                if (target_v == 'drybc'):
                    model_name += '_b' if target_v == 'drybc' and f_name == partner_name else '_a'
                if contains(years, 1850):
                    if model_name not in main_dict:
                        main_dict[model_name] = {'s_file': f_name, 'e_file': None, 's_year': years[0]}
                    else:
                        main_dict[model_name]['s_file'] = f_name
                        main_dict[model_name]['s_year'] = years[0]
                if contains(years, 1980):
                    if model_name not in main_dict:
                        main_dict[model_name] = {'e_file': f_name, 's_file': None, 'e_year': years[1]}
                    else:
                        main_dict[model_name]['e_file'] = f_name
                        main_dict[model_name]['e_year'] = years[1]

#commands to extract file timeslices and decadally average
to_eval += 'echo "extracting timeslices..." && '
for model_name, d in main_dict.items():
    if d['s_file'] != None and d['e_file'] != None:
        for year in (1850, 1980):
            #print(model_name, year, filename)
            file_index = 's_file' if year == 1850 else 'e_file'
            year_index = 's_year' if year == 1850 else 'e_year'
            file_suffix = '_pi' if year == 1850 else '_pd'
            filename = d[file_index]
            try:
                f = Dataset(root + '/' + filename)
            except OSError:
                print('wrong format:', model_name)
                bads.add(model_name)
                continue
            time_var = f.variables['time']
            times = f['time'][:]
            if np.max(times) >= 365 * 1850:
                times = np.divide(times, 365)
            assert 'days since' in time_var.units
            for i in times:
                print(i)
            i_start_decade = T.nearest_search(times, year)
            end_target = year + avg_window if year == 1850 else year - avg_window
            i_end_decade = T.nearest_search(times, end_target)
            f.close()
            #average times
            og_new_name_map[filename] = year
            new_filename = model_name + file_suffix + '.nc'
            to_eval += 'ncwa -b -a time -d time,' + str(i_start_decade) + ',' + str(i_end_decade) + ' ' + filename + ' ' + new_filename + ' -O && '
            #to_eval += 'ncks -d time,' + str(time_index) + ' ' + filename + ' ' + new_filename + ' -O && '
    else:
        #print('doesnt have start and end:', model_name)
        bads.add(model_name)

print('all files:')
print(og_new_name_map)

to_eval = evaluate(to_eval)

#comands to combine files with their partners (subtraction)
if target_v == 'drybc':
    to_eval += 'echo "combining files with partners..." && '
    valid_models = list(set(main_dict.keys()).difference(bads))
    valid_er_models = set()
    for model_name in valid_models:
        if '_b' in model_name:
            continue
        partner = model_name.replace('_a', '_b')
        for suffix in ['_pi.nc', '_pd.nc']:
            m_suffix = model_name + suffix
            p_suffix = partner + suffix
            new_name = m_suffix.replace('_a', '').replace('.nc', '')
            #get sign of wetbc
            f = Dataset(root + '/' + p_suffix)
            wet_arr = f['wetbc'][:]
            f.close()
            if np.min(wet_arr) >= 0 and not np.max(wet_arr) <= 0:
                operation = 'add'
            elif np.max(wet_arr) <= 0 and not np.min(wet_arr) >= 0:
                operation = 'sub'
            if np.max(wet_arr) > 0 and np.min(wet_arr) < 0:
                raise Exception('this wetbc file contains both negative and positive values: ' + p_suffix)
            to_eval += 'ncrename -h -O -v wetbc,drybc ' + p_suffix + ' && '
            to_eval += 'ncbo --op_typ=' + operation + ' ' + m_suffix + ' ' + p_suffix + ' ' + new_name + '.nc -O && '
            valid_er_models.add(new_name.replace('_pi', '').replace('_pd', ''))
    to_eval = evaluate(to_eval)
else:
    valid_er_models = list(set(main_dict.keys()).difference(bads))

#commands to divide files
to_eval += 'echo "dividing..." && '
for model_name in valid_er_models:
    pi = model_name + '_pi.nc'
    pd = model_name + '_pd.nc'
    new_name = model_name + '.nc'
    to_eval += 'ncbo -v ' + target_v + ' --op_typ=dvd ' + pd + ' ' + pi + ' ' + new_name + ' -O && '

filenames = [model_name + '.nc' for model_name in valid_er_models]
to_eval = evaluate(to_eval)

#commands to remove time_bnds variable
to_eval += 'echo "removing time_bnds variable..." && '
for file_name in filenames:
    to_eval += 'ncks -C -O -x -v time_bnds ' + file_name + ' ' + file_name + ' -O && '
    pass

to_eval = evaluate(to_eval)

#commands to regrid all models
to_eval += 'echo "regriding..." && '
for i in range(len(filenames)):
    file_name = filenames[i]
    f = Dataset(root + '/' + file_name)
    to_eval += "ncap2 -O -s '" + target_v + "=double(" + target_v + ");' " + file_name + ' ' + file_name + ' && '
    if f.variables['lat'].shape[0] > 64 or f.variables['lon'].shape[0] > 128:
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
    if target_v == 'mmrbc':
        if 'CESM2-WACCM_re.nc' in files:
            files.remove('CESM2-WACCM_re.nc')
        elif 'CNRM-ESM2-1_re.nc' in files:
            files.remove('CNRM-ESM2-1_re.nc')
    if len(files) > 1:
        to_eval += 'ncra ' + ' '.join(files) + ' ' + base + '.nc -O && '
    if len(files) == 1:
        to_eval += 'mv ' + files[0] + ' ' + base + '.nc && '
    if len(files) < 1:
        to_eval += 'echo "\tall files from ' + base + ' removed." && '
        continue
    bases.append(base + '.nc')

to_eval = evaluate(to_eval)

#comand to average files
to_eval += 'echo "averaging..." && '
to_eval += 'ncra ' + ' '.join(bases) + ' output.nc -O && '
to_eval += 'echo "nco workflow done!"'
evaluate(to_eval)



print('begin python workflow...')
import pandas as pd #this has to be here otherwise var "pd" is overwritten

index_path = 'data/standardized-ice-cores/index.csv'
dupe_path = 'data/standardized-ice-cores/index-dup-cores.csv'
ice_coords = T.get_ice_coords(index_path, dupe_path)
df = pd.DataFrame(columns=['model'] + list(ice_coords.keys()))

for file in bases:
    row = pd.Series()
    row.at['model'] = file.replace('.nc', '')
    f = Dataset(os.path.join(root, file))
    lats, lons = T.adjust_lat_lon_format(f['lat'][:], f['lon'][:])
    times = f['time']
    v = f[target_v][:]
    for core_name in ice_coords.keys():
        y, x = ice_coords[core_name]
        lat = T.nearest_search(lats, y)
        lon = T.nearest_search(lons, x)
        assert T.within(lats[lat], y, 5) and T.within(lons[lon], x, 5)
        row.at[core_name] = v[0, lat, lon] if target_v != 'mmrbc' else v[0, 0, lat, lon]
    f.close()
    df.loc[len(df)] = row

#save csv data files
var2subfolder = {'drybc': 'cmip6', 'loadbc': 'loadbc', 'sootsn': 'cesm-sootsn', 'mmrbc': 'mmrbc'}
subfolder = var2subfolder[target_v]
if subfolder == 'cmip' and cesm_mode:
    subfolder = 'cesm-wetdry'
output_path = os.path.join(os.getcwd(), 'data', 'model-ice-depo', subfolder, 'nco.csv') if not bad_boy_mode else os.path.join(os.getcwd(), target_v + '.csv')
df.to_csv(output_path)

#save csv list of models
with open(os.path.join(root, "output_models.txt"), "w") as text_file:
    text_file.write("Models used in output: " + ' '.join(bases))

print('nco.csv saved to ' + output_path + '!')

#todo:
#remove nan and infinity from all files, I can do this using my notes