import os
import platform
from tools import ToolBox
from netCDF4 import Dataset
T = ToolBox()

def get_years(filename):
    years = filename[filename.rfind('_') + 1:len(filename) - 3].split('-')
    years = [int(year[0:4]) for year in years]
    return years

def contains(years, target_year):
    s_year, e_year = years
    if s_year <= target_year <= e_year:
        return True
    return False

def get_model_name(filename):
    #prsn_day_MIROC-ES2L_historical_r21i1p1f2_gn_19800101-19801231.nc
    return filename[T.find_nth(filename, '_', 2) + 1:filename.find('_historical')]

main_dict = {}
root = '/glade/derecho/scratch/nlbills/cmip-precip'
system = platform.system() #differentiate local and derecho env by sys platform
if system == "Darwin":
    import pyperclip
    files = pyperclip.paste().split('\n')
else:
    files = os.listdir(root)
bads = set([])
to_eval = 'cd ' + root + ' && '

#find start and end files
for filename in files:
    if 'prsn' in filename:
        model_name = get_model_name(filename)
        years = get_years(filename)
        if contains(years, 1850):
            if model_name not in main_dict:
                main_dict[model_name] = {'s_file': filename, 'e_file': None, 's_year': years[0]}
            else:
                main_dict[model_name]['s_file'] = filename
                main_dict[model_name]['s_year'] = years[0]
        if contains(years, 1980):
            if model_name not in main_dict:
                main_dict[model_name] = {'e_file': filename, 's_file': None, 'e_year': years[0]}
            else:
                main_dict[model_name]['e_file'] = filename
                main_dict[model_name]['e_year'] = years[0]

#commands to extract file timeslices
for model_name, d in main_dict.items():
    if d['s_file'] != None and d['e_file'] != None:
        for year in (1850, 1980):
            file_index = 's_file' if year == 1850 else 'e_file'
            year_index = 's_year' if year == 1850 else 'e_year'
            file_suffix = '_pi' if year == 1850 else '_pd'
            filename = d[file_index]
            try:
                f = Dataset(root + '/' + filename)
            except OSError:
                #print('wrong format:', model_name)
                bads.add(model_name)
                continue
            time_var = f.variables['time']
            times = f['time'][:]
            assert 'days since' in time_var.units
            #print(f.variables['prsn'].shape)
            time_index = T.nearest_search(times, year)
            f.close()
            new_filename = model_name + file_suffix + '.nc'
            to_eval += 'ncks -d time,' + str(time_index) + ' ' + filename + ' ' + new_filename + ' -O && '
    else:
        #print('doesnt have start and end:', model_name)
        bads.add(model_name)

#commands to divide files
valid_models = list(set(main_dict.keys()).difference(bads))
for model_name in valid_models:
    pi = model_name + '_pi.nc'
    pd = model_name + '_pd.nc'
    new_name = model_name + '.nc'
    to_eval += 'ncbo --op_typ=dvd ' + pd + ' ' + pi + ' ' + new_name + ' -O && '

filenames = [model_name + '.nc' for model_name in valid_models]

#commands to remove time_bnds variable
for file_name in filenames:
    to_eval += 'ncks -C -O -x -v time_bnds ' + file_name + ' ' + file_name + ' -O && '

#commands to resize all models
for i in range(len(filenames)):
    file_name = filenames[i]
    f = Dataset(root + '/' + file_name)
    print(f.variables['lat'].shape[0], f.variables['lon'].shape[0], file_name)
    if f.variables['lat'].shape[0] > 64 or f.variables['lon'].shape[0] > 128:
        to_eval += "ncap2 -O -s 'prsn=double(prsn);' " + file_name + ' ' + file_name + ' && '
        to_eval += 'ncremap -d CanESM5-1.nc ' + file_name + ' ' + file_name.replace('.nc', '_re.nc') + ' && '
        filenames[i] = file_name.replace('.nc', '_re.nc')
    f.close()

filenames = [model_name + '_re.nc' for model_name in valid_models]

#comand to average files
to_eval += 'ncra ' + ' '.join(filenames) + ' output.nc -O'

#evaluate
print(to_eval)
os.system(to_eval)
#print(list(bads))

#todo:
#remove nan and infinity from all files, I can do this using my notes