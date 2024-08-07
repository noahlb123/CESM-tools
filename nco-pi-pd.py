import os
import sys
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
    return filename[T.find_nth(filename, '_', 2) + 1:filename.find('_historical')]

#DRY MUST COME FIRST IF USING WET DRY PAIRS
main_dict = {}
if len(sys.argv) < 3:
    raise Exception('3 command line arguments required: <varaible name common in all desired files> <root directory> <name of .nc file with lowest resolution grid>')
common_var = sys.argv[1]
root = sys.argv[2]
smallest_grid = sys.argv[3]
#python3 process-precip.py drybc /glade/derecho/scratch/nlbills/cmip6-snow-dep/all drybc_AERmon_CanESM5-1_historical_r11i1p2f1_gn_185001-201412.nc
system = platform.system() #differentiate local and derecho env by sys platform
partners = {}
if system == "Darwin":
    import pyperclip
    files = pyperclip.paste().split('\n')
else:
    files = os.listdir(root)
bads = set([])
to_eval = 'cd ' + root + ' && '

#find start and end files
for filename in files:
    if common_var in filename:
        partner_name = filename.replace('wetbc', 'drybc') if 'wetbc' in filename else filename.replace('drybc', 'wetbc')
        if os.path.isfile(os.path.join(root, partner_name)):
            for f_name in [filename, partner_name]:
                model_name = get_model_name(f_name)
                model_name += '_b' if f_name == partner_name else '_a'
                years = get_years(f_name)
                if contains(years, 1850):
                    if model_name not in main_dict:
                        main_dict[model_name] = {'s_file': f_name, 'e_file': None, 's_year': years[0]}
                    else:
                        main_dict[model_name]['s_file'] = f_name
                        main_dict[model_name]['s_year'] = years[0]
                if contains(years, 1980):
                    if model_name not in main_dict:
                        main_dict[model_name] = {'e_file': f_name, 's_file': None, 'e_year': years[0]}
                    else:
                        main_dict[model_name]['e_file'] = f_name
                        main_dict[model_name]['e_year'] = years[0]

#commands to extract file timeslices
to_eval += 'echo "extracting timeslices..." && '
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
            time_index = T.nearest_search(times, year)
            f.close()
            new_filename = model_name + file_suffix + '.nc'
            to_eval += 'ncks -d time,' + str(time_index) + ' ' + filename + ' ' + new_filename + ' -O && '
    else:
        #print('doesnt have start and end:', model_name)
        bads.add(model_name)

#comands to combine files with their partners
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
        to_eval += 'ncbo --op_typ=sub ' + m_suffix + ' ' + p_suffix + ' ' + new_name + '.nc -O && '
        valid_er_models.add(new_name.replace('_pi', '').replace('_pd', ''))

#commands to divide files
to_eval += 'echo "dividing..." && '
for model_name in valid_er_models:
    pi = model_name + '_pi.nc'
    pd = model_name + '_pd.nc'
    new_name = model_name + '.nc'
    to_eval += 'ncbo --op_typ=dvd ' + pd + ' ' + pi + ' ' + new_name + ' -O && '

filenames = [model_name + '.nc' for model_name in valid_er_models]

#commands to remove time_bnds variable
to_eval += 'echo "removing time_bnds variable..." && '
for file_name in filenames:
    to_eval += 'ncks -C -O -x -v time_bnds ' + file_name + ' ' + file_name + ' -O && '

print('evaluating section 1/2 ...')
os.system(to_eval[0:len(to_eval) - 4]) #remove trailing ' && '
to_eval = 'cd ' + root + ' && '

#commands to regrid all models
to_eval += 'echo "regriding..." && '
for i in range(len(filenames)):
    file_name = filenames[i]
    f = Dataset(root + '/' + file_name)
    print(file_name, list(f.variables.keys()))
    to_eval += "ncap2 -O -s '" + common_var + "=double(" + common_var + ");' " + file_name + ' ' + file_name + ' && '
    if f.variables['lat'].shape[0] > 64 or f.variables['lon'].shape[0] > 128:
        to_eval += 'echo "regridding ' + file_name + '" && '
        to_eval += 'ncremap -d ' + smallest_grid + ' ' + file_name + ' ' + file_name.replace('.nc', '_re.nc') + ' && '
        filenames[i] = file_name.replace('.nc', '_re.nc')
    f.close()

#comand to average files
to_eval += 'echo "averaging..." && '
to_eval += 'ncra ' + ' '.join(filenames) + ' output.nc -O && '
to_eval += 'echo "done!"'

#evaluate
#print(to_eval)
print('evaluating section 2/2 ...')
os.system(to_eval)
#print(list(bads))

#todo:
#wetbc and drybc are being removed in subtraction step because the main var in each files is different, use different subtraction method I wrote in notes
#remove nan and infinity from all files, I can do this using my notes

#ncap2 -s [wetdep=wetdep*-1] in.nc out.nc
#ncdivide 1.nc 2.nc 3.nc