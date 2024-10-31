import os
import sys

target_vars = ['loadbc', 'sootsn', 'mmrbc', 'drybc', 'wetbc']
var_prefix_map = {'loadbc': 'Eday_', 'sootsn': 'LImon_', 'mmrbc': 'AERmon_', 'wetbc': 'AERmon_', 'drybc': 'AERmon_'}
start_year = int(sys.argv[1])
end_year = int(sys.argv[2])
root = sys.argv[3]
if len(sys.argv) < 4:
    #python3 remove-bad-dates.py 1850 1980 /glade/derecho/scratch/nlbills/all-ice-core-data/sootsn
    raise Exception('3 command line arguments required: <int: start year> <int: end year> <root directory>')
assert os.path.isdir(root)

def has_any_target_var(filename):
    for v in target_vars:
        if v in filename:
            return True
    return False

def filename2modelname(filename):
    target_v = ''
    for v in target_vars:
        if v in filename:
            target_v = v
            break
    start_year = filename[filename.rfind("_") + 1:filename.rfind("-")][0:4]
    end_year = filename[filename.rfind("-") + 1:filename.rfind(".")][0:4]
    model_name = filename[filename.index(prefix) + len(prefix): filename.index('_historical')]
    return int(start_year), int(end_year)

def valid_range(s_year, e_year):
    for year in [s_year, e_year]:
        if start_year <= year <= end_year:
            return True
    return False

def get_years(filename):
    years = filename[filename.rfind('_') + 1:len(filename) - 3].split('-')
    years = [int(year[0:4]) for year in years]
    return years

to_eval = 'cd ' + root + ' && rm '
n = 0
for filename in os.listdir(root):
    if 'wget' not in filename and has_any_target_var(filename) and filename[len(filename) - 3:len(filename)] == '.nc':
        s_year, e_year = filename2modelname(filename)
        if not valid_range(s_year, e_year):
            to_eval += filename + ' '
            n += 1

to_eval += '&& echo "removed ' + str(n) + ' files"'
print(to_eval)
print('sample year extraction: ' + list(os.listdir(root))[0], filename2modelname(list(os.listdir(root))[0]))

if input('Run the command above? (y/n): ').lower() == 'y':
    os.system(to_eval)