from netCDF4 import Dataset
import pandas as pd
import tools
import sys
import os

if len(sys.argv) < 2:
    #python div-var-robinsons.py 2
    raise Exception('1 command line arguments required: <steps: (a/1/2)>')
step = sys.argv[1]
T = tools.ToolBox()
root = '/glade/derecho/scratch/nlbills/all-ice-core-data'
work_dir = os.path.join(root, 'ratios')
dirs = ['loadbc', 'mmrbc', 'sootsn', 'wet-dry']
name_var_map = {'loadbc': 'loadbc', 'mmrbc': 'mmrbc', 'sootsn': 'sootsn', 'wet-dry': 'drybc'}
file2dir = {}

if step == '1' or step == 'a': #combine nc files
    columns = []
    index = []
    for dir in dirs:
        p_i = os.path.join(root, dir, 'CESM2_pi.nc')
        p_d = os.path.join(root, dir, 'CESM2_pd.nc')
        main = os.path.join(root, dir, 'CESM2.nc')
        if os.path.isfile(p_i) and os.path.isfile(p_d) and os.path.isfile(main):
            columns.append(dir)
            index.append(dir)
    df_mult = pd.DataFrame(columns=columns, index=index)
    df_div = pd.DataFrame(columns=columns, index=index)
    smallest_grid = T.smallest_grid(columns)
    #regrid
    old_new_name = {}
    for dir in columns:
        new_name = os.path.join(work_dir, dir, 'CESM2.nc')
        old_name = os.path.join(root, dir, 'CESM2.nc')
        old_new_name[old_name] = new_name
        os.system('ncremap -d ' + smallest_grid + ' ' + old_name + ' ' + new_name + ' -O')
        #rename var
        os.system('ncrename -h -O -v ' + name_var_map[dir] + ',' + 'X' + ' ' + old_name)
    for numo in columns:
        for deno in index:
            if numo == deno:
                df_mult.loc[deno, numo] = os.path.join(root, dir, 'CESM2.nc')
            else:
                numo_path = os.path.join(work_dir, numo, 'CESM2.nc')
                deno_path = os.path.join(work_dir, deno, 'CESM2.nc')
                new_path = os.path.join(work_dir, deno, numo + '_X_' + deno + '.nc')
                #math
                'ncbo --op_typ=multiply ' + numo_path + ' ' + deno_path + ' ' + new_path + '.nc -O && '
                df_mult.loc[deno, numo] = new_path
if step == '2' or step == 'a': #plot
    pass