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
    for numo in columns:
        for deno in index:
            if numo == deno:
                df_mult.loc[deno, numo] = os.path.join(root, dir, 'CESM2.nc')
            else:
                nums = [Dataset(os.path.join(root, numo, 'CESM2_' + era + '.nc')) for era in ('pi', 'pd')]
                denos = [Dataset(os.path.join(root, deno, 'CESM2_' + era + '.nc')) for era in ('pi', 'pd')]
                smallest_grid = T.smallest_grid(nums + denos)
                name_var_map = {}
                for file in nums + denos:
                    f = Dataset(file)
                    vars = list(f.variables())
                    for v in vars:
                        #print(v)
                        pass
                '''
                #rename var
                t_var = 
                os.system('ncrename -h -O -v wetbc,drybc ' + p_suffix + ' && ')
                #math
                'ncbo --op_typ=' + operation + ' ' + m_suffix + ' ' + p_suffix + ' ' + new_name + '.nc -O && '
                df_mult.loc[deno, numo] = os.path.join(root, dir, 'CESM2.nc')
if step == '2' or step == 'a': #plot
    pass'''