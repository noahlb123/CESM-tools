from netCDF4 import Dataset
import pandas as pd
import numpy as np
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
to_eval = ''

def evaluate(s):
    l = len(s)
    if s[l - 4:l] == ' && ':
        s = s[0:l - 4]
    os.system(s)
    return 'cd ' + root + ' && '

#setup
columns = []
index = []
for dir in dirs:
    p_i = os.path.join(root, dir, 'CESM2_pi.nc')
    p_d = os.path.join(root, dir, 'CESM2_pd.nc')
    main = os.path.join(root, dir, 'CESM2.nc')
    if os.path.isfile(p_i) and os.path.isfile(p_d) and os.path.isfile(main):
        columns.append(dir)
        index.append(dir)

if step == '1' or step == 'a': #combine nc files
    df_mult = pd.DataFrame(columns=columns, index=index)
    df_div = pd.DataFrame(columns=columns, index=index)
    smallest_grid = T.smallest_grid([os.path.join(root, dir, 'CESM2.nc') for dir in columns])
    #regrid
    old_new_name = {}
    to_eval += 'echo "regridding..." && '
    for dir in columns:
        new_name = os.path.join(work_dir, dir + '.nc')
        old_name = os.path.join(root, dir, 'CESM2.nc')
        old_new_name[old_name] = new_name
        #regrid
        to_eval += 'ncremap -d ' + smallest_grid + ' ' + old_name + ' ' + new_name + ' && '
    to_eval = evaluate(to_eval)
    to_eval += 'echo "renaming..." && '
    for dir in columns:
        new_name = os.path.join(work_dir, dir + '.nc')
        old_name = os.path.join(root, dir, 'CESM2.nc')
        #rename var
        f = Dataset(new_name)
        vars = list(f.variables.keys())
        if not 'X' in vars:
            to_eval += 'ncrename -h -O -v ' + name_var_map[dir] + ',' + 'X' + ' ' + new_name + ' && '
    to_eval = evaluate(to_eval)
    to_eval += 'echo "combining..." && '
    for numo in columns:
        for deno in index:
            if numo == deno:
                df_mult.loc[deno, numo] = os.path.join(root, dir, 'CESM2.nc')
                df_div.loc[deno, numo] = os.path.join(root, dir, 'CESM2.nc')
            else:
                numo_path = os.path.join(work_dir, numo + '.nc')
                deno_path = os.path.join(work_dir, deno + '.nc')
                new_path = os.path.join(work_dir, numo + '_X_' + deno + '.nc')
                #multiply
                to_eval += 'ncbo --op_typ=multiply ' + numo_path + ' ' + deno_path + ' ' + new_path + ' -O && '
                #divide
                to_eval += 'ncbo --op_typ=divide ' + numo_path + ' ' + deno_path + ' ' + new_path.replace('_X_', '_D_') + ' -O && '
                df_mult.loc[deno, numo] = new_path
    to_eval = evaluate(to_eval)
if step == '2' or step == 'a': #plot
    print('plotting...')
    import cartopy
    from matplotlib import colormaps
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm
    from matplotlib.colors import Normalize

    for op in ('D', 'X'):
        fig, ax = plt.subplots(len(columns), len(index), subplot_kw={'projection': cartopy.crs.Robinson()})
        for numo_i in range(len(columns)):
            numo = columns[numo_i]
            for deno_i in range(len(index)):
                deno = index[deno_i]
                if numo == deno:
                    filename = os.path.join(root, numo, 'CESM2.nc')
                else:
                    filename = os.path.join(work_dir, numo + '_' + op + '_' + deno + '.nc')
                
                #title
                ax[numo_i, deno_i].set_title(numo + '_' + op + '_' + deno)
                
                #setup cartopy
                ax[numo_i, deno_i].add_feature(cartopy.feature.COASTLINE, edgecolor='black')

                #get data
                f = Dataset(filename)
                lats = f['lat'][:]
                lons = f['lon'][:]
                x = f['X'][0,0,:,:] if 'mmrbc' in filename else f['X'][0,:,:]

                #color
                cmap = colormaps['BrBG_r']
                max = np.nanmax(x)
                print(max, 1/max, filename)
                c_norm = LogNorm(vmin=1/max, vmax=np.nanmax(x))
                sm = ScalarMappable(cmap=cmap, norm=c_norm)

                #plot
                ax[numo_i, deno_i].pcolormesh(lons, lats, x, cmap=cmap, norm=c_norm, transform=cartopy.crs.PlateCarree())
                #plt.colorbar(mappable=sm, label=var_name, orientation="horizontal", ax=ax, extend='both')
        plt.savefig(os.path.join(os.getcwd(), op + '.png'), dpi=200)
        print('saved to ' + os.path.join(os.getcwd(), op + '.png'))

print('done.')