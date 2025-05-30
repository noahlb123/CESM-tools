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
    main = os.path.join(root, dir, 'CESM2.nc')
    if os.path.isfile(main):
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
            to_eval += 'echo "renaming ' + new_name + ',' + name_var_map[dir] + '..." && '
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
    from matplotlib.colors import to_rgb
    #from matplotlib.colors import Normalize
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.colors import BoundaryNorm

    def mask_zeros(m):
        mask = np.logical_and(np.ones(np.shape(m)).astype(bool), m == 0)
        m[mask] = np.nan
        return m

    def normalize(m):
        return m
        m = mask_zeros(m)
        min = np.nanmin(np.ma.masked_invalid(m))
        max = np.nanmax(np.ma.masked_invalid(m))
        #return (m - min) / (max - min)
        return m / min

    for op in ['D', 'X']:
        fig, ax = plt.subplots(len(columns), len(index), subplot_kw={'projection': cartopy.crs.Robinson()})
        fig.tight_layout(rect=(0, 0, 1, 1))
        for numo_i in range(len(columns)):
            numo = columns[numo_i]
            print(numo)
            for deno_i in range(len(index)):
                deno = index[deno_i]

                #title
                if numo_i == 0:
                    ax[numo_i, deno_i].set_title(deno, size=12)
                if deno_i == 0:
                    ax[numo_i, deno_i].text(-0.2, 0.5, numo, size=12, verticalalignment='center', rotation=90, transform=ax[numo_i, deno_i].transAxes)
                
                #setup cartopy
                ax[numo_i, deno_i].add_feature(cartopy.feature.COASTLINE, edgecolor='black', linewidth=0.5)

                #get data
                if numo == deno:
                    filename = os.path.join(root, numo, 'CESM2.nc')
                    f = Dataset(filename)
                    lats = f['lat'][:]
                    lons = f['lon'][:]
                    x = normalize(f['X'][0,0,:,:]) if 'mmrbc' in filename else normalize(f['X'][0,:,:])
                else:
                    filename = os.path.join(work_dir, numo + '_' + op + '_' + deno + '.nc')
                    f_numo = Dataset(os.path.join(work_dir, numo + '.nc'))
                    f_deno = Dataset(os.path.join(work_dir, deno + '.nc'))
                    lats = f_numo['lat'][:]
                    lons = f_numo['lon'][:]
                    x_n = normalize(f_numo['X'][0,0,:,:]) if 'mmrbc' in numo else normalize(f_numo['X'][0,:,:])
                    x_d = normalize(f_deno['X'][0,0,:,:]) if 'mmrbc' in deno else normalize(f_deno['X'][0,:,:])
                    if op == 'X':
                        x = np.multiply(x_n, x_d)
                    elif op == 'D':
                        x = np.divide(x_n, x_d)

                #color
                cmap = colormaps['BrBG_r'] if op == 'D' else colormaps['viridis']
                if op == 'D':
                    cmaplist = [cmap(i) for i in range(cmap.N)]
                    cmap2 = T.custom_cmap([(0.3254901960784314, 0.19215686274509805, 0.01568627450980392), to_rgb('#da00db')])
                    cmaplist2 = [cmap2(i) for i in range(cmap.N)]
                    n_extra_colors = 6
                    extra_colors = [cmaplist2[i * int(np.floor(len(cmaplist2) / (n_extra_colors + 1)))] for i in range(n_extra_colors + 1)]
                    custom_list = ['#003c30', '#0c7169', '#59b0a7', '#b4e1da', '#ffffff', '#f1deb3', '#d0a255', '#995d14', '#533104'] + extra_colors[1:len(extra_colors)]
                    cmap = LinearSegmentedColormap.from_list('Custom cmap', custom_list, cmap.N)
                    #cmap.set_extremes(over='#ff0000')
                    bounds = [round(x, 1) for x in np.linspace(0, 2, 10)] + [2 + i * (6 - 2) / (n_extra_colors) for i in range((n_extra_colors))]
                    c_norm = BoundaryNorm(bounds, cmap.N)
                else:
                    c_norm = LogNorm(vmin=1, vmax=10)
                sm = ScalarMappable(cmap=cmap, norm=c_norm)

                #plot
                ax[numo_i, deno_i].pcolormesh(lons, lats, x, cmap=cmap, norm=c_norm, transform=cartopy.crs.PlateCarree())
        op_s = '*' if op == 'X' else '÷'
        label = 'X1 (PD/PI) ' + op_s + ' X2 (PD/PI)'
        plt.colorbar(mappable=sm, label=label, orientation="horizontal", ax=ax, extend='both')
        labels = ('1/Max', '1', 'Max') if op == 'D' else ('0.1', '', 'Max')
        plt.savefig(os.path.join(os.getcwd(), op + '.png'), dpi=200, bbox_inches='tight')
        print('saved to ' + os.path.join(os.getcwd(), op + '.png'))

print('done.')