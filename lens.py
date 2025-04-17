from netCDF4 import Dataset
import pandas as pd
import numpy as np
import platform
import tools
import csv
import sys
import os

if len(sys.argv) < 2:
    #python lens.py new
    raise Exception('1 command line arguments required: <mode (setup, old, new)>')

if sys.argv[1] == 'setup':
    #gather files
    files = []
    root = '/glade/campaign/cesm/collections/cesmLE/restarts'
    for sub_dir in os.listdir(root):
        if ('B20TRC5CNBDRD' in sub_dir) and (not 'OIC' in sub_dir):
            i = int(sub_dir.split('.')[-1])
            years = [1980, 1975, 1970, 1965, 1960, 1955] if i != 34 else [1982, 1977, 1972, 1970, 1965, 1960, 1955]
            for year in years:
                path = os.path.join(root, sub_dir, 'b.e11.B20TRC5CNBDRD.f09_g16.0' + "{:02d}".format(i) + '.rest.' + str(year) + '-01-01-00000.tar')
                if os.path.isfile(path):
                    files.append(path)
                else:
                    assert os.path.isfile(path)
    #copy files
    print('cp ' + ' '.join(files) + ' .')
    #delete unneeded files
    print('\n\n\n\n')
    print('for file in *; do mkdir "${{file:0:52}}" && tar -xf "$file" -C "${{file:0:52}}" && my_dir="${{file:0:31}}"; for subfile in "${{file:0:52}}"/*/*; do if [ "${{subfile:102:3}}" != "cam" ]; then rm -r $subfile; fi; done; done\n')
    #average files that have bc_a vars
    run_file_map = {}
    token = 'b.e11.B20TRC5CNBDRD.f09_g16.'
    root = '/glade/derecho/scratch/nlbills/lens/pd'
    for file in os.listdir(root):
        if not token in file:
            continue
        i = int(file[file.index(token)+len(token):file.index(token)+len(token)+3])
        f = Dataset(os.path.join(root, file))
        if 'bc_a1_SRF' in f.variables.keys():
            if i in run_file_map.keys():
                run_file_map[i].append(file)
            else:
                run_file_map[i] = [file]
        f.close()
    s = ''
    for run, files in run_file_map.items():
        s += 'ncra ' + ' '.join(files) + ' ' + str(run) + '.nc && '
    print(s[0:len(s) - 4])
    print('\n')
    #remove unneeded vars
    print('ncks -v bc_a1_SRF,hyai,hybi,P0,PSL,TREFHT,lat,lon pd-avg.nc pd-small.nc')



elif sys.argv[1] == 'old' or sys.argv[1] == 'new':

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

    for mode in ['pd', 'pi']:
        if sys.argv[1] == 'old':
            print(mode)
            h = 1 if mode == "pd" else 0
            #get viable models
            for model in models:
                if mode == 'pd':
                    years = [1980 + 5 * (x - 2) for x in range(5)] if model != 34 else (1977, 1982, 1987, 1972, 1970)
                else:
                    years = (1865, 1855, 1860)
                for year in years:
                    #get path
                    year_s = str(year) + "-01-01-00000"
                    working_path = '/glade/derecho/scratch/nlbills/lens-all-' + mode
                    file_paths = modelN2fnames(model, year_s, h)
                    print(file_paths)
                    continue
                continue
            if 'cringe':
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

        if sys.argv[1] == 'new':
            root = '/glade/derecho/scratch/nlbills/lens/pd'
            file = 'pi-small.nc'
            viable_models.append(file)
            model_var_map[file] = ['bc_a1_SRF']
            model_year_map[file] = 1850
            model_path_map[file] = os.path.join('/glade/derecho/scratch/nlbills/lens/pi', 'pi-small.nc')
            for file in os.listdir(root):
                if file.replace('.nc', '').isnumeric():
                    viable_models.append(file)
                    model_var_map[file] = ['bc_a1_SRF']
                    model_year_map[file] = 1980
                    model_path_map[file] = os.path.join(root, file)


        
        #get ice core lat, lon
        index_path = 'data/standardized-ice-cores/index.csv'
        dupe_path = 'data/standardized-ice-cores/index-dup-cores.csv'
        ice_coords = T.get_ice_coords(index_path, dupe_path)

        #get bc depo
        for lv in lvls:
            print(lv)
            for shape in [[[0, 0]]]:
                print('shape len:', len(shape))
                csv_dict = []
                for model in viable_models:
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
                filename = mode + "-lv" + str(lv) + '-s' + str(len(shape)) + ".csv"
                print('saving to ' + os.path.join(os.getcwd(), filename))
                filepath = os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'lens', filename) #if sys.argv[1] == 'old' else os.path.join(os.getcwd(), filename)
                with open(filepath, 'w') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(csv_dict)

    pd.DataFrame(data=[model_path_map[model] for model in viable_models], columns=['files']).to_csv(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'lens', 'fileuse-index.csv'))

    print("done lens.py.")
print('evaluating lens-avg.py...')
if sys.argv[1] == 'old':
    os.system('python lens-avg.py && echo "everything completed!"')
elif sys.argv[1] == 'new':
    df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'lens', 'pd-lv30-s1.csv')).drop(['BC_vars', 'year'], axis=1)
    df = df.set_index('model number')
    df.div(df.loc['pi-small.nc']).drop(['pi-small.nc'], axis=0).to_csv(os.path.join(os.getcwd(), 'lens.csv'))