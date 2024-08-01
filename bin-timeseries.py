import pandas as pd
import os

def get_base_model(path):
    if 'cmip6-snow-dep/all' in path and 'CESM' in path:
        return ['CESM', 'CMIP6']
    elif 'cmip6-snow-dep/all' in path:
        return 'CMIP6'
    elif 'cmip6-snow-dep' in path:
        return 'CESM-SOOTSN'
    elif 'lens' in path:
        return 'LENS'

unbinned = pd.read_csv(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'model-timeseries.csv')).abs()
binned = pd.DataFrame(index=unbinned.index)
path_model_map = {'CMIP6': [], 'CESM': [], 'CESM-SOOTSN': [], 'LENS': [],}

def manage_paths(path):
    model = get_base_model(path)
    if type(model) == type(''):
        path_model_map[model].append(path)
    elif type(model) == type([]):
        for m in model:
            path_model_map[m].append(path)

[manage_paths(path) for path in unbinned.columns]

for model, paths in path_model_map.items():
    model_avg = unbinned[paths].mean(axis=1)
    binned[model] = model_avg

binned.to_csv(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'binned-timeseries.csv'))
print('done.')