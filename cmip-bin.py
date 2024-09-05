import pandas as pd
import numpy as np
import json
import sys
import os

#cmip6, loadbc
if len(sys.argv) < 2:
    raise Exception('format command as:\npython3 cmip-bin.py <MODEL-ICE-DEPO SUBFOLDER> <...>')
dirs = sys.argv[1:len(sys.argv)]

for dir in dirs:
    root_path = os.path.join(os.getcwd(), 'data', 'model-ice-depo', dir)
    file_suffix = '-lv30.csv' if dir == 'lens' else '.csv'
    base_model_dict = {}
    data_dict = {
        'pi': {},#keys: base_model str, vals: (sum vector, n)
        'pd': {}
    }
    olds = {
        'pi': pd.read_csv(os.path.join(root_path, "pi" + file_suffix)).drop(['window', 'n ensemble members'], axis=1),
        'pd': pd.read_csv(os.path.join(root_path, "pd" + file_suffix)).drop(['window', 'n ensemble members'], axis=1)
    }

    def base_model(model):
        if dir != 'loadbc':
            if 'MIROC6' in model:
                base_model = 'MIROC'
            if '-' in model:
                base_model = model[0:model.index('-')]
            else:
                base_model = model
        else:
            base_model = model
        if base_model in base_model_dict:
            base_model_dict[base_model].append(model)
        else:
            base_model_dict[base_model] = [model]
        return base_model

    #get data by base model
    for key in olds.keys():
        old = olds[key]
        for index, row in old.iterrows():
            model = base_model(row['model'])
            if model in data_dict:
                new_s = np.add(list(row.drop(['model'])), data_dict[model][0])
                new_n = data_dict[model][1] + 1
                data_dict[key][model] = (new_s, new_n)
            else:
                data_dict[key][model] = (row.drop(['model']), 1)

    #recomile data_dict into data_array
    data_array = {
        'pi': [],
        'pd': []
    }
    for era in olds.keys():
        for key, value in data_dict[era].items():
            data_array[era].append([key] + list(np.divide(value[0], value[1])))

    #save as csv and json
    for era in olds.keys():
        pd.DataFrame(data=data_array[era]).set_axis(olds[era].columns, axis='columns').to_csv(os.path.join(root_path, "binned-" + era + ".csv"), index=False)
    with open(os.path.join(root_path, 'model-bins.json'), "w") as outfile:
        json.dump(base_model_dict, outfile)
    
    #read josn and output contents
    models = sum(json.load(open(os.path.join(root_path, 'model-bins.json'))).values(), [])
    models.sort()
    for model in models:
        print(model, end=', ')

print('done.')