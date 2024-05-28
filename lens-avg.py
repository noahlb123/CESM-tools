import pandas as pd
import os

def relavant_indicies(indexes, n, type):
    mid_index = len(indexes) // 2
    mid_n = n // 2
    if type == 'pd':
        return [indexes[mid_index + i - mid_n] for i in range(n)]
    else:
        return indexes[0:n // 2 + 1]

root_path = os.path.join(os.getcwd(), 'data', 'lens')
lvls = (30, 29, 28)
windows = (5, 10, 20)
n_samples = {5: 1, 10: 3, 20: 5}

for lvl in lvls:
    #read files
    p_i = pd.read_csv(os.path.join(root_path, "pi-lv" + str(lvl) + '.csv'))
    p_d = pd.read_csv(os.path.join(root_path, "pd-lv" + str(lvl) + '.csv'))
    for window in windows:
        #create pd map
        mode_indexes_pd = {}
        mode_indexes_pi = {}
        pd_df_dict = {}
        pi_df_dict = {}
        for index, row in p_d.iterrows():
            model_string = row["model number"]
            model_n = model_string[0:2]
            year = model_string[5:9]
            if model_n not in mode_indexes_pd:
                mode_indexes_pd[model_n] = [index]
            else:
                mode_indexes_pd[model_n].append(index)
        #create pi map
        for index, row in p_i.iterrows():
            model_string = row["model number"]
            model_n = model_string[0:2]
            year = model_string[5:9]
            if model_n not in mode_indexes_pi:
                mode_indexes_pi[model_n] = [index]
            else:
                mode_indexes_pi[model_n].append(index)
        #calc avgs
        for column_name in p_d:#should be same for pi and pd
            if column_name not in ('model number', 'BC_vars', 'year'):
                #pd
                for model_n, indexes in mode_indexes_pd.items():
                    mean = p_d.iloc[relavant_indicies(indexes, n_samples[window], 'pi'), p_d.columns.get_loc(column_name)].mean()
                    if model_n not in pd_df_dict:
                        pd_df_dict[model_n] = [mean]
                    else:
                        pd_df_dict[model_n].append(mean)
                #pi
                for model_n, indexes in mode_indexes_pi.items():
                    mean = p_i.iloc[relavant_indicies(indexes, n_samples[window], 'pi'), p_i.columns.get_loc(column_name)].mean()
                    if model_n not in pi_df_dict:
                        pi_df_dict[model_n] = [mean]
                    else:
                        pi_df_dict[model_n].append(mean)

        #divide and save
        avg_pd = pd.DataFrame.from_dict(pd_df_dict, columns=list(p_d.columns[3:len(p_d.columns)]), orient='index')
        avg_pi = pd.DataFrame.from_dict(pi_df_dict, columns=list(p_i.columns[3:len(p_i.columns)]), orient='index')
        print('saving', 'a' + str(window) + 'lv' + str(lvl) + '.csv')
        avg_pd.div(avg_pi.iloc[0], axis='columns').to_csv('a' + str(window) + 'lv' + str(lvl) + '.csv')

print('done.')