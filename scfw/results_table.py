import os

import numpy as np
import pandas as pd  
from sklearn.datasets import load_svmlight_file


def results_table(results_data, error_hist_data, data_list, threshold):
    policies = ['scopt', 'standard', 'line_search', 'icml', 'backtracking']
    df_dict = {}
    dfs = []
    for p in policies:
        if p == 'scopt':
            columns = ['N', 'n', 'iter', 'time', 'error', 'time per iter', 'f_val']
        else:
            columns = ['iter', 'time', 'error', 'time per iter', 'f_val']
        for dn in sorted(data_list):
            data_path = os.path.join('data', dn)
            Phi, _ = load_svmlight_file(data_path)

            N, n = Phi.shape

            error_i = np.where(error_hist_data[p][dn] >= threshold)[0][-1]
            error = error_hist_data[p][dn][error_i]
            iter = error_i + 1
            time = sum(results_data[dn][dn][p]['time_hist'][:error_i])
            time_per_iter = time / iter
            f_val = results_data[dn][dn][p]['Q_hist'][error_i]
            
            if p == 'scopt':
                df_dict[dn] = [N, n, iter, round(time, 3), round(error, 3), round(time_per_iter, 3), round(f_val, 3)]
            else:
                df_dict[dn] = [iter, round(time, 3), round(error, 3), round(time_per_iter, 3), round(f_val, 3)]
        
        dfs.append(pd.DataFrame.from_dict(df_dict, orient='index', columns=columns))
    return pd.concat(dfs, axis=1)
