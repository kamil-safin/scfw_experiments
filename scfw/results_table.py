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
            results = results_data[dn][dn][p]

            if results['time_hist'][0] == 0:
                results['time_hist'] = results[data_name][data_name][p]['time_hist'][1:]
                results['Q_hist'] = results[data_name][data_name][p]['Q_hist'][1:]

            data_path = os.path.join('data', dn)
            Phi, _ = load_svmlight_file(data_path)

            N, n = Phi.shape
            error_hist = error_hist_data[p][dn]
            if any(error_hist <= threshold):
                # index of the first error which is <= than threshold
                error_i = np.where(error_hist <= threshold)[0][0]
            else:
                error_i = len(error_hist) - 1
            error = error_hist[error_i]
            iter = error_i + 1
            time = sum(results['time_hist'][:iter])
            time_per_iter = time / iter
            f_val = results['Q_hist'][error_i]
            
            if p == 'scopt':
                df_dict[dn] = [N, n, iter, round(time, 3), round(error, 3), round(time_per_iter, 3), round(f_val, 3)]
            else:
                df_dict[dn] = [iter, round(time, 3), round(error, 3), round(time_per_iter, 3), round(f_val, 3)]
        
        dfs.append(pd.DataFrame.from_dict(df_dict, orient='index', columns=columns))
    return pd.concat(dfs, axis=1)
