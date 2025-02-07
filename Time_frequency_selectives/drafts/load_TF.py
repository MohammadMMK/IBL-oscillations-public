import numpy as np
import pandas as pd
import mne
import os
import re
def aggregate_TFs(pids, selectives, condition = 'BiasRight_BiasLeft_anticip', side_selective = 'left' , regions = ['VISp'], tmin = -0.3, tmax = 0.8):
    
    c1_TF_all = []
    c2_TF_all = []
    n_trials_c1_all = []
    n_trials_c2_all = []
    df_all = []
    for i, pid in enumerate(pids):

        df = selectives.loc[selectives['pid'] == pid].reset_index(drop = True)
        

        path = f'results/{pid}_{condition}.npy'
        if len(df) == 0 or not os.path.exists(path):
            continue
        c1_TF , c2_TF = np.load(f'results/{pid}_{condition}.npy', allow_pickle = True)
        # get number of trials for each condition

        c1_TF = c1_TF.crop(tmin = tmin, tmax = tmax)
        c2_TF = c2_TF.crop(tmin = tmin, tmax = tmax)
        c1 = np.log(c1_TF.data)
        c2 = np.log(c2_TF.data)
        c1_TF_all.append(c1)
        c2_TF_all.append(c2)

        n_trials_c1 = [c1_TF.nave] * c1.shape[0]
        n_trials_c2 = [c2_TF.nave] * c1.shape[0]
        n_trials_c1_all.append(n_trials_c1)
        n_trials_c2_all.append(n_trials_c2)
        df_all.append(df)


    c1_TF_all = np.concatenate(c1_TF_all, axis = 0)
    c2_TF_all = np.concatenate(c2_TF_all, axis = 0)
    df_all = pd.concat(df_all)
    df_all = df_all.reset_index(drop = True)
    df_all['index'] = np.arange(len(df_all))

    # filter channels
    acronyms = df_all['acronyms'].tolist()
    keep_ch_region = [i for i, ch in enumerate(acronyms) if any(re.match(rf'^{reg}[12456]', ch) for reg in regions)] 

    if  side_selective == 'right':
        keep_ch_side = np.where(~((df_all['p_value_c1'] > 0.05) | (df_all['accuracies_c1'] < 0.6)))[0]
    elif side_selective == 'left':
        keep_ch_side = np.where(~((df_all['p_value_c2'] > 0.05)| (df_all['accuracies_c2'] < 0.6)))[0]


    keep_ch = np.intersect1d(keep_ch_region, keep_ch_side)
    c1_TF_all = c1_TF_all[ keep_ch, :, :]
    c2_TF_all = c2_TF_all[ keep_ch, :, :]
    df_all = df_all.loc[keep_ch]
    ch_names = df_all['acronyms'].tolist()
    n_trials_c1_all = np.concatenate(n_trials_c1_all)[keep_ch]
    n_trials_c2_all = np.concatenate(n_trials_c2_all)[keep_ch]

    times = c1_TF.times
    freqs = c1_TF.freqs
    return c1_TF_all, c2_TF_all, ch_names, times, freqs, np.array(n_trials_c1_all), np.array(n_trials_c2_all)

