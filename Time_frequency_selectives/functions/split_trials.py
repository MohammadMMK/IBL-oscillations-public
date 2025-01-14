import numpy as np
import mne

def split_trials(epoch, condition, min_trial = 0, remove_first_trials_of_block = False):

    meta = epoch.metadata.reset_index(drop=True)
    if condition == 'Stim_NoStim':
        condition1_trial = np.where(((meta['contrastLeft'] ==1) | (meta['contrastRight'] == 1)))[0]
        condition2_trial = np.where(((meta['contrastLeft'] <0.1) | (meta['contrastRight'] < 0.1)))[0]
    elif condition == 'Right_Left':
        condition1_trial = np.where(((meta['contrastRight'].isna()) & (meta['contrastLeft'] > 0.8)))[0]
        condition2_trial = np.where(((meta['contrastLeft'].isna()) & (meta['contrastRight'] > 0.8)))[0]
    elif condition == 'BiasRight_BiasLeft_anticip':
        condition1_trial = np.where(meta['probabilityLeft'] == 0.2 )[0]
        condition2_trial = np.where(meta['probabilityLeft'] == 0.8 )[0]
    elif condition == 'BiasLeft_BiasRight_stimLeft':
        condition1_trial = np.where((meta['probabilityLeft'] == 0.8) & (meta['contrastLeft'] > 0.2))[0]
        condition2_trial = np.where((meta['probabilityLeft'] == 0.2) & (meta['contrastLeft'] > 0.2))[0]
    elif condition == 'BiasLeft_BiasRight_stimRight':
        condition1_trial = np.where((meta['probabilityLeft'] == 0.8) & (meta['contrastRight'] > 0.2))[0]
        condition2_trial = np.where((meta['probabilityLeft'] == 0.2) & (meta['contrastRight'] > 0.2))[0]
    
    elif condition == 'BiasLeft_BiasRight_NoStimLeft':
        condition1_trial = np.where((meta['probabilityLeft'] == 0.8) & (~(meta['contrastLeft'] > 0)))[0]
        condition2_trial = np.where((meta['probabilityLeft'] == 0.2) & (~(meta['contrastLeft'] > 0)))[0]
    elif condition == 'BiasLeft_BiasRight_NoStimRight':
        condition1_trial = np.where((meta['probabilityLeft'] == 0.8) & (~(meta['contrastRight'] > 0)))[0]
        condition2_trial = np.where((meta['probabilityLeft'] == 0.2) & (~(meta['contrastRight'] > 0)))[0]

    elif condition == 'success_error':
        condition1_trial = np.where(meta['feedbackType'] == 1)[0]
        condition2_trial = np.where(meta['feedbackType'] == -1)[0]
    elif condition == 'PrevSuccess_PrevFail':
        condition1_trial = np.where(meta['feedbackType'].shift(1) == 1)[0]
        condition2_trial = np.where(meta['feedbackType'].shift(1) == -1)[0]  
    elif condition == 'expected_unexpected_stim':
        condition1_trial = np.where(((meta['probabilityLeft'] == 0.8) & (meta['contrastLeft'] > 0.2)) |
                                    ((meta['probabilityLeft'] == 0.2) & (meta['contrastRight'] > 0.2)))[0]
        condition2_trial = np.where(((meta['probabilityLeft'] == 0.8) & (meta['contrastRight'] > 0.2)) |
                                    ((meta['probabilityLeft'] == 0.2) & (meta['contrastLeft'] > 0.2)))[0]
    elif condition == 'expected_unexpected_NoStim':
        condition1_trial = np.where(((meta['probabilityLeft'] == 0.8) & (meta['contrastLeft'] < 0.1)) |
                                    ((meta['probabilityLeft'] == 0.2) & (meta['contrastRight'] < 0.1)))[0]
        condition2_trial = np.where(((meta['probabilityLeft'] == 0.8) & (meta['contrastRight'] < 0.1)) |
                                    ((meta['probabilityLeft'] == 0.2) & (meta['contrastLeft'] < 0.1)))[0]
    elif condition == 'Right_left_choice':
        condition1_trial = np.where(meta['choice'] == 1)[0]
        condition2_trial = np.where(meta['choice'] == -1)[0]
    else:
        raise ValueError('Invalid condition')

    
    change_indices = meta['probabilityLeft'].ne(meta['probabilityLeft'].shift()).to_numpy().nonzero()[0]
    change_indices =  change_indices[1:] # remove the first change
    change_indices_10 = []
    for change in change_indices:
        change_indices_10.extend(range(change, change + 11))


    if remove_first_trials_of_block:
        condition1_trial = [i for i in condition1_trial if i not in change_indices_10]
        condition2_trial = [i for i in condition2_trial if i not in change_indices_10]

    if len(condition1_trial) <= min_trial or len(condition2_trial) <= min_trial:
        print(f'Not enough trials for condition {condition}')
        return 0 
    epochs_1 = epoch[condition1_trial]
    epochs_2 = epoch[condition2_trial]

    return epochs_1, epochs_2