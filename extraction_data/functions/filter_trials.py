def filter_trials(trials_df, prob_left='all', contrasts='all', stim_side='right', remove_first_trials_of_block=False):
    """
    Filters trials based on probability left, contrast, stimulus side, and block transitions.

    Parameters:
    - trials_df (pd.DataFrame): The trials DataFrame (behavior metadata or MNE epochs metadata).
    - prob_left (list or 'all'): List of probability left values to keep.
    - contrasts (list or 'all'): List of contrast values to keep.
    - stim_side (str): 'right', 'left', or 'both'.
    - remove_first_trials_of_block (bool): If True, removes first 10 trials after a block change.

    Returns:
    - pd.DataFrame: Filtered trials DataFrame.
    """

    filtered_df = trials_df.copy()

    # Filter by probabilityLeft if not 'all'
    if isinstance(prob_left, list) and prob_left != 'all':
        filtered_df = filtered_df[filtered_df['probabilityLeft'].isin(prob_left)]

    # Filter by stim_side (right/left)
    if stim_side == 'right':
        filtered_df = filtered_df[~filtered_df['contrastRight'].isna()]
    elif stim_side == 'left':
        filtered_df = filtered_df[~filtered_df['contrastLeft'].isna()]

    # Filter by contrast values if not 'all'
    if isinstance(contrasts, list) and contrasts != 'all':
        if stim_side == 'right':
            filtered_df = filtered_df[filtered_df['contrastRight'].isin(contrasts)]
        elif stim_side == 'left':
            filtered_df = filtered_df[filtered_df['contrastLeft'].isin(contrasts)]

    # Remove first trials of each block if required
    if remove_first_trials_of_block:
        change_indices = filtered_df['probabilityLeft'].ne(filtered_df['probabilityLeft'].shift()).to_numpy().nonzero()[0]
        change_indices = change_indices[1:]  # Remove first block boundary
        remove_indices = [i for change in change_indices for i in range(change, change + 11)]
        remove_indices = [i for i in remove_indices if i < len(filtered_df)]
        filtered_df = filtered_df.drop(remove_indices).reset_index(drop=True)

    return filtered_df
