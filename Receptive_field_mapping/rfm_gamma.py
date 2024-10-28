def rfm_gamma(pid, eid, region):
    from scipy.signal import butter, filtfilt, hilbert
    import numpy as np
    import pandas as pd
    from one.api import ONE
    import brainbox.io.one as bbone
    import brainbox.task.passive as passive
    from brainbox.io.one import SpikeSortingLoader
    import ibldsp.voltage as voltage
    import mne

    def bandpass_filter(lfp_data, low_freq, high_freq, fs):
        nyquist = 0.5 * fs
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = butter(2, [low, high], btype='band')
        return filtfilt(b, a, lfp_data, axis=1)

    def compute_gamma_power(lfp_data, fs):
        gamma_band1 = bandpass_filter(lfp_data, 25, 45, fs)
        gamma_band2 = bandpass_filter(lfp_data, 55, 200, fs)
        combined_gamma = gamma_band1 + gamma_band2
        analytic_signal = hilbert(combined_gamma, axis=1)
        gamma_power = np.abs(analytic_signal) ** 2
        return gamma_power

    #################
    # initialize ONE
    #################
    one = ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True)
    one = ONE(password='international')
    ssl = SpikeSortingLoader(pid=pid, one=one)
    
    #################
    # Load channels data 
    #################
    path_channels = f'/mnt/data/AdaptiveControl/IBLrawdata/eid_data/{eid}/probe_{pid}.pkl'
    channels = pd.read_pickle(path_channels)
    pattern = fr'^{region}[12456]$'
    condition = channels['acronym'].str.contains(pattern)
    channels = channels[condition]
    channels = channels[channels['channel_labels'] == 0]
    indices = channels['Unnamed: 0'].values
    ch_coords = channels[['axial_um', 'lateral_um']].to_numpy()
    ch_dist = np.sqrt(np.sum(np.diff(ch_coords, axis=0)**2, axis=1))
    n_channels = len(indices)
    rf_map = np.zeros((15, 15, n_channels-2))

    #################
    # Load passive RF mapping data ( stimulus times, positions, etc. )
    #################
    map = bbone.load_passive_rfmap(eid, one=one)
    RF_frame_times, RF_frame_pos, RF_frame_stim = passive.get_on_off_times_and_positions(map)
    
    # Loop over each stimulus location (x, y) in the 15x15 grid
    for x in range(15):
        for y in range(15):

            #################
            # Extract stimulus times for this pixel
            #################
            pixel_idx = np.bitwise_and(RF_frame_pos[:, 0] == x, RF_frame_pos[:, 1] == y)
            stim_on_frames = RF_frame_stim['on'][pixel_idx]
            stim_on_times = RF_frame_times[stim_on_frames[0][0]]

            gamma_power_responses = []

            for t_event in stim_on_times:

                #################
                # LFP extraction for the stimulus event in passive recording
                #################
                sr_lf = ssl.raw_electrophysiology(band="lf", stream=True)
                sample_lf = int(ssl.samples2times(t_event, direction='reverse') // 12)
                window_secs_ap = [-0.4, 1]

                first, last = (int(window_secs_ap[0] * sr_lf.fs + sample_lf), int(window_secs_ap[1] * sr_lf.fs + sample_lf))
                raw_lf = sr_lf[first:last, :-sr_lf.nsync].T
                
                destriped = voltage.destripe_lfp(raw_lf.astype(float), fs=sr_lf.fs)
                decimated = mne.filter.resample(destriped.astype(float), up=2.0, down=10.0, window='boxcar', npad='auto', pad='reflect_limited', verbose=False)
                decimated = decimated[indices, :] # Select only the channels in the region and less noisy ones

                #################
                # CSD computation
                #################
                v_diff1 = np.diff(decimated[:-1, :], axis=0)
                v_diff2 = np.diff(decimated[1:, :], axis=0)
                ch_dist_mat_expanded = np.tile(ch_dist[:, np.newaxis], (1, decimated.shape[1]))
                csd = (v_diff2 / ch_dist_mat_expanded[1:]) - (v_diff1 / ch_dist_mat_expanded[:-1])
                
                #################
                # Compute gamma power for this stimulus event
                #################
                gamma_power = compute_gamma_power(csd, 500)
                gamma_power_responses.append(gamma_power)


            #################
            # baseline correct gamma power with z-score method ( baseline : 0.4 seconds duration before stimulus onset )
            #################

            # Convert gamma power responses to an array
            gamma_power_array = np.array(gamma_power_responses)
            # Compute baseline mean and std for z-score calculation
            baseline_duration = int(0.4 * 500)  # 0.4 seconds
            after_stim_duration = int(1.0 * 500)  # 1 second
            baseline_means = np.mean(gamma_power_array[:, :, :baseline_duration], axis=2)
            baseline_stds = np.std(gamma_power_array[:, :, :baseline_duration], axis=2)
            # Compute z-scores for 1 second after stimulus
            after_stim_gamma_power = gamma_power_array[:, :, baseline_duration:baseline_duration + after_stim_duration]
            z_scores = (np.mean(after_stim_gamma_power, axis=2) - baseline_means) / baseline_stds
            mean_z_scores = np.mean(z_scores, axis=0) # Average z-scores across stimulus events (approximately 8 to 12 events 'on' per pixel)
            #################
            # Save RF map
            #################
            rf_map[x, y, :] = mean_z_scores  

    # Save RF map 
    np.save(f'rf_map_{pid}_csd_zscore_gamma.npy', rf_map)
    return