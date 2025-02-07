def plot_time_frequency_diff( diff, times, freqs, ch_names, BL_n_trials, BR_n_trials, side_selective , selected_regions, order):
    layers = ['1', '2/3', '4', '5', '6a', '6b']

    # Plot for each region by layer and channel
    for region in selected_regions:
        print(f"Processing region: {region}")
        fig, axes_dict = {}, {}
        
        for layer in layers:
            # Find channels for the specific region and layer
            layer_pattern = rf'^{region}{layer}$'
            keep_ch = [i for i, ch in enumerate(ch_names) if re.match(layer_pattern, ch)]
            
            if not keep_ch:  # Skip if no data for the layer
                continue
            
            n_channels = len(keep_ch)
            ncols = 4 if n_channels > 1 else 1
            nrows = (n_channels + ncols - 1) // ncols  # Calculate required rows
            
            # Create a figure for each layer
            fig[layer], axes_dict[layer] = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(5 * ncols, 5 * nrows),
                squeeze=False,
            )
            # fig[layer].suptitle(f' {region} : Left - Right Blocks for {side_selective} selective', fontsize=16)
            
            for i, ch_idx in enumerate(keep_ch):
                row, col = divmod(i, ncols)
                ax = axes_dict[layer][row, col]
                im = ax.imshow(
                    diff[ch_idx, :, :],  # Use the individual channel's data
                    aspect='auto',
                    extent=[times[0], times[-1], freqs[0], freqs[-1]],
                    origin='lower',
                    cmap='RdBu_r',
                )
                if order == 'Left-Right':
                    ax.set_title(f'L{layer}, nTrials ( {BL_n_trials[ch_idx]} -{BR_n_trials[ch_idx]})', fontsize=12)
                elif order == 'Right-Left':
                    ax.set_title(f'L{layer}, nTrials ({BR_n_trials[ch_idx]} - {BL_n_trials[ch_idx]})', fontsize=12)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Frequency (Hz)')
                fig[layer].colorbar(im, ax=ax, orientation='vertical', label='Power Difference')
            
            # Hide any empty subplots
            for idx in range(n_channels, nrows * ncols):
                row, col = divmod(idx, ncols)
                fig[layer].delaxes(axes_dict[layer][row, col])
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout
            plt.show()