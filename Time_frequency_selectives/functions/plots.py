import matplotlib.pyplot as plt

def plot_all_tf_by_layers(tf_data_list, title, region_prefix='VISp'):
    """
    Plot all TF channels from multiple sessions arranged by layer.

    The channels are grouped by layer (extracted from the channel's acronym by removing
    the region prefix). The expected layer order is:
        ["1", "2/3", "4", "5", "6a", "6b"]

    If the maximum number of channels per layer is more than 5, each layer is plotted
    in a separate figure with a maximum of 5 columns and any needed number of rows.
    Layers with no channels are removed from the plot.

    Parameters
    ----------
    tf_data_list : list of dict
        List of TF dictionaries (one per channel).
    title : str
        The title for the plot(s).
    region_prefix : str, optional
        The prefix of the region in the channel acronym (default is "VISp").
    """
    # Define the expected layer order.
    layer_order = ["1", "2/3", "4", "5", "6a", "6b"]

    # Group the channels by layer.
    layers_dict = {layer: [] for layer in layer_order}
    for item in tf_data_list:
        acro = item['acronym']
        if acro.startswith(region_prefix):
            # Extract the layer string; for example, "VISp2/3" -> "2/3"
            layer = acro[len(region_prefix):].strip()
            if layer in layers_dict:
                layers_dict[layer].append(item)
            else:
                print(f"Warning: channel with acronym '{acro}' does not match expected layers.")
        else:
            print(f"Warning: channel acronym '{acro}' does not start with '{region_prefix}'.")

    # Determine the maximum number of channels across layers
    max_channels = max(len(ch_list) for ch_list in layers_dict.values())

    # If max_channels > 5, plot each layer in a separate figure
    if max_channels > 5:
        for layer in layer_order:
            channels = layers_dict[layer]
            if not channels:
                continue

            # Calculate the number of rows needed for this layer
            n_rows = (len(channels) + 4) // 5  # Ceiling division to ensure all channels fit
            n_cols = 5

            # Create a new figure for this layer
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
            fig.suptitle(f"{title} - Layer {layer}")

            # Plot each channel in the layer
            for i, item in enumerate(channels):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col]

                tf_data = item['TF']         # shape: (n_freqs, n_times)
                times = item['times']
                freqs = item['freqs']
                accuracy_right = item['accuracy_right']
                accuracy_left = item['accuracy_left']
                pvalue_right = item['pvalue_right']
                pvalue_left = item['pvalue_left']
                acro = item['acronym']
                pid = item['pid']

                # Plot the TF data.
                cax = ax.pcolormesh(times, freqs, tf_data, shading='auto', cmap='RdBu_r')
                # Optionally, add a colorbar for each subplot.
                fig.colorbar(cax, ax=ax)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Frequency (Hz)")
                ax.set_title(f"{acro}\nAcc_R: {accuracy_right:.2f}, pVal_R: {pvalue_right:.3f},\n Acc_L: {accuracy_left:.2f}, pVal_L: {pvalue_left:.3f}")

            # Hide any extra subplots in this figure
            for i in range(len(channels), n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axes[row, col].axis('off')

            plt.tight_layout()
            plt.show()

    else:
        # If max_channels <= 5, plot all layers in one figure
        # Filter out layers with no channels
        layers_with_channels = {layer: channels for layer, channels in layers_dict.items() if channels}
        filtered_layer_order = [layer for layer in layer_order if layer in layers_with_channels]

        n_layers = len(filtered_layer_order)
        fig, axes = plt.subplots(n_layers, max_channels, figsize=(4 * max_channels, 3 * n_layers), squeeze=False)
        fig.suptitle(title)

        # Loop over layers (rows)
        for i, layer in enumerate(filtered_layer_order):
            channels = layers_with_channels[layer]
            # For each column in the current row:
            for j in range(max_channels):
                ax = axes[i, j]
                if j < len(channels):
                    item = channels[j]
                    tf_data = item['TF']         # shape: (n_freqs, n_times)
                    times = item['times']
                    freqs = item['freqs']
                    accuracy_right = item['accuracy_right']
                    accuracy_left = item['accuracy_left']
                    pvalue_right = item['pvalue_right']
                    pvalue_left = item['pvalue_left']
                    acro = item['acronym']
                    pid = item['pid']

                    # Plot the TF data.
                    cax = ax.pcolormesh(times, freqs, tf_data, shading='auto', cmap='RdBu_r')
                    # Optionally, add a colorbar for each subplot.
                    fig.colorbar(cax, ax=ax)
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Frequency (Hz)")
                    ax.set_title(f"{acro}\nAcc_R: {accuracy_right:.2f}, pVal_R: {pvalue_right:.3f},\n Acc_L: {accuracy_left:.2f}, pVal_L: {pvalue_left:.3f}")
                else:
                    # Hide any extra subplots in this row.
                    ax.axis('off')

        plt.tight_layout()
        plt.show()