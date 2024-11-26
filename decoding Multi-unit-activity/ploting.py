import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


def plot_firing_rate_depth_and_ids(right_activity_mean, left_activity_mean, times, depths, ids):
    
    from brainbox.ephys_plots import plot_brain_regions

    true_depths = depths
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, width_ratios=[4, 1])  # 2 rows, 2 columns

    # First column, first row: Right Trials
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(right_activity_mean, aspect='auto', origin='lower',
                     extent=[times[0], times[-1], depths[0], depths[-1]], 
                     cmap='viridis')
    ax1.set_title('Firing Rate for Right Trials')
    ax1.set_ylabel('Depth (µm)')
    ax1.set_ylim(0, 3840)
    plt.colorbar(im1, ax=ax1, label='Z-score Firing Rate')

    # Second column, first row: Brain Regions for Right Trials
    ax2 = fig.add_subplot(gs[0, 1])  # Second column, first row
    plot_brain_regions(ids, true_depths, display=True, ax=ax2, 
                       title='Brain Regions for Right Trials', label='right')

    # First column, second row: Left Trials
    ax3 = fig.add_subplot(gs[1, 0])
    im2 = ax3.imshow(left_activity_mean, aspect='auto', origin='lower',
                     extent=[times[0], times[-1], depths[0], depths[-1]], 
                     cmap='viridis')
    ax3.set_title('Firing Rate for Left Trials')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Depth (µm)')
    ax3.set_ylim(0, 3840)
    plt.colorbar(im2, ax=ax3, label='Z-score Firing Rate')

    # Second column, second row: Brain Regions for Left Trials
    ax4 = fig.add_subplot(gs[1, 1])  # Second column, second row
    plot_brain_regions(ids, true_depths, display=True, ax=ax4, 
                       title='Brain Regions for Left Trials', label='left')

    # Layout adjustment
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()