# Right-Left Stimuli Selectivity Analysis

This project provides tools to analyze channel and cluster selectivity in response to right- or left-side visual stimuli using IBL datasets. The analysis aims to identify neural responses with selectivity based on firing rates and gamma power, using time-based comparisons across depth levels and clusters.

## Structure

### Functions

-   **compute_decodability.py**: Computes the decodability of each channel or cluster using the Area Under the ROC Curve (AUC).
-   **firing_rate.py**: Calculates firing rates across specified conditions, time periods, and depths.

### Notebooks

-   **SpikesBasedSelectivity_OnClusters_example.ipynb**: Demonstrates a spiking-based selectivity analysis across clusters and time.
-   **SpikesBasedSelectivity_OnDepths_example.ipynb**: Demonstrates a spiking-based selectivity analysis across depths and time
-   **Spikes_based_selectivity_overall.ipynb**: Computes selectivity across multiple probe IDs (PIDs) and session IDs (EIDs) for a given list, analyzing responses across brain regions.

## Usage

### 1. Environment Setup

Ensure the IBL environment is installed and activated.

### 2. Data Extraction

Data can be loaded in two modes:

-   **Load mode**: If you have already downloaded the relevant IBL Ephys datasets, set `mode='load'` and specify the data directory with `path='path/to/datasets'`.
-   **Download mode**: If datasets are not pre-downloaded, set `mode='download'` to directly retrieve data from the IBL server (note: data is extracted but not saved to your local directory).

## Authors and Acknowledgments

**Author**: Mohammad Keshtkar\
**Email**: mohammad.m.keshtkar\@gmail.com

This repository is developed during my work at the Lyon Neuroscience Research Center (CRNL) under the supervision of Dr. Romain Ligneul. Please note that this repository is actively evolving, with certain analyses and computations still awaiting review by my supervisor.