# Acknowledgments**Time-Frequency Representations (TFR)**

The goal of this project is to compute and visualize TFR for different task conditions across Visual processing related regions.

## **Regions:**

-   primary visual area (VISp) **(V1)**
-   Anteromedial visual area (VISam) **(AM)**
-   lateral visual area (VISl) **(LM)**

## **Conditions:**

-   **Bias Blocks**: Right-bias trials (probability_left = 0.2) and left-bias trials (probability_left = 0.8). The first ten trials of each bias block are excluded to ensure the mouse is likely aware of the block it is in.

-   **Expected/Unexpected**: Trials where the stimulus appears on the side matching the bias block (e.g., on the right when the probability for the right side is 80%) are classified as *expected*, while those appearing on the opposite side are *unexpected*. The first ten trials of each bias block and trials with contrast levels below 20% are excluded.

-   **Previous Success/Fail**: Trials where the previous trial was successful or unsuccessful.

-   **Stim/NoStim**: Trials with high contrast (100%) stimulus and low contrast (\<10%) stimulus.

## **Time-Frequency Parameters:**

-   `freqs` = `np.concatenate([np.arange(1, 10, 0.5), np.arange(10, 45, 1)])`
-   `n_cycles` = `freqs / 2.`
-   `time_bandwidth` = 3.5

## **Usage:**

Begin by downloading the LFP data for your sessions of interest using the [data extraction project](https://github.com/MohammadMMK/IBL_projects/tree/main/extraction/). (A future update will enable direct analysis by loading data directly from the IBL server.)

Current scripts rely on parallel computation using Submitit on a server (local computation will be added soon).

**Authors and Acknowledgments:**

-   **Author**: Mohammad Keshtkar
-   **Email**: mohammad.m.keshtkar\@gmail.com

This repository is developed during my work at the Lyon Neuroscience Research Center (CRNL) under the supervision of Dr. Romain Ligneul. Please note that this repository is actively evolving, with certain analyses and computations still awaiting review by my supervisor.