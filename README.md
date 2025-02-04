This project is dedicated to analyzing International Brain Laboratory (IBL) Electrophysiology datasets. The repository is under active development. The repository is linked to our Mattermost channel through webhook.

# Introduction

According to predictive coding and communication by oscillations theories, high-frequency feedforward (FF) and low-frequency feedback (FB) oscillations may contribute to the hierarchical processing of sensory stimuli by generating predictions and attaching behavioral context to the sensory world [(Aggarwalet al. 2022)](https://www.nature.com/articles/s41467-022-32378-x). Yet, the role of these oscillations in shaping perception remains poorly under stood, particularly in mice. The International Brain Laboratory (IBL) open-access datasets, with their sophisticated experimental paradigms and large-scale neural recordings, provide an unprecedented opportunity to study the principles of predictive coding in mice. In this project, we analyzed the IBL electrophysiology recordings (LFP, spikes) from the visual areas as a first step towards utilizing this dataset to understand the principles of predictive coding and communication through oscillations.

# IBL Task

The International Brain Laboratory (IBL) [(Benson et al. 2023)](https://www.biorxiv.org/content/10.1101/2023.07.04.547681v2.abstract) provides an extensive open-access dataset recorded from more than 100 mice trained to perform a perceptual decision-making task. In this task, mice are presented with a visual stimulus of controlled contrast and are required to move the stimulus to the center of the screen using a steering wheel. The stimulus appears on the right or left side of the screen, with a fixed probability for blocks of trials to create a predictable pattern. Yet, these bias blocks change unpredictably, requiring the mice to constantly update their predictions and internal model of the environment.

# Usage

1.  **Clone the repository**\
    Clone this repository on CRLN drive or your local machine.

2.  **Set up the environment**\
    Install the IBL environment and dependencies (see instructions [here](https://github.com/int-brain-lab/iblenv)).

3.  **Install additional dependencies**
    Ensure you are at the root of the repository and run:
    ``` bash
    pip install --requirement requirements.txt
    ```
4.  **Install the project as a module locally**

    Ensure you are at the root of the repository and run:

    ``` bash
    pip install --requirement requirements.txt
    ```

4.  **Download LFP data**\
    Use the [`get_LFP_data.ipynb`](_analyses/extraction_data/get_LFP_data.ipynb) notebook in the [`extraction_data`](_analyses/extraction_data) module to download and save LFP data for your sessions of interest.

    -   The path to the LFP data is configured in [`config.py`](_analyses/config.py) and depends on your operating system.

    -   Refer to the corresponding [`ReadMe.md`](_analyses/extraction_data/ReadMe.md) file for detailed instructions.

5.  **Identify selective channels**\
    Use the [`submit_decoding.ipynb`](_analyses/Decoding_spikes/submit_decoding.ipynb) notebook in the [`Decoding_spikes`](_analyses/Decoding_spikes) module to find right/left selective channels.

6.  **Compute time-frequency representations (TFRs)**\
    Use the [`submit_TFR_computation.ipynb`](_analyses/Time_frequency_selectives/submit_TFR_computation.ipynb) notebook in the [`Time_frequency_selectives`](_analyses/Time_frequency_selectives) module to compute TFRs for different conditions (e.g., comparing TFRs for left stimulus presentations with different probabilities on left-selective electrodes).

## Quarto

Refer to the [documentation](https://cophyteam.github.io/project-template/about.html) of the Cophy project template to use Quarto. For convenience, the conda environments for Quarto are provided as .ylm files in \_functions/envs.

# Notes

-   The paths to the saved LFP datasets and the IBL cache are defined in the [`config.py`](_analyses/config.py) file and are OS-dependent.

-   To prevent duplicate files, ensure that if you update the save path on the server, you also relocate any existing files.

-   A lot of old codes that were not usable anymore have removed from the git repository to avoid confusion. However, the ones that we will need in future will be refine and uploaded soon.

    a- In the terminal, make sure you have activate the iblenv
    ``` bash
    conda activate iblenv
    ```
    b- cd to the _analyses folder where the setup file is located. 
    ``` bash
    cd ibl-oscillations/_analyses
    ```
    c- Then install with the option of editing in future:
    ``` bash
    pip install -e .
    ```
    
5.  **Download LFP data**\
    Use the [`get_LFP_data.ipynb`](_analyses/extraction_data/get_LFP_data.ipynb) notebook in the [`extraction_data`](_analyses/extraction_data) module to download and save LFP data for your sessions of interest.

    -   The path to the LFP data is configured in [`config.py`](_analyses/config.py) and depends on your operating system.

    -   Refer to the corresponding [`ReadMe.md`](_analyses/extraction_data/ReadMe.md) file for detailed instructions.
  
    -   **in the new updates of the scripts, the lfp will be saved as fif format (MNE format). Also there will be option to download directly from IBL server instead of loading from saved files**


6.  **Identify selective channels**\
    Use the [`submit_decoding.ipynb`](_analyses/Decoding_spikes/submit_decoding.ipynb) notebook in the [`Decoding_spikes`](_analyses/Decoding_spikes) module to find right/left selective channels.

7.  **Compute time-frequency representations (TFRs)**\
    Use the [`submit_TFR_computation.ipynb`](_analyses/Time_frequency_selectives/submit_TFR_computation.ipynb) notebook in the [`Time_frequency_selectives`](_analyses/Time_frequency_selectives) module to compute TFRs for different conditions (e.g., comparing TFRs for left stimulus presentations with different probabilities on left-selective electrodes).

## Quarto

Refer to the [documentation](https://cophyteam.github.io/project-template/about.html) of the Cophy project template to use Quarto. For convenience, the conda environments for Quarto are provided as .ylm files in \_functions/envs.

# Notes

-   The paths to the saved LFP datasets and the IBL cache are defined in the [`config.py`](_analyses/config.py) file and are OS-dependent.

-   To prevent duplicate files, ensure that if you update the save path on the server, you also relocate any existing files.

-   A lot of old codes that were not usable anymore have removed from the git repository to avoid confusion. However, the ones that we will need in future will be refine and uploaded soon.
