
# Overview

This project is dedicated to analyzing International Brain Laboratory (IBL) Electrophysiology datasets. It is Developed during my time at the Lyon Neuroscience Research Center (CRNL) under the supervision of Dr. Romain Ligneul (March 2024 - Dec 2024). The repository is actively evolving, with some analyses still under review by my supervisor.

# Goals

According to predictive coding and communication by oscillations theories, high-frequency feedforward (FF) and low-frequency feedback (FB) oscillations may contribute to the hierarchical processing of sensory stimuli by generating predictions and attaching behavioral context to the sensory world [(Aggarwalet al. 2022)](https://www.nature.com/articles/s41467-022-32378-x). Yet, the role of these oscillations in shaping perception remains poorly under stood, particularly in mice. The International Brain Laboratory (IBL) open-access datasets, with their sophisticated experimental paradigms and large-scale neural recordings, provide an unprecedented opportunity to study the principles of predictive coding in mice. In this project, we analyzed the IBL electrophysiology recordings (LFP, spikes) from the visual areas as a first step towards utilizing this dataset to understand the principles of predictive coding and communication through oscillations.

# IBL Task

The International Brain Laboratory (IBL) [(Benson et al. 2023)](https://www.biorxiv.org/content/10.1101/2023.07.04.547681v2.abstract) provides an extensive open-access dataset recorded from more than 100 mice trained to perform a perceptual decision-making task. In this task, mice are presented with a visual stimulus of controlled contrast and are required to move the stimulus to the center of the screen using a steering wheel. The stimulus appears on the right or left side of the screen, with a fixed probability for blocks of trials to create a predictable pattern. Yet, these bias blocks change unpredictably, requiring the mice to constantly update their predictions and internal model of the environment.

# Methods

The project relies on a data processing pipeline built with Python. The key analyses include:

-   Time-frequency analysis: To investigate neural oscillatory patterns over time.

-   Decoding Task Variables from neural data: Using machine learning techniques (such as Support Vector Classifi cation, logistic regression, area under the ROC curve, and singular value decomposition) to decode variables like expectation and left vs. right stimulus from raw LFP, time-frequency data, and spike activity data.

-   Phase-Amplitude Coupling: To understand the relationship between the phase of low-frequency oscillations and the amp litude of higher frequencies.

-   Phase Analysis: To examine the role of phase coherence in neural dynamics.

-   Receptive Field Mapping: To explore the spatial response properties of neurons

Given the large-scale datasets, I focused on optimizing computations using cluster management tools like Submitit.

# Repository Contents (From Latest to Older)

-   [**Decoding_spikes**](./Decoding_spikes/)
-   [**Decoding on Time-Frequency Representations (TFR)**](./decoding_onTFR/)
-   [**Receptive Field Mapping**](./Receptive_field_mapping/)
-   [**Time_frequency_selectives**](./Time_frequency_selectives/)
-   [**extraction_data**](./extraction_data/)
-   [**Phase-Amplitude Coupling (PAC) Analysis**](./Phase_amplitude_coupling(PAC)/)
-   [**Behavioral Analysis**](./behavioral_analysis/)

You can find my Master thesis, defended in September 2024, in the [Writings folder](./Writings/)


# Usage

1.  **Clone the repository**\
    Clone this repository on CRLN drive or your local machine.

2.  **Set up the environment**\
    Install the IBL environment and dependencies (see instructions [here](https://github.com/int-brain-lab/iblenv)).

3.  **Install additional dependencies** Ensure you are at the root of the repository and run: `bash     pip install --requirement requirements.txt`

4.  **Install the project as a module locally**

    a- In the terminal, make sure you have activate the iblenv

    ``` bash
    conda activate iblenv
    ```

    b- cd to the \_analyses folder where the setup file is located.

    ``` bash
    cd ibl-oscillations/_analyses
    ```

    c- Then install with the option of editing in future:

    ``` bash
    pip install -e .
    ```

5.  **Download LFP data (optional)**

    If you wish to proceed the analysis by loading the LFP data from IBL server directly, in `config.py` file put `` LFP_mode = `download` ``.  The ONE instance will operate in remote mode without caching REST responses on disk By setting `cache_rest=None` and `mode='remote'` while initializing the ONE instance. See the [ONE documentation](https://int-brain-lab.github.io/ONE/notebooks/one_modes.html) for more information.

    Alternatively, you can Use the [`get_LFP_data.ipynb`](_analyses/extraction_data/get_LFP_data.ipynb) notebook in the [`extraction_data`](_analyses/extraction_data) module to download and save LFP data for your sessions of interest and then change teh `LFP_mode` to `load` in the `config.py` file. Then, you will load the LFP data from the saved files in further analyzis. Make sure the path to the saved LFP data is correct in the `config.py` file.

    -   Refer to the corresponding [`ReadMe.md`](_analyses/extraction_data/ReadMe.md) file for detailed instructions.

    -   In the new updates of the scripts, the LFP will be load as fif format (MNE format)

6.  **Identify selective channels**\
    Use the [`submit_decoding.ipynb`](_analyses/Decoding_spikes/submit_decoding.ipynb) notebook in the [`Decoding_spikes`](_analyses/Decoding_spikes) module to find right/left selective channels.

7.  **Compute time-frequency representations (TFRs)**\
    Use the [`submit_TFR_computation.ipynb`](_analyses/Time_frequency_selectives/submit_TFR_computation.ipynb) notebook in the [`Time_frequency_selectives`](_analyses/Time_frequency_selectives) module to compute TFRs for different conditions (e.g., comparing TFRs for left stimulus presentations with different probabilities on left-selective electrodes).

## Quarto

Refer to the [documentation](https://cophyteam.github.io/project-template/about.html) of the Cophy project template to use Quarto. For convenience, the conda environments for Quarto are provided as .ylm files in \_functions/envs.

# Notes

-   To prevent duplicate files, ensure that if you update the save path on the server, you also relocate any existing files.

-   A lot of old codes that were not usable anymore have removed from the git repository to avoid confusion. However, the ones that we will be need in future will be refine and uploaded soon.
