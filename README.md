bibliography: references.bib
---

# Overview

This project is dedicated to analyzing International Brain Laboratory (IBL) Electrophysiology datasets. It is Developed during my time at the Lyon Neuroscience Research Center (CRNL) under the supervision of Dr. Romain Ligneul (March 2024 - Dec 2024). The repository is actively evolving, with some analyses still under review by my supervisor.

# Goals

According to predictive coding and communication by oscillations theories, high-frequency feedforward (FF) and low-frequency feedback (FB) oscillations may contribute to the hierarchical processing of sensory stimuli by generating predictions and attaching behavioral context to the sensory world [@aggarwal2022]. Yet, the role of these oscillations in shaping perception remains poorly under stood, particularly in mice. The International Brain Laboratory (IBL) open-access datasets, with their sophisticated experimental paradigms and large-scale neural recordings, provide an unprecedented opportunity to study the principles of predictive coding in mice. In this project, we analyzed the IBL electrophysiology recordings (LFP, spikes) from the visual areas as a first step towards utilizing this dataset to understand the principles of predictive coding and communication through oscillations.

# IBL Task

The International Brain Laboratory (IBL) [@benson2023] provides an extensive open-access dataset recorded from more than 100 mice trained to perform a perceptual decision-making task. In this task, mice are presented with a visual stimulus of controlled contrast and are required to move the stimulus to the center of the screen using a steering wheel. The stimulus appears on the right or left side of the screen, with a fixed probability for blocks of trials to create a predictable pattern. Yet, these bias blocks change unpredictably, requiring the mice to constantly update their predictions and internal model of the environment.

# Contents (From Latest to Older)

The more recent sub-projects in this repository are structured with greater code organization and comprehensive documentation, reflecting improvements in coding practices and project management over time.

-   **Right-Left Selectivity Analysis**
-   **Receptive Field Mapping**
-   **Time_Frequency_Representations(TFR)**
-   **Extraction**
-   **Decoding on Time-Frequency Representations (TFR)**
-   **Phase-Amplitude Coupling (PAC) Analysis**
-   **Behavioral Analysis**

You can find my Master thesis, defended in September 2024, in the [Writings folder](./Writings/)

# Usage

-   Install IBL dependencies (see [here](https://github.com/int-brain-lab/iblenv))

-   The majority of project is based on computing on the server and using parallel computation with Submitit module. However, such in the right_left_selectivity project, the scripts will be modified to be able to run it on both local and remote server.

-   More detail will be added soon

# Authors and Acknowledgments

-   **Author**: Mohammad Keshtkar [mohammad.m.keshtkar\@gmail.com](mohammad.m.keshtkar@gmail.com)
-   **Supervisor**: Dr. Romain Ligneul [romain.ligneul\@inserm.fr](romain.ligneul@inserm.fr)