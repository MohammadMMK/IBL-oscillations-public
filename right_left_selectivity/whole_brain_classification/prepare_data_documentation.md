
# `prepare_data` Function

## Goal

The primary goal of the `prepare_data` function is to generate a structured DataFrame for each probe recording session (pid) that includes the firing rates and associated metadata for right and left stimulus trials. This prepared dataset serves as input for machine learning models aimed at identifying brain areas that exhibit selectivity for right or left stimuli presentations. By averaging firing rates across a specified period, the function reduces dimensionality while retaining critical trial-by-trial information.

## Workflow and Data Preparation

1. **Extract Firing Rates**: Using `right_left_firingRates_onDepths`, we extract firing rates for trials where stimuli appeared on the right and left. These rates are then standardized (z-scored) and averaged over a predefined time window (e.g., 0.1–1 seconds post-stimulus).
  
2. **Metadata Preparation**: The function compiles essential metadata for each channel and trial, such as:
   - **Spatial Coordinates** (`x`, `y`, `z`): Provides the 3D location of each channel.
   - **Brain Acronym**: Labels each channel with its associated brain area acronym.
   - **Trial Index**: Indicates the sequential position of each trial within the experiment, allowing insights into session progression.
   - **Distance to Change**: Measures how far each trial is from the most recent bias block change, helping capture adaptation in responses.
   - **Bias Block Probability (`prob_left`)**: Reflects the probability that the stimulus appeared on the left, based on bias blocks.

3. **Data Structuring**: The function consolidates firing rates and metadata into a single DataFrame. This structure supports efficient querying and compatibility with machine learning frameworks for analysis.

## Output DataFrame Structure

The final DataFrame contains the following columns:

| Column           | Description                                                                            |
|------------------|----------------------------------------------------------------------------------------|
| `firing_rate`    | Averaged firing rate per trial and channel.                                            |
| `label`          | Binary label indicating right (1) or left (0) trials.                                  |
| `trial_index`    | Sequential index of each trial in the session.                                         |
| `distance_to_change` | Distance in trials to the last change in bias block.                               |
| `prob_left`      | Bias block probability for left-side stimulus occurrence.                              |
| `x`, `y`, `z`    | 3D spatial coordinates for each channel.                                               |
| `ch_index`       | Channel index within the probe.                                                        |
| `acronym`        | Brain region acronym for the channel.                                                  |
| `probe_id`       | Unique identifier for the probe.                                                       |
| `experiment_id`  | Unique identifier for the experimental session.                                        |

## Usage and Future Steps

The resulting DataFrames from all probe sessions (742 probes) can be merged into a comprehensive dataset, suitable for input to machine learning classifiers. The inclusion of spatial, temporal, and probabilistic features enables a flexible selection of input variables to explore various aspects of neural selectivity and regional stimulus sensitivity.



The above documentation is build with help of chatGPT 