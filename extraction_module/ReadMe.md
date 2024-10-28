# README

## Project Overview

This sub-project is designed to:

1.  Generate a table listing IBL sessions that contain recordings for a specified pair of brain regions.
2.  Extract various datasets (behavioral, wheel, DLC, spikes, and LFP) for these sessions.

## Project Structure

-   **eid_probe_info_table.py**: Generates a table containing session IDs and the count of channels for specified brain regions.
-   **submit_table_generation.py**: Submits jobs to generate the table of session IDs and channel counts.
-   **data_extraction.py**: Extracts various types of data (behavior, wheel, DLC, spikes, LFP) for specified sessions and probes.
-   **submit_extraction.py**: Submits jobs to extract data for the specified brain region pairs.
-   **config.py**: Contains configuration parameters for the project, including brain region pairs, extraction parameters, and `submitit` parameters.
-   **data/** directory for saving tables
-   logs/ directory to save the log files

## Usage

### 1. Configuration

Before running any scripts, ensure that the configuration parameters in `config.py` are set according to your requirements. Key parameters include:

-   `BRAIN_REGION_PAIRS`: Pairs of brain regions to analyze.
-   `minimum_number_of_channels`: Minimum number of channels per region to consider for a session.
-   `extraction_parameters`: Parameters for data extraction.
-   `SUBMITIT_PARAMS`: Parameters for job submission using `submitit`.

### 2. Make sure the IBL environment is activate in the terminal

In your terminal (connected to the cluster), ensure that the IBL environment is activated. You will also need to install the `submitit` library if it is not already installed. and you should cd to the extraction_module directory

``` bash
conda activate iblenv
pip install submitit
cd extraction_module
```

### 3. Generate Session Table

Run `python submit_table_generation.py` to generate a table containing session IDs and channel counts for the specified brain region pairs. This script will submit jobs to generate the table and save it as a CSV file.

### 4. Extract Data

Run `python submit_extraction.py` to extract data for the specified brain region pairs. This script will submit jobs to extract various types of data (behavior, wheel, DLC, spikes, LFP) for the sessions and probes listed in the generated table.

## Notes

-   The `BRAIN_REGION_PAIRS` parameter in `config.py` should contain only one pair of brain regions for the `data_extraction.py` script.
-   The `submit_table_generation.py` and `submit_extraction.py` scripts use the `submitit` library to submit jobs for parallel processing. Adjust the `SUBMITIT_PARAMS` in `config.py` according to your computational resources. For example, to generate the tables you do not need high memory limit but for data extraction the memory limit should be at least 16 GB.

##