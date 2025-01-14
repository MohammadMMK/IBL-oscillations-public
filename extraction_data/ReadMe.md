## Overview

This sub-project is designed to define functions for extracting data from the IBL database. The functions can extract various types of data, including behavior, wheel, DLC, spikes, and LFP, for a given experiment ID (eid) and probe label.

### LFP Data Extraction

Due to the large size of LFP data, retrieving it every time can be time-consuming. It is recommended to use the `get_LFP_data.ipynb` Jupyter notebook to extract and save the LFP data, facilitating faster access for future use. No need to download and save other datasets, we download them directly from IBL server during further analysis as they have light size. However, if you prefer, you can download them and use 'load' mode instead of 'download' mode. For example you want to get spikes datasets:

``` python
from functions import get_spikes
spikes = get_spikes(pid, modee='load', path= 'path/to/already_saved/spikes')
```

### New LFP Extraction

a)  The new function extracts LFP data as epochs around the stimulus onset and saves it. This approach contrasts with the previous function, which extracted LFP data for one second before and after the first and last trial. The new version is believed to be more accurate for converting the trial clock to each probe clock.

b)  The LFP is saved as NPY due to RAM limitation. see the last blocks of `get_LFP_data.ipynb` to convert to MNE

c)  you can change the path to save the LFP data at the `_analyses/config.py` based on your operating system.

Note: Bad channels and bad trials are not computed in the raw epoched LFP

## Main Functions

`functions/get_data.py`:

1- `get_behavior`

2- `get_spikes`

3- `get_channels` channels information for a given probe

4- `get_epoch_StimOn`

5- `get_pid_eid_pairs`