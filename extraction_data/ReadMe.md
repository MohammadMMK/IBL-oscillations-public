# Overview

This sub-project is designed to define functions for extracting data from the IBL database. The functions can extract various types of data, including behavior, spikes, channels, and LFP, for a given experiment ID (eid) and probe label.

## New LFP Extraction

1.  The new function extracts LFP data as epochs around the stimulus onset and saves it. This approach contrasts with the previous function, which extracted LFP data for one second before and after the first and last trial. The new version is believed to be more accurate for converting the trial clock to each probe clock.

2.  You can download and save data using the `submit_LFP_extraction.ipynb` notebook. The path to LFP folder can be modify in the `_analyses/config.py` based on your operating system.

3.  You can also skip this step and in further processing download each time the part of LFP data that you are interested in. To do this, change the `LFP_mode` variable to `'download'` in the `_analyses/config.py`

4.  In the new update you can download/load only the selected trials:

``` {.python .Python}
from extraction_data import get_epoch_StimOn
BiasLeft_filter = { 'tmin': -1, 'tmax': 0, 'contrasts': 'all',
                          'stim_side': "both", 'prob_left': [0.8], 'remove_first_trials_of_block': True}
epochs = get_epoch_StimOn(pid, **BiasLeft_filter) # in MNE format
```

Note: Bad channels and bad trials are not computed in the raw epoched LFP

## Other datasets

``` python
from extraction_data import get_spikes, get_behavior, get_channels 
spikes = get_spikes(pid, modee='load', path= 'path/to/already_saved/spikes')
behavior = get_behavior(eid,modee='load', path= 'path/to/already_saved/behavior' )
channels = get_channels(pid)
```

## Listing sessions

listing sessions that include **any** regions in a list:

``` python
from extraction_data import get_pid_eid_pairs
pid_eid_pairs_list = get_pid_eid_pairs(only_passive=True, regions = ['VISp', 'VISpm', 'VISam', 'VISa', 'VISrl', 'VISal', 'VISli', 'VISl'])
# only_passive means including sessions that passive recordings are available
```

## Main Functions

`functions/get_data.py`:

1- `get_behavior`

2- `get_spikes`

3- `get_channels` channels information for a given probe

4- `get_epoch_StimOn`

5- `get_pid_eid_pairs`