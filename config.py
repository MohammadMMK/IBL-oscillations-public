# change the paths based on the operating system
import platform
import os 


# either 'download' or 'load' the LFP data during analysis
LFP_mode = 'download'

system = platform.system()
# windows
if system == 'Windows':
    LFP_dir = r"C:\Users\gmoha\OneDrive\文档\ibl_data"
# linux
if system == 'Linux':
    if 'workspaces' in os.getcwd():
        # on git codespace
        LFP_dir = '/workspaces/onedrive1/文档/ibl_data'
    else:
        # on crln server
        LFP_dir = '/mnt/data/AdaptiveControl/IBLrawdata/LFP/'


REST_cache_expirry_min = 2