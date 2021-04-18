"""
Functions for working with GOES ABI imagery.

Steven Pestana, April 2021 (spestana@uw.edu)

"""

import os
import pandas as pd

def goes_timestamps(goes_file_list, ext='tif', fn_sep='_', fn_sep_number=-4):
    # create an empty dictionary we'll fill with filenames and timestamps for each subdirectory we search
    goes_dict = {}
    
    for this_goes_file in goes_file_list:
    
        this_goes_filename = os.path.normpath(this_goes_file).split('\\')[-1]
        if this_goes_filename.split('.')[-1] == ext:
            
            # parse the timstamp in the filename 
            this_goes_datetime_UTC = this_goes_filename.split(fn_sep)[fn_sep_number].split('.')[0][1:-1]
            #print(this_goes_datetime_UTC)
            this_goes_datetime_UTC = pd.to_datetime(this_goes_datetime_UTC, format="%Y%j%H%M%S")
            this_goes_datetime_UTC = pd.Timestamp(this_goes_datetime_UTC, tz='UTC')
            #print('\t{}'.format(this_goes_datetime_UTC))
    
            # add these to our dictionary, use the date as the key
            goes_dict[this_goes_datetime_UTC] = {}
            goes_dict[this_goes_datetime_UTC]['filepath'] = this_goes_file
            
    return goes_dict