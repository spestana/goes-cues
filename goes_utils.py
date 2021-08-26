"""
Functions for working with GOES ABI imagery.

Steven Pestana, April 2021 (spestana@uw.edu)

"""

import os
import pandas as pd
import numpy as np

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

def goesBrightnessTemp(rad, band=None, fk1=None, fk2=None, bc1=None, bc2=None): 
    # Convert Radiance (in [mW m-2 sr-1 (cm-1)-1]) to Brightness Temperature for GOES-R ABI emissive bands
    
    # values from the GOES-R Product Definition and User’s Guide
    # Table 7.1.3.6.1.1-2 Radiances to Brightness Temperature Planck Constants
    planck_constants = {7:  {'fk1':2.02263e+05, 'fk2':3.69819e+03, 'bc1':0.43361, 'bc2':0.99939},
                        8:  {'fk1':5.06871e+04, 'fk2':2.33158e+03, 'bc1':1.55228, 'bc2':0.99667},
                        9:  {'fk1':3.58283e+04, 'fk2':2.07695e+03, 'bc1':0.34427, 'bc2':0.99918},
                        10: {'fk1':3.01740e+04, 'fk2':1.96138e+03, 'bc1':0.05651, 'bc2':0.99986},
                        11: {'fk1':1.97799e+04, 'fk2':1.70383e+03, 'bc1':0.18733, 'bc2':0.99948},
                        12: {'fk1':1.34321e+04, 'fk2':1.49761e+03, 'bc1':0.09102, 'bc2':0.99971},
                        13: {'fk1':1.08033e+04, 'fk2':1.39274e+03, 'bc1':0.07550, 'bc2':0.99975},
                        14: {'fk1':8.51022e+03, 'fk2':1.28627e+03, 'bc1':0.22516, 'bc2':0.99920},
                        15: {'fk1':6.45462e+03, 'fk2':1.17303e+03, 'bc1':0.21702, 'bc2':0.99916},
                        16: {'fk1':5.10127e+03, 'fk2':1.08453e+03, 'bc1':0.06266, 'bc2':0.99974}}
    
    # if the function is supplied a band number, use values from the table above
    if band!=None:
        assert band in range(7,17), "Band number must be 7-16"
        fk1 = planck_constants[band]['fk1']
        fk2 = planck_constants[band]['fk2']
        bc1 = planck_constants[band]['bc1']
        bc2 = planck_constants[band]['bc2']
    # otherwise use the supplied constant values
    
    # compute brightness temperature
    Tb = ( fk2 / (np.log((fk1 / rad) + 1)) - bc1 ) / bc2
    
    return Tb

def planck(l, T):
    
    '''
    Planck function to compute the spectral radiance (B) at wavelength (l) for an object at temperature (T).
    Note that this does not take emissivity into account.
    (From Dozier & Warren, 1982)
    
    Inputs:
        l: wavelength [m]
        T: thermodynamic temperature [K]
        
    Output:
        B: spectral radiance [W m-2 sr-1 m-1]
    '''
    
    h = 6.63e-34 # planck's constant [Js]
    c = 299792458 #speed of light [m/s]
    k = 1.38e-23 # boltzmann's constant [J/K]
    
    B = (2 * h * (c**(2)) * (l**(-5))) / (np.exp( (h*c) / (k*l*T) ) - 1)
    
    return B

def abi_radiance_wavenumber_to_wavelength(goes, channel, rad_wn):
    ''' Convert GOES ABI Radiance units from
                                                mW / m^2 sr cm^-1
                                        to
                                                W / m^2 sr um
    Inputs
     - goes = 16 or 17 (int); GOES-16 or GOES-17
     - channel = 1-16 (int); GOES ABI channel/band
     - rad_wn = GOES ABI Radiance in "wavenumber" units [mW / m^2 sr cm^-1]
    Outputs
     - rad_wl = GOES ABI Radiance in "wavelength" units [W / m^2 sr um]
    '''
    
    # Read in Band Equivalent Widths file for GOES16 or GOES17
    eqw = pd.read_csv('GOES{goes}_ABI_ALLBANDS_MAR2016.eqw'.format(goes=str(goes)), sep='\s+', skiprows=1, index_col='CHANNEL')    
    
    # Divide milliwats by 1000 to get watts
    scale_milliwatts_by = 1000 
    
    # Convert units
    rad_wl = (rad_wn / scale_milliwatts_by) * (eqw['EQW(cm-1)'][channel]/eqw['EQW(um)'][channel])
    
    return rad_wl

def convert_radiance_units(rad_in, l):
    '''
    convert radiance
        rad_in: radiance in [W m-2 sr-1 m-1]
        l: wavelength [m]
        rad_out: radiance in [mW m-2 sr-1 (cm-1)-1]
    '''
        
    rad_out = 1000 * rad_in * ((l**2)/(10**-2))
    
    return rad_out