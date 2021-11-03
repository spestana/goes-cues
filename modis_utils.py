"""
Functions for working with MODIS emissive band imagery.

Steven Pestana, Feb 2021 (spestana@uw.edu)

"""

import xarray as xr
import numpy as np
from pyspectral.blackbody import blackbody_rad2temp


   

def emissive_convert_dn(modis_src):
    '''Convert MODIS Emissive MxD021KM Digital Number ("scaled integer") values
    to Radiance and Brightness Temperature.
    
    Inputs:
        modis_src = xr.open_rasterio(MxD021KM_filepath)'''

    center_wavelengths = np.array([3.75, 3.959, 3.959, 4.05, 4.4655, 4.5155, 6.715, 7.325, 8.55, 9.73, 11.03, 12.02, 13.335, 13.635, 13.935, 14.235]) * 1e-6
    
    # unpack scales and offsets from metadata
    scales = modis_src.radiance_scales
    scales = np.array( [float(s) for s in scales.split(',')] )
    offsets = modis_src.radiance_offsets
    offsets = np.array( [float(o) for o in offsets.split(',')] )
    
    # make empty np arrays of the same shape as the DN array
    modis_rad = np.empty_like(modis_src.values)
    modis_tb = np.empty_like(modis_src.values)
    modis_tb_c = np.empty_like(modis_src.values)
    
    # replace nodata values (0) with np.nan in DN array
    modis_src_masked = modis_src.where(modis_src!=0)
    
    for band in range(modis_src_masked.shape[0]):
        # apply the scale and offset to convert from DN to radiance
        # radiance = radiance_scale*( scaled_integer - radiance_offset ) 
        modis_rad[band] = scales[band] * ( modis_src_masked.isel(band=band) - offsets[band] )
        # use pyspectral function to convert radiance to brightness temperature
        modis_tb[band] = blackbody_rad2temp(center_wavelengths[band], modis_rad[band]*1e6) # multiply radiance here by 1e6 to convert micron to meter
        # for brightness temperature in C
        modis_tb_c[band] = modis_tb[band] - 273.15
    
    # set brightness temperature array to nan where it should be masked out
    modis_tb[np.isnan(modis_rad)] = np.nan
        
    # edit attributes for new radiance data array
    rad_attrs = modis_src_masked.attrs.copy()
    rad_attrs['long_name'] = 'Radiance'
    # make a data array out of the radiance np array
    modis_rad_da = xr.DataArray(modis_rad, dims=["band", "y", "x"], 
                                coords=[modis_src_masked.band.values, 
                                        modis_src_masked.y.values, 
                                        modis_src_masked.x.values], 
                                attrs=rad_attrs)
    
    # edit attributes for new brightness temperature data array
    tb_attrs = modis_src_masked.attrs.copy()
    tb_attrs['long_name'] = 'Brightness Temperature'
    # make a data array out of the brightness temperature np array
    modis_tb_da = xr.DataArray(modis_tb, dims=["band", "y", "x"], 
                                coords=[modis_src_masked.band.values, 
                                        modis_src_masked.y.values, 
                                        modis_src_masked.x.values], 
                                attrs=tb_attrs)
    # tb in C
    modis_tb_c_da = xr.DataArray(modis_tb_c, dims=["band", "y", "x"], 
                                coords=[modis_src_masked.band.values, 
                                        modis_src_masked.y.values, 
                                        modis_src_masked.x.values], 
                                attrs=tb_attrs)
    
    # combine the original DN array and new Radiance data arrays into a single dataset
    modis_ds = xr.Dataset(data_vars={"dn": modis_src_masked, 
                                     "radiance": modis_rad_da, 
                                     "tb": modis_tb_da,
                                     "tb_c": modis_tb_c_da},
                         attrs=modis_src_masked.attrs)
    
    return modis_ds
