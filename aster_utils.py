"""
Functions for working with ASTER TIR imagery.

Steven Pestana, July 2020 (spestana@uw.edu)

"""

import pandas as pd
import numpy as np
import glob
import geopandas as gpd
import rasterio as rio
import rasterio.plot as rioplt
from rasterio.mask import mask
import xarray as xr
import xrspatial as xrs
import rioxarray


def tir_dn2rad(DN, band):
    '''Convert AST_L1T Digital Number values to At-Sensor Radiance for the TIR bands (bands 10-14).'''
    ucc = [6.822e-3, 6.780e-3, 6.590e-3, 5.693e-3, 5.225e-3]
    rad = (DN-1.) * ucc[band-10]
    return rad

def tir_rad2tb(rad, band):
    '''Convert AST_L1T At-Sensor Radiance to Brightness Temperature [K] for the TIR bands (bands 10-14).'''
    k1 = [3047.47, 2480.93, 1930.80, 865.65, 649.60]
    k2 = [1736.18, 1666.21, 1584.72,1349.82, 1274.49]
    tb = k2[band-10] /  np.log((k1[band-10]/rad) + 1)
    return tb

def aster_timestamps(directory, ext='hdf'):

    '''Given a directory of ASTER files, read the timestamps of ASTER observations (in UTC) from their filenames.
       Option to search for HDF of TIF files.
       
       Returns timestamp and filepath for each file found.'''
    
    assert (ext == 'hdf') or (ext == 'tif') , "File extension must be either hdf or tif"
    
    # Find all our ASTER files
    search_path = '{directory}/**/*.{ext}'.format(directory=directory, ext=ext)
    aster_files = glob.glob(search_path, recursive=True)
    
    # Create empty list to hold timestamps as we step through all files in the list
    aster_timestamps_UTC = []
    
    # for each filepath in the list of ASTER files
    for fpath in aster_files:
        # Parse the date and time from ASTER filename
        fn = fpath.split('/')[-1]
        MM = fn.split('_')[2][3:5]
        DD = fn.split('_')[2][5:7]
        YYYY = fn.split('_')[2][7:11]
        hh = fn.split('_')[2][11:13]
        mm = fn.split('_')[2][13:15]
        ss = fn.split('_')[2][15:17]
        # create pandas timestamp and append to the list
        aster_timestamps_UTC.append(pd.Timestamp('{}-{}-{} {}:{}:{}'.format(YYYY, MM, DD, hh, mm, ss),tz='UTC'))
    
    # Create pandas dataframe, sort, and reset index
    aster_df = pd.DataFrame({'timestampUTC': aster_timestamps_UTC, 'filepath': aster_files})
    aster_df.sort_values('timestampUTC',inplace=True)
    aster_df.reset_index(inplace=True, drop=True)
    
    return aster_df


def zonal_stats(aster_filepath, aster_band, shapefile_filepath, return_masked_array=False):
	'''Calculate zonal statistics for an ASTER TIR geotiff image within a single polygon from a shapefile.'''

	with rio.open(aster_filepath) as src:
		
		# Open the shapefile
		zone_shape = gpd.read_file(shapefile_filepath)

		# Make sure our shapefile is the same CRS as the ASTER TIR image
		zone_shape = zone_shape.to_crs(src.crs)

		# Mask the ASTER TIR image to the area of the shapefile
		try:
			masked_aster_band_DN, mask_transform = mask(dataset=src, 
											shapes=zone_shape.geometry,
											crop=True,
											all_touched=True,
											filled=True)
		# Note that we still have a "bands" axis (of size 1) even though there's only one band, we can remove it below
		except ValueError as e: 
			# ValueError when shape doesn't overlap raster
			return
		
		# change data type to float64 so we can fill in DN=0 with NaN values
		masked_aster_band_DN = masked_aster_band_DN.astype('float64')
		masked_aster_band_DN[masked_aster_band_DN==0] = np.nan
				
		# Convert DN to Radiance
		masked_aster_band_rad = tir_dn2rad(masked_aster_band_DN, aster_band)
		
		# Convert Radiance to Brightness Temperature
		masked_aster_band_tb = tir_rad2tb(masked_aster_band_rad, aster_band)
		
		# Remove the extra dimension (bands, we only have one band here)
		masked_aster_band_tb = masked_aster_band_tb.squeeze()
		
		# Get all pixel values in our masked area
		values = masked_aster_band_tb.flatten() # flatten to 1-D
		values = values[~np.isnan(values)] # remove NaN pixel values
		
		
		# Calculate zonal statistics for this area (mean, max, min, std:)
		try:
			masked_aster_band_tb_mean = values.mean()
			masked_aster_band_tb_max = values.max()
			masked_aster_band_tb_min = values.min()
			masked_aster_band_tb_std = values.std()
		except ValueError as e:
			# ValueError when the shapefile is empty I think
			return
		
		if return_masked_array == True:
			return masked_aster_band_tb_mean, masked_aster_band_tb_max, masked_aster_band_tb_min, masked_aster_band_tb_std, masked_aster_band_tb
		else:
			return masked_aster_band_tb_mean, masked_aster_band_tb_max, masked_aster_band_tb_min, masked_aster_band_tb_std


        
def mapZonalStats(zones, zonalstats, stat_name):
    ''' Function for mapping the zonal statistics back to the original grid to get a 2D map of the chosen statistic'''
    # create an empty array for this summary stat
    zonal_stat = np.zeros_like(zones.values, dtype=np.float64)

    # for each zone
    for zone_n in zonalstats.dim_0.values:
        # get the summary stat for that zone, 
        # and assign it to the correct locations in the zonal_stat array
        try:
            zonal_stat[zones.values==zone_n] = zonalstats['{}'.format(stat_name)].sel(dim_0=zone_n).values
        except: #MaskError: Cannot convert masked element to a Python int.
            zonal_stat[zones.values==zone_n] = -9999

    # convert this to an xarray data array with the proper name
    zonal_stat_da = xr.DataArray(zonal_stat, 
                                 dims=["y", "x"],
                                 coords=dict(
                                             x=(["x"], zones.x),
                                             y=(["y"], zones.y),
                                             ),
                                 name='zonal_{}'.format(stat_name))
    # remove nodata values
    zonal_stat_da = zonal_stat_da.where(zonal_stat_da!=-9999, np.nan)

    return zonal_stat_da        
        
        
        
def upscale_aster_goes_rad_zonal_stats(aster_rad_filepath, goes_rad_filepath, goes_zones_filepath, bounding_geometry, zonal_count_threshold=None, goes_tb_filepath=None, output_filepath=None):
    '''Given an ASTER thermal infrared radiance GeoTiff image, a GOES ABI thermal infrared radiance GeoTiff image,
       and a GOES ABI GeoTiff image defining pixel footprint "zones", compute zonal statistics for each GOES pixel
       footprint. Compute the difference between the GOES ABI radiance and ASTER zonal mean radiance.
       Return a dataset, or save dataset to a netcdf file, with the three input dataarrays plus zonal statistic data arrays.'''
    
    ### Open and clean-up input datasets ###
    # Use rioxarray to open a GeoTIFF of GOES-16 ABI Radiance that has been orthorectified for our study area
    goes_rad = xr.open_rasterio(goes_rad_filepath)
    # Open its associated GOES "zone_labels" GeoTIFF raster that was generated at the same time the image was orthorectified
    goes_zones = xr.open_rasterio(goes_zones_filepath)
    # Open the coincident ASTER thermal infrared radiance GeoTIFF image we want to compare with
    aster_src = xr.open_rasterio(aster_rad_filepath)
    
    ### Use reproject_match to align the GOES radiance and GOES zones rasters to the ASTER image, clip to the geometry defined above, and set zone values to integers. ###
    # Reproject match goes rad raster to aster, clip to geometry,  and squeeze out extra dim
    goes_rad_repr = goes_rad.rio.reproject_match(aster_src).rio.clip(bounding_geometry).squeeze().rename('goes_rad')
    # convert GOES radiance (mW m^-2 sr^-1 1/cm^-1) to match MODIS and ASTER (W m^-2 sr^-1 um^-1)
    goes_rad_repr = (goes_rad_repr / 1000) * (61.5342/0.7711)
    # Reproject match goes zones raster to aster, clip to geometry, set datatype to integer, and squeeze out extra dim
    goes_zones_repr = goes_zones.rio.reproject_match(aster_src).rio.clip(bounding_geometry).astype('int').squeeze().rename('goes_zones')
    
    ### If we are given a GOES Brightness Temperature geotiff, open and include it in the final dataset output
    if goes_tb_filepath != None:
        # Use rioxarray to open a GeoTIFF of GOES-16 ABI Brightness Temperature that has been orthorectified for our study area
        goes_tb = xr.open_rasterio(goes_tb_filepath)
        # Reproject match goes tb raster to aster, clip to geometry,  and squeeze out extra dim
        goes_tb_repr = goes_tb.rio.reproject_match(aster_src).rio.clip(bounding_geometry).squeeze().rename('goes_tb')
    
    ### Clean up the ASTER image by replacing nodata values with NaN, removing an extra dimension, converting the digital number values to radiance values, and finally clipping to the geometry defined above. ###
    # Replace the nodatavals with NaN, squeeze out the band dim we don't need
    aster_src = aster_src.where(aster_src!=aster_src.nodatavals, np.nan).squeeze()
    # Convert ASTER DN to Radiance
    aster_band = 14
    aster_rad = tir_dn2rad(aster_src, aster_band)
    # set crs back
    aster_rad.rio.set_crs(aster_src.crs, inplace=True)
    # clip to geometry
    aster_rad_clipped = aster_rad.rio.clip(bounding_geometry).rename('aster_rad')
    
    ### Compute zonal statistics and format results ###
    # Compute zonal statistics from the ASTER image, using the GOES zone labels:
    zonalstats = xrs.zonal.stats(goes_zones_repr, 
                             aster_rad_clipped, 
                             stat_funcs=['mean', 'max', 'min', 'std', 'var', 'count'])
    # Convert zonal statistics dataframe to xarray dataset
    zonalstats = xr.Dataset(zonalstats)
    
    ### Map the results from xrs.zonal.stats() back into the original zones grid ###
    # Map each zonal stat back into the original zones grid
    zonal_means = mapZonalStats(goes_zones_repr, zonalstats, 'mean').rename('mean_rad')
    zonal_max = mapZonalStats(goes_zones_repr, zonalstats, 'max').rename('max_rad')
    zonal_min = mapZonalStats(goes_zones_repr, zonalstats, 'min').rename('min_rad')
    zonal_std = mapZonalStats(goes_zones_repr, zonalstats, 'std').rename('std_rad')
    zonal_var = mapZonalStats(goes_zones_repr, zonalstats, 'var').rename('var_rad')
    zonal_count = mapZonalStats(goes_zones_repr, zonalstats, 'count')
        
    ### Compute brightness temperatures for the original ASTER image, and each zonal statistic from Radiance
    rad2tb_stats = []
    for this_stat_da in [aster_rad_clipped, zonal_means, zonal_max, zonal_min, zonal_std, zonal_var]:
        # ASTER Radiance to Brightness Temperature
        aster_this_stat_da_tb_K =  tir_rad2tb(this_stat_da, aster_band)
        aster_this_stat_da_tb_K.rio.set_crs(aster_src.crs, inplace=True)
        aster_this_stat_da_tb_K = aster_this_stat_da_tb_K.rename('{}2tbK'.format(this_stat_da.name))
        rad2tb_stats.append(aster_this_stat_da_tb_K)
        # convert brightness temperature in K to C
        #aster_this_stat_da_tb_C = aster_this_stat_da_tb_K.values - 273.15
        #aster_this_stat_da_tb_C = xr.DataArray(aster_this_stat_da_tb_C, name='{}2tbC'.format(this_stat_da.name))
        #rad2tb_stats.append(aster_this_stat_da_tb_C)
    
    ### Compute the difference between GOES Radiance and the ASTER zonal mean radiance ###
    # Compute the difference between GOES Radiance and the ASTER zonal mean radiance
    mean_diff_rad = goes_rad_repr.values - zonal_means.values
    # Create a data array for the mean difference values
    mean_diff_rad_da = xr.DataArray(mean_diff_rad, name='mean_diff_rad', dims=["y", "x"])
    
    ### Merge all zonal stats back with the original ASTER data to create a single dataset ###
    # Merge all the zonal statistics data arrays and the mean difference data array
    if goes_tb_filepath == None:
        # without the GOES Brightness Temperature data array
        ds = xr.merge([aster_rad_clipped, goes_rad_repr, goes_zones_repr, zonal_means, zonal_max, zonal_min, zonal_std, zonal_var, zonal_count, mean_diff_rad_da] + rad2tb_stats)
    else:
        # With the GOES Brightness Temperature data array
        ds = xr.merge([aster_rad_clipped, goes_rad_repr, goes_tb_repr, goes_zones_repr, zonal_means, zonal_max, zonal_min, zonal_std, zonal_var, zonal_count, mean_diff_rad_da] + rad2tb_stats)
    
    
    
    if goes_tb_filepath != None:
        ### If we have a GOES Brightness Temperature data array,
        # Compute the difference between GOES Brightness Temperature and the ASTER zonal mean brightness temperature
        mean_diff_tb = goes_tb_repr.values - ds.mean_rad2tbK.values
        # Create a data array for the mean difference values
        mean_diff_tb_da = xr.DataArray(mean_diff_tb, name='mean_diff_tb', dims=["y", "x"])
        # Add it to our dataset
        ds['mean_diff_tb'] = mean_diff_tb_da
    
    
    # Clip this dataset to the ASTER image extent
    ds = ds.where(~np.isnan(aster_rad_clipped))
    # Remove data where we have overlap between a GOES pixel footprint "zone" and the edge of the ASTER image
    if zonal_count_threshold != None:
        # Use the "zonal_count" (the number of ASTER pixels in each GOES pixel footprint) to determine this
        # Example: zonal_count_threshold = 800 90m ASTER pixels is 800x90x90 square meters, or approximately 6.48 square kilometers or,
        # about 2.5x2.5 km, about the size of a full GOES-16 ABI pixel here
        # NOTE: The number of ASTER pixels per GOES pixel footprint is not constant, so there are some "edge pixels" still included here
        ds = ds.where(ds.zonal_count > zonal_count_threshold)
    
    # If an output_filepath was specified, save the resulting dataset to a netcdf file
    if output_filepath != None:
        ds.to_netcdf(output_filepath)
    
    return ds