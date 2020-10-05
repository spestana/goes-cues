# goes-cues

Comparison of [GOES](https://www.goes-r.gov/) satellite observations against [CUES](https://snow.ucsb.edu/) station data

---

## Functions/scripts:

### lw_clr.py

Functions for estimating clear-sky downward longwave radition and atmospheric emissivity. I'm using these here to compare against observations of all-sky downward longwave at the CUES site to identify periods of cloud cover (especially at night when we cannot use shortwave observations).

### cloud_detect.py

Detecting cloud cover using ground-based observations (similar to the clear-sky index of Marty & Philipona, 2000). First uses ground-based observations of air temperature and relative humidity to run an ensemble of clear-sky downward longwave (LWd) estimation methods (lw_clr.py). Then compares these estimates against LWd observations at the site. Where the LWd is greater than the clear-sky estimates, we likely have cloud-cover.

### aster_utils.py

Functions for working with ASTER TIR imagery(from the [AST_L1T product](https://lpdaac.usgs.gov/products/ast_l1tv003/)): converting DN to radiance, radiance to brightness temperature, and computing zonal statistics given a shapefile.
Also take a look at [these AST_L1T utilities from LP DAAC](https://git.earthdata.nasa.gov/projects/LPDUR/repos/aster-l1t/browse).

### resampled_stats.py

Functions for computing statistics on [DataArrayResample](https://xarray.pydata.org/en/stable/generated/xarray.core.resample.DataArrayResample.html) objects.

---

## Data processing/analysis notebooks:

### merge-cues.ipynb

Notebook for cleaning up and combining CUES Level 1 data (temperature and radiation datasets) retrieved from [snow.ucsb.edu](https://snow.ucsb.edu/index.php/query-db/).

### cloud-detect.ipynb

Estimate when we have cloud-cover and add a cloud_flag to the CUES dataset.

### merge-goes-cues.ipynb

Merge the CUES dataset with GOES brightness temperature observations of the site. (Also resamples to 5-minute mean values) (See [data](/data/data.md) here.)

### goes-cues-2017-2020.ipynb

Analysis and plotting of GOES (single pixel) and CUES temperature timeseries.

### goes-aster-2017-2020.ipynb

Analysis and plotting of GOES, CUES, and ASTER zonal statistics all together.

---

## Notebooks for testing/examples:

### test-lw-clr.ipynb

Testing the lw_clr.py functions.

### test-cloud-detect.ipynb

Testing the cloud_detect.py functions, and brute-force parameter test to find optimal clear-sky index thresholds. Read in [RESULTS.pkl](/misc/RESULTS.pkl) to get a pandas dataframe of the brute force parameter test results.

![cloud_detect_threshold_options.png](/images/cloud_detect_threshold_options.png "cloud_detect_threshold_options")

### example-plot.ipynb

Produces this plot:

![example plot](/images/goes-vs-cues.jpg "example plot")

### zonal-statistics-aster.ipynb

Example notebook to read in an AST_L1T geotiff, shapefile, and compute zonal statistics.

![zonal_stats_example.png](/images/zonal_stats_example.png "zonal_stats_example")

### resampled-stats-example.ipynb

Testing the [resampled_stats.py](./resampled_stats.py) functions, to compute summary statistics on [DataArrayResample](https://xarray.pydata.org/en/stable/generated/xarray.core.resample.DataArrayResample.html) objects.

---


---