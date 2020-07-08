# goes-cues

Comparison of [GOES](https://www.goes-r.gov/) satellite observations against [CUES](https://snow.ucsb.edu/) station data

![flowchart](https://github.com/spestana/goes-cues/blob/master/flowchart.png "goes-cues flowchart")

---

### example-plot.ipynb

Produces this plot:

![example plot](https://github.com/spestana/goes-cues/blob/master/goes-vs-cues.jpg "example plot")

---

### lw_clr.py

Functions for estimating clear-sky downward longwave radition and atmospheric emissivity. I'm using these here to compare against observations of all-sky downward longwave at the CUES site to identify periods of cloud cover (especially at night when we cannot use shortwave observations).

---

### merge-cues.ipynb

Notebook for cleaning up and combining CUES Level 1 data (temperature and radiation datasets) retrieved from [snow.ucsb.edu](https://snow.ucsb.edu/index.php/query-db/).

---

### Notebooks for testing:

#### test-lw-clr.ipynb

Testing the lw_clr.py functions.

#### test-cloud-detect.ipynb

Testing the cloud_detect.py functions, and brute-force parameter test to find optimal clear-sky index thresholds.

![cloud_detect_threshold_options.png](https://github.com/spestana/goes-cues/blob/master/cloud_detect_threshold_options.png "cloud_detect_threshold_options")

---

### cloud-detect.ipynb

Estimate when we have cloud-cover and add a cloud_flag to the CUES dataset.

---

### merge-goes-cues.ipynb

Merge the CUES dataset with GOES brightness temperature observations of the site. (Also resamples to 5-minute mean values)

---

### goes-cues-2017-2020.ipynb

Analysis and plotting.