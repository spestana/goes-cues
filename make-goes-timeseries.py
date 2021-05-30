# Make Timeseries of GOES-16 LST and ACM Products

import goes_ortho
import pandas as pd

# Set location of a study site

# Gaylor Pit, Tuolumne
site_name = 'Tuolumne'
lon = -119.31212
lat = 37.88175
#z = 2811 # elevation in meters
z = 0 # no orthorectification

# CUES, Mammoth Mtn
#site_name = 'CUES'
#lon = -119.029146
#lat = 37.643103
#z = 2940 # elevation in meters
#z = 0 # no orthorectification

# Make ABI L1b Radiance timeseries and export to csv files
for year in [2017]:#,2018,2019,2020]:
    _ = goes_ortho.make_abi_timeseries('/storage/GOES/goes16/{year}/'.format(year=year), 'RadC-*C14*', ['Rad'], lon, lat, z, '/storage/GOES/pkl/GaylorPit_not_orthorectified/goes16-c14-{year}-{site_name}.csv'.format(site_name=site_name,year=year))

## Make ABI Cloud Mask (ACM) timeseries and export to csv files
#for year in [2017,2018,2019,2020]:
#    for month in range(1,13):
#        _ = goes_ortho.make_abi_timeseries('/storage/GOES/goes16/{year}/{month}'.format(year=year,month=month), 'ACMC', ['BCM','DQF'], #lon, lat, z, './GOES-16_ABI-L2_timeseries/ACMC/Tuolumne/GOES-16_ABI-L2-ACMC_{site_name}_{month}-#{year}.csv'.format(site_name=site_name,year=year,month=month))

## Make Land Surface Temperature (LST) timeseries and export to csv files
#for year in [2017,2018,2019,2020]:
#    for month in range(1,13):
#        _ = goes_ortho.make_abi_timeseries('/storage/GOES/goes16/{year}/{month}'.format(year=year,month=month), 'LSTC', ['LST'], lon, #lat, z, './GOES-16_ABI-L2_timeseries/LSTC/Tuolumne/GOES-16_ABI-L2-LSTC_{site_name}_{month}-#{year}.csv'.format(site_name=site_name,year=year,month=month))
