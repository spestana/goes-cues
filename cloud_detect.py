"""
Some rough experimental methods for detecting cloud cover using ground-based and/or satellite observations.

Steven Pestana, June 2020 (spestana@uw.edu)

"""

# To do:
# - 


import pandas as pd
import numpy as np
import xarray as xr

# for longwave estimation functions
import lw_clr


#----------------- Longwave comparison method for cloud detection -------------------#
def lw_cloud_detect(Tair, RH, LWd_obs, sun_flag_obs, elev, threshold=0):
	'''
	Using ground-based observations of air temperature and relative humidity, run an ensemble of clear-sky downward longwave (LWd) estimation methods.
	Then compare these methods against LWd observations at the site. 
	Where the LWd is greater than the clear-sky estimates, we likely have cloud-cover.
	'''

	# Input variables: 
	#					Tair (K) 			[xarray.DataArray with datetime index]
	#					RH (%) 				[xarray.DataArray with datetime index]
	#					LWd_obs (Wm-2) 		[xarray.DataArray with datetime index]
	#					sun_flag_obs (0/1)	[xarray.DataArray with datetime index]
	#					elevation (m) 		[integer or float value]
	# 					threshold (Wm-2)    [integer or float value] (threshold above ensemble mean)

	####################
	### Estimate LWd ###
	####################
	# Run the ensemble function
	LWd_pred = lw_clr.ensemble(Tair, RH, elev)

	# Create a "day_flag" for hours throughout the year (so including shorter winter days) when we exect to see sunlight if there are no clouds. 
	# This is used in the evaluation of the performance of this cloud detection method. (I've chosen here to designate 8am-4pm as "day", but these can be changed)
	daystarthour = 8
	dayendhour = 16
	day_flag =  [1 if (pd.Timestamp(x).hour > daystarthour) & (pd.Timestamp(x).hour < dayendhour) else 0 for x in LWd_pred['datetime'].values]
	LWd_pred['day_flag'] = (['datetime'],  day_flag)

	################################################################
	### Now make a cloud flag where we think there's cloud cover ###
	################################################################
	# We can set a threshold of allowance above the ensemble mean estimated LWd value where we say there are still no clouds.
	# (Defaulting to threshold = 0. I'm not sure what this value should be, so we can do some tests to find a good value later)

	# create array of zeros (cloud_flag = 0, no clouds)
	cloud_flag = np.zeros_like(LWd_pred.lclr_mean.values) 

	# Conditional statement to determine if we think we have clouds:
	# if LWd_observed > LWd_clearsky + threshold
	lw_cloud_condition = LWd_obs > LWd_pred.lclr_mean + threshold

	# Set our cloud_flag = 1 whenever this condition is true
	cloud_flag[lw_cloud_condition] = 1

	# Add the cloud flag to the dataset
	LWd_pred['cloud_flag'] = (['datetime'],  cloud_flag)

	##################################
	### Compute a confusion matrix ###
	##################################
	# Predictions: get the cloud flag values where we know we're in daytime hours
	y_pred = LWd_pred.cloud_flag.where(LWd_pred.day_flag == 1)
	# (note that I am inverting the value of the cloud flag so it is now a "sun flag")
	y_pred = np.abs( y_pred - 1 )

	# Actual: get the sun flag values where we know we're in daytime hours
	y_actual = sun_flag_obs.where(LWd_pred.day_flag == 1)

	# Set up a dataframe with these the sun and cloud flags for daytime hours
	data = {
			'sun_actual':    y_actual,
			'sun_predicted': y_pred 
			}
	df = pd.DataFrame(data, columns=['sun_actual','sun_predicted'])

	# Compute the confusion matrix:
	confusion_matrix = pd.crosstab(df['sun_actual'], df['sun_predicted'], rownames=['Observed'], colnames=['Predicted'])
	# Evaluation metrics
	tp = confusion_matrix[1][1] # true positives
	fp = confusion_matrix[0][1] # false positives
	fn = confusion_matrix[1][0] # false negatives
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f1_score = 2 * ( (precision * recall)/(precision + recall) )
	
	
	return LWd_pred, confusion_matrix, precision, recall, f1_score


def optimize_lw_cloud_detect(Tair, RH, LWd_obs, sun_flag_obs, elev):
	'''
	Try to optimize the threshold used in lw_cloud_detect for a given set of observations.
	
	Our objective function we want to minimize is the F1-score of the confusion matrix from the lw_cloud_detect method:
	LWd_pred, confusion_matrix = lw_cloud_detect(Tair, RH, LWd_obs, sun_flag_obs, elev, threshold)
	'''
	
	# Generate a random boolean array to select approximately some % of the original data for testing
	random_numbers = np.random.rand(len(Tair))
	random_bools = random_numbers < 0.005 # adjust this to select more or 

	# Select this random subset from the input data
	Tair_sample = Tair[random_bools]
	RH_sample = RH[random_bools]
	LWd_obs_sample = LWd_obs[random_bools]
	sun_flag_obs_sample = sun_flag_obs[random_bools]


	
	optimized_threshold = None
	
	return optimized_threshold


#----------------- Surface temperature comparison method for cloud detection -------------------# 
'''
Uses ground-based observations of surface temperature and satellite (GOES-R) observations of brightness temperature (in the thermal infrared).
Compute a linear regression between the two sources of temperature information over some time period (such as 3 hours).
If there are no clouds blocking the satellite's view of the surface, the slope of this linear regression is likely close to 1.
If there are clouds blocking the satellite's view of the surface (cloud-top temperatures much colder than ground-based surface temperatures),
then the slope of the linear regression is >> 1.
'''
