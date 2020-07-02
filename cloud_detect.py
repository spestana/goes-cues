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

# for optimization
from scipy.optimize import minimize, minimize_scalar


#----------------- Longwave comparison method for cloud detection -------------------#
def lw_cloud_detect(Tair, RH, LWd_obs, sun_flag_obs, elev, lw_threshold=0, csi_threshold=0):
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
		
	# LW-based Clear-Sky Index, and cloud flag
	# This CSI is a ratio of observed to estimated LWd+lw_threshold
	# where we have clouds when LWd_observed > LWd_clearsky + lw_threshold (CSI > 1)
	LWd_pred['CSI_lw_and_threshold'] = (['datetime'],  LWd_obs / (LWd_pred.lclr_mean + lw_threshold) )
	# create array of zeros (cloud_flag = 0, no clouds)
	cloud_flag = np.zeros_like(LWd_pred.lclr_mean.values) 
	# Conditional statement to determine if we think we have clouds:
	# if CSI > 1 + csi_threshold
	csi_cloud_condition = LWd_pred.CSI_lw_and_threshold > 1 + csi_threshold
	# Set our simple cloud_flag = 1 whenever this condition is true
	cloud_flag[csi_cloud_condition] = 1
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
	try:
		# Evaluation metrics
		tp = confusion_matrix[1][1] # true positives
		fp = confusion_matrix[0][1] # false positives
		fn = confusion_matrix[1][0] # false negatives
		precision = tp / (tp + fp)
		recall = tp / (tp + fn)
		f1_score = 2 * ( (precision * recall)/(precision + recall) )
	except KeyError:
		precision = None
		recall = None
		f1_score = None

	
	return LWd_pred, confusion_matrix, precision, recall, f1_score


#----------------- Optimization functions for longwave cloud detection -------------------#
#def min_f1_score(threshold, *args):
#	''' Objective function for minimizing based on -f1_score of confusion matrix'''
#
#	Tair = args[0]
#	RH = args[1]
#	LWd_obs = args[2]
#	sun_flag_obs = args[3]
#	elev = args[4]
#
#	_, _, _, _, f1_score = lw_cloud_detect(Tair, RH, LWd_obs, sun_flag_obs, elev, threshold)
#
#	return -1*f1_score
#	
#	
#
#def optimize_lw_cloud_detect(Tair, RH, LWd_obs, sun_flag_obs, elev, iterations=1):
#	'''
#	Try to optimize the threshold used in lw_cloud_detect for a given set of observations.
#	
#	Our objective function we want to minimize is (1 - F1-score) of the confusion matrix from the lw_cloud_detect method:
#			LWd_pred, confusion_matrix, precision, recall, f1_score 
#			= cloud_detect.lw_cloud_detect(Tair_sample, RH_sample, LWd_obs_sample, sun_flag_obs_sample, elev, threshold)
#
#	'''
#	optimized_thresholds = []
#	
#	for n in range(iterations):
#		# Generate a random boolean array to select approximately some % of the original data for testing
#		random_numbers = np.random.rand(len(Tair))
#		random_bools = random_numbers < 0.001 # adjust this to select more or 
#
#		# Select this random subset from the input data
#		Tair_sample = Tair[random_bools]
#		RH_sample = RH[random_bools]
#		LWd_obs_sample = LWd_obs[random_bools]
#		sun_flag_obs_sample = sun_flag_obs[random_bools]
#
#		# LWd_pred, confusion_matrix, precision, recall, f1_score = cloud_detect.lw_cloud_detect(Tair_sample, RH_sample, LWd_obs_sample, sun_flag_obs_sample, elev, threshold)
#		#res = minimize(
#		#			fun = min_f1_score,
#		#			x0=0,
#		#			args=(Tair_sample, RH_sample, LWd_obs_sample, sun_flag_obs_sample, elev),
#		#			method='Nelder-Mead'
#		#)
#		res = minimize_scalar(
#					min_f1_score,
#					args=(Tair_sample, RH_sample, LWd_obs_sample, sun_flag_obs_sample, elev)
#		)
#	
#	
#		optimized_thresholds.append(res)
#	
#	return optimized_thresholds


##----------------- Surface temperature comparison method for cloud detection -------------------# 
#'''
#Uses ground-based observations of surface temperature and satellite (GOES-R) observations of brightness temperature (in the thermal infrared).
#Compute a linear regression between the two sources of temperature information over some time period (such as 3 hours).
#If there are no clouds blocking the satellite's view of the surface, the slope of this linear regression is likely close to 1.
#If there are clouds blocking the satellite's view of the surface (cloud-top temperatures much colder than ground-based surface temperatures),
#then the slope of the linear regression is >> 1.
#'''
