"""
Functions for computing statistics on DataArrayResample objects.

Steven Pestana, Sept 2020 (spestana@uw.edu)

"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import xarray as xr
import matplotlib.pyplot as plt

def clean_nans_from_dict(d):
    '''For a dictionary where each key holds a list or np.array of values, remove all nan values.
    Based on: https://stackoverflow.com/questions/24068306/is-there-a-way-to-remove-nan-from-a-dictionary-filled-with-data'''
    
    # create empty dict to put the cleaned-up dictionary key:value pairs
    cleaned_d = {}
    
    # loop through each key in the original dict
    for key_name in d:
              
        # for each item in the values for this key, if it is not nan then add to new list
        items_cleaned = [item for item in d[key_name] if not np.isnan(item)]
        
        # add this list of cleaned items to the cleaned dictionary
        cleaned_d[key_name] = items_cleaned

    # return the cleaned dictionary
    return cleaned_d


def summary_stats(_a, _b):
    '''Compute summary statistics for the difference between two sets.
    Input two flattened (1-D) arrays with NaN values removed'''
    
    # remove nan values
    a = _a[(np.isnan(_a)==False) & (np.isnan(_b)==False)]
    b = _b[(np.isnan(_a)==False) & (np.isnan(_b)==False)]
    
    # for difference stats
    diff = b - a
    
    # for linear regression stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(a, b)
    
    # populate dict with summary stats
    summary_stats_dict = {
        #'diff' : diff ,
        'n' : len(diff) ,
        'mean_diff' : np.nanmean( diff ),
        'median_diff' : np.nanmedian( diff ),
        'mean_squared_diff' : np.nanmean( diff**2 ),
        'rms_diff' : np.sqrt( np.nanmean( diff**2 ) ),
        'std_diff' : np.nanstd( diff ),
        'slope' : slope,
        'intercept' : intercept,
        'r_value' : r_value,
        'p_value' : p_value,
        'std_err' : std_err
        }
    
    return summary_stats_dict

def compute_modes(resampled, n=0):
    
    '''Given a resampled xarray object (xarray.core.resample.DataArrayResample),
       compute the modes ( rounding values with np.round(_,n) ), 
       return a pandas dataframe with the modes, counts, and groups (datetime)'''
    
    # Compute modes
    resampled_ModeResults = [stats.mode(np.round(x[1],n)) for x in resampled]
    
    # Get all the modes
    resampled_modes = [x.mode for x in resampled_ModeResults]
    # Reshape result into 1D array
    resampled_modes = np.array(resampled_modes).reshape(len(resampled))
    
    # Get the count for each mode
    resampled_counts = [x.count for x in resampled_ModeResults]
    # Reshape result into 1D array
    resampled_counts = np.array(resampled_counts).reshape(len(resampled))
    
    # Get the group (datetime) for each mode
    resampled_groups = np.array( list(resampled.groups.keys()) )
    
    # Create pandas dataframe
    d = {'modes': resampled_modes, 'counts': resampled_counts}
    df = pd.DataFrame(data=d, index=resampled_groups)
    
    return df

def resampled_stats(resampled, n=0, q=(.25,.75)):
    
    '''Given a resampled xarray object (xarray.core.resample.DataArrayResample),
    return a pandas dataframe with the mean, median, modes, counts, and groups (datetime)'''
    
    # Compute modes
    resampled_modes = compute_modes(resampled, n)
    
    # Compute and add mean, medians, modes, counts to a dataframe
    d = {'means': resampled.mean().values, 
         'medians': resampled.median().values, 
         'modes': resampled_modes.modes.values,
         'counts': resampled_modes.counts.values,
         'maxs': resampled.max().values,
         'mins': resampled.min().values,
         'ranges': resampled.max().values - resampled.min().values,
         'stds': resampled.std().values,
         'qUpper': resampled.quantile(q[1]).values,
         'qLower': resampled.quantile(q[0]).values,}
    # remove any nans from the dictionary
    cleaned_d = clean_nans_from_dict(d)

    # Create a dataframe from the cleaned-up dictionary
    df = pd.DataFrame(data=cleaned_d, index=resampled_modes.index)
    
    return df


def resampled_plot(original_df, resampled_df, ymin=-20, ymax=20, xmin=0, xmax=400, nbins=100):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,4), 
                           tight_layout=True, sharey=True,
                           gridspec_kw={'width_ratios': [1, 3]})
    
    ### Original Dataframe, Histogram ###
    original_df.plot.hist(ax=ax[0],
                          bins=nbins, 
                          orientation='horizontal',
                          color = '#000000',
                          ec='none',
                          lw=1, legend=False)
    ax[0].axhline(0,color='lightgrey',linestyle='-')
    ax[0].set_title('Difference Histogram')
    ax[0].set_xlabel('Number of Observations\nTotal: {}'.format(int(original_df.count())))
    ax[0].set_xlim((xmin,xmax))
    
    ### Resampled Dataframe, Timeseries "Boxplots" ###
    # mean marker
    resampled_df.means.plot(linestyle='none',marker='_',color='k', label='Mean', ax=ax[1])
    # median marker
    resampled_df.medians.plot(linestyle='none',marker='o',color='k', label='Median', ax=ax[1])
    # mode marker
    resampled_df.modes.plot(linestyle='none',marker='o',markerfacecolor='w',markeredgecolor='k', label='Mode', ax=ax[1])
    
    # lower and upper quartile error bars
    ax[1].errorbar(x=resampled_df.index, 
                y=resampled_df.means,
                yerr=np.array([np.abs(resampled_df.qLower-resampled_df.means), 
                               np.abs(resampled_df.qUpper-resampled_df.means)]),
                fmt='none',
                linewidth=4,
                color='k',
                alpha=0.3,
                capsize=None,
                label='IQR')
    
    ax[1].axhline(0,linestyle='-',color='k',linewidth=1)
    plt.legend()
    
    # Format shared y-axis
    ax[0].set_ylim(ymin,ymax);
    ax[0].set_ylabel('GOES $T_{B}$ - CUES $T_{SS}$ [$\Delta\degree C$]')
    
    return fig, ax
