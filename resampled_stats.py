"""
Functions for computing statistics on DataArrayResample objects.

Steven Pestana, Sept 2020 (spestana@uw.edu)

"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import xarray as xr
import matplotlib.pyplot as plt

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
    
    df = pd.DataFrame(data=d, index=resampled_modes.index)
    
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
                          lw=1)
    ax[0].axhline(0,color='lightgrey',linestyle='-')
    ax[0].set_title('Difference Histogram')
    ax[0].set_xlabel('Number of Observations\nTotal: {}'.format(original_df.count()))
    ax[0].set_xlim((xmin,xmax))
    
    ### Resampled Dataframe, Timeseries "Boxplots" ###
    # mean marker
    resampled_df.means.plot(linestyle='none',marker='o',color='k', label='Mean', ax=ax[1])
    # median marker
    resampled_df.medians.plot(linestyle='none',marker='_',color='k', label='Median', ax=ax[1])
    # mode marker
    resampled_df.modes.plot(linestyle='none',marker='o',markerfacecolor='w',markeredgecolor='k', label='Mode', ax=ax[1])
    
    # lower and upper quartile error bars
    ax[1].errorbar(x=resampled_df.index, 
                y=resampled_df.means,
                yerr=np.array([np.abs(resampled_df.qLower-resampled_df.means), 
                               np.abs(resampled_df.qUpper-resampled_df.means)]),
                fmt='none',
                linewidth=1,
                color='k',
                alpha=0.4,
                capsize=None,
                label='IQR')
    
    ax[1].axhline(0,linestyle='-',color='lightgrey',linewidth=1)
    plt.legend()
    
    # Format shared y-axis
    ax[0].set_ylim(ymin,ymax);
    ax[0].set_ylabel('GOES $T_{B}$ - CUES $T_{SS}$ [$\Delta\degree C$]')
    
    return fig, ax