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
        'min_diff' : np.nanmin( diff ),
        'max_diff' : np.nanmax( diff ),
        'range_diff' : np.nanmax( diff ) - np.nanmin( diff ),
        'n' : len(diff) ,
        'mean_diff' : np.nanmean( diff ),
        'median_diff' : np.nanmedian( diff ),
        'mean_squared_diff' : np.nanmean( diff**2 ),
        'rms_diff' : np.sqrt( np.nanmean( diff**2 ) ),
        'std_diff' : np.nanstd( diff ),
        'slope' : slope,
        'intercept' : intercept,
        'r_value' : r_value,
        'r_squared' : r_value**2,
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
    #resampled_counts = [x.count for x in resampled_ModeResults]
    # Reshape result into 1D array
    #resampled_counts = np.array(resampled_counts).reshape(len(resampled))
    
    # Get the group (datetime) for each mode
    #resampled_groups = np.array( list(resampled.groups.keys()) )
    
    # Create pandas dataframe
    #d = {'modes': resampled_modes, 'counts': resampled_counts}
    #df = pd.DataFrame(data=d, index=resampled_groups)
    
    return resampled_modes

def reconcile_groups_dict(d, my_groups, all_groups):
    # make a new dictionary
    new_d = {}
    # for each key in the dictionary
    for key_name in d:
        # make a temporary empty list we'll populate with values that correspond to our original groups
        cleaned_list = []
        # now for each group (and it's index) in "all_groups" (which have too many groups but include the "my_groups" I want to keep)
        for i, group in enumerate(all_groups):
            # if this group from "all_groups" is one of "my_groups"
            if group in my_groups:
                # append the value from this key_name to the temporary list
                cleaned_list.append(d[key_name][i])
        # at the end of the items in this key, add the cleaned_list to the new dictionary with the same key
        new_d[key_name] = cleaned_list
    return new_d

def resampled_stats(resampled, n=0, q=(.25,.75)):
    
    '''Given a resampled xarray object (xarray.core.resample.DataArrayResample),
    return a pandas dataframe with the mean, median, modes, counts, and groups (datetime)'''
    
    # get the groups that we are resampling or grouping by
    my_groups = list(resampled.groups.keys())
        
    # Compute and add mean, medians, modes, counts to a dataframe
    d = {'means': resampled.mean().values, 
         'medians': resampled.median().values, 
         'counts': resampled.count().values,
         'maxs': resampled.max().values,
         'mins': resampled.min().values,
         'ranges': resampled.max().values - resampled.min().values,
         'stds': resampled.std().values,
         'qUpper': resampled.quantile(q[1]).values,
         'qLower': resampled.quantile(q[0]).values,}
    
    # if we are using a "DataArrayResample" obect, we may need to fix the groups
    if type(resampled) == xr.core.resample.DataArrayResample:
        # fix the groups in the dictionary
        all_groups = list(resampled.mean().time.values)
        fixed_d = reconcile_groups_dict(d, my_groups, all_groups)
    # otherwise, we're working with an "xr.core.resample.DataArrayGroupBy" object and this shouldn't be an issue
    else:
        fixed_d = d
    
    # Create a dataframe from the cleaned-up dictionary
    df = pd.DataFrame(data=fixed_d, index=my_groups)
    
    # Compute modes and add to dataframe
    df['modes'] = compute_modes(resampled, n)
        
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
    ax[0].axhline(0,color='lightgrey',linestyle=':',lw=1)
    ax[0].set_title('Difference Histogram')
    ax[0].set_xlabel('Number of Observations\nTotal: {}'.format(int(original_df.count())))
    ax[0].set_xlim((xmin,xmax))
    
    ### Resampled Dataframe, Timeseries "Boxplots" ###
    # mean marker
    resampled_df.means.plot(linestyle='none', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=99, label='Mean difference', ax=ax[1])
    # median marker
    #resampled_df.medians.plot(linestyle='none', marker='^', markerfacecolor='w', markeredgecolor='k', zorder=98, label='Median', ax=ax[1])
    # mode marker
    #resampled_df.modes.plot(linestyle='none', marker='+', color='k', label='Mode', zorder=97, ax=ax[1])
    
    ## lower and upper quartile error bars
    #ax[1].errorbar(x=resampled_df.index, 
    #            y=resampled_df.means,
    #            yerr=np.array([np.abs(resampled_df.qLower-resampled_df.means), 
    #                           np.abs(resampled_df.qUpper-resampled_df.means)]),
    #            fmt='none',
    #            linewidth=4,
    #            color='k',
    #            alpha=0.3,
    #            capsize=None,
    #            label='IQR')
    
    # +/- 1 standard deviation error bars
    ax[1].errorbar(x=resampled_df.index, 
                y=resampled_df.means,
                yerr=resampled_df.stds,
                fmt='none',
                linewidth=4,
                color='k',
                alpha=0.3,
                capsize=None,
                label='$\pm 1 \sigma$ difference')
    
    ax[1].axhline(0,color='lightgrey',linestyle=':',lw=1)
    plt.legend()
    
    # Format shared y-axis
    ax[0].set_ylim(ymin,ymax);
    ax[0].set_ylabel('$T_{B}$ - $T_{SS}$ [$\Delta\degree C$]')
    
    return fig, ax

def resampled_plot2(original_df, resampled_df, ymin=-20, ymax=20, xmin=0, xmax=400, nbins=100):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4), 
                           tight_layout=True, sharey=True,
                           gridspec_kw={'width_ratios': [1, 1]})
    
    ### Original Dataframe, Histogram ###
    original_df.plot.hist(ax=ax[0],
                          bins=nbins, 
                          orientation='horizontal',
                          color = '#000000',
                          ec='none',
                          lw=1, legend=False)
    ax[0].axhline(0,color='lightgrey',linestyle=':',lw=1)
    ax[0].text(0.79, 0.9, f'mean:{np.round(original_df.mean().values[0],1)}\n$\sigma$:{np.round(original_df.std().values[0],1)}', horizontalalignment='left',verticalalignment='center', transform=ax[0].transAxes)
    ax[0].set_title('Difference Histogram')
    ax[0].set_xlabel('Number of Observations\nTotal: {}'.format(int(original_df.count())))
    ax[0].set_xlim((xmin,xmax))
    
    ### Resampled Dataframe, Timeseries "Boxplots" ###
    # mean marker
    resampled_df.means.plot(linestyle='none', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=99, label='Mean difference', ax=ax[1])
    # median marker
    #resampled_df.medians.plot(linestyle='none', marker='^', markerfacecolor='w', markeredgecolor='k', zorder=98, label='Median', ax=ax[1])
    # mode marker
    #resampled_df.modes.plot(linestyle='none', marker='+', color='k', label='Mode', zorder=97, ax=ax[1])
    
    ## lower and upper quartile error bars
    #ax[1].errorbar(x=resampled_df.index, 
    #            y=resampled_df.means,
    #            yerr=np.array([np.abs(resampled_df.qLower-resampled_df.means), 
    #                           np.abs(resampled_df.qUpper-resampled_df.means)]),
    #            fmt='none',
    #            linewidth=4,
    #            color='k',
    #            alpha=0.3,
    #            capsize=None,
    #            label='IQR')
    
    # +/- 1 standard deviation error bars
    ax[1].errorbar(x=resampled_df.index, 
                y=resampled_df.means,
                yerr=resampled_df.stds,
                fmt='none',
                linewidth=4,
                color='k',
                alpha=0.3,
                capsize=None,
                label='$\pm 1 \sigma$ difference')
    
    ax[1].axhline(0,color='lightgrey',linestyle=':',lw=1)
    plt.legend()
    
    # Format shared y-axis
    ax[0].set_ylim(ymin,ymax);
    ax[0].set_ylabel('$T_{B}$ - $T_{SS}$ [$\Delta\degree C$]')
    
    return fig, ax

def resampled_plot3(original_df1, resampled_df1, original_df2, resampled_df2, original_df3, resampled_df3, ymin=-20, ymax=20, xmin=0, xmax=400, nbins=100, color1='#a25f2c', color2='#2ca25f', color3='k'):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9,4), 
                           tight_layout=True, sharey=True,
                           gridspec_kw={'width_ratios': [1, 1]})
    
    ### Original Dataframe 1, Histogram ###
    original_df1.plot.hist(ax=ax[0],
                          bins=nbins, 
                          orientation='horizontal',
                          color = color1,
                          ec='none',
                          alpha=0.5,
                          lw=1, legend=False)
    try:
        ax[0].text(0.7, 0.9, f'mean={np.round(original_df1.mean().values[0],1)}\n$\sigma$={np.round(original_df1.std().values[0],1)}', horizontalalignment='left',verticalalignment='center', transform=ax[0].transAxes, color=color1)
    except AttributeError:
        #AttributeError: 'numpy.float64' object has no attribute 'values'
        ax[0].text(0.7, 0.9, f'mean={np.round(original_df1.mean(),1)}\n$\sigma$={np.round(original_df1.std(),1)}', horizontalalignment='left',verticalalignment='center', transform=ax[0].transAxes, color=color1)
    ax[0].set_title('Difference Histogram')
    ax[0].set_xlabel('Number of Observations\n n={}'.format(int(original_df1.count())))
    ax[0].set_xlim((xmin,xmax))
    ### Original Dataframe 2, Histogram ###
    original_df2.plot.hist(ax=ax[0],
                          bins=nbins, 
                          orientation='horizontal',
                          color = color2,
                          ec='none',
                          alpha=0.5,
                          lw=1, legend=False)
    try:
        ax[0].text(0.7, 0.1, f'mean={np.round(original_df2.mean().values[0],1)}\n$\sigma$={np.round(original_df2.std().values[0],1)}', horizontalalignment='left',verticalalignment='center', transform=ax[0].transAxes, color=color2)
    except AttributeError:
        ax[0].text(0.7, 0.1, f'mean={np.round(original_df2.mean(),1)}\n$\sigma$={np.round(original_df2.std(),1)}', horizontalalignment='left',verticalalignment='center', transform=ax[0].transAxes, color=color2)
    ax[0].set_title('Difference Histogram')
    ax[0].set_xlabel('Number of Observations\n n={}'.format(int(original_df2.count())))
    ax[0].set_xlim((xmin,xmax))
    ### Original Dataframe 3, Histogram ###
    original_df3.plot.hist(ax=ax[0],
                          bins=nbins, 
                          orientation='horizontal',
                          color = color3,
                          ec='none',
                          alpha=0.5,
                          lw=1, legend=False)
    try:
        ax[0].text(0.7, 0.7, f'mean={np.round(original_df2.mean().values[0],1)}\n$\sigma$={np.round(original_df3.std().values[0],1)}', horizontalalignment='left',verticalalignment='center', transform=ax[0].transAxes, color=color3)
    except AttributeError:
        ax[0].text(0.7, 0.7, f'mean={np.round(original_df3.mean(),1)}\n$\sigma$={np.round(original_df3.std(),1)}', horizontalalignment='left',verticalalignment='center', transform=ax[0].transAxes, color=color3)
    ax[0].set_title('Difference Histogram')
    ax[0].set_xlabel('Number of Observations\n n={}'.format(int(original_df3.count())))
    ax[0].set_xlim((xmin,xmax))
    
    
    
    ### Resampled Dataframe 1, Timeseries "Boxplots" ###
    # mean marker
    resampled_df1.means.plot(linestyle='none', marker='o', markerfacecolor=color1, markeredgecolor='w', zorder=99, label='mean $T_{B} - T_{ss}$', ax=ax[1])    
    # +/- 1 standard deviation error bars
    ax[1].errorbar(x=resampled_df1.index, 
                y=resampled_df1.means,
                yerr=resampled_df1.stds,
                fmt='none',
                linewidth=4,
                color=color1,
                alpha=0.3,
                capsize=None,
                label='$\pm 1 \sigma \, T_{B} - T_{ss}$')
    ### Resampled Dataframe 2, Timeseries "Boxplots" ###
    # mean marker
    resampled_df2.means.plot(linestyle='none', marker='o', markerfacecolor=color2, markeredgecolor='w', zorder=99, label='mean $T_{B} - T_{a}$', ax=ax[1])    
    # +/- 1 standard deviation error bars
    ax[1].errorbar(x=resampled_df2.index, 
                y=resampled_df2.means,
                yerr=resampled_df2.stds,
                fmt='none',
                linewidth=4,
                color=color2,
                alpha=0.3,
                capsize=None,
                label='$\pm 1 \sigma \, T_{B} - T_{a}$')
    ### Resampled Dataframe 3, Timeseries "Boxplots" ###
    resampled_df3.means.plot(linestyle='none', marker='o', markerfacecolor=color3, markeredgecolor='w', zorder=99, label='mean $T_{B} - T_{sfc}$', ax=ax[1])    
    # +/- 1 standard deviation error bars
    ax[1].errorbar(x=resampled_df3.index, 
                y=resampled_df3.means,
                yerr=resampled_df3.stds,
                fmt='none',
                linewidth=4,
                color=color3,
                alpha=0.3,
                capsize=None,
                label='$\pm 1 \sigma \, T_{B} - T_{sfc}$')
    
    # add zero lines
    ax[0].axhline(0,color='lightgrey',linestyle=':',lw=1)
    ax[1].axhline(0,color='lightgrey',linestyle=':',lw=1)
    plt.legend()
    
    # Format shared y-axis
    ax[0].set_ylim(ymin,ymax);
    ax[0].set_ylabel('[$\Delta\degree C$]')
    
    return fig, ax
