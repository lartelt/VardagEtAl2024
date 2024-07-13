# Author: Lukas Artelt
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cf
import numpy as np
import datetime
import geopandas
import netCDF4 as nc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
#sns.set_theme(style="darkgrid")
sns.set(rc={'axes.facecolor':'gainsboro'})


# calculate background xco2 timeseries
def background_xco2_time_series(annual_growth_rate_starting_from_2009: list[int], start_co2_ppm_2009: int, min_year: int, max_year: int) -> pd.DataFrame:
    '''# Documentation
    Function to create a timeseries of background xco2 values
    # input:
            annual_growth_rate_starting_from_2009: array of annual xco2 growth rates from min_year to max_year, in ppm
            min_year: 2009 for GOSAT
            max_year: always +1 than the last available year, for GOSAT use 2020 because data only exists until 2019
            start_co2_ppm_2009: xco2 offset that is assumed for min_year, in ppm
    # output:
            df: timeseries of monthly mean xCO2 background values, including Month_Date entries
    '''
    background_all_month = []
    background_month = []
    background_year = []
    
    for y in range(min_year, max_year):
            if y == 2009:
                for m in range(5,13):
                    background_all_month.append(start_co2_ppm_2009+annual_growth_rate_starting_from_2009[y-2009]/12*(m-5)) # rechnen Startwert * 6*1/12 des Jahresanstiegs für Juni
                    background_month.append(m)
                    background_year.append(y)
            else:
                for m in range(1,13):
                    background_all_month.append(background_all_month[-1]+annual_growth_rate_starting_from_2009[y-min_year]/12)
                    background_month.append(m)
                    background_year.append(y)
    
    background_dict = {'background_xco2' : background_all_month,'year':background_year,'month': background_month}
    background_time_series = pd.DataFrame(data=background_dict)
    background_MonthDate = background_time_series.apply(lambda x: datetime.date(int(x.year),int(x.month),15),axis=1)
    background_time_series.insert(loc = 1, column = 'MonthDate', value = background_MonthDate)
    return background_time_series

# calculate monthly means for xCO2 modelled by TM5
def calculate_monthly_means_modelled_xco2( gdf: geopandas, cosampled: bool=True) -> pd.DataFrame:
    '''# Documentation
    Function to calculate monthly mean xco2 values for a given time period as gdf. Use data for whole South America.
    # input:
               date_min (datetime.datetime object): datetime.datetime(2009,5,1) is the first day for a whole month. 
               date_max (datetime.datetime object): insert "datetime.datetime(2009,7,1)" to get months 5 & 6 because: when no time is inserted: h=m=s=0
               gdf: geodataframe created from dataframe of GOSAT ACOS data
    
    # output:
               MonthMeans: dataframe with the following values: 
                           MonthDate, MonthMeans_xco2, xco2_monthly_std_of_mean, xco2_monthly_mean_measurement_uncertainty, MonthMeans_tot_error
    '''
    MonthMeans = gdf.groupby(['MonthDate'])[['modeled_XCO2']].mean().reset_index() # group the gdf2 by MonthDate and get the mean of xco2 for each month as gdf
    # MonthMeans has 3 columns: Index, MonthDate, xco2. For one month, there is one entry in each column. xco2=mean xco2 value
    MonthMeans.rename(columns={'modeled_XCO2': 'modelled_xco2_monthly_mean'}, inplace=True) # rename the column xco2 because the entries are the std_dev values of the mean xco2 of this month
    if cosampled == True:
        MonthMeans_measured = gdf.groupby(['MonthDate'])[['measured_XCO2']].mean().reset_index() # group the gdf2 by MonthDate and get the mean of xco2 for each month as gdf
        MonthMeans_measured.rename(columns={'measured_XCO2': 'measured_xco2_monthly_mean'}, inplace=True) # rename the column xco2 because the entries are the std_dev values of the mean xco2 of this month
    MonthMeans_std = gdf.groupby(['MonthDate'])[['modeled_XCO2']].std().reset_index() # create another gdf, but now the entries are the standard deviation of the mean xco2
    MonthMeans_std.rename(columns={'modeled_XCO2': 'modelled_xco2_monthly_std_of_mean'}, inplace=True) # rename the column xco2 because the entries are the std_dev values of the mean xco2 of this month
    MonthMeans_std.insert(loc=1, column='modelled_xco2_monthly_mean',value=MonthMeans.modelled_xco2_monthly_mean) # insert the monthly mean xco2 value into the gdf
    if cosampled == True:
        MonthMeans_std.insert(loc=1, column='measured_xco2_monthly_mean',value=MonthMeans_measured.measured_xco2_monthly_mean) # insert the monthly mean xco2 value into the gdf
    MonthMeans_count = gdf.groupby(['MonthDate'])['MonthDate'].count().reset_index(name='counts')
    MonthMeans_std.insert(loc=1, column='counts_of_MonthDate',value=MonthMeans_count.counts) # insert number of entries of each month
    sigma_of_mean = MonthMeans_std.modelled_xco2_monthly_std_of_mean/np.sqrt(MonthMeans_std.counts_of_MonthDate)
    MonthMeans_std.insert(loc=3, column='modelled_xco2_error_of_mean', value=sigma_of_mean) # insert error of the mean
    
    return MonthMeans_std

# calculate the MSC
def calculateMeanSeasonalCycle(gdf_timeseries: pd.DataFrame, background_xco2_time_series: pd.DataFrame, cosampled: bool=True) -> pd.DataFrame:
    '''# Documentation
    Function to generate a df containing the mean detrended xCO2 per month and...
        the std_dev of the mean &
        the number of datapoints used for the mean &
        the sigma of the mean
    # arguments:
        gdf_timeseries: timeseries of monthly mean xCO2 values
        background_xco2_time_series: timeseries of monthly mean background xCO2 values
    '''
    Monthly_Means_gdf_timeseries = calculate_monthly_means_modelled_xco2(gdf_timeseries, cosampled=cosampled)
    Monthly_Means_gdf_timeseries = Monthly_Means_gdf_timeseries.merge(background_xco2_time_series, left_on='MonthDate', right_on='MonthDate', how='inner')
    Monthly_Means_gdf_timeseries.insert(loc=2, column='detrended_xco2_monthly_mean',
                                        value = Monthly_Means_gdf_timeseries.modelled_xco2_monthly_mean - Monthly_Means_gdf_timeseries.background_xco2)
    #calc mean seasonal month values for xco2_modelled - background
    gdf_mean_seasonal_cycle = Monthly_Means_gdf_timeseries.groupby(['month'])['detrended_xco2_monthly_mean'].mean().reset_index()
    gdf_mean_seasonal_cycle_std = Monthly_Means_gdf_timeseries.groupby(['month'])['detrended_xco2_monthly_mean'].std(ddof=0).reset_index()
    gdf_mean_seasonal_cycle_count = Monthly_Means_gdf_timeseries.groupby(['month'])['detrended_xco2_monthly_mean'].count().reset_index()
    gdf_mean_seasonal_cycle = gdf_mean_seasonal_cycle.merge(gdf_mean_seasonal_cycle_std, on='month', suffixes=('', '_std'))
    gdf_mean_seasonal_cycle = gdf_mean_seasonal_cycle.merge(gdf_mean_seasonal_cycle_count, on='month', suffixes=('', '_count'))
    gdf_mean_seasonal_cycle['detrended_xco2_monthly_sigma'] = gdf_mean_seasonal_cycle['detrended_xco2_monthly_mean_std'] / np.sqrt(gdf_mean_seasonal_cycle['detrended_xco2_monthly_mean_count'])
    return gdf_mean_seasonal_cycle


# calculations for measured xCO2
def calculate_monthly_means_for_measurements_with_xco2_raw(gdf: geopandas) -> pd.DataFrame:
    '''# Documentation
    Function to calculate monthly mean xco2 values for a given time period as gdf. Use data for whole South America.
    # input:
            gdf: geodataframe created from dataframe of GOSAT ACOS data
    
    # output:
            MonthMeans: dataframe with the following values: 
                        MonthDate, MonthMeans_xco2, xco2_monthly_std_of_mean, xco2_monthly_mean_measurement_uncertainty, MonthMeans_tot_error
    '''
    MonthMeans = gdf.groupby(['MonthDate'])[['xco2']].mean().reset_index() # group the gdf2 by MonthDate and get the mean of xco2 for each month as gdf
    MonthMeans_raw = gdf.groupby(['MonthDate'])[['xco2_raw']].mean().reset_index() # group the gdf2 by MonthDate and get the mean of xco2 for each month as gdf
    MonthMeans.insert(loc=2, column='xco2_raw_monthly_mean', value=MonthMeans_raw.xco2_raw)
    # MonthMeans has 3 columns: Index, MonthDate, xco2. For one month, there is one entry in each column. xco2=mean xco2 value
    MonthMeans.rename(columns={'xco2': 'xco2_monthly_mean'}, inplace=True) # rename the column xco2 because the entries are the std_dev values of the mean xco2 of this month
    
    MonthMeans_std = gdf.groupby(['MonthDate'])[['xco2']].std().reset_index() # create another gdf, but now the entries are the standard deviation of the mean xco2
    MonthMeans_std.rename(columns={'xco2': 'xco2_monthly_std_of_mean'}, inplace=True) # rename the column xco2 because the entries are the std_dev values of the mean xco2 of this month
    MonthMeans_std_raw = gdf.groupby(['MonthDate'])[['xco2_raw']].std().reset_index()
    MonthMeans_std.insert(loc=2, column='xco2_raw_monthly_std_of_mean', value=MonthMeans_std_raw.xco2_raw) # create another gdf, but now the entries are the standard deviation of the mean xco2
    
    #MonthMeans_std.insert(loc=2, column='xco2_monthly_mean_measurement_uncertainty',value=np.mean(gdf2.xco2_uncertainty)) # calc & insert the mean xco2 measurement uncertainty as column in the gdf
    MonthMeans_std.insert(loc=1, column='xco2_monthly_mean',value=MonthMeans.xco2_monthly_mean) # insert the monthly mean xco2 value into the gdf
    MonthMeans_std.insert(loc=1, column='xco2_raw_monthly_mean',value=MonthMeans.xco2_raw_monthly_mean) # insert the monthly mean xco2 value into the gdf
    
    MonthMeans_count = gdf.groupby(['MonthDate'])['MonthDate'].count().reset_index(name='counts')
    MonthMeans_std.insert(loc=1, column='counts_of_MonthDate',value=MonthMeans_count.counts) # insert number of entries of each month
    sigma_of_mean = MonthMeans_std.xco2_monthly_std_of_mean/np.sqrt(MonthMeans_std.counts_of_MonthDate)
    sigma_of_mean_raw = MonthMeans_std.xco2_raw_monthly_std_of_mean/np.sqrt(MonthMeans_std.counts_of_MonthDate)
    MonthMeans_std.insert(loc=2, column='xco2_error_of_mean', value=sigma_of_mean) # insert error of the mean
    MonthMeans_std.insert(loc=2, column='xco2_raw_error_of_mean', value=sigma_of_mean_raw) # insert error of the mean
    
    return MonthMeans_std

def calculateMeanSeasonalCycle_measurements_with_xco2_raw(gdf_timeseries: pd.DataFrame = None, background_xco2_time_series: pd.DataFrame = None):
    '''# Documentation
    Function to generate a df containing the mean detrended xCO2 per month and...
        the std_dev of the mean &
        the number of datapoints used for the mean &
        the sigma of the mean
    # arguments:
        gdf_timeseries: timeseries of monthly mean xCO2 values
        background_xco2_time_series: timeseries of monthly mean background xCO2 values
    '''
    Monthly_Means_gdf_timeseries = calculate_monthly_means_for_measurements_with_xco2_raw(gdf_timeseries)
    Monthly_Means_gdf_timeseries = Monthly_Means_gdf_timeseries.merge(background_xco2_time_series, left_on='MonthDate', right_on='MonthDate', how='inner')
    Monthly_Means_gdf_timeseries.insert(loc=2, column='detrended_xco2_raw_monthly_mean',
                                        value = Monthly_Means_gdf_timeseries.xco2_raw_monthly_mean - Monthly_Means_gdf_timeseries.background_xco2)
    Monthly_Means_gdf_timeseries.insert(loc=2, column='detrended_xco2_monthly_mean',
                                        value = Monthly_Means_gdf_timeseries.xco2_monthly_mean - Monthly_Means_gdf_timeseries.background_xco2)
    #calc mean seasonal month values for xco2_modelled - background
    gdf_mean_seasonal_cycle = Monthly_Means_gdf_timeseries.groupby(['month'])['detrended_xco2_monthly_mean'].mean().reset_index()
    gdf_mean_seasonal_cycle_std = Monthly_Means_gdf_timeseries.groupby(['month'])['detrended_xco2_monthly_mean'].std(ddof=0).reset_index()
    gdf_mean_seasonal_cycle_count = Monthly_Means_gdf_timeseries.groupby(['month'])['counts_of_MonthDate'].mean().reset_index()
    gdf_mean_seasonal_cycle_count_std = Monthly_Means_gdf_timeseries.groupby(['month'])['counts_of_MonthDate'].std().reset_index()
    gdf_mean_seasonal_cycle = gdf_mean_seasonal_cycle.merge(gdf_mean_seasonal_cycle_std, on='month', suffixes=('', '_std'))
    gdf_mean_seasonal_cycle = gdf_mean_seasonal_cycle.merge(gdf_mean_seasonal_cycle_count, on='month')
    gdf_mean_seasonal_cycle = gdf_mean_seasonal_cycle.merge(gdf_mean_seasonal_cycle_count_std, on='month', suffixes=('', '_std'))
    
    gdf_mean_seasonal_cycle_raw = Monthly_Means_gdf_timeseries.groupby(['month'])['detrended_xco2_raw_monthly_mean'].mean().reset_index()
    gdf_mean_seasonal_cycle_std_raw = Monthly_Means_gdf_timeseries.groupby(['month'])['detrended_xco2_raw_monthly_mean'].std(ddof=0).reset_index()
    gdf_mean_seasonal_cycle = gdf_mean_seasonal_cycle.merge(gdf_mean_seasonal_cycle_raw, on='month')
    gdf_mean_seasonal_cycle = gdf_mean_seasonal_cycle.merge(gdf_mean_seasonal_cycle_std_raw, on='month', suffixes=('', '_std'))
    
    return gdf_mean_seasonal_cycle


# GENERAL plotting functions
def plot_xCO2_Timeseries_and_MSC_general(start_month: int = 5, start_year: int=2009, end_year: int=2019, 
                                         background_xCO2_timeseries: pd.DataFrame=pd.DataFrame(),
                                         gdf_list: list=[],
                                         cosampled_list: list=[True, False],
                                         model_ass_cos_list: list=['TM5-4DVar_apri_cos_RT', 'TM5-4DVar_IS_RT_NOT_cos'],
                                         label_list: list=[],
                                         what_is_plotted: str='Modelled_xCO2',
                                         color_list: list=['dimgrey', 'forestgreen', 'royalblue', 'firebrick'],
                                         linestyle_list: list=['dashed', 'solid'],
                                         region_name: str='SAT', savepath: str='/home/lartelt/MA/software/', compare_case: int=1):
    '''# Documentation
    Function to plot timeseries of the mean seasonal cycle of xCO2 modelled by TM5, MIP & next to it the mean seasonal cycle
    # arguments:
        - gdf_list: list of the gdfs that should be used in this plot
        - variable_to_plot_list: list of variables that should be plotted for the respective gdf type
        - model_assimilation_list: list of model and assimilation types that are used in this plot. All combinations of the following list should work:
            - ['TM5-4DVar prior', 'TM5-4DVar/IS', 'TM5-4DVar/ACOS+IS', 'TM5-4DVar/RT+IS', 'TM5-4DVar/IS+OCO2', 'MIP/IS_ens', 'MIP/IS+OCO2_ens']
        - what_is_plotted: str; only used for title & savename, e.g. 'NEE+fire_flux_regional' or in future maybe also sth with 'xCO2'
        - color_list: list of colors for the assimilation types
        - linestyle_list: list of linestyles for the assimilation types
        - savefig: bool, save fig or do not save fig
        - compare_case: int
            - Case 1: eight timeseries: Prior & IS & IS_ACOS & IS_RT for TM5_regional_Sourish & TM5_gridded_3x2 or 1x1 --> expect that regional & gridded are equal for 3x2 grid
        - region_name: str
            - 'SAT', 'SATr', 'SAT_SATr'
                    
    '''
    print('Start plotting...')
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5), sharey=True, gridspec_kw={'width_ratios':[8,3]})
    fig.subplots_adjust(wspace=0)
    
    first_month = start_month-2 # needed because df.loc[:5] will also affect month=7
    savename = what_is_plotted
    
    if compare_case==1:
        for i,gdf in enumerate(gdf_list):
            Monthly_Means = calculate_monthly_means_modelled_xco2(gdf, cosampled=cosampled_list[i])
            Monthly_Means = Monthly_Means.merge(background_xCO2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
            detrended_values = Monthly_Means.modelled_xco2_monthly_mean - Monthly_Means.background_xco2
            #ax1.plot(Monthly_Means.MonthDate, detrended_values, color=color_list[i], linestyle=linestyle_list[i], label=model_ass_cos_list[i].replace('_', ' '), linewidth=1.25)
            ax1.plot(Monthly_Means.MonthDate, detrended_values, color=color_list[i], linestyle=linestyle_list[i], label=label_list[i], linewidth=1.25)
            print('MSC is calculated & plotted')
            msc = calculateMeanSeasonalCycle(gdf, background_xCO2_timeseries, cosampled=cosampled_list[i])
            msc.loc[:first_month,'month'] += 12
            msc_new = pd.DataFrame(pd.concat([msc.loc[first_month+1:, 'month'], msc.loc[:first_month, 'month']], ignore_index=True))
            msc = pd.merge(msc_new, msc, on='month', how='outer')
            ax2.plot(msc.month, msc.detrended_xco2_monthly_mean, color=color_list[i], linestyle=linestyle_list[i], label=model_ass_cos_list[i], linewidth=1.25)
            ax2.fill_between(msc.month,
                             msc.detrended_xco2_monthly_mean - msc.detrended_xco2_monthly_mean_std, 
                             msc.detrended_xco2_monthly_mean + msc.detrended_xco2_monthly_mean_std, 
                             color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)
            if i!=len(gdf_list)-1:
                savename = savename + '_' + model_ass_cos_list[i] + '_&'
            else:
                savename = savename + '_' + model_ass_cos_list[i] + '_timeseries_AND_msc_' + region_name + '.png'

    fig.suptitle(what_is_plotted+' for '+region_name, y=0.92, fontsize=12)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('xCO$_2$ [ppm]', color='black')
    ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax1.minorticks_on()
    ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax1.tick_params(bottom=True, left=True, color='gray')
    ax1.legend(framealpha=0.4, facecolor='grey', loc='lower left', fontsize=9.5, ncol=2, handletextpad=0.3, columnspacing=0.1)#columnspacing=0.6)
    ax1.set_yticks([-2, -1, 0, 1, 2])
    ax1.set_yticklabels([-2, -1, 0, 1, 2])
    ax1.set_ylim([-2.2, 2.2])
    #ax1.set_title('TM5-4DVar XCO$_2$ timeseries '+region, pad=-13)
    #MSC axis
    ax2.set_xlabel('Month')
    ax2.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax2.set_xticks([5,10,15])
    ax2.set_xticklabels([5, 10, 3]) # set the x-axis tick labels
    ax2.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax2.tick_params(bottom=True, color='gray')
    ax2.set_title('MSC '+str(start_year)+'-'+str(end_year))#, pad=-13)
    ax2.set_ylim([-2.2, 2.2])
    
    print('Saving plot...')
    if compare_case==1:
        plt.savefig(savepath + savename, dpi=300, bbox_inches='tight')
    print('DONE plotting!')



# compare ONE subregion: Prior, TM5/IS, TM5/IS+ACOS, TM5/IS+RT
def plotTimeseries_MSC_start_May_subregion(start_month: int=5,
                                           TM5_xCO2_prior_subregion: pd.DataFrame = None,
                                           TM5_xCO2_IS_subregion: pd.DataFrame = None,
                                           TM5_xCO2_IS_ACOS_subregion: pd.DataFrame = None,
                                           TM5_xCO2_IS_RT_loc_subregion: pd.DataFrame = None,
                                           background_xco2_timeseries: pd.DataFrame = None, 
                                           cosampled_on: str = 'RT', region: str='SAT', 
                                           savepath: str='', compare_case: int=1):
    '''# Documentation
    Compare ONE subregion: Prior, TM5/IS, TM5/IS+ACOS, TM5/IS+RT
    Function to plot timeseries & mean seasonal cycle of xCO2 by TM5 for each subregion
    # arguments:
            X
            savefig: bool, save fig or do not save fig
            compare_case: int
                    Case 1: four timeseries plotted: TM5_prior_cos_RT & TM5_IS_cos_RT & TM5_IS_ACOS_cos_RT & TM5_IS_RT_cos_RT
                    Case 2: two timeseries plotted: NOT_cosampled TM5_IS & TM5_IS_RT
            region: str
                    - SAT
                    - north_SAT
                    - mid_SAT
                    - south_SAT
                    - west_SAT
                    - mid_long_SAT
                    - east_SAT
    '''
    #ACOS_color = 'royalblue'
    ACOS_color = 'firebrick'
    #RT_color='firebrick'
    RT_color = 'coral'
    IS_color = 'indianred'
    #IS_color = 'forestgreen'
    Apri_color = 'dimgrey'
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5), sharey=True, gridspec_kw={'width_ratios':[8,3]})
    fig.subplots_adjust(wspace=0)
    
    if compare_case==1:
        print('Case 1: four timeseries plotted: TM5_prior_cos_RT & TM5_IS_cos_RT & TM5_IS_ACOS_cos_RT & TM5_IS_RT_cos_RT')
        # TM5/apri cos RT
        Monthly_Means_apri_cos_RT = calculate_monthly_means_modelled_xco2(TM5_xCO2_prior_subregion)
        Monthly_Means_apri_cos_RT = Monthly_Means_apri_cos_RT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_apri_cos_RT = Monthly_Means_apri_cos_RT.modelled_xco2_monthly_mean - Monthly_Means_apri_cos_RT.background_xco2
        # TM5/IS_cos_RT
        Monthly_Means_IS_ass_cos_RT = calculate_monthly_means_modelled_xco2(TM5_xCO2_IS_subregion)
        Monthly_Means_IS_ass_cos_RT = Monthly_Means_IS_ass_cos_RT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_IS_ass_cos_RT = Monthly_Means_IS_ass_cos_RT.modelled_xco2_monthly_mean - Monthly_Means_IS_ass_cos_RT.background_xco2
        # TM5/IS_ACOS_cos_RT
        Monthly_Means_IS_ACOS_ass_cos_RT = calculate_monthly_means_modelled_xco2(TM5_xCO2_IS_ACOS_subregion)
        Monthly_Means_IS_ACOS_ass_cos_RT = Monthly_Means_IS_ACOS_ass_cos_RT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_IS_ACOS_ass_cos_RT = Monthly_Means_IS_ACOS_ass_cos_RT.modelled_xco2_monthly_mean - Monthly_Means_IS_ACOS_ass_cos_RT.background_xco2
        # TM5/IS_RT_cos_RT
        Monthly_Means_IS_RT_ass_cos_RT = calculate_monthly_means_modelled_xco2(TM5_xCO2_IS_RT_loc_subregion)
        Monthly_Means_IS_RT_ass_cos_RT = Monthly_Means_IS_RT_ass_cos_RT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_IS_RT_ass_cos_RT = Monthly_Means_IS_RT_ass_cos_RT.modelled_xco2_monthly_mean - Monthly_Means_IS_RT_ass_cos_RT.background_xco2
        
        ax1.plot(Monthly_Means_apri_cos_RT.MonthDate, detrended_values_modelled_apri_cos_RT, color=Apri_color, linestyle='dashed', label = 'TM5-4DVar apri cos. RT', linewidth=1.25)
        ax1.plot(Monthly_Means_IS_ass_cos_RT.MonthDate, detrended_values_modelled_IS_ass_cos_RT, color=IS_color, linestyle='dashed', label = 'TM5-4DVar/IS cos. RT', linewidth=1.25)
        ax1.plot(Monthly_Means_IS_ACOS_ass_cos_RT.MonthDate, detrended_values_modelled_IS_ACOS_ass_cos_RT, color=ACOS_color, label = 'TM5-4DVar/IS+ACOS cos. RT', linewidth=1.25)
        ax1.plot(Monthly_Means_IS_RT_ass_cos_RT.MonthDate, detrended_values_modelled_IS_RT_ass_cos_RT, color=RT_color, label = 'TM5-4DVar/IS+RT cos. RT', linewidth=1.25)
        
        print('Case 1: four MSCs plotted: TM5_prior_cos_RT & TM5_IS_cos_RT & TM5_IS_ACOS_cos_RT & TM5_IS_RT_cos_RT')
        msc_apri_cos_RT = calculateMeanSeasonalCycle(TM5_xCO2_prior_subregion, background_xco2_timeseries)
        msc_IS_ass_cos_RT = calculateMeanSeasonalCycle(TM5_xCO2_IS_subregion, background_xco2_timeseries)
        msc_IS_ACOS_ass_cos_RT = calculateMeanSeasonalCycle(TM5_xCO2_IS_ACOS_subregion, background_xco2_timeseries)
        msc_IS_RT_ass_cos_RT = calculateMeanSeasonalCycle(TM5_xCO2_IS_RT_loc_subregion, background_xco2_timeseries)
        # if start_month=5 -> Mai will be the first month that is plotted
        first_month = start_month-2 # needed because df.loc[:5] will also affect month=7
        msc_apri_cos_RT.loc[:first_month,'month'] += 12
        msc_IS_ass_cos_RT.loc[:first_month,'month'] += 12
        msc_IS_ACOS_ass_cos_RT.loc[:first_month,'month'] += 12
        msc_IS_RT_ass_cos_RT.loc[:first_month,'month'] += 12
        # concatenate the two data frames vertically
        msc_apri_cos_RT_new = pd.DataFrame(pd.concat([msc_apri_cos_RT.loc[first_month+1:, 'month'], msc_apri_cos_RT.loc[:first_month, 'month']], ignore_index=True))
        msc_IS_ass_cos_RT_new = pd.DataFrame(pd.concat([msc_IS_ass_cos_RT.loc[first_month+1:, 'month'], msc_IS_ass_cos_RT.loc[:first_month, 'month']], ignore_index=True))
        msc_IS_ACOS_ass_cos_RT_new = pd.DataFrame(pd.concat([msc_IS_ACOS_ass_cos_RT.loc[first_month+1:, 'month'], msc_IS_ACOS_ass_cos_RT.loc[:first_month, 'month']], ignore_index=True))
        msc_IS_RT_ass_cos_RT_new = pd.DataFrame(pd.concat([msc_IS_RT_ass_cos_RT.loc[first_month+1:, 'month'], msc_IS_RT_ass_cos_RT.loc[:first_month, 'month']], ignore_index=True))
        # merge 
        msc_apri_cos_RT = pd.merge(msc_apri_cos_RT_new, msc_apri_cos_RT, on='month', how='outer')
        msc_IS_ass_cos_RT = pd.merge(msc_IS_ass_cos_RT_new, msc_IS_ass_cos_RT, on='month', how='outer')
        msc_IS_ACOS_ass_cos_RT = pd.merge(msc_IS_ACOS_ass_cos_RT_new, msc_IS_ACOS_ass_cos_RT, on='month', how='outer')
        msc_IS_RT_ass_cos_RT = pd.merge(msc_IS_RT_ass_cos_RT_new, msc_IS_RT_ass_cos_RT, on='month', how='outer')
        
        ax2.plot(msc_apri_cos_RT.month, msc_apri_cos_RT.detrended_xco2_monthly_mean, color = Apri_color, linestyle='dashed', label = 'TM5-4DVar apri cos. RT', linewidth=1.25)
        ax2.plot(msc_IS_ass_cos_RT.month, msc_IS_ass_cos_RT.detrended_xco2_monthly_mean, color = IS_color, linestyle='dashed', label = 'TM5-4DVar/IS cos. RT', linewidth=1.25)
        ax2.plot(msc_IS_ACOS_ass_cos_RT.month, msc_IS_ACOS_ass_cos_RT.detrended_xco2_monthly_mean, color = ACOS_color, label = 'TM5-4DVar/IS+ACOS cos. RT', linewidth=1.25)
        ax2.plot(msc_IS_RT_ass_cos_RT.month, msc_IS_RT_ass_cos_RT.detrended_xco2_monthly_mean, color = RT_color, label = 'TM5-4DVar/IS+RT cos. RT', linewidth=1.25)
        
        ax2.fill_between(msc_apri_cos_RT.month,
                        msc_apri_cos_RT.detrended_xco2_monthly_mean - msc_apri_cos_RT.detrended_xco2_monthly_mean_std, 
                        msc_apri_cos_RT.detrended_xco2_monthly_mean + msc_apri_cos_RT.detrended_xco2_monthly_mean_std, 
                        color=Apri_color, linestyle='dashed', alpha=0.2)
        ax2.fill_between(msc_IS_ass_cos_RT.month,
                        msc_IS_ass_cos_RT.detrended_xco2_monthly_mean - msc_IS_ass_cos_RT.detrended_xco2_monthly_mean_std, 
                        msc_IS_ass_cos_RT.detrended_xco2_monthly_mean + msc_IS_ass_cos_RT.detrended_xco2_monthly_mean_std, 
                        color=IS_color, linestyle='dashed', alpha=0.2)
        ax2.fill_between(msc_IS_ACOS_ass_cos_RT.month,
                        msc_IS_ACOS_ass_cos_RT.detrended_xco2_monthly_mean - msc_IS_ACOS_ass_cos_RT.detrended_xco2_monthly_mean_std, 
                        msc_IS_ACOS_ass_cos_RT.detrended_xco2_monthly_mean + msc_IS_ACOS_ass_cos_RT.detrended_xco2_monthly_mean_std, 
                        color=ACOS_color, alpha=0.2)
        ax2.fill_between(msc_IS_RT_ass_cos_RT.month,
                        msc_IS_RT_ass_cos_RT.detrended_xco2_monthly_mean - msc_IS_RT_ass_cos_RT.detrended_xco2_monthly_mean_std, 
                        msc_IS_RT_ass_cos_RT.detrended_xco2_monthly_mean + msc_IS_RT_ass_cos_RT.detrended_xco2_monthly_mean_std, 
                        color=RT_color, alpha=0.2)
    if compare_case==2:
        print('Case 2: for NOT cosampled: three timeseries plotted: TM5_IS & TM5_IS_ACOS & TM5_IS_RT')
        # TM5/IS_cos_RT
        Monthly_Means_IS_ass_cos_RT = calculate_monthly_means_modelled_xco2(TM5_xCO2_IS_subregion, cosampled=False)
        Monthly_Means_IS_ass_cos_RT = Monthly_Means_IS_ass_cos_RT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_IS_ass_cos_RT = Monthly_Means_IS_ass_cos_RT.modelled_xco2_monthly_mean - Monthly_Means_IS_ass_cos_RT.background_xco2
        # TM5/IS_ACOS_cos_RT
        Monthly_Means_IS_ACOS_ass_cos_RT = calculate_monthly_means_modelled_xco2(TM5_xCO2_IS_ACOS_subregion, cosampled=False)
        Monthly_Means_IS_ACOS_ass_cos_RT = Monthly_Means_IS_ACOS_ass_cos_RT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_IS_ACOS_ass_cos_RT = Monthly_Means_IS_ACOS_ass_cos_RT.modelled_xco2_monthly_mean - Monthly_Means_IS_ACOS_ass_cos_RT.background_xco2
        # TM5/IS_RT_cos_RT
        Monthly_Means_IS_RT_ass_cos_RT = calculate_monthly_means_modelled_xco2(TM5_xCO2_IS_RT_loc_subregion, cosampled=False)
        Monthly_Means_IS_RT_ass_cos_RT = Monthly_Means_IS_RT_ass_cos_RT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_IS_RT_ass_cos_RT = Monthly_Means_IS_RT_ass_cos_RT.modelled_xco2_monthly_mean - Monthly_Means_IS_RT_ass_cos_RT.background_xco2
        
        ax1.plot(Monthly_Means_IS_ass_cos_RT.MonthDate, detrended_values_modelled_IS_ass_cos_RT, color=IS_color, linestyle='dashed', label = 'TM5-4DVar/IS NOT cos', linewidth=1.25)
        ax1.plot(Monthly_Means_IS_ACOS_ass_cos_RT.MonthDate, detrended_values_modelled_IS_ACOS_ass_cos_RT, color=ACOS_color, label = 'TM5-4DVar/IS+ACOS NOT cos', linewidth=1.25)
        ax1.plot(Monthly_Means_IS_RT_ass_cos_RT.MonthDate, detrended_values_modelled_IS_RT_ass_cos_RT, color=RT_color, label = 'TM5-4DVar/IS+RT NOT cos', linewidth=1.25)
        
        print('Case 2: 3 MSCs plotted')
        msc_IS_ass_cos_RT = calculateMeanSeasonalCycle(TM5_xCO2_IS_subregion, background_xco2_timeseries, cosampled=False)
        msc_IS_ACOS_ass_cos_RT = calculateMeanSeasonalCycle(TM5_xCO2_IS_ACOS_subregion, background_xco2_timeseries, cosampled=False)
        msc_IS_RT_ass_cos_RT = calculateMeanSeasonalCycle(TM5_xCO2_IS_RT_loc_subregion, background_xco2_timeseries, cosampled=False)
        # if start_month=5 -> Mai will be the first month that is plotted
        first_month = start_month-2 # needed because df.loc[:5] will also affect month=7
        msc_IS_ass_cos_RT.loc[:first_month,'month'] += 12
        msc_IS_ACOS_ass_cos_RT.loc[:first_month,'month'] += 12
        msc_IS_RT_ass_cos_RT.loc[:first_month,'month'] += 12
        # concatenate the two data frames vertically
        msc_IS_ass_cos_RT_new = pd.DataFrame(pd.concat([msc_IS_ass_cos_RT.loc[first_month+1:, 'month'], msc_IS_ass_cos_RT.loc[:first_month, 'month']], ignore_index=True))
        msc_IS_ACOS_ass_cos_RT_new = pd.DataFrame(pd.concat([msc_IS_ACOS_ass_cos_RT.loc[first_month+1:, 'month'], msc_IS_ACOS_ass_cos_RT.loc[:first_month, 'month']], ignore_index=True))
        msc_IS_RT_ass_cos_RT_new = pd.DataFrame(pd.concat([msc_IS_RT_ass_cos_RT.loc[first_month+1:, 'month'], msc_IS_RT_ass_cos_RT.loc[:first_month, 'month']], ignore_index=True))
        # merge 
        msc_IS_ass_cos_RT = pd.merge(msc_IS_ass_cos_RT_new, msc_IS_ass_cos_RT, on='month', how='outer')
        msc_IS_ACOS_ass_cos_RT = pd.merge(msc_IS_ACOS_ass_cos_RT_new, msc_IS_ACOS_ass_cos_RT, on='month', how='outer')
        msc_IS_RT_ass_cos_RT = pd.merge(msc_IS_RT_ass_cos_RT_new, msc_IS_RT_ass_cos_RT, on='month', how='outer')
        
        ax2.plot(msc_IS_ass_cos_RT.month, msc_IS_ass_cos_RT.detrended_xco2_monthly_mean, color = IS_color, linestyle='dashed', label = 'TM5-4DVar/IS NOT cos', linewidth=1.25)
        ax2.plot(msc_IS_ACOS_ass_cos_RT.month, msc_IS_ACOS_ass_cos_RT.detrended_xco2_monthly_mean, color = ACOS_color, label = 'TM5-4DVar/IS+ACOS NOT cos', linewidth=1.25)
        ax2.plot(msc_IS_RT_ass_cos_RT.month, msc_IS_RT_ass_cos_RT.detrended_xco2_monthly_mean, color = RT_color, label = 'TM5-4DVar/IS+RT NOT cos', linewidth=1.25)
        
        ax2.fill_between(msc_IS_ass_cos_RT.month,
                        msc_IS_ass_cos_RT.detrended_xco2_monthly_mean - msc_IS_ass_cos_RT.detrended_xco2_monthly_mean_std, 
                        msc_IS_ass_cos_RT.detrended_xco2_monthly_mean + msc_IS_ass_cos_RT.detrended_xco2_monthly_mean_std, 
                        color=IS_color, linestyle='dashed', alpha=0.2)
        ax2.fill_between(msc_IS_ACOS_ass_cos_RT.month,
                        msc_IS_ACOS_ass_cos_RT.detrended_xco2_monthly_mean - msc_IS_ACOS_ass_cos_RT.detrended_xco2_monthly_mean_std, 
                        msc_IS_ACOS_ass_cos_RT.detrended_xco2_monthly_mean + msc_IS_ACOS_ass_cos_RT.detrended_xco2_monthly_mean_std, 
                        color=ACOS_color, alpha=0.2)
        ax2.fill_between(msc_IS_RT_ass_cos_RT.month,
                        msc_IS_RT_ass_cos_RT.detrended_xco2_monthly_mean - msc_IS_RT_ass_cos_RT.detrended_xco2_monthly_mean_std, 
                        msc_IS_RT_ass_cos_RT.detrended_xco2_monthly_mean + msc_IS_RT_ass_cos_RT.detrended_xco2_monthly_mean_std, 
                        color=RT_color, alpha=0.2)

    
    # timeseries axis
    ax1.set_xlabel('Year')
    ax1.set_ylabel('XCO$_2$ [ppm]', color='black')
    ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax1.minorticks_on()
    ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax1.tick_params(bottom=True, left=True, color='gray')
    ax1.legend(framealpha=0.25, facecolor='grey', loc='lower left', fontsize=9.5, ncol=2, columnspacing=0.8)
    ax1.set_title('TM5-4DVar XCO$_2$ timeseries '+region, pad=-13)
    ax1.set_yticks([-2, -1, 0, 1, 2])
    ax1.set_yticklabels([-2, -1, 0, 1, 2])
    ax1.set_ylim([-2.2, 2.2])
    #MSC axis
    ax2.set_xlabel('Month')
    ax2.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax2.set_xticks([5,10,15])
    ax2.set_xticklabels([5, 10, 3]) # set the x-axis tick labels
    ax2.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax2.tick_params(bottom=True, color='gray')
    ax2.set_title('MSC 2009-2019', pad=-13)
    ax2.set_ylim([-2.2, 2.2])

    if compare_case==1:
        plt.savefig(savepath + 'TM5_apri_ass_TM5_IS_ass_TM5_IS_ACOS_ass_TM5_IS_RT_all_cos_'+cosampled_on+'_xCO2_timeseries_AND_mean_seasonal_cycle_std_dev_' + region + '.png', dpi=300, bbox_inches='tight')
    if compare_case==2:
        plt.savefig(savepath + 'TM5_IS_ass_TM5_IS_ACOS_ass_TM5_IS_RT_NOT_cos_xCO2_timeseries_AND_mean_seasonal_cycle_std_dev_' + region + '.png', dpi=300, bbox_inches='tight')
    


def plotTimeseries_MSC_start_May_subregion_comparison(start_month: int=5,
                                           TM5_xCO2_SAT: pd.DataFrame = None,
                                           TM5_xCO2_north_SAT: pd.DataFrame = None,
                                           TM5_xCO2_mid_SAT: pd.DataFrame = None,
                                           TM5_xCO2_south_SAT: pd.DataFrame = None,
                                           background_xco2_timeseries: pd.DataFrame = None, 
                                           cosampled_on: str = 'RemoteC',
                                           assimilated: str='IS', savepath: str='', compare_case: int=1, plot_std_error:bool=True, savefig: bool=True):
    '''# Documentation
    Function to plot timeseries & mean seasonal cycle of TM5 xCO2 for each subregion
    # arguments:
            X
            savefig: bool, save fig or do not save fig
            assimilated: str
                    - IS
                    - IS_ACOS
                    - IS_RT
            compare_case: int
                    Case 1: for north,mid,south SAT:    four timeseries plotted: SAT, north SAT, mid SAT, south SAT gridded
                    Case 2: for west,mid_long,east SAT: four timeseries plotted: SAT, west SAT, mid_long SAT, east SAT gridded
                            Fluxes_ACOS_IS_north_SAT -> Fluxes_ACOS_IS_west_SAT
                            Fluxes_ACOS_IS_mid_SAT   -> Fluxes_ACOS_IS_mid_long_SAT
                            Fluxes_ACOS_IS_south_SAT -> Fluxes_ACOS_IS_east_SAT
                    Case 11: for NOT cosampled: north,mid,south SAT: four timeseries plotted: SAT, north SAT, mid SAT, south SAT gridded
                    Case 22: for NOT cosampled: west,mid_long,east SAT: four timeseries plotted: SAT, west SAT, mid_long SAT, east SAT gridded
                             Fluxes_ACOS_IS_north_SAT -> Fluxes_ACOS_IS_west_SAT
                             Fluxes_ACOS_IS_mid_SAT   -> Fluxes_ACOS_IS_mid_long_SAT
                             Fluxes_ACOS_IS_south_SAT -> Fluxes_ACOS_IS_east_SAT
    '''
    SAT_color = 'dimgrey'
    nSAT_color = 'royalblue'
    mSAT_color = 'firebrick'
    sSAT_color = 'forestgreen'
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5), sharey=True, gridspec_kw={'width_ratios':[8,3]})
    fig.subplots_adjust(wspace=0)
    
    if compare_case==1:
        print('Case 1: four timeseries plotted: SAT, north SAT, mid SAT, south SAT gridded')
        # TM5/apri cos RT
        Monthly_Means_SAT = calculate_monthly_means_modelled_xco2(TM5_xCO2_SAT, cosampled=True)
        Monthly_Means_SAT = Monthly_Means_SAT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_SAT = Monthly_Means_SAT.modelled_xco2_monthly_mean - Monthly_Means_SAT.background_xco2
        # TM5/IS_cos_RT
        Monthly_Means_north_SAT = calculate_monthly_means_modelled_xco2(TM5_xCO2_north_SAT, cosampled=True)
        Monthly_Means_north_SAT = Monthly_Means_north_SAT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_north_SAT = Monthly_Means_north_SAT.modelled_xco2_monthly_mean - Monthly_Means_north_SAT.background_xco2
        # TM5/IS_ACOS_cos_RT
        Monthly_Means_mid_SAT = calculate_monthly_means_modelled_xco2(TM5_xCO2_mid_SAT, cosampled=True)
        Monthly_Means_mid_SAT = Monthly_Means_mid_SAT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_mid_SAT = Monthly_Means_mid_SAT.modelled_xco2_monthly_mean - Monthly_Means_mid_SAT.background_xco2
        # TM5/IS_RT_cos_RT
        Monthly_Means_south_SAT = calculate_monthly_means_modelled_xco2(TM5_xCO2_south_SAT, cosampled=True)
        Monthly_Means_south_SAT = Monthly_Means_south_SAT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_south_SAT = Monthly_Means_south_SAT.modelled_xco2_monthly_mean - Monthly_Means_south_SAT.background_xco2
        
        ax1.plot(Monthly_Means_SAT.MonthDate, detrended_values_modelled_SAT, color=SAT_color, linestyle='dashed', label = 'TM5-4DVar SAT', linewidth=1.25)
        ax1.plot(Monthly_Means_north_SAT.MonthDate, detrended_values_modelled_north_SAT, color=nSAT_color, linestyle='solid', label = 'TM5-4DVar north SAT', linewidth=1.25)
        ax1.plot(Monthly_Means_mid_SAT.MonthDate, detrended_values_modelled_mid_SAT, color=mSAT_color, label = 'TM5-4DVar mid SAT', linewidth=1.25)
        ax1.plot(Monthly_Means_south_SAT.MonthDate, detrended_values_modelled_south_SAT, color=sSAT_color, label = 'TM5-4DVar south SAT', linewidth=1.25)
        
        print('Case 1: four MSCs plotted: SAT, north SAT, mid SAT, south SAT gridded')
        msc_SAT = calculateMeanSeasonalCycle(TM5_xCO2_SAT, background_xco2_timeseries, cosampled=True)
        msc_north_SAT = calculateMeanSeasonalCycle(TM5_xCO2_north_SAT, background_xco2_timeseries, cosampled=True)
        msc_mid_SAT = calculateMeanSeasonalCycle(TM5_xCO2_mid_SAT, background_xco2_timeseries, cosampled=True)
        msc_south_SAT = calculateMeanSeasonalCycle(TM5_xCO2_south_SAT, background_xco2_timeseries, cosampled=True)
        # if start_month=5 -> Mai will be the first month that is plotted
        first_month = start_month-2 # needed because df.loc[:5] will also affect month=7
        msc_SAT.loc[:first_month,'month'] += 12
        msc_north_SAT.loc[:first_month,'month'] += 12
        msc_mid_SAT.loc[:first_month,'month'] += 12
        msc_south_SAT.loc[:first_month,'month'] += 12
        # concatenate the two data frames vertically
        msc_SAT_new = pd.DataFrame(pd.concat([msc_SAT.loc[first_month+1:, 'month'], msc_SAT.loc[:first_month, 'month']], ignore_index=True))
        msc_north_SAT_new = pd.DataFrame(pd.concat([msc_north_SAT.loc[first_month+1:, 'month'], msc_north_SAT.loc[:first_month, 'month']], ignore_index=True))
        msc_mid_SAT_new = pd.DataFrame(pd.concat([msc_mid_SAT.loc[first_month+1:, 'month'], msc_mid_SAT.loc[:first_month, 'month']], ignore_index=True))
        msc_south_SAT_new = pd.DataFrame(pd.concat([msc_south_SAT.loc[first_month+1:, 'month'], msc_south_SAT.loc[:first_month, 'month']], ignore_index=True))
        # merge 
        msc_SAT = pd.merge(msc_SAT_new, msc_SAT, on='month', how='outer')
        msc_north_SAT = pd.merge(msc_north_SAT_new, msc_north_SAT, on='month', how='outer')
        msc_mid_SAT = pd.merge(msc_mid_SAT_new, msc_mid_SAT, on='month', how='outer')
        msc_south_SAT = pd.merge(msc_south_SAT_new, msc_south_SAT, on='month', how='outer')
        
        ax2.plot(msc_SAT.month, msc_SAT.detrended_xco2_monthly_mean, color = SAT_color, linestyle='dashed', label = 'TM5-4DVar SAT', linewidth=1.25)
        ax2.plot(msc_north_SAT.month, msc_north_SAT.detrended_xco2_monthly_mean, color = nSAT_color, linestyle='solid', label = 'TM5-4DVar north SAT', linewidth=1.25)
        ax2.plot(msc_mid_SAT.month, msc_mid_SAT.detrended_xco2_monthly_mean, color = mSAT_color, label = 'TM5-4DVar mid SAT', linewidth=1.25)
        ax2.plot(msc_south_SAT.month, msc_south_SAT.detrended_xco2_monthly_mean, color = sSAT_color, label = 'TM5-4DVar south SAT', linewidth=1.25)
        if plot_std_error:
            ax2.fill_between(msc_SAT.month,
                            msc_SAT.detrended_xco2_monthly_mean - msc_SAT.detrended_xco2_monthly_mean_std, 
                            msc_SAT.detrended_xco2_monthly_mean + msc_SAT.detrended_xco2_monthly_mean_std, 
                            color=SAT_color, linestyle='dashed', alpha=0.2)
            ax2.fill_between(msc_north_SAT.month,
                            msc_north_SAT.detrended_xco2_monthly_mean - msc_north_SAT.detrended_xco2_monthly_mean_std, 
                            msc_north_SAT.detrended_xco2_monthly_mean + msc_north_SAT.detrended_xco2_monthly_mean_std, 
                            color=nSAT_color, linestyle='solid', alpha=0.2)
            ax2.fill_between(msc_mid_SAT.month,
                            msc_mid_SAT.detrended_xco2_monthly_mean - msc_mid_SAT.detrended_xco2_monthly_mean_std, 
                            msc_mid_SAT.detrended_xco2_monthly_mean + msc_mid_SAT.detrended_xco2_monthly_mean_std, 
                            color=mSAT_color, alpha=0.2)
            ax2.fill_between(msc_south_SAT.month,
                            msc_south_SAT.detrended_xco2_monthly_mean - msc_south_SAT.detrended_xco2_monthly_mean_std, 
                            msc_south_SAT.detrended_xco2_monthly_mean + msc_south_SAT.detrended_xco2_monthly_mean_std, 
                            color=sSAT_color, alpha=0.2)
    if compare_case==2:
        print('Case 2: four timeseries plotted: SAT, west SAT, mid_long SAT, east SAT gridded')
        # TM5/apri cos RT
        Monthly_Means_SAT = calculate_monthly_means_modelled_xco2(TM5_xCO2_SAT, cosampled=True)
        Monthly_Means_SAT = Monthly_Means_SAT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_SAT = Monthly_Means_SAT.modelled_xco2_monthly_mean - Monthly_Means_SAT.background_xco2
        # TM5/IS_cos_RT
        Monthly_Means_north_SAT = calculate_monthly_means_modelled_xco2(TM5_xCO2_north_SAT, cosampled=True)
        Monthly_Means_north_SAT = Monthly_Means_north_SAT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_north_SAT = Monthly_Means_north_SAT.modelled_xco2_monthly_mean - Monthly_Means_north_SAT.background_xco2
        # TM5/IS_ACOS_cos_RT
        Monthly_Means_mid_SAT = calculate_monthly_means_modelled_xco2(TM5_xCO2_mid_SAT, cosampled=True)
        Monthly_Means_mid_SAT = Monthly_Means_mid_SAT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_mid_SAT = Monthly_Means_mid_SAT.modelled_xco2_monthly_mean - Monthly_Means_mid_SAT.background_xco2
        # TM5/IS_RT_cos_RT
        Monthly_Means_south_SAT = calculate_monthly_means_modelled_xco2(TM5_xCO2_south_SAT, cosampled=True)
        Monthly_Means_south_SAT = Monthly_Means_south_SAT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_south_SAT = Monthly_Means_south_SAT.modelled_xco2_monthly_mean - Monthly_Means_south_SAT.background_xco2
        
        ax1.plot(Monthly_Means_SAT.MonthDate, detrended_values_modelled_SAT, color=SAT_color, linestyle='dashed', label = 'TM5-4DVar SAT', linewidth=1.25)
        ax1.plot(Monthly_Means_north_SAT.MonthDate, detrended_values_modelled_north_SAT, color=nSAT_color, linestyle='solid', label = 'TM5-4DVar west SAT', linewidth=1.25)
        ax1.plot(Monthly_Means_mid_SAT.MonthDate, detrended_values_modelled_mid_SAT, color=mSAT_color, label = 'TM5-4DVar mid_long SAT', linewidth=1.25)
        ax1.plot(Monthly_Means_south_SAT.MonthDate, detrended_values_modelled_south_SAT, color=sSAT_color, label = 'TM5-4DVar east SAT', linewidth=1.25)
        
        print('Case 2: four timeseries plotted: SAT, west SAT, mid_long SAT, east SAT gridded')
        msc_SAT = calculateMeanSeasonalCycle(TM5_xCO2_SAT, background_xco2_timeseries, cosampled=True)
        msc_north_SAT = calculateMeanSeasonalCycle(TM5_xCO2_north_SAT, background_xco2_timeseries, cosampled=True)
        msc_mid_SAT = calculateMeanSeasonalCycle(TM5_xCO2_mid_SAT, background_xco2_timeseries, cosampled=True)
        msc_south_SAT = calculateMeanSeasonalCycle(TM5_xCO2_south_SAT, background_xco2_timeseries, cosampled=True)
        # if start_month=5 -> Mai will be the first month that is plotted
        first_month = start_month-2 # needed because df.loc[:5] will also affect month=7
        msc_SAT.loc[:first_month,'month'] += 12
        msc_north_SAT.loc[:first_month,'month'] += 12
        msc_mid_SAT.loc[:first_month,'month'] += 12
        msc_south_SAT.loc[:first_month,'month'] += 12
        # concatenate the two data frames vertically
        msc_SAT_new = pd.DataFrame(pd.concat([msc_SAT.loc[first_month+1:, 'month'], msc_SAT.loc[:first_month, 'month']], ignore_index=True))
        msc_north_SAT_new = pd.DataFrame(pd.concat([msc_north_SAT.loc[first_month+1:, 'month'], msc_north_SAT.loc[:first_month, 'month']], ignore_index=True))
        msc_mid_SAT_new = pd.DataFrame(pd.concat([msc_mid_SAT.loc[first_month+1:, 'month'], msc_mid_SAT.loc[:first_month, 'month']], ignore_index=True))
        msc_south_SAT_new = pd.DataFrame(pd.concat([msc_south_SAT.loc[first_month+1:, 'month'], msc_south_SAT.loc[:first_month, 'month']], ignore_index=True))
        # merge 
        msc_SAT = pd.merge(msc_SAT_new, msc_SAT, on='month', how='outer')
        msc_north_SAT = pd.merge(msc_north_SAT_new, msc_north_SAT, on='month', how='outer')
        msc_mid_SAT = pd.merge(msc_mid_SAT_new, msc_mid_SAT, on='month', how='outer')
        msc_south_SAT = pd.merge(msc_south_SAT_new, msc_south_SAT, on='month', how='outer')
        
        ax2.plot(msc_SAT.month, msc_SAT.detrended_xco2_monthly_mean, color = SAT_color, linestyle='dashed', label = 'TM5-4DVar SAT', linewidth=1.25)
        ax2.plot(msc_north_SAT.month, msc_north_SAT.detrended_xco2_monthly_mean, color = nSAT_color, linestyle='solid', label = 'TM5-4DVar west SAT', linewidth=1.25)
        ax2.plot(msc_mid_SAT.month, msc_mid_SAT.detrended_xco2_monthly_mean, color = mSAT_color, label = 'TM5-4DVar mid_long SAT', linewidth=1.25)
        ax2.plot(msc_south_SAT.month, msc_south_SAT.detrended_xco2_monthly_mean, color = sSAT_color, label = 'TM5-4DVar east SAT', linewidth=1.25)
        if plot_std_error:
            ax2.fill_between(msc_SAT.month,
                            msc_SAT.detrended_xco2_monthly_mean - msc_SAT.detrended_xco2_monthly_mean_std, 
                            msc_SAT.detrended_xco2_monthly_mean + msc_SAT.detrended_xco2_monthly_mean_std, 
                            color=SAT_color, linestyle='dashed', alpha=0.2)
            ax2.fill_between(msc_north_SAT.month,
                            msc_north_SAT.detrended_xco2_monthly_mean - msc_north_SAT.detrended_xco2_monthly_mean_std, 
                            msc_north_SAT.detrended_xco2_monthly_mean + msc_north_SAT.detrended_xco2_monthly_mean_std, 
                            color=nSAT_color, linestyle='solid', alpha=0.2)
            ax2.fill_between(msc_mid_SAT.month,
                            msc_mid_SAT.detrended_xco2_monthly_mean - msc_mid_SAT.detrended_xco2_monthly_mean_std, 
                            msc_mid_SAT.detrended_xco2_monthly_mean + msc_mid_SAT.detrended_xco2_monthly_mean_std, 
                            color=mSAT_color, alpha=0.2)
            ax2.fill_between(msc_south_SAT.month,
                            msc_south_SAT.detrended_xco2_monthly_mean - msc_south_SAT.detrended_xco2_monthly_mean_std, 
                            msc_south_SAT.detrended_xco2_monthly_mean + msc_south_SAT.detrended_xco2_monthly_mean_std, 
                            color=sSAT_color, alpha=0.2)
    if compare_case==11:
        print('Case 11: for NOT cosampled: north,mid,south SAT: four timeseries plotted: SAT, north SAT, mid SAT, south SAT gridded')
        # TM5/apri cos RT
        Monthly_Means_SAT = calculate_monthly_means_modelled_xco2(TM5_xCO2_SAT, cosampled=False)
        Monthly_Means_SAT = Monthly_Means_SAT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_SAT = Monthly_Means_SAT.modelled_xco2_monthly_mean - Monthly_Means_SAT.background_xco2
        # TM5/IS_cos_RT
        Monthly_Means_north_SAT = calculate_monthly_means_modelled_xco2(TM5_xCO2_north_SAT, cosampled=False)
        Monthly_Means_north_SAT = Monthly_Means_north_SAT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_north_SAT = Monthly_Means_north_SAT.modelled_xco2_monthly_mean - Monthly_Means_north_SAT.background_xco2
        # TM5/IS_ACOS_cos_RT
        Monthly_Means_mid_SAT = calculate_monthly_means_modelled_xco2(TM5_xCO2_mid_SAT, cosampled=False)
        Monthly_Means_mid_SAT = Monthly_Means_mid_SAT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_mid_SAT = Monthly_Means_mid_SAT.modelled_xco2_monthly_mean - Monthly_Means_mid_SAT.background_xco2
        # TM5/IS_RT_cos_RT
        Monthly_Means_south_SAT = calculate_monthly_means_modelled_xco2(TM5_xCO2_south_SAT, cosampled=False)
        Monthly_Means_south_SAT = Monthly_Means_south_SAT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_south_SAT = Monthly_Means_south_SAT.modelled_xco2_monthly_mean - Monthly_Means_south_SAT.background_xco2
        
        ax1.plot(Monthly_Means_SAT.MonthDate, detrended_values_modelled_SAT, color=SAT_color, linestyle='dashed', label = 'TM5-4DVar SAT', linewidth=1.25)
        ax1.plot(Monthly_Means_north_SAT.MonthDate, detrended_values_modelled_north_SAT, color=nSAT_color, linestyle='solid', label = 'TM5-4DVar north SAT', linewidth=1.25)
        ax1.plot(Monthly_Means_mid_SAT.MonthDate, detrended_values_modelled_mid_SAT, color=mSAT_color, label = 'TM5-4DVar mid SAT', linewidth=1.25)
        ax1.plot(Monthly_Means_south_SAT.MonthDate, detrended_values_modelled_south_SAT, color=sSAT_color, label = 'TM5-4DVar south SAT', linewidth=1.25)
        
        print('Case 11: for NOT cosampled: north,mid,south SAT: four timeseries plotted: SAT, north SAT, mid SAT, south SAT gridded')
        msc_SAT = calculateMeanSeasonalCycle(TM5_xCO2_SAT, background_xco2_timeseries, cosampled=False)
        msc_north_SAT = calculateMeanSeasonalCycle(TM5_xCO2_north_SAT, background_xco2_timeseries, cosampled=False)
        msc_mid_SAT = calculateMeanSeasonalCycle(TM5_xCO2_mid_SAT, background_xco2_timeseries, cosampled=False)
        msc_south_SAT = calculateMeanSeasonalCycle(TM5_xCO2_south_SAT, background_xco2_timeseries, cosampled=False)
        # if start_month=5 -> Mai will be the first month that is plotted
        first_month = start_month-2 # needed because df.loc[:5] will also affect month=7
        msc_SAT.loc[:first_month,'month'] += 12
        msc_north_SAT.loc[:first_month,'month'] += 12
        msc_mid_SAT.loc[:first_month,'month'] += 12
        msc_south_SAT.loc[:first_month,'month'] += 12
        # concatenate the two data frames vertically
        msc_SAT_new = pd.DataFrame(pd.concat([msc_SAT.loc[first_month+1:, 'month'], msc_SAT.loc[:first_month, 'month']], ignore_index=True))
        msc_north_SAT_new = pd.DataFrame(pd.concat([msc_north_SAT.loc[first_month+1:, 'month'], msc_north_SAT.loc[:first_month, 'month']], ignore_index=True))
        msc_mid_SAT_new = pd.DataFrame(pd.concat([msc_mid_SAT.loc[first_month+1:, 'month'], msc_mid_SAT.loc[:first_month, 'month']], ignore_index=True))
        msc_south_SAT_new = pd.DataFrame(pd.concat([msc_south_SAT.loc[first_month+1:, 'month'], msc_south_SAT.loc[:first_month, 'month']], ignore_index=True))
        # merge 
        msc_SAT = pd.merge(msc_SAT_new, msc_SAT, on='month', how='outer')
        msc_north_SAT = pd.merge(msc_north_SAT_new, msc_north_SAT, on='month', how='outer')
        msc_mid_SAT = pd.merge(msc_mid_SAT_new, msc_mid_SAT, on='month', how='outer')
        msc_south_SAT = pd.merge(msc_south_SAT_new, msc_south_SAT, on='month', how='outer')
        
        ax2.plot(msc_SAT.month, msc_SAT.detrended_xco2_monthly_mean, color = SAT_color, linestyle='dashed', label = 'TM5-4DVar SAT', linewidth=1.25)
        ax2.plot(msc_north_SAT.month, msc_north_SAT.detrended_xco2_monthly_mean, color = nSAT_color, linestyle='solid', label = 'TM5-4DVar north SAT', linewidth=1.25)
        ax2.plot(msc_mid_SAT.month, msc_mid_SAT.detrended_xco2_monthly_mean, color = mSAT_color, label = 'TM5-4DVar mid SAT', linewidth=1.25)
        ax2.plot(msc_south_SAT.month, msc_south_SAT.detrended_xco2_monthly_mean, color = sSAT_color, label = 'TM5-4DVar south SAT', linewidth=1.25)
        if plot_std_error:
            ax2.fill_between(msc_SAT.month,
                            msc_SAT.detrended_xco2_monthly_mean - msc_SAT.detrended_xco2_monthly_mean_std, 
                            msc_SAT.detrended_xco2_monthly_mean + msc_SAT.detrended_xco2_monthly_mean_std, 
                            color=SAT_color, linestyle='dashed', alpha=0.2)
            ax2.fill_between(msc_north_SAT.month,
                            msc_north_SAT.detrended_xco2_monthly_mean - msc_north_SAT.detrended_xco2_monthly_mean_std, 
                            msc_north_SAT.detrended_xco2_monthly_mean + msc_north_SAT.detrended_xco2_monthly_mean_std, 
                            color=nSAT_color, linestyle='solid', alpha=0.2)
            ax2.fill_between(msc_mid_SAT.month,
                            msc_mid_SAT.detrended_xco2_monthly_mean - msc_mid_SAT.detrended_xco2_monthly_mean_std, 
                            msc_mid_SAT.detrended_xco2_monthly_mean + msc_mid_SAT.detrended_xco2_monthly_mean_std, 
                            color=mSAT_color, alpha=0.2)
            ax2.fill_between(msc_south_SAT.month,
                            msc_south_SAT.detrended_xco2_monthly_mean - msc_south_SAT.detrended_xco2_monthly_mean_std, 
                            msc_south_SAT.detrended_xco2_monthly_mean + msc_south_SAT.detrended_xco2_monthly_mean_std, 
                            color=sSAT_color, alpha=0.2)
    if compare_case==22:
        print('Case 22: NOT cos: four timeseries plotted: SAT, west SAT, mid_long SAT, east SAT gridded')
        # TM5/apri cos RT
        Monthly_Means_SAT = calculate_monthly_means_modelled_xco2(TM5_xCO2_SAT, cosampled=False)
        Monthly_Means_SAT = Monthly_Means_SAT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_SAT = Monthly_Means_SAT.modelled_xco2_monthly_mean - Monthly_Means_SAT.background_xco2
        # TM5/IS_cos_RT
        Monthly_Means_north_SAT = calculate_monthly_means_modelled_xco2(TM5_xCO2_north_SAT, cosampled=False)
        Monthly_Means_north_SAT = Monthly_Means_north_SAT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_north_SAT = Monthly_Means_north_SAT.modelled_xco2_monthly_mean - Monthly_Means_north_SAT.background_xco2
        # TM5/IS_ACOS_cos_RT
        Monthly_Means_mid_SAT = calculate_monthly_means_modelled_xco2(TM5_xCO2_mid_SAT, cosampled=False)
        Monthly_Means_mid_SAT = Monthly_Means_mid_SAT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_mid_SAT = Monthly_Means_mid_SAT.modelled_xco2_monthly_mean - Monthly_Means_mid_SAT.background_xco2
        # TM5/IS_RT_cos_RT
        Monthly_Means_south_SAT = calculate_monthly_means_modelled_xco2(TM5_xCO2_south_SAT, cosampled=False)
        Monthly_Means_south_SAT = Monthly_Means_south_SAT.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
        detrended_values_modelled_south_SAT = Monthly_Means_south_SAT.modelled_xco2_monthly_mean - Monthly_Means_south_SAT.background_xco2
        
        ax1.plot(Monthly_Means_SAT.MonthDate, detrended_values_modelled_SAT, color=SAT_color, linestyle='dashed', label = 'TM5-4DVar SAT', linewidth=1.25)
        ax1.plot(Monthly_Means_north_SAT.MonthDate, detrended_values_modelled_north_SAT, color=nSAT_color, linestyle='solid', label = 'TM5-4DVar west SAT', linewidth=1.25)
        ax1.plot(Monthly_Means_mid_SAT.MonthDate, detrended_values_modelled_mid_SAT, color=mSAT_color, label = 'TM5-4DVar mid_long SAT', linewidth=1.25)
        ax1.plot(Monthly_Means_south_SAT.MonthDate, detrended_values_modelled_south_SAT, color=sSAT_color, label = 'TM5-4DVar east SAT', linewidth=1.25)
        
        print('Case 22: NOT cos: four MSCs plotted: SAT, west SAT, mid_long SAT, east SAT gridded')
        msc_SAT = calculateMeanSeasonalCycle(TM5_xCO2_SAT, background_xco2_timeseries, cosampled=False)
        msc_north_SAT = calculateMeanSeasonalCycle(TM5_xCO2_north_SAT, background_xco2_timeseries, cosampled=False)
        msc_mid_SAT = calculateMeanSeasonalCycle(TM5_xCO2_mid_SAT, background_xco2_timeseries, cosampled=False)
        msc_south_SAT = calculateMeanSeasonalCycle(TM5_xCO2_south_SAT, background_xco2_timeseries, cosampled=False)
        # if start_month=5 -> Mai will be the first month that is plotted
        first_month = start_month-2 # needed because df.loc[:5] will also affect month=7
        msc_SAT.loc[:first_month,'month'] += 12
        msc_north_SAT.loc[:first_month,'month'] += 12
        msc_mid_SAT.loc[:first_month,'month'] += 12
        msc_south_SAT.loc[:first_month,'month'] += 12
        # concatenate the two data frames vertically
        msc_SAT_new = pd.DataFrame(pd.concat([msc_SAT.loc[first_month+1:, 'month'], msc_SAT.loc[:first_month, 'month']], ignore_index=True))
        msc_north_SAT_new = pd.DataFrame(pd.concat([msc_north_SAT.loc[first_month+1:, 'month'], msc_north_SAT.loc[:first_month, 'month']], ignore_index=True))
        msc_mid_SAT_new = pd.DataFrame(pd.concat([msc_mid_SAT.loc[first_month+1:, 'month'], msc_mid_SAT.loc[:first_month, 'month']], ignore_index=True))
        msc_south_SAT_new = pd.DataFrame(pd.concat([msc_south_SAT.loc[first_month+1:, 'month'], msc_south_SAT.loc[:first_month, 'month']], ignore_index=True))
        # merge 
        msc_SAT = pd.merge(msc_SAT_new, msc_SAT, on='month', how='outer')
        msc_north_SAT = pd.merge(msc_north_SAT_new, msc_north_SAT, on='month', how='outer')
        msc_mid_SAT = pd.merge(msc_mid_SAT_new, msc_mid_SAT, on='month', how='outer')
        msc_south_SAT = pd.merge(msc_south_SAT_new, msc_south_SAT, on='month', how='outer')
        
        ax2.plot(msc_SAT.month, msc_SAT.detrended_xco2_monthly_mean, color = SAT_color, linestyle='dashed', label = 'TM5-4DVar SAT', linewidth=1.25)
        ax2.plot(msc_north_SAT.month, msc_north_SAT.detrended_xco2_monthly_mean, color = nSAT_color, linestyle='solid', label = 'TM5-4DVar west SAT', linewidth=1.25)
        ax2.plot(msc_mid_SAT.month, msc_mid_SAT.detrended_xco2_monthly_mean, color = mSAT_color, label = 'TM5-4DVar mid_long SAT', linewidth=1.25)
        ax2.plot(msc_south_SAT.month, msc_south_SAT.detrended_xco2_monthly_mean, color = sSAT_color, label = 'TM5-4DVar east SAT', linewidth=1.25)
        if plot_std_error:
            ax2.fill_between(msc_SAT.month,
                            msc_SAT.detrended_xco2_monthly_mean - msc_SAT.detrended_xco2_monthly_mean_std, 
                            msc_SAT.detrended_xco2_monthly_mean + msc_SAT.detrended_xco2_monthly_mean_std, 
                            color=SAT_color, linestyle='dashed', alpha=0.2)
            ax2.fill_between(msc_north_SAT.month,
                            msc_north_SAT.detrended_xco2_monthly_mean - msc_north_SAT.detrended_xco2_monthly_mean_std, 
                            msc_north_SAT.detrended_xco2_monthly_mean + msc_north_SAT.detrended_xco2_monthly_mean_std, 
                            color=nSAT_color, linestyle='solid', alpha=0.2)
            ax2.fill_between(msc_mid_SAT.month,
                            msc_mid_SAT.detrended_xco2_monthly_mean - msc_mid_SAT.detrended_xco2_monthly_mean_std, 
                            msc_mid_SAT.detrended_xco2_monthly_mean + msc_mid_SAT.detrended_xco2_monthly_mean_std, 
                            color=mSAT_color, alpha=0.2)
            ax2.fill_between(msc_south_SAT.month,
                            msc_south_SAT.detrended_xco2_monthly_mean - msc_south_SAT.detrended_xco2_monthly_mean_std, 
                            msc_south_SAT.detrended_xco2_monthly_mean + msc_south_SAT.detrended_xco2_monthly_mean_std, 
                            color=sSAT_color, alpha=0.2)
    
    # timeseries axis
    ax1.set_xlabel('Year')
    ax1.set_ylabel('XCO$_2$ [ppm]', color='black')
    ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax1.minorticks_on()
    ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax1.tick_params(bottom=True, left=True, color='gray')
    ax1.legend(framealpha=0.25, facecolor='grey', loc='lower left', fontsize=9.5, ncol=2, columnspacing=0.8)
    ax1.set_title('TM5-4DVar/'+assimilated+' cos. '+cosampled_on+' xCO2 per region', pad=-13)
    ax1.set_yticks([-2, -1, 0, 1, 2])
    ax1.set_yticklabels([-2, -1, 0, 1, 2])
    ax1.set_ylim([-2.2, 2.2])
    #MSC axis
    ax2.set_xlabel('Month')
    ax2.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax2.set_xticks([5,10,15])
    ax2.set_xticklabels([5, 10, 3]) # set the x-axis tick labels
    ax2.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax2.tick_params(bottom=True, color='gray')
    ax2.set_title('MSC 2009-2019', pad=-13)
    ax2.set_ylim([-2.2, 2.2])

    if compare_case==1:
        if plot_std_error:          
            plt.savefig(savepath + 'TM5_'+assimilated+'_all_cos_'+cosampled_on+'_gridded_SAT_nSAT_mSAT_sSAT_xCO2_timeseries_AND_mean_seasonal_cycle_std_dev.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(savepath + 'TM5_'+assimilated+'_all_cos_'+cosampled_on+'_gridded_SAT_nSAT_mSAT_sSAT_xCO2_timeseries_AND_mean_seasonal_cycle.png', dpi=300, bbox_inches='tight')
    if compare_case==2:
        if plot_std_error:          
            plt.savefig(savepath + 'TM5_'+assimilated+'_all_cos_'+cosampled_on+'_gridded_SAT_wSAT_mSAT_eSAT_xCO2_timeseries_AND_mean_seasonal_cycle_std_dev.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(savepath + 'TM5_'+assimilated+'_all_cos_'+cosampled_on+'_gridded_SAT_wSAT_mSAT_eSAT_xCO2_timeseries_AND_mean_seasonal_cycle.png', dpi=300, bbox_inches='tight')
    if compare_case==11:
        if plot_std_error:          
            plt.savefig(savepath + 'TM5_'+assimilated+'_all_NOT_cos_gridded_SAT_nSAT_mSAT_sSAT_xCO2_timeseries_AND_mean_seasonal_cycle_std_dev.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(savepath + 'TM5_'+assimilated+'_all_NOT_cos_gridded_SAT_nSAT_mSAT_sSAT_xCO2_timeseries_AND_mean_seasonal_cycle.png', dpi=300, bbox_inches='tight')
    if compare_case==22:
        if plot_std_error:
            plt.savefig(savepath + 'TM5_'+assimilated+'_all_NOT_cos_gridded_SAT_wSAT_mSAT_eSAT_xCO2_timeseries_AND_mean_seasonal_cycle_std_dev.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(savepath + 'TM5_'+assimilated+'_all_NOT_cos_gridded_SAT_wSAT_mSAT_eSAT_xCO2_timeseries_AND_mean_seasonal_cycle.png', dpi=300, bbox_inches='tight')



# plot measured xCO2
# plot monthly mean fluxes & mean seasonal cycle next to it WITH mean seasonal cycle start in MAY BEFORE/afer 2014
def plot_measured_xCO2_Timeseries_and_MSC_general(start_month: int = 5, background_xco2_timeseries: pd.DataFrame = None, start_year: int=2009, end_year: int=2019,
                                            gdf_list = [],
                                            plot_second_MSCs: bool = False,
                                            second_MSC_start_year: int = 2014,
                                            second_MSC_end_year: int = 2018,
                                            plot_third_MSCs: bool = False,
                                            third_MSC_start_year: int = 2009,
                                            third_MSC_end_year: int = 2014,
                                            columns_plotted = ['xco2_monthly_mean', 'xco2_raw_monthly_mean'],
                                            satellite_list = ['GOSAT_ACOS_CT_mask', 'GOSAT_ACOS_old_mask', 'GOSAT_RT_old_mask', 'GOSAT_RT_CT_mask', 'OCO2'],
                                            color_list = [],
                                            linestyle_list = [],
                                            region_name: str = 'SAT',
                                            savepath: str = '',
                                            compare_case: int = 1):
    '''# Documentation
    Plots satellite measurements for the specified RegionName (subregion)
    # arguments:
        - gdf_list: list of the gdfs that should be used in this plot
        - columns_plotted: list of variables that should be plotted for the respective gdf type
        - satellite_list: list of satellite types that are used in this plot. All combinations of the following list should work:
        - color_list: list of colors for the satellite types
        - linestyle_list: list of linestyles for the satellite types
        - savefig: bool, save fig or do not save fig
    '''
    if plot_second_MSCs:
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,5), sharey=True, gridspec_kw={'width_ratios':[6,2,2]})#figsize=(7,5)
        savename = 'Satellite_measurements_TWO_MSCs'
    if plot_third_MSCs:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(11,5), sharey=True, gridspec_kw={'width_ratios':[6,2,2,2]})#figsize=(7,5)
        savename = 'Satellite_measurements_THREE_MSCs'
    else:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5), sharey=True, gridspec_kw={'width_ratios':[8,3]})#figsize=(7,5)
        savename = 'Satellite_measurements'
    fig.subplots_adjust(wspace=0)
    
    first_month = start_month-2 # needed because df.loc[:5] will also affect month=7
    if compare_case==1:
        for i,gdf in enumerate(gdf_list):
            print('Plotting '+satellite_list[i])
            Monthly_Means = calculate_monthly_means_for_measurements_with_xco2_raw(gdf)
            Monthly_Means = Monthly_Means.merge(background_xco2_timeseries, left_on='MonthDate', right_on='MonthDate', how='inner')
            detrended_values = Monthly_Means[columns_plotted[i]] - Monthly_Means.background_xco2
            ax1.plot(Monthly_Means.MonthDate, detrended_values, color = color_list[i], linestyle=linestyle_list[i], label = satellite_list[i], linewidth=1.25)
            msc = calculateMeanSeasonalCycle_measurements_with_xco2_raw(gdf, background_xco2_timeseries)
            msc.loc[:first_month,'month'] += 12
            msc_new = pd.DataFrame(pd.concat([msc.loc[first_month+1:, 'month'], msc.loc[:first_month, 'month']], ignore_index=True))
            msc = pd.merge(msc_new, msc, on='month', how='outer')
            ax2.plot(msc.month, msc['detrended_'+columns_plotted[i]], color = color_list[i], linestyle=linestyle_list[i], label = satellite_list[i], linewidth=1.25)
            ax2.fill_between(msc.month,
                            msc['detrended_'+columns_plotted[i]] - msc['detrended_'+columns_plotted[i]+'_std'], 
                            msc['detrended_'+columns_plotted[i]] + msc['detrended_'+columns_plotted[i]+'_std'], 
                            color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)
            if plot_second_MSCs and satellite_list[i]!='OCO2':
                gdf_second_MSC = gdf.drop(gdf[(gdf['year'] < second_MSC_start_year)].index, inplace=False)
                gdf_second_MSC = gdf_second_MSC.drop(gdf[(gdf['year'] > second_MSC_end_year)].index, inplace=False)
                #gdf_second_MSC = gdf[(gdf['year']>=second_MSC_start_year) and (gdf['year']<=second_MSC_end_year)]
                msc_snd_msc = calculateMeanSeasonalCycle_measurements_with_xco2_raw(gdf_second_MSC, background_xco2_timeseries)
                msc_snd_msc.loc[:first_month,'month'] += 12
                msc_snd_msc_new = pd.DataFrame(pd.concat([msc_snd_msc.loc[first_month+1:, 'month'], msc_snd_msc.loc[:first_month, 'month']], ignore_index=True))
                msc_snd_msc = pd.merge(msc_snd_msc_new, msc_snd_msc, on='month', how='outer')
                ax3.plot(msc_snd_msc.month, msc_snd_msc['detrended_'+columns_plotted[i]], color = color_list[i], linestyle=linestyle_list[i], label = satellite_list[i], linewidth=1.25)
                ax3.fill_between(msc_snd_msc.month,
                                msc_snd_msc['detrended_'+columns_plotted[i]] - msc_snd_msc['detrended_'+columns_plotted[i]+'_std'], 
                                msc_snd_msc['detrended_'+columns_plotted[i]] + msc_snd_msc['detrended_'+columns_plotted[i]+'_std'], 
                                color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)
            if plot_third_MSCs:
                gdf_third_MSC = gdf.drop(gdf[(gdf['year'] < third_MSC_start_year)].index, inplace=False)
                gdf_third_MSC = gdf_third_MSC.drop(gdf[(gdf['year'] > third_MSC_end_year)].index, inplace=False)
                msc_trd_msc = calculateMeanSeasonalCycle_measurements_with_xco2_raw(gdf_third_MSC, background_xco2_timeseries)
                msc_trd_msc.loc[:first_month,'month'] += 12
                msc_trd_msc_new = pd.DataFrame(pd.concat([msc_trd_msc.loc[first_month+1:, 'month'], msc_trd_msc.loc[:first_month, 'month']], ignore_index=True))
                msc_trd_msc = pd.merge(msc_trd_msc_new, msc_trd_msc, on='month', how='outer')
                ax4.plot(msc_trd_msc.month, msc_trd_msc['detrended_'+columns_plotted[i]], color = color_list[i], linestyle=linestyle_list[i], label = satellite_list[i], linewidth=1.25)
                ax4.fill_between(msc_trd_msc.month,
                                msc_trd_msc['detrended_'+columns_plotted[i]] - msc_trd_msc['detrended_'+columns_plotted[i]+'_std'], 
                                msc_trd_msc['detrended_'+columns_plotted[i]] + msc_trd_msc['detrended_'+columns_plotted[i]+'_std'], 
                                color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)
            if i!=len(gdf_list)-1:
                savename = savename + '_' + satellite_list[i] + '_&'
            else:
                savename = savename + '_' + satellite_list[i] + '_timeseries_AND_msc_' + str(start_year) + '-' + str(end_year) +'_' + region_name + '.png'

    # timeseries axis
    ax1.set_xlabel('Year')
    ax1.set_ylabel('xCO$_2$ [ppm]', color='black')
    ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax1.minorticks_on()
    ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax1.tick_params(which='major', bottom=True, left=True, right=False, color='gray')
    ax1.tick_params(which='minor', bottom=False, left=True, right=False, color='gray')
        #ax1.grid(which='major', color='white', axis='x', linestyle='-', linewidth='0.75')
    ax1.legend(framealpha=0.25, facecolor='grey', loc='lower left', fontsize=9.5, ncol=3, columnspacing=0.5)
    #ax1.set_yticks([-2, -1, 0, 1])
    #ax1.set_yticklabels([-2, -1, 0, 1])
    #ax1.set_ylim([-2.2, 1.6])
    ax1.set_title('measured xCO2 timeseries '+region_name)#, pad=-13)
    # MSC axis
    ax2.set_xlabel('Month')
    ax2.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax2.set_xticks([5,10,15])
    ax2.set_xticklabels([5, 10, 3]) # set the x-axis tick labels
    ax2.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax2.tick_params(which='major', bottom=True, left=False, color='gray')
    ax2.tick_params(which='minor', bottom=False, left=False, color='gray')
    ax2.set_title('MSC '+str(start_year)+'-'+str(end_year))#, pad=-13)
    #ax2.set_ylim([-2.2, 1.6])
    if plot_second_MSCs:
        # MSC axis
        ax3.set_xlabel('Month')
        ax3.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
        ax3.set_xticks([5,10,15])
        ax3.set_xticklabels([5, 10, 3]) # set the x-axis tick labels
        ax3.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax3.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax3.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
        ax3.tick_params(which='major', bottom=True, left=False, color='gray')
        ax3.tick_params(which='minor', bottom=False, left=False, color='gray')
        ax3.set_title('MSC '+str(second_MSC_start_year)+'-'+str(second_MSC_end_year))#, pad=-13)
    if plot_third_MSCs:
        # MSC axis
        ax4.set_xlabel('Month')
        ax4.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
        ax4.set_xticks([5,10,15])
        ax4.set_xticklabels([5, 10, 3]) # set the x-axis tick labels
        ax4.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax4.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax4.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
        ax4.tick_params(which='major', bottom=True, left=False, color='gray')
        ax4.tick_params(which='minor', bottom=False, left=False, color='gray')
        ax4.set_title('MSC '+str(third_MSC_start_year)+'-'+str(third_MSC_end_year))#, pad=-13)
        
    if compare_case==1:
        plt.savefig(savepath + savename, dpi=300, bbox_inches='tight')
    
