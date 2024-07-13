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
import function_to_load_datasets as load_datasets


# calculations:
def calculateMeanSeasonalCycle_of_fluxes(Flux_per_subregion_monthly_timeseries: pd.DataFrame, model: str='MIP'):
    '''## Documentation
    ### input:
    - Flux_per_subregion_monthly_timeseries: df with the fluxes per subregion and month
    - model: 
        - 'MIP'
        - 'TM5_flux_regional_Sourish'
                - df contains CO2 values in Tg_CO2/region/month -> additional column for flux in TgC/region/month
        - 'TM5_flux_gridded'
    ### output:
    - Flux_msc: df with the mean seasonal cycle of the fluxes per subregion
    '''
    if model=='MIP':
        print('start calculating MSC for MIP fluxes')
        Flux_msc = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['Landtot']].mean().reset_index()
        Flux_msc_std = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['Landtot']].std(ddof=0).reset_index()
        Flux_msc_count = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['Landtot']].count().reset_index()
        Flux_msc = Flux_msc.merge(Flux_msc_std, on='Month', suffixes=('', '_std'))
        Flux_msc = Flux_msc.merge(Flux_msc_count, on='Month', suffixes=('', '_count'))
        return Flux_msc
    elif model=='TM5_flux_regional_Sourish':
        print('start calculating MSC for TM5_fluxes regional Sourish')
        Flux_region_mean_seasonal_cycle = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['Flux_fire','Flux_bio','Flux_ocn','Flux_fire_apri','Flux_bio_apri','Flux_ocn_apri']].mean().reset_index()
        Flux_region_msc_std = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['Flux_fire','Flux_bio','Flux_ocn','Flux_fire_apri','Flux_bio_apri','Flux_ocn_apri']].std(ddof=0).reset_index()
        Flux_region_msc_count = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['Flux_fire','Flux_bio','Flux_ocn','Flux_fire_apri','Flux_bio_apri','Flux_ocn_apri']].count().reset_index()
        Flux_region_mean_seasonal_cycle = Flux_region_mean_seasonal_cycle.merge(Flux_region_msc_std, on='Month', suffixes=('', '_std'))
        Flux_region_mean_seasonal_cycle = Flux_region_mean_seasonal_cycle.merge(Flux_region_msc_count, on='Month', suffixes=('', '_count'))
        Flux_region_mean_seasonal_cycle['Flux_fire_plus_bio'] = Flux_region_mean_seasonal_cycle['Flux_fire'] + Flux_region_mean_seasonal_cycle['Flux_bio']
        Flux_region_mean_seasonal_cycle['Flux_fire_plus_bio_std'] = np.sqrt(Flux_region_mean_seasonal_cycle['Flux_bio_std']**2 + Flux_region_mean_seasonal_cycle['Flux_fire_std']**2)
        Flux_region_mean_seasonal_cycle['Flux_fire_apri_plus_bio_apri_std'] = np.sqrt(Flux_region_mean_seasonal_cycle['Flux_bio_apri_std']**2 + Flux_region_mean_seasonal_cycle['Flux_fire_apri_std']**2)
        Flux_region_mean_seasonal_cycle['Flux_fire_TgC'] = Flux_region_mean_seasonal_cycle['Flux_fire']*(12/44)
        Flux_region_mean_seasonal_cycle['Flux_fire_apri_TgC'] = Flux_region_mean_seasonal_cycle['Flux_fire_apri']*(12/44)
        Flux_region_mean_seasonal_cycle['Flux_bio_TgC'] = Flux_region_mean_seasonal_cycle['Flux_bio']*(12/44)
        Flux_region_mean_seasonal_cycle['Flux_bio_apri_TgC'] = Flux_region_mean_seasonal_cycle['Flux_bio_apri']*(12/44)
        Flux_region_mean_seasonal_cycle['Flux_ocn_TgC'] = Flux_region_mean_seasonal_cycle['Flux_ocn']*(12/44)
        Flux_region_mean_seasonal_cycle['Flux_ocn_apri_TgC'] = Flux_region_mean_seasonal_cycle['Flux_ocn_apri']*(12/44)
        Flux_region_mean_seasonal_cycle['Flux_fire_plus_bio_TgC'] = Flux_region_mean_seasonal_cycle['Flux_fire_TgC'] + Flux_region_mean_seasonal_cycle['Flux_bio_TgC']
        Flux_region_mean_seasonal_cycle['Flux_fire_apri_plus_bio_apri_TgC'] = Flux_region_mean_seasonal_cycle['Flux_fire_apri_TgC'] + Flux_region_mean_seasonal_cycle['Flux_bio_apri_TgC']
        Flux_region_mean_seasonal_cycle['Flux_fire_std_TgC'] = Flux_region_mean_seasonal_cycle['Flux_fire_std']*(12/44)
        Flux_region_mean_seasonal_cycle['Flux_fire_apri_std_TgC'] = Flux_region_mean_seasonal_cycle['Flux_fire_apri_std']*(12/44)
        Flux_region_mean_seasonal_cycle['Flux_bio_std_TgC'] = Flux_region_mean_seasonal_cycle['Flux_bio_std']*(12/44)
        Flux_region_mean_seasonal_cycle['Flux_bio_apri_std_TgC'] = Flux_region_mean_seasonal_cycle['Flux_bio_apri_std']*(12/44)
        Flux_region_mean_seasonal_cycle['Flux_ocn_std_TgC'] = Flux_region_mean_seasonal_cycle['Flux_ocn_std']*(12/44)
        Flux_region_mean_seasonal_cycle['Flux_ocn_apri_std_TgC'] = Flux_region_mean_seasonal_cycle['Flux_ocn_apri_std']*(12/44)
        Flux_region_mean_seasonal_cycle['Flux_fire_plus_bio_std_TgC'] = Flux_region_mean_seasonal_cycle['Flux_fire_plus_bio_std']*(12/44)
        Flux_region_mean_seasonal_cycle['Flux_fire_apri_plus_bio_apri_std_TgC'] = Flux_region_mean_seasonal_cycle['Flux_fire_apri_plus_bio_apri_std']*(12/44)
        return Flux_region_mean_seasonal_cycle
    elif model=='TM5_flux_gridded' or model=='TM5':
        print('Calculating MSC for TM5_fluxes gridded')
        Flux_per_subregion_monthly_timeseries['Month'] = Flux_per_subregion_monthly_timeseries['MonthDate'].apply(lambda x: x.month)
        Flux_msc = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['CO2_NEE_flux_monthly_TgC_per_subregion', 
                                                                             'CO2_fire_flux_monthly_TgC_per_subregion', 
                                                                             'CO2_ocean_flux_monthly_TgC_per_subregion',
                                                                             'CO2_fossil_flux_monthly_TgC_per_subregion',
                                                                             'CO2_NEE_fire_flux_monthly_TgC_per_subregion']].mean()
        Flux_msc_std = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['CO2_NEE_flux_monthly_TgC_per_subregion', 
                                                                                 'CO2_fire_flux_monthly_TgC_per_subregion', 
                                                                                 'CO2_ocean_flux_monthly_TgC_per_subregion',
                                                                                 'CO2_fossil_flux_monthly_TgC_per_subregion',
                                                                                 'CO2_NEE_fire_flux_monthly_TgC_per_subregion']].std(ddof=0).reset_index()
        Flux_msc_count = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['CO2_NEE_flux_monthly_TgC_per_subregion', 
                                                                                   'CO2_fire_flux_monthly_TgC_per_subregion', 
                                                                                   'CO2_ocean_flux_monthly_TgC_per_subregion',
                                                                                   'CO2_fossil_flux_monthly_TgC_per_subregion',
                                                                                   'CO2_NEE_fire_flux_monthly_TgC_per_subregion']].count().reset_index()
        Flux_msc = Flux_msc.merge(Flux_msc_std, on='Month', suffixes=('', '_std'))
        Flux_msc = Flux_msc.merge(Flux_msc_count, on='Month', suffixes=('', '_count'))
        return Flux_msc


# plotting functions
def plotFluxTimeseries_and_MSC_general(start_month: int = 5, start_year: int=2009, end_year: int=2019, fix_y_scale: bool=False,
                               gdf_list: list=[],
                               plot_second_MSCs: bool = False,
                               second_MSC_start_year: int = 2014,
                               second_MSC_end_year: int = 2018,
                               variable_to_plot_list: list=['CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'Landtot'],
                               label_list: list=None,
                               model_assimilation_list: list=['TM5-4DVar_prior', 'TM5-4DVar_IS', 'TM5-4DVar_IS+ACOS', 'TM5-4DVar_IS+RT', 'TM5-4DVar_IS+OCO2', 'MIP_IS_ens', 'MIP_IS+OCO2_ens'],
                               what_is_plotted: str='NEE+fire_flux',
                               color_list: list=['dimgrey', 'forestgreen', 'royalblue', 'firebrick'],
                               linestyle_list: list=['dashed', 'solid'],
                               region_name: str='SAT', savepath: str='/home/lartelt/MA/software/', compare_case: int=1):
    '''# Documentation
    Function to plot timeseries of the mean seasonal cycle of CO2 fluxes modelled by TM5, MIP & next to it the mean seasonal cycle
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
    if plot_second_MSCs:
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,5), sharey=True, gridspec_kw={'width_ratios':[6,2,2]})#figsize=(7,5)
    else:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5), sharey=True, gridspec_kw={'width_ratios':[8,3]})
    fig.subplots_adjust(wspace=0)
    
    first_month = start_month-2 # needed because df.loc[:5] will also affect month=7
    savename = what_is_plotted
    if compare_case==1:
        print('Case 1 timeseries')
        for i,model in enumerate(model_assimilation_list):
            if label_list is not None:
                ax1.plot(gdf_list[i].MonthDate, gdf_list[i][variable_to_plot_list[i]], color = color_list[i], linestyle=linestyle_list[i], label = label_list[i], linewidth=1.25)
            else:
                ax1.plot(gdf_list[i].MonthDate, gdf_list[i][variable_to_plot_list[i]], color = color_list[i], linestyle=linestyle_list[i], label = model, linewidth=1.25)
            if model=='TM5-4DVar_IS+OCO2':
                MSC = calculateMeanSeasonalCycle_of_fluxes(gdf_list[i], model='MIP')
            else:
                MSC = calculateMeanSeasonalCycle_of_fluxes(gdf_list[i], model=model_assimilation_list[i][:3])
            MSC.loc[:first_month,'Month'] += 12
            MSC_new = pd.DataFrame(pd.concat([MSC.loc[first_month+1:, 'Month'], MSC.loc[:first_month, 'Month']], ignore_index=True))
            MSC = pd.merge(MSC_new, MSC, on='Month', how='outer')
            ax2.plot(MSC.Month, MSC[variable_to_plot_list[i]], color = color_list[i], linestyle=linestyle_list[i], label = model, linewidth=1.25)
            ax2.fill_between(MSC.Month,
                             MSC[variable_to_plot_list[i]] - MSC[(variable_to_plot_list[i]+'_std')], 
                             MSC[variable_to_plot_list[i]] + MSC[(variable_to_plot_list[i]+'_std')], 
                             color = color_list[i], linestyle=linestyle_list[i], alpha=0.2)
            if plot_second_MSCs and model_assimilation_list[i] not in ['TM5-4DVar_prior', 'TM5-4DVar_IS+ACOS', 'TM5-4DVar_IS+RT']:
                gdf_second_MSC = gdf_list[i].drop(gdf_list[i][(gdf_list[i]['year'] < second_MSC_start_year)].index, inplace=False)
                gdf_second_MSC = gdf_second_MSC.drop(gdf_list[i][(gdf_list[i]['year'] > second_MSC_end_year)].index, inplace=False)
                if model=='TM5-4DVar_IS+OCO2':
                    MSC_snd_msc = calculateMeanSeasonalCycle_of_fluxes(gdf_second_MSC, model='MIP')
                else:
                    MSC_snd_msc = calculateMeanSeasonalCycle_of_fluxes(gdf_second_MSC, model=model_assimilation_list[i][:3])
                MSC_snd_msc.loc[:first_month,'Month'] += 12
                MSC_snd_msc_new = pd.DataFrame(pd.concat([MSC_snd_msc.loc[first_month+1:, 'Month'], MSC_snd_msc.loc[:first_month, 'Month']], ignore_index=True))
                MSC_snd_msc = pd.merge(MSC_snd_msc_new, MSC_snd_msc, on='Month', how='outer')
                ax3.plot(MSC_snd_msc.Month, MSC_snd_msc[variable_to_plot_list[i]], color = color_list[i], linestyle=linestyle_list[i], label = model, linewidth=1.25)
                ax3.fill_between(MSC_snd_msc.Month,
                                MSC_snd_msc[variable_to_plot_list[i]] - MSC_snd_msc[(variable_to_plot_list[i]+'_std')], 
                                MSC_snd_msc[variable_to_plot_list[i]] + MSC_snd_msc[(variable_to_plot_list[i]+'_std')], 
                                color = color_list[i], linestyle=linestyle_list[i], alpha=0.2)
            if i!=len(model_assimilation_list)-1:
                savename = savename + '_' + model_assimilation_list[i] + '_&'
            else:
                if plot_second_MSCs:
                    savename = savename + '_TWO_MSCs_' + model_assimilation_list[i] + '_timeseries_AND_msc_' + region_name + '.png'
                else:
                    savename = savename + '_' + model_assimilation_list[i] + '_timeseries_AND_msc_' + region_name + '.png'

    #fig.suptitle(what_is_plotted+' for '+region_name, y=0.92, fontsize=12)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('CO$_2$ flux [TgC/month/region]', color='black')
    ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax1.minorticks_on()
    ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax1.tick_params(bottom=True, left=True, color='gray')
    ax1.legend(framealpha=0.25, facecolor='grey', loc='lower left', fontsize=9.5, ncol=2, columnspacing=0.6)
    #ax1.set_title('TM5-4DVar & MIP land fluxes '+region_name, pad=-13)
    ax1.set_title('TM5-4DVar land fluxes '+region_name)#, pad=-13)
    if fix_y_scale==True:
        ax1.set_yticks([-400, -200, 0, 200, 400])
        ax1.set_yticklabels([-400, -200, 0, 200, 400])
        ax1.set_ylim([-450, 500])
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
    if fix_y_scale==True:
        ax2.set_ylim([-450, 500])
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
        if fix_y_scale==True:
            ax3.set_ylim([-450, 500])
    
    
    print('Saving plot...')
    if compare_case==1:
        plt.savefig(savepath + savename, dpi=300, bbox_inches='tight')
    print('DONE plotting!')


def plotFluxTimeseries_and_MSC_compare_gridded_and_regional(start_month: int = 5,
                                             Fluxes_prior_region: pd.DataFrame = None,
                                             Fluxes_prior_region_gridded: pd.DataFrame = None,
                                             Fluxes_ACOS_IS_region: pd.DataFrame = None,
                                             Fluxes_ACOS_IS_region_gridded: pd.DataFrame = None,
                                             Fluxes_IS_region: pd.DataFrame = None,
                                             Fluxes_IS_region_gridded: pd.DataFrame = None,
                                             Fluxes_RT_IS_loc_region: pd.DataFrame = None,
                                             Fluxes_RT_IS_loc_region_gridded: pd.DataFrame = None,
                                             region_name: str='SAT', savepath: str='/home/lartelt/MA/software/', compare_case: int=1, grid_res='3x2'):
    '''# Documentation
    Function to plot timeseries of the mean seasonal cycle of CO2 fluxes modelled by TM5 & next to it the mean seasonal cycle
    Use: Fluxes_prior_region = Fluxes_IS_region --> Is done automatically in function_to_load_datasets.py 
    # arguments:
            X
            savefig: bool, save fig or do not save fig
            compare_case: int
                    Case 1: eight timeseries: Prior & IS & IS_ACOS & IS_RT for TM5_regional_Sourish & TM5_gridded_3x2 or 1x1 --> expect that regional & gridded are equal for 3x2 grid
            region_name: str
                    - 'SAT', 'SATr', 'SAT_SATr'
                    
    '''
    print('Start plotting Case '+ str(compare_case))
    ACOS_color = 'royalblue'
    RT_color='firebrick'
    IS_color = 'forestgreen'
    Apri_color = 'dimgrey'
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5), sharey=True, gridspec_kw={'width_ratios':[8,3]})
    fig.subplots_adjust(wspace=0)
    
    if compare_case==1:
        print('Case 1: eight timeseries: Prior & IS & IS_ACOS & IS_RT for TM5_regional_Sourish & TM5_gridded')
        ax1.plot(Fluxes_prior_region.MonthDate, (Fluxes_prior_region.Flux_bio_apri+Fluxes_prior_region.Flux_fire_apri)*(12/44), color = Apri_color, linestyle='solid', label = 'TM5-4DVar prior', linewidth=1.25)
        ax1.plot(Fluxes_prior_region_gridded.MonthDate, Fluxes_prior_region_gridded.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = Apri_color, linestyle='dashed', label = 'TM5-4DVar prior grid '+grid_res, linewidth=1.25)
        ax1.plot(Fluxes_IS_region.MonthDate, (Fluxes_IS_region.Flux_bio+Fluxes_IS_region.Flux_fire)*(12/44), color = IS_color, linestyle='solid', label = 'TM5-4DVar/IS', linewidth=1.25)
        ax1.plot(Fluxes_IS_region_gridded.MonthDate, Fluxes_IS_region_gridded.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = IS_color, linestyle='dashed', label = 'TM5-4DVar/IS grid '+grid_res, linewidth=1.25)
        ax1.plot(Fluxes_ACOS_IS_region.MonthDate, (Fluxes_ACOS_IS_region.Flux_bio+Fluxes_ACOS_IS_region.Flux_fire)*(12/44), color = ACOS_color, label = 'TM5-4DVar/IS+ACOS', linewidth=1.25)
        ax1.plot(Fluxes_ACOS_IS_region_gridded.MonthDate, Fluxes_ACOS_IS_region_gridded.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = ACOS_color, linestyle='dashed', label = 'TM5-4DVar/IS+ACOS grid '+grid_res, linewidth=1.25)
        ax1.plot(Fluxes_RT_IS_loc_region.MonthDate, (Fluxes_RT_IS_loc_region.Flux_bio+Fluxes_RT_IS_loc_region.Flux_fire)*(12/44), color = RT_color, label = 'TM5-4DVar/IS+RT', linewidth=1.25)
        ax1.plot(Fluxes_RT_IS_loc_region_gridded.MonthDate, Fluxes_RT_IS_loc_region_gridded.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = RT_color, linestyle='dashed', label = 'TM5-4DVar/IS+RT grid '+grid_res, linewidth=1.25)
        
        print('Case 1: eight timeseries: Prior & IS & IS_ACOS & IS_RT for TM5_regional_Sourish & TM5_gridded_3x2')
        Prior_region_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_prior_region, model='TM5_flux_regional_Sourish')
        Prior_region_gridded_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_prior_region_gridded, model='TM5_flux_gridded')
        IS_region_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_IS_region, model='TM5_flux_regional_Sourish')
        IS_region_gridded_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_IS_region_gridded, model='TM5_flux_gridded')
        ACOS_IS_region_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_ACOS_IS_region, model='TM5_flux_regional_Sourish')
        ACOS_IS_region_gridded_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_ACOS_IS_region_gridded, model='TM5_flux_gridded')
        RT_IS_loc_region_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_RT_IS_loc_region, model='TM5_flux_regional_Sourish')
        RT_IS_loc_region_gridded_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_RT_IS_loc_region_gridded, model='TM5_flux_gridded')
        # if start_month=5 -> Mai will be the first month that is plotted
        first_month = start_month-2 # needed because df.loc[:5] will also affect month=7
        Prior_region_msc.loc[:first_month,'Month'] += 12
        Prior_region_gridded_msc.loc[:first_month,'Month'] += 12
        IS_region_msc.loc[:first_month,'Month'] += 12
        IS_region_gridded_msc.loc[:first_month,'Month'] += 12
        ACOS_IS_region_msc.loc[:first_month,'Month'] += 12
        ACOS_IS_region_gridded_msc.loc[:first_month,'Month'] += 12
        RT_IS_loc_region_msc.loc[:first_month,'Month'] += 12
        RT_IS_loc_region_gridded_msc.loc[:first_month,'Month'] += 12
        # concatenate the two data frames vertically
        Prior_region_msc_new = pd.DataFrame(pd.concat([Prior_region_msc.loc[first_month+1:, 'Month'], Prior_region_msc.loc[:first_month, 'Month']], ignore_index=True))
        Prior_region_gridded_msc_new = pd.DataFrame(pd.concat([Prior_region_gridded_msc.loc[first_month+1:, 'Month'], Prior_region_gridded_msc.loc[:first_month, 'Month']], ignore_index=True))
        IS_region_msc_new = pd.DataFrame(pd.concat([IS_region_msc.loc[first_month+1:, 'Month'], IS_region_msc.loc[:first_month, 'Month']], ignore_index=True))
        IS_region_gridded_msc_new = pd.DataFrame(pd.concat([IS_region_gridded_msc.loc[first_month+1:, 'Month'], IS_region_gridded_msc.loc[:first_month, 'Month']], ignore_index=True))
        ACOS_IS_region_msc_new = pd.DataFrame(pd.concat([ACOS_IS_region_msc.loc[first_month+1:, 'Month'], ACOS_IS_region_msc.loc[:first_month, 'Month']], ignore_index=True))
        ACOS_IS_region_gridded_msc_new = pd.DataFrame(pd.concat([ACOS_IS_region_gridded_msc.loc[first_month+1:, 'Month'], ACOS_IS_region_gridded_msc.loc[:first_month, 'Month']], ignore_index=True))
        RT_IS_loc_region_msc_new = pd.DataFrame(pd.concat([RT_IS_loc_region_msc.loc[first_month+1:, 'Month'], RT_IS_loc_region_msc.loc[:first_month, 'Month']], ignore_index=True))
        RT_IS_loc_region_gridded_msc_new = pd.DataFrame(pd.concat([RT_IS_loc_region_gridded_msc.loc[first_month+1:, 'Month'], RT_IS_loc_region_gridded_msc.loc[:first_month, 'Month']], ignore_index=True))
        
        # merge
        Prior_region_msc = pd.merge(Prior_region_msc_new, Prior_region_msc, on='Month', how='outer')
        Prior_region_gridded_msc = pd.merge(Prior_region_gridded_msc_new, Prior_region_gridded_msc, on='Month', how='outer')
        IS_region_msc = pd.merge(IS_region_msc_new, IS_region_msc, on='Month', how='outer')
        IS_region_gridded_msc = pd.merge(IS_region_gridded_msc_new, IS_region_gridded_msc, on='Month', how='outer')
        ACOS_IS_region_msc = pd.merge(ACOS_IS_region_msc_new, ACOS_IS_region_msc, on='Month', how='outer')
        ACOS_IS_region_gridded_msc = pd.merge(ACOS_IS_region_gridded_msc_new, ACOS_IS_region_gridded_msc, on='Month', how='outer')
        RT_IS_loc_region_msc = pd.merge(RT_IS_loc_region_msc_new, RT_IS_loc_region_msc, on='Month', how='outer')
        RT_IS_loc_region_gridded_msc = pd.merge(RT_IS_loc_region_gridded_msc_new, RT_IS_loc_region_gridded_msc, on='Month', how='outer')

        ax2.plot(Prior_region_msc.Month, Prior_region_msc.Flux_fire_apri_plus_bio_apri_TgC, color = Apri_color, linestyle='solid', label = 'TM5-4DVar prior', linewidth=1.25)
        ax2.plot(Prior_region_gridded_msc.Month, Prior_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = Apri_color, linestyle='dashed', label = 'TM5-4DVar prior', linewidth=1.25)
        ax2.plot(IS_region_msc.Month, IS_region_msc.Flux_fire_plus_bio_TgC, color = IS_color, linestyle='solid', label = 'TM5-4DVar/IS', linewidth=1.25)
        ax2.plot(IS_region_gridded_msc.Month, IS_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = IS_color, linestyle='dashed', label = 'TM5-4DVar/IS gridded', linewidth=1.25)
        ax2.plot(ACOS_IS_region_msc.Month, ACOS_IS_region_msc.Flux_fire_plus_bio_TgC, color = ACOS_color, label='TM5-4DVar/IS+ACOS', linewidth=1.25)
        ax2.plot(ACOS_IS_region_gridded_msc.Month, ACOS_IS_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = ACOS_color, linestyle='dashed', label='TM5-4DVar/IS+ACOS gridded', linewidth=1.25)
        ax2.plot(RT_IS_loc_region_msc.Month, RT_IS_loc_region_msc.Flux_fire_plus_bio_TgC, color = RT_color, label='TM5-4DVar/IS+RT', linewidth=1.25)
        ax2.plot(RT_IS_loc_region_gridded_msc.Month, RT_IS_loc_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = RT_color, linestyle='dashed', label='TM5-4DVar/IS+RT gridded', linewidth=1.25)
        
        ax2.fill_between(Prior_region_msc.Month, 
                        (Prior_region_msc.Flux_fire_apri_plus_bio_apri_TgC)-Prior_region_msc.Flux_fire_apri_plus_bio_apri_std_TgC, 
                        (Prior_region_msc.Flux_fire_apri_plus_bio_apri_TgC)+Prior_region_msc.Flux_fire_apri_plus_bio_apri_std_TgC, 
                        color=Apri_color, linestyle='solid', alpha=0.2)
        ax2.fill_between(Prior_region_gridded_msc.Month,
                        (Prior_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)-Prior_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                        (Prior_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)+Prior_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                        color=Apri_color, linestyle='dashed', alpha=0.2)
        ax2.fill_between(IS_region_msc.Month,
                        (IS_region_msc.Flux_fire_plus_bio_TgC)-IS_region_msc.Flux_fire_plus_bio_std_TgC, 
                        (IS_region_msc.Flux_fire_plus_bio_TgC)+IS_region_msc.Flux_fire_plus_bio_std_TgC, 
                        color=IS_color, linestyle='solid', alpha=0.2)
        ax2.fill_between(IS_region_gridded_msc.Month,
                        (IS_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)-IS_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                        (IS_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)+IS_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                        color=IS_color, linestyle='dashed', alpha=0.2)
        ax2.fill_between(ACOS_IS_region_msc.Month,
                        (ACOS_IS_region_msc.Flux_fire_plus_bio_TgC)-ACOS_IS_region_msc.Flux_fire_plus_bio_std_TgC, 
                        (ACOS_IS_region_msc.Flux_fire_plus_bio_TgC)+ACOS_IS_region_msc.Flux_fire_plus_bio_std_TgC, 
                        color=ACOS_color, linestyle='solid', alpha=0.2)
        ax2.fill_between(ACOS_IS_region_gridded_msc.Month,
                        (ACOS_IS_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)-ACOS_IS_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                        (ACOS_IS_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)+ACOS_IS_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                        color=ACOS_color, linestyle='dashed', alpha=0.2)
        ax2.fill_between(RT_IS_loc_region_msc.Month,
                        (RT_IS_loc_region_msc.Flux_fire_plus_bio_TgC)-RT_IS_loc_region_msc.Flux_fire_plus_bio_std_TgC, 
                        (RT_IS_loc_region_msc.Flux_fire_plus_bio_TgC)+RT_IS_loc_region_msc.Flux_fire_plus_bio_std_TgC, 
                        color=RT_color, linestyle='solid', alpha=0.2)
        ax2.fill_between(RT_IS_loc_region_gridded_msc.Month,
                        (RT_IS_loc_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)-RT_IS_loc_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                        (RT_IS_loc_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)+RT_IS_loc_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                        color=RT_color, linestyle='dashed', alpha=0.2)
    
    fig.suptitle('TM5-4DVar land fluxes '+region_name+' regional Sourish & gridded '+grid_res, y=0.92, fontsize=12)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('CO$_2$ flux [TgC/month/region]', color='black')
    ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax1.minorticks_on()
    ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax1.tick_params(bottom=True, left=True, color='gray')
    #ax1.grid(which='major', color='white', axis='x', linestyle='-', linewidth='0.75')
    ax1.legend(framealpha=0.25, facecolor='grey', loc='lower left', fontsize=9.5, ncol=2, columnspacing=0.6)
    #ax1.set_title('TM5-4DVar land fluxes '+region_name+' regional Sourish & gridded '+grid_res, pad=-13)
    #MSC axis
    ax2.set_xlabel('Month')
    ax2.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    # also ok: ax2.set_xticks(np.arange(5,17,2)) # set the x-axis ticks
    # also ok: ax2.set_xticklabels([5, 7, 9, 11, 1, 3])
    ax2.set_xticks([5,10,15])
    ax2.set_xticklabels([5, 10, 3]) # set the x-axis tick labels
    ax2.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax2.tick_params(bottom=True, color='gray')
    ax2.set_title('MSC 2009-2019', pad=-13)
    #ax2.grid(which='major', color='grey', axis='x', linestyle='-', linewidth='0.75')
    
    print('Saving plot...')
    if compare_case==1:
        plt.savefig(savepath + 'TM5_prior_&_IS_&_IS_ACOS_&_IS_RT_ass_NEE+fire_flux_regional_sourish_&_gridded_'+grid_res+'_timeseries_AND_mean_seasonal_cycle_' + region_name + '.png', dpi=300, bbox_inches='tight')
    print('DONE plotting!')



# compare ONE subregion: Prior, TM5/IS, TM5/IS+ACOS, TM5/IS+RT, MIP
def plotTimeseries_MSC_start_May_subregion(start_month: int=5, include_MIP: bool=False, fix_y_scale:bool=False, start_year: int=2009,
                                           Fluxes_prior_region_gridded: pd.DataFrame = None,
                                           Fluxes_ACOS_IS_region_gridded: pd.DataFrame = None,
                                           Fluxes_IS_region_gridded: pd.DataFrame = None,
                                           Fluxes_RT_IS_loc_region_gridded: pd.DataFrame = None,
                                           Fluxes_MIP_IS_region_gridded: pd.DataFrame = None,
                                           Fluxes_MIP_IS_OCO2_region_gridded: pd.DataFrame = None,
                                           region: str='SAT', savepath: str='', compare_case: int=1, plot_std_error:bool=True, savefig: bool=True):
    '''# Documentation
    Function to plot timeseries & mean seasonal cycle of CO2 fluxes by TM5 for each subregion
    # arguments:
            include_MIP: bool, include MIP fluxes in plot or not; Then included gdfs should be loaded with specified start_ & end_year
            fix_y_scale: bool, fix y-scale or not
            savefig: bool, save fig or do not save fig
            compare_case: int
                    Case 1: four timeseries plotted: TM5_prior_gridded & TM5_IS_gridded & TM5_IS_ACOS_gridded & TM5_IS_RT_gridded
            region: str
                    - north_SAT
                    - mid_SAT
                    - south_SAT
                    - SAT
    '''
    ACOS_color = 'firebrick'
    RT_color = 'coral'
    IS_color = 'indianred'
    Apri_color = 'dimgrey'
    MIP_IS_color = 'royalblue'
    MIP_IS_OCO2_color = 'royalblue'
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5), sharey=True, gridspec_kw={'width_ratios':[8,3]})
    fig.subplots_adjust(wspace=0)
    
    if compare_case==1:
        print('Case 1: four timeseries plotted: TM5_prior_gridded & TM5_IS_gridded & TM5_IS_ACOS_gridded & TM5_IS_RT_gridded')
        ax1.plot(Fluxes_prior_region_gridded.MonthDate, Fluxes_prior_region_gridded.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = Apri_color, linestyle='dashed', label = 'TM5-4DVar prior', linewidth=1.25)
        ax1.plot(Fluxes_IS_region_gridded.MonthDate, Fluxes_IS_region_gridded.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = IS_color, linestyle='dashed', label = 'TM5-4DVar/IS', linewidth=1.25)
        ax1.plot(Fluxes_ACOS_IS_region_gridded.MonthDate, Fluxes_ACOS_IS_region_gridded.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = ACOS_color, linestyle='solid', label = 'TM5-4DVar/IS+ACOS', linewidth=1.25)
        ax1.plot(Fluxes_RT_IS_loc_region_gridded.MonthDate, Fluxes_RT_IS_loc_region_gridded.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = RT_color, linestyle='solid', label = 'TM5-4DVar/IS+RT', linewidth=1.25)
        if include_MIP==True:
            ax1.plot(Fluxes_MIP_IS_region_gridded.MonthDate, Fluxes_MIP_IS_region_gridded.Landtot, color = MIP_IS_color, linestyle='dashed', label = 'MIP/IS mean', linewidth=1.25)
            ax1.plot(Fluxes_MIP_IS_OCO2_region_gridded.MonthDate, Fluxes_MIP_IS_OCO2_region_gridded.Landtot, color = MIP_IS_OCO2_color, linestyle='solid', label = 'MIP/IS+OCO2 mean', linewidth=1.25)
        
        print('Case 1: four timeseries plotted: TM5_prior_gridded & TM5_IS_gridded & TM5_IS_ACOS_gridded & TM5_IS_RT_gridded')
        Prior_region_gridded_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_prior_region_gridded, model='TM5_flux_gridded')
        IS_region_gridded_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_IS_region_gridded, model='TM5_flux_gridded')
        ACOS_IS_region_gridded_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_ACOS_IS_region_gridded, model='TM5_flux_gridded')
        RT_IS_loc_region_gridded_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_RT_IS_loc_region_gridded, model='TM5_flux_gridded')
        if include_MIP==True:
            MIP_IS_region_gridded_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_MIP_IS_region_gridded, model='MIP')
            MIP_IS_OCO2_region_gridded_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_MIP_IS_OCO2_region_gridded, model='MIP')
        # if start_month=5 -> Mai will be the first month that is plotted
        first_month = start_month-2 # needed because df.loc[:5] will also affect month=7
        Prior_region_gridded_msc.loc[:first_month,'Month'] += 12
        IS_region_gridded_msc.loc[:first_month,'Month'] += 12
        ACOS_IS_region_gridded_msc.loc[:first_month,'Month'] += 12
        RT_IS_loc_region_gridded_msc.loc[:first_month,'Month'] += 12
        if include_MIP==True:
            MIP_IS_region_gridded_msc.loc[:first_month,'Month'] += 12
            MIP_IS_OCO2_region_gridded_msc.loc[:first_month,'Month'] += 12
        # concatenate the two data frames vertically
        Prior_region_gridded_msc_new = pd.DataFrame(pd.concat([Prior_region_gridded_msc.loc[first_month+1:, 'Month'], Prior_region_gridded_msc.loc[:first_month, 'Month']], ignore_index=True))
        IS_region_gridded_msc_new = pd.DataFrame(pd.concat([IS_region_gridded_msc.loc[first_month+1:, 'Month'], IS_region_gridded_msc.loc[:first_month, 'Month']], ignore_index=True))
        ACOS_IS_region_gridded_msc_new = pd.DataFrame(pd.concat([ACOS_IS_region_gridded_msc.loc[first_month+1:, 'Month'], ACOS_IS_region_gridded_msc.loc[:first_month, 'Month']], ignore_index=True))
        RT_IS_loc_region_gridded_msc_new = pd.DataFrame(pd.concat([RT_IS_loc_region_gridded_msc.loc[first_month+1:, 'Month'], RT_IS_loc_region_gridded_msc.loc[:first_month, 'Month']], ignore_index=True))
        if include_MIP==True:
            MIP_IS_region_gridded_msc_new = pd.DataFrame(pd.concat([MIP_IS_region_gridded_msc.loc[first_month+1:, 'Month'], MIP_IS_region_gridded_msc.loc[:first_month, 'Month']], ignore_index=True))
            MIP_IS_OCO2_region_gridded_msc_new = pd.DataFrame(pd.concat([MIP_IS_OCO2_region_gridded_msc.loc[first_month+1:, 'Month'], MIP_IS_OCO2_region_gridded_msc.loc[:first_month, 'Month']], ignore_index=True))
        # merge
        Prior_region_gridded_msc = pd.merge(Prior_region_gridded_msc_new, Prior_region_gridded_msc, on='Month', how='outer')
        IS_region_gridded_msc = pd.merge(IS_region_gridded_msc_new, IS_region_gridded_msc, on='Month', how='outer')
        ACOS_IS_region_gridded_msc = pd.merge(ACOS_IS_region_gridded_msc_new, ACOS_IS_region_gridded_msc, on='Month', how='outer')
        RT_IS_loc_region_gridded_msc = pd.merge(RT_IS_loc_region_gridded_msc_new, RT_IS_loc_region_gridded_msc, on='Month', how='outer')
        if include_MIP==True:
            MIP_IS_region_gridded_msc = pd.merge(MIP_IS_region_gridded_msc_new, MIP_IS_region_gridded_msc, on='Month', how='outer')
            MIP_IS_OCO2_region_gridded_msc = pd.merge(MIP_IS_OCO2_region_gridded_msc_new, MIP_IS_OCO2_region_gridded_msc, on='Month', how='outer')
            
        ax2.plot(Prior_region_gridded_msc.Month, Prior_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = Apri_color, linestyle='dashed', label = 'TM5-4DVar prior', linewidth=1.25)
        ax2.plot(IS_region_gridded_msc.Month, IS_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = IS_color, linestyle='dashed', label = 'TM5-4DVar/IS', linewidth=1.25)
        ax2.plot(ACOS_IS_region_gridded_msc.Month, ACOS_IS_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = ACOS_color, linestyle='solid', label='TM5-4DVar/IS+ACOS', linewidth=1.25)
        ax2.plot(RT_IS_loc_region_gridded_msc.Month, RT_IS_loc_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = RT_color, linestyle='solid', label='TM5-4DVar/IS+RT', linewidth=1.25)
        if include_MIP==True:
            ax2.plot(MIP_IS_region_gridded_msc.Month, MIP_IS_region_gridded_msc.Landtot, color = MIP_IS_color, linestyle='dashed', label='MIP/IS mean', linewidth=1.25)
            ax2.plot(MIP_IS_OCO2_region_gridded_msc.Month, MIP_IS_OCO2_region_gridded_msc.Landtot, color = MIP_IS_OCO2_color, linestyle='solid', label='MIP/IS+OCO2 mean', linewidth=1.25)
        if plot_std_error:
            ax2.fill_between(Prior_region_gridded_msc.Month,
                            (Prior_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)-Prior_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            (Prior_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)+Prior_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            color=Apri_color, linestyle='dashed', alpha=0.2)
            ax2.fill_between(IS_region_gridded_msc.Month,
                            (IS_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)-IS_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            (IS_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)+IS_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            color=IS_color, linestyle='dashed', alpha=0.2)
            ax2.fill_between(ACOS_IS_region_gridded_msc.Month,
                            (ACOS_IS_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)-ACOS_IS_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            (ACOS_IS_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)+ACOS_IS_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            color=ACOS_color, linestyle='solid', alpha=0.2)
            ax2.fill_between(RT_IS_loc_region_gridded_msc.Month,
                            (RT_IS_loc_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)-RT_IS_loc_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            (RT_IS_loc_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)+RT_IS_loc_region_gridded_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            color=RT_color, linestyle='solid', alpha=0.2)
            if include_MIP==True:
                ax2.fill_between(MIP_IS_region_gridded_msc.Month,
                             MIP_IS_region_gridded_msc.Landtot-MIP_IS_region_gridded_msc.Landtot_std,
                             MIP_IS_region_gridded_msc.Landtot+MIP_IS_region_gridded_msc.Landtot_std,
                             color=MIP_IS_color, linestyle='dashed', alpha=0.2)
                ax2.fill_between(MIP_IS_OCO2_region_gridded_msc.Month,
                                MIP_IS_OCO2_region_gridded_msc.Landtot-MIP_IS_OCO2_region_gridded_msc.Landtot_std,
                                MIP_IS_OCO2_region_gridded_msc.Landtot+MIP_IS_OCO2_region_gridded_msc.Landtot_std,
                                color=MIP_IS_OCO2_color, linestyle='solid', alpha=0.2)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('CO$_2$ flux [TgC/month/region]', color='black')
    ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax1.minorticks_on()
    ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax1.tick_params(bottom=True, left=True, color='gray')
    ax1.legend(framealpha=0.25, facecolor='grey', loc='lower left', fontsize=9.5, ncol=2)
    ax1.set_title('TM5-4DVar land fluxes '+region, pad=-13)
    if fix_y_scale==True:
        if include_MIP==True:
            ax1.set_yticks([-300, -200, -100, 0, 100, 200])
            ax1.set_yticklabels([-300, -200, -100, 0, 100, 200])
            ax1.set_ylim([-350, 230])
        else:
            ax1.set_yticks([-400, -200, 0, 200, 400])
            ax1.set_yticklabels([-400, -200, 0, 200, 400])
            ax1.set_ylim([-450, 500])
    #MSC axis
    ax2.set_xlabel('Month')
    ax2.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax2.set_xticks([5,10,15])
    ax2.set_xticklabels([5, 10, 3]) # set the x-axis tick labels
    ax2.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax2.tick_params(bottom=True, color='gray')
    ax2.set_title('MSC '+str(start_year)+'-2019', pad=-13)
    if fix_y_scale==True:
        if include_MIP==True:
            ax2.set_ylim([-350, 230])
        else:
            ax2.set_ylim([-450, 500])
    
    if compare_case==1:
        if include_MIP==True:
            plt.savefig(savepath + 'TM5_prior_&_TM5_IS_&_TM5_IS_ACOS_&_TM5_IS_RT_ass_&_MIP_IS_&_MIP_IS_OCO2_ass_ens_gridded_bio+fire_flux_timeseries_AND_mean_seasonal_cycle_' + region + '.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(savepath + 'TM5_prior_&_TM5_IS_&_TM5_IS_ACOS_&_TM5_IS_RT_ass_gridded_bio+fire_flux_timeseries_AND_mean_seasonal_cycle_' + region + '.png', dpi=300, bbox_inches='tight')


# compare 3 subregions + SAT region for one model/assimilation configuration:
def plotTimeseries_MSC_start_May_subregion_comparison(start_month: int=5, is_MIP: bool=False, fix_y_scale: bool=False, start_year: int=2009,
                                                      Fluxes_ACOS_IS_SAT: pd.DataFrame = None,
                                                      Fluxes_ACOS_IS_north_SAT: pd.DataFrame = None,
                                                      Fluxes_ACOS_IS_mid_SAT: pd.DataFrame = None,
                                                      Fluxes_ACOS_IS_south_SAT: pd.DataFrame = None,
                                                      assimilated: str='IS', savepath: str='', compare_case: int=1, plot_std_error:bool=True, savefig: bool=True):
    '''# Documentation
    Compare fluxes in one plot from different subregions
    Function to plot timeseries & mean seasonal cycle of CO2 fluxes by TM5 for each subregion.
    # arguments:
            X
            savefig: bool, save fig or do not save fig
            assimilated: str
                    - IS
                    - IS+ACOS
                    - IS+RT
                    - IS+OCO2 if is_MIP==True
            compare_case: int
                    Case 1: for north,mid,south SAT:    four timeseries plotted: TM5_prior_gridded & TM5_IS_gridded & TM5_IS_ACOS_gridded & TM5_IS_RT_gridded
                    Case 2: for west,mid_long,east SAT: four timeseries plotted: TM5_prior_gridded & TM5_IS_gridded & TM5_IS_ACOS_gridded & TM5_IS_RT_gridded
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
        if is_MIP==True:
            ax1.plot(Fluxes_ACOS_IS_SAT.MonthDate, Fluxes_ACOS_IS_SAT.Landtot, color = SAT_color, linestyle='dashed', label = 'MIP SAT', linewidth=1.25)
            ax1.plot(Fluxes_ACOS_IS_north_SAT.MonthDate, Fluxes_ACOS_IS_north_SAT.Landtot, color = nSAT_color, linestyle='solid', label = 'MIP north SAT', linewidth=1.25)
            ax1.plot(Fluxes_ACOS_IS_mid_SAT.MonthDate, Fluxes_ACOS_IS_mid_SAT.Landtot, color = mSAT_color, linestyle='solid', label = 'MIP mid SAT', linewidth=1.25)
            ax1.plot(Fluxes_ACOS_IS_south_SAT.MonthDate, Fluxes_ACOS_IS_south_SAT.Landtot, color = sSAT_color, linestyle='solid', label = 'MIP south SAT', linewidth=1.25)    
        else:
            ax1.plot(Fluxes_ACOS_IS_SAT.MonthDate, Fluxes_ACOS_IS_SAT.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = SAT_color, linestyle='dashed', label = 'TM5-4DVar SAT', linewidth=1.25)
            ax1.plot(Fluxes_ACOS_IS_north_SAT.MonthDate, Fluxes_ACOS_IS_north_SAT.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = nSAT_color, linestyle='solid', label = 'TM5-4DVar north SAT', linewidth=1.25)
            ax1.plot(Fluxes_ACOS_IS_mid_SAT.MonthDate, Fluxes_ACOS_IS_mid_SAT.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = mSAT_color, linestyle='solid', label = 'TM5-4DVar mid SAT', linewidth=1.25)
            ax1.plot(Fluxes_ACOS_IS_south_SAT.MonthDate, Fluxes_ACOS_IS_south_SAT.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = sSAT_color, linestyle='solid', label = 'TM5-4DVar south SAT', linewidth=1.25)    
        print('Case 1: four timeseries plotted: SAT, north SAT, mid SAT, south SAT gridded')
        if is_MIP==True:
            Fluxes_ACOS_IS_SAT_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_ACOS_IS_SAT, model='MIP')
            Fluxes_ACOS_IS_north_SAT_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_ACOS_IS_north_SAT, model='MIP')
            Fluxes_ACOS_IS_mid_SAT_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_ACOS_IS_mid_SAT, model='MIP')
            Fluxes_ACOS_IS_south_SAT_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_ACOS_IS_south_SAT, model='MIP')
        else:
            Fluxes_ACOS_IS_SAT_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_ACOS_IS_SAT, model='TM5_flux_gridded')
            Fluxes_ACOS_IS_north_SAT_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_ACOS_IS_north_SAT, model='TM5_flux_gridded')
            Fluxes_ACOS_IS_mid_SAT_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_ACOS_IS_mid_SAT, model='TM5_flux_gridded')
            Fluxes_ACOS_IS_south_SAT_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_ACOS_IS_south_SAT, model='TM5_flux_gridded')
        # if start_month=5 -> Mai will be the first month that is plotted
        first_month = start_month-2 # needed because df.loc[:5] will also affect month=7
        Fluxes_ACOS_IS_SAT_msc.loc[:first_month,'Month'] += 12
        Fluxes_ACOS_IS_north_SAT_msc.loc[:first_month,'Month'] += 12
        Fluxes_ACOS_IS_mid_SAT_msc.loc[:first_month,'Month'] += 12
        Fluxes_ACOS_IS_south_SAT_msc.loc[:first_month,'Month'] += 12
        # concatenate the two data frames vertically
        Fluxes_ACOS_IS_SAT_msc_new = pd.DataFrame(pd.concat([Fluxes_ACOS_IS_SAT_msc.loc[first_month+1:, 'Month'], Fluxes_ACOS_IS_SAT_msc.loc[:first_month, 'Month']], ignore_index=True))
        Fluxes_ACOS_IS_north_SAT_msc_new = pd.DataFrame(pd.concat([Fluxes_ACOS_IS_north_SAT_msc.loc[first_month+1:, 'Month'], Fluxes_ACOS_IS_north_SAT_msc.loc[:first_month, 'Month']], ignore_index=True))
        Fluxes_ACOS_IS_mid_SAT_msc_new = pd.DataFrame(pd.concat([Fluxes_ACOS_IS_mid_SAT_msc.loc[first_month+1:, 'Month'], Fluxes_ACOS_IS_mid_SAT_msc.loc[:first_month, 'Month']], ignore_index=True))
        Fluxes_ACOS_IS_south_SAT_msc_new = pd.DataFrame(pd.concat([Fluxes_ACOS_IS_south_SAT_msc.loc[first_month+1:, 'Month'], Fluxes_ACOS_IS_south_SAT_msc.loc[:first_month, 'Month']], ignore_index=True))
        
        # merge
        Fluxes_ACOS_IS_SAT_msc = pd.merge(Fluxes_ACOS_IS_SAT_msc_new, Fluxes_ACOS_IS_SAT_msc, on='Month', how='outer')
        Fluxes_ACOS_IS_north_SAT_msc = pd.merge(Fluxes_ACOS_IS_north_SAT_msc_new, Fluxes_ACOS_IS_north_SAT_msc, on='Month', how='outer')
        Fluxes_ACOS_IS_mid_SAT_msc = pd.merge(Fluxes_ACOS_IS_mid_SAT_msc_new, Fluxes_ACOS_IS_mid_SAT_msc, on='Month', how='outer')
        Fluxes_ACOS_IS_south_SAT_msc = pd.merge(Fluxes_ACOS_IS_south_SAT_msc_new, Fluxes_ACOS_IS_south_SAT_msc, on='Month', how='outer')

        if is_MIP==True:
            ax2.plot(Fluxes_ACOS_IS_SAT_msc.Month, Fluxes_ACOS_IS_SAT_msc.Landtot, color = SAT_color, linestyle='dashed', label = 'MIP SAT', linewidth=1.25)
            ax2.plot(Fluxes_ACOS_IS_north_SAT_msc.Month, Fluxes_ACOS_IS_north_SAT_msc.Landtot, color = nSAT_color, linestyle='solid', label = 'MIP nSAT', linewidth=1.25)
            ax2.plot(Fluxes_ACOS_IS_mid_SAT_msc.Month, Fluxes_ACOS_IS_mid_SAT_msc.Landtot, color = mSAT_color, linestyle='solid', label='MIP mSAT', linewidth=1.25)
            ax2.plot(Fluxes_ACOS_IS_south_SAT_msc.Month, Fluxes_ACOS_IS_south_SAT_msc.Landtot, color = sSAT_color, linestyle='solid', label='MIP sSAT', linewidth=1.25)
        else:
            ax2.plot(Fluxes_ACOS_IS_SAT_msc.Month, Fluxes_ACOS_IS_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = SAT_color, linestyle='dashed', label = 'TM5-4DVar SAT', linewidth=1.25)
            ax2.plot(Fluxes_ACOS_IS_north_SAT_msc.Month, Fluxes_ACOS_IS_north_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = nSAT_color, linestyle='solid', label = 'TM5-4DVar nSAT', linewidth=1.25)
            ax2.plot(Fluxes_ACOS_IS_mid_SAT_msc.Month, Fluxes_ACOS_IS_mid_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = mSAT_color, linestyle='solid', label='TM5-4DVar mSAT', linewidth=1.25)
            ax2.plot(Fluxes_ACOS_IS_south_SAT_msc.Month, Fluxes_ACOS_IS_south_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = sSAT_color, linestyle='solid', label='TM5-4DVar sSAT', linewidth=1.25)
        if is_MIP==True:
            ax2.fill_between(Fluxes_ACOS_IS_SAT_msc.Month,
                            (Fluxes_ACOS_IS_SAT_msc.Landtot)-Fluxes_ACOS_IS_SAT_msc.Landtot_std, 
                            (Fluxes_ACOS_IS_SAT_msc.Landtot)+Fluxes_ACOS_IS_SAT_msc.Landtot_std, 
                            color=SAT_color, linestyle='dashed', alpha=0.2)
            ax2.fill_between(Fluxes_ACOS_IS_north_SAT_msc.Month,
                            (Fluxes_ACOS_IS_north_SAT_msc.Landtot)-Fluxes_ACOS_IS_north_SAT_msc.Landtot_std, 
                            (Fluxes_ACOS_IS_north_SAT_msc.Landtot)+Fluxes_ACOS_IS_north_SAT_msc.Landtot_std, 
                            color=nSAT_color, linestyle='solid', alpha=0.2)
            ax2.fill_between(Fluxes_ACOS_IS_mid_SAT_msc.Month,
                            (Fluxes_ACOS_IS_mid_SAT_msc.Landtot)-Fluxes_ACOS_IS_mid_SAT_msc.Landtot_std, 
                            (Fluxes_ACOS_IS_mid_SAT_msc.Landtot)+Fluxes_ACOS_IS_mid_SAT_msc.Landtot_std, 
                            color=mSAT_color, linestyle='solid', alpha=0.2)
            ax2.fill_between(Fluxes_ACOS_IS_south_SAT_msc.Month,
                            (Fluxes_ACOS_IS_south_SAT_msc.Landtot)-Fluxes_ACOS_IS_south_SAT_msc.Landtot_std, 
                            (Fluxes_ACOS_IS_south_SAT_msc.Landtot)+Fluxes_ACOS_IS_south_SAT_msc.Landtot_std, 
                            color=sSAT_color, linestyle='solid', alpha=0.2)
        else:
            ax2.fill_between(Fluxes_ACOS_IS_SAT_msc.Month,
                            (Fluxes_ACOS_IS_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)-Fluxes_ACOS_IS_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            (Fluxes_ACOS_IS_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)+Fluxes_ACOS_IS_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            color=SAT_color, linestyle='dashed', alpha=0.2)
            ax2.fill_between(Fluxes_ACOS_IS_north_SAT_msc.Month,
                            (Fluxes_ACOS_IS_north_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)-Fluxes_ACOS_IS_north_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            (Fluxes_ACOS_IS_north_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)+Fluxes_ACOS_IS_north_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            color=nSAT_color, linestyle='solid', alpha=0.2)
            ax2.fill_between(Fluxes_ACOS_IS_mid_SAT_msc.Month,
                            (Fluxes_ACOS_IS_mid_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)-Fluxes_ACOS_IS_mid_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            (Fluxes_ACOS_IS_mid_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)+Fluxes_ACOS_IS_mid_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            color=mSAT_color, linestyle='solid', alpha=0.2)
            ax2.fill_between(Fluxes_ACOS_IS_south_SAT_msc.Month,
                            (Fluxes_ACOS_IS_south_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)-Fluxes_ACOS_IS_south_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            (Fluxes_ACOS_IS_south_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)+Fluxes_ACOS_IS_south_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            color=sSAT_color, linestyle='solid', alpha=0.2)

    if compare_case==2:
        print('Case 2: for west,mid_long,east SAT: four timeseries plotted: SAT, west SAT, mid_long SAT, east SAT gridded')
        if is_MIP==True:
            ax1.plot(Fluxes_ACOS_IS_SAT.MonthDate, Fluxes_ACOS_IS_SAT.Landtot, color = SAT_color, linestyle='dashed', label = 'MIP SAT', linewidth=1.25)
            ax1.plot(Fluxes_ACOS_IS_north_SAT.MonthDate, Fluxes_ACOS_IS_north_SAT.Landtot, color = nSAT_color, linestyle='solid', label = 'MIP west SAT', linewidth=1.25)
            ax1.plot(Fluxes_ACOS_IS_mid_SAT.MonthDate, Fluxes_ACOS_IS_mid_SAT.Landtot, color = mSAT_color, linestyle='solid', label = 'MIP mid-long SAT', linewidth=1.25)
            ax1.plot(Fluxes_ACOS_IS_south_SAT.MonthDate, Fluxes_ACOS_IS_south_SAT.Landtot, color = sSAT_color, linestyle='solid', label = 'MIP east SAT', linewidth=1.25)    
        else:
            ax1.plot(Fluxes_ACOS_IS_SAT.MonthDate, Fluxes_ACOS_IS_SAT.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = SAT_color, linestyle='dashed', label = 'TM5-4DVar SAT', linewidth=1.25)
            ax1.plot(Fluxes_ACOS_IS_north_SAT.MonthDate, Fluxes_ACOS_IS_north_SAT.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = nSAT_color, linestyle='solid', label = 'TM5-4DVar west SAT', linewidth=1.25)
            ax1.plot(Fluxes_ACOS_IS_mid_SAT.MonthDate, Fluxes_ACOS_IS_mid_SAT.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = mSAT_color, linestyle='solid', label = 'TM5-4DVar mid-long SAT', linewidth=1.25)
            ax1.plot(Fluxes_ACOS_IS_south_SAT.MonthDate, Fluxes_ACOS_IS_south_SAT.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = sSAT_color, linestyle='solid', label = 'TM5-4DVar east SAT', linewidth=1.25)    
        print('Case 1: four timeseries plotted: SAT, north SAT, mid SAT, south SAT gridded')
        if is_MIP==True:
            Fluxes_ACOS_IS_SAT_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_ACOS_IS_SAT, model='MIP')
            Fluxes_ACOS_IS_north_SAT_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_ACOS_IS_north_SAT, model='MIP')
            Fluxes_ACOS_IS_mid_SAT_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_ACOS_IS_mid_SAT, model='MIP')
            Fluxes_ACOS_IS_south_SAT_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_ACOS_IS_south_SAT, model='MIP')
        else:
            Fluxes_ACOS_IS_SAT_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_ACOS_IS_SAT, model='TM5_flux_gridded')
            Fluxes_ACOS_IS_north_SAT_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_ACOS_IS_north_SAT, model='TM5_flux_gridded')
            Fluxes_ACOS_IS_mid_SAT_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_ACOS_IS_mid_SAT, model='TM5_flux_gridded')
            Fluxes_ACOS_IS_south_SAT_msc = calculateMeanSeasonalCycle_of_fluxes(Fluxes_ACOS_IS_south_SAT, model='TM5_flux_gridded')
        # if start_month=5 -> Mai will be the first month that is plotted
        first_month = start_month-2 # needed because df.loc[:5] will also affect month=7
        Fluxes_ACOS_IS_SAT_msc.loc[:first_month,'Month'] += 12
        Fluxes_ACOS_IS_north_SAT_msc.loc[:first_month,'Month'] += 12
        Fluxes_ACOS_IS_mid_SAT_msc.loc[:first_month,'Month'] += 12
        Fluxes_ACOS_IS_south_SAT_msc.loc[:first_month,'Month'] += 12
        # concatenate the two data frames vertically
        Fluxes_ACOS_IS_SAT_msc_new = pd.DataFrame(pd.concat([Fluxes_ACOS_IS_SAT_msc.loc[first_month+1:, 'Month'], Fluxes_ACOS_IS_SAT_msc.loc[:first_month, 'Month']], ignore_index=True))
        Fluxes_ACOS_IS_north_SAT_msc_new = pd.DataFrame(pd.concat([Fluxes_ACOS_IS_north_SAT_msc.loc[first_month+1:, 'Month'], Fluxes_ACOS_IS_north_SAT_msc.loc[:first_month, 'Month']], ignore_index=True))
        Fluxes_ACOS_IS_mid_SAT_msc_new = pd.DataFrame(pd.concat([Fluxes_ACOS_IS_mid_SAT_msc.loc[first_month+1:, 'Month'], Fluxes_ACOS_IS_mid_SAT_msc.loc[:first_month, 'Month']], ignore_index=True))
        Fluxes_ACOS_IS_south_SAT_msc_new = pd.DataFrame(pd.concat([Fluxes_ACOS_IS_south_SAT_msc.loc[first_month+1:, 'Month'], Fluxes_ACOS_IS_south_SAT_msc.loc[:first_month, 'Month']], ignore_index=True))
        
        # merge
        Fluxes_ACOS_IS_SAT_msc = pd.merge(Fluxes_ACOS_IS_SAT_msc_new, Fluxes_ACOS_IS_SAT_msc, on='Month', how='outer')
        Fluxes_ACOS_IS_north_SAT_msc = pd.merge(Fluxes_ACOS_IS_north_SAT_msc_new, Fluxes_ACOS_IS_north_SAT_msc, on='Month', how='outer')
        Fluxes_ACOS_IS_mid_SAT_msc = pd.merge(Fluxes_ACOS_IS_mid_SAT_msc_new, Fluxes_ACOS_IS_mid_SAT_msc, on='Month', how='outer')
        Fluxes_ACOS_IS_south_SAT_msc = pd.merge(Fluxes_ACOS_IS_south_SAT_msc_new, Fluxes_ACOS_IS_south_SAT_msc, on='Month', how='outer')

        if is_MIP==True:
            ax2.plot(Fluxes_ACOS_IS_SAT_msc.Month, Fluxes_ACOS_IS_SAT_msc.Landtot, color = SAT_color, linestyle='dashed', label = 'MIP SAT', linewidth=1.25)
            ax2.plot(Fluxes_ACOS_IS_north_SAT_msc.Month, Fluxes_ACOS_IS_north_SAT_msc.Landtot, color = nSAT_color, linestyle='solid', label = 'MIP wSAT', linewidth=1.25)
            ax2.plot(Fluxes_ACOS_IS_mid_SAT_msc.Month, Fluxes_ACOS_IS_mid_SAT_msc.Landtot, color = mSAT_color, linestyle='solid', label='MIP mSAT', linewidth=1.25)
            ax2.plot(Fluxes_ACOS_IS_south_SAT_msc.Month, Fluxes_ACOS_IS_south_SAT_msc.Landtot, color = sSAT_color, linestyle='solid', label='MIP eSAT', linewidth=1.25)
        else:
            ax2.plot(Fluxes_ACOS_IS_SAT_msc.Month, Fluxes_ACOS_IS_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = SAT_color, linestyle='dashed', label = 'TM5-4DVar SAT', linewidth=1.25)
            ax2.plot(Fluxes_ACOS_IS_north_SAT_msc.Month, Fluxes_ACOS_IS_north_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = nSAT_color, linestyle='solid', label = 'TM5-4DVar wSAT', linewidth=1.25)
            ax2.plot(Fluxes_ACOS_IS_mid_SAT_msc.Month, Fluxes_ACOS_IS_mid_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = mSAT_color, linestyle='solid', label='TM5-4DVar mSAT', linewidth=1.25)
            ax2.plot(Fluxes_ACOS_IS_south_SAT_msc.Month, Fluxes_ACOS_IS_south_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = sSAT_color, linestyle='solid', label='TM5-4DVar eSAT', linewidth=1.25)
        if is_MIP==True:
            ax2.fill_between(Fluxes_ACOS_IS_SAT_msc.Month,
                            (Fluxes_ACOS_IS_SAT_msc.Landtot)-Fluxes_ACOS_IS_SAT_msc.Landtot_std, 
                            (Fluxes_ACOS_IS_SAT_msc.Landtot)+Fluxes_ACOS_IS_SAT_msc.Landtot_std, 
                            color=SAT_color, linestyle='dashed', alpha=0.2)
            ax2.fill_between(Fluxes_ACOS_IS_north_SAT_msc.Month,
                            (Fluxes_ACOS_IS_north_SAT_msc.Landtot)-Fluxes_ACOS_IS_north_SAT_msc.Landtot_std, 
                            (Fluxes_ACOS_IS_north_SAT_msc.Landtot)+Fluxes_ACOS_IS_north_SAT_msc.Landtot_std, 
                            color=nSAT_color, linestyle='solid', alpha=0.2)
            ax2.fill_between(Fluxes_ACOS_IS_mid_SAT_msc.Month,
                            (Fluxes_ACOS_IS_mid_SAT_msc.Landtot)-Fluxes_ACOS_IS_mid_SAT_msc.Landtot_std, 
                            (Fluxes_ACOS_IS_mid_SAT_msc.Landtot)+Fluxes_ACOS_IS_mid_SAT_msc.Landtot_std, 
                            color=mSAT_color, linestyle='solid', alpha=0.2)
            ax2.fill_between(Fluxes_ACOS_IS_south_SAT_msc.Month,
                            (Fluxes_ACOS_IS_south_SAT_msc.Landtot)-Fluxes_ACOS_IS_south_SAT_msc.Landtot_std, 
                            (Fluxes_ACOS_IS_south_SAT_msc.Landtot)+Fluxes_ACOS_IS_south_SAT_msc.Landtot_std, 
                            color=sSAT_color, linestyle='solid', alpha=0.2)
        else:
            ax2.fill_between(Fluxes_ACOS_IS_SAT_msc.Month,
                            (Fluxes_ACOS_IS_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)-Fluxes_ACOS_IS_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            (Fluxes_ACOS_IS_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)+Fluxes_ACOS_IS_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            color=SAT_color, linestyle='dashed', alpha=0.2)
            ax2.fill_between(Fluxes_ACOS_IS_north_SAT_msc.Month,
                            (Fluxes_ACOS_IS_north_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)-Fluxes_ACOS_IS_north_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            (Fluxes_ACOS_IS_north_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)+Fluxes_ACOS_IS_north_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            color=nSAT_color, linestyle='solid', alpha=0.2)
            ax2.fill_between(Fluxes_ACOS_IS_mid_SAT_msc.Month,
                            (Fluxes_ACOS_IS_mid_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)-Fluxes_ACOS_IS_mid_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            (Fluxes_ACOS_IS_mid_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)+Fluxes_ACOS_IS_mid_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            color=mSAT_color, linestyle='solid', alpha=0.2)
            ax2.fill_between(Fluxes_ACOS_IS_south_SAT_msc.Month,
                            (Fluxes_ACOS_IS_south_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)-Fluxes_ACOS_IS_south_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            (Fluxes_ACOS_IS_south_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion)+Fluxes_ACOS_IS_south_SAT_msc.CO2_NEE_fire_flux_monthly_TgC_per_subregion_std, 
                            color=sSAT_color, linestyle='solid', alpha=0.2)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('CO$_2$ flux [TgC/month/region]', color='black')
    ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax1.minorticks_on()
    ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax1.tick_params(bottom=True, left=True, color='gray')
    ax1.legend(framealpha=0.25, facecolor='grey', loc='lower left', fontsize=9.5, ncol=2)
    if is_MIP==True:
        ax1.set_title('MIP/'+assimilated+' land fluxes per region')#, pad=-13)
    else:
        ax1.set_title('TM5-4DVar/'+assimilated+' land fluxes per region')#, pad=-13)
    if fix_y_scale==True:
        if is_MIP==True:
            ax1.set_yticks([-300, -200, -100, 0, 100, 200])
            ax1.set_yticklabels([-300, -200, -100, 0, 100, 200])
            ax1.set_ylim([-350, 250])
        else:
            ax1.set_yticks([-300, -200, -100, 0, 100, 200, 300])
            ax1.set_yticklabels([-300, -200, -100, 0, 100, 200, 300])
            ax1.set_ylim([-330, 300])
    #MSC axis
    ax2.set_xlabel('Month')
    ax2.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax2.set_xticks([5,10,15])
    ax2.set_xticklabels([5, 10, 3]) # set the x-axis tick labels
    ax2.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax2.tick_params(bottom=True, color='gray')
    ax2.set_title('MSC 2009-2019')#, pad=-13)
    if fix_y_scale==True:
        if is_MIP==True:
            ax2.set_ylim([-350, 250])
        else:
            ax2.set_ylim([-330, 300])

    if compare_case==1:
        if is_MIP==True:
            plt.savefig(savepath + 'MIP_'+assimilated+'_gridded_SAT_nSAT_mSAT_sSAT_bio+fire_flux_timeseries_AND_mean_seasonal_cycle_std_dev_.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(savepath + 'TM5_'+assimilated+'_gridded_SAT_nSAT_mSAT_sSAT_bio+fire_flux_timeseries_AND_mean_seasonal_cycle_std_dev_.png', dpi=300, bbox_inches='tight')
    if compare_case==2:
        if is_MIP==True:
            plt.savefig(savepath + 'MIP'+assimilated+'_gridded_SAT_wSAT_mSAT_eSAT_bio+fire_flux_timeseries_AND_mean_seasonal_cycle_std_dev.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(savepath + 'TM5_'+assimilated+'_gridded_SAT_wSAT_mSAT_eSAT_bio+fire_flux_timeseries_AND_mean_seasonal_cycle_std_dev.png', dpi=300, bbox_inches='tight')


# plotting functions
def plotFluxTimeseries_and_MSC_general_with_second_MSC_before_main_MSC(start_month: int = 5, start_year: int=2009, end_year: int=2019, fix_y_scale: bool=False,
                               gdf_list: list=[],
                               plot_second_MSCs: bool = False,
                               second_MSC_start_year: int = 2014,
                               second_MSC_end_year: int = 2018,
                               variable_to_plot_list: list=['CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'Landtot'],
                               model_assimilation_list: list=['TM5-4DVar_prior', 'TM5-4DVar_IS', 'TM5-4DVar_IS+ACOS', 'TM5-4DVar_IS+RT', 'TM5-4DVar_IS+OCO2', 'MIP_IS_ens', 'MIP_IS+OCO2_ens'],
                               what_is_plotted: str='NEE+fire_flux',
                               color_list: list=['dimgrey', 'forestgreen', 'royalblue', 'firebrick'],
                               linestyle_list: list=['dashed', 'solid'],
                               region_name: str='SAT', savepath: str='/home/lartelt/MA/software/', compare_case: int=1):
    '''# Documentation
    Function to plot timeseries of the mean seasonal cycle of CO2 fluxes modelled by TM5, MIP & next to it the mean seasonal cycle
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
    if plot_second_MSCs:
        fig, (ax1, ax3, ax2) = plt.subplots(1,3, figsize=(10,5), sharey=True, gridspec_kw={'width_ratios':[6,2,2]})#figsize=(7,5)
    else:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5), sharey=True, gridspec_kw={'width_ratios':[8,3]})
    fig.subplots_adjust(wspace=0)
    
    first_month = start_month-2 # needed because df.loc[:5] will also affect month=7
    savename = what_is_plotted
    if compare_case==1:
        print('Case 1 timeseries')
        for i,model in enumerate(model_assimilation_list):
            ax1.plot(gdf_list[i].MonthDate, gdf_list[i][variable_to_plot_list[i]], color = color_list[i], linestyle=linestyle_list[i], label = model, linewidth=1.25)
            if model_assimilation_list[i] not in ['TM5-4DVar_prior', 'TM5-4DVar_IS+ACOS', 'TM5-4DVar_IS+RT']:
                if model=='TM5-4DVar_IS+OCO2':
                    MSC = calculateMeanSeasonalCycle_of_fluxes(gdf_list[i], model='MIP')
                else:
                    MSC = calculateMeanSeasonalCycle_of_fluxes(gdf_list[i], model=model_assimilation_list[i][:3])
                MSC.loc[:first_month,'Month'] += 12
                MSC_new = pd.DataFrame(pd.concat([MSC.loc[first_month+1:, 'Month'], MSC.loc[:first_month, 'Month']], ignore_index=True))
                MSC = pd.merge(MSC_new, MSC, on='Month', how='outer')
                ax2.plot(MSC.Month, MSC[variable_to_plot_list[i]], color = color_list[i], linestyle=linestyle_list[i], label = model, linewidth=1.25)
                ax2.fill_between(MSC.Month,
                                MSC[variable_to_plot_list[i]] - MSC[(variable_to_plot_list[i]+'_std')], 
                                MSC[variable_to_plot_list[i]] + MSC[(variable_to_plot_list[i]+'_std')], 
                                color = color_list[i], linestyle=linestyle_list[i], alpha=0.2)
            if plot_second_MSCs:
                gdf_second_MSC = gdf_list[i].drop(gdf_list[i][(gdf_list[i]['year'] < second_MSC_start_year)].index, inplace=False)
                gdf_second_MSC = gdf_second_MSC.drop(gdf_list[i][(gdf_list[i]['year'] > second_MSC_end_year)].index, inplace=False)
                if model=='TM5-4DVar_IS+OCO2':
                    MSC_snd_msc = calculateMeanSeasonalCycle_of_fluxes(gdf_second_MSC, model='MIP')
                else:
                    MSC_snd_msc = calculateMeanSeasonalCycle_of_fluxes(gdf_second_MSC, model=model_assimilation_list[i][:3])
                MSC_snd_msc.loc[:first_month,'Month'] += 12
                MSC_snd_msc_new = pd.DataFrame(pd.concat([MSC_snd_msc.loc[first_month+1:, 'Month'], MSC_snd_msc.loc[:first_month, 'Month']], ignore_index=True))
                MSC_snd_msc = pd.merge(MSC_snd_msc_new, MSC_snd_msc, on='Month', how='outer')
                ax3.plot(MSC_snd_msc.Month, MSC_snd_msc[variable_to_plot_list[i]], color = color_list[i], linestyle=linestyle_list[i], label = model, linewidth=1.25)
                ax3.fill_between(MSC_snd_msc.Month,
                                MSC_snd_msc[variable_to_plot_list[i]] - MSC_snd_msc[(variable_to_plot_list[i]+'_std')], 
                                MSC_snd_msc[variable_to_plot_list[i]] + MSC_snd_msc[(variable_to_plot_list[i]+'_std')], 
                                color = color_list[i], linestyle=linestyle_list[i], alpha=0.2)
            if i!=len(model_assimilation_list)-1:
                savename = savename + '_' + model_assimilation_list[i] + '_&'
            else:
                if plot_second_MSCs:
                    savename = savename + '_TWO_MSCs_' + model_assimilation_list[i] + '_timeseries_AND_msc_' + region_name + '.png'
                else:
                    savename = savename + '_' + model_assimilation_list[i] + '_timeseries_AND_msc_' + region_name + '.png'

    #fig.suptitle(what_is_plotted+' for '+region_name, y=0.92, fontsize=12)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('CO$_2$ flux [TgC/month/region]', color='black')
    ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax1.minorticks_on()
    ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax1.tick_params(bottom=True, left=True, color='gray')
    ax1.legend(framealpha=0.25, facecolor='grey', loc='lower left', fontsize=9.5, ncol=2, columnspacing=0.6)
    #ax1.set_title('TM5-4DVar & MIP land fluxes '+region_name, pad=-13)
    ax1.set_title('TM5-4DVar land fluxes '+region_name)#, pad=-13)
    if fix_y_scale==True:
        ax1.set_yticks([-400, -200, 0, 200, 400])
        ax1.set_yticklabels([-400, -200, 0, 200, 400])
        ax1.set_ylim([-450, 500])
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
    if fix_y_scale==True:
        ax2.set_ylim([-450, 500])
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
        if fix_y_scale==True:
            ax3.set_ylim([-450, 500])
    
    
    print('Saving plot...')
    if compare_case==1:
        plt.savefig(savepath + savename, dpi=300, bbox_inches='tight')
    print('DONE plotting!')



if __name__=='__main__':
    print('Do not use this file as main file, only to import in different files')


