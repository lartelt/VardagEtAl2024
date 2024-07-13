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
import functions_to_load_datasets as load_datasets

TrendyModels = ['CABLE-POP', 'CLASSIC', 'DLEM', 'IBIS', 'ISAM', 'JSBACH', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'ORCHIDEE-CNP', 'ORCHIDEEv3', 'CLM5.0',
                'ISBA-CTRIP', 'JULES-ES-1p0', 'LPJ', 'SDGVM', 'VISIT', 'YIBs']

# calculations:
def calculateMeanSeasonalCycle_of_fluxes(Flux_per_subregion_monthly_timeseries: pd.DataFrame, model: str='MIP', TRENDY_variable: str=None):
    '''## Documentation
    ### input:
    - Flux_per_subregion_monthly_timeseries: df with the fluxes per subregion and month
    - model: 
        - 'MIP'
        - 'TM5_flux_regional_Sourish'
                - df contains CO2 values in Tg_CO2/region/month -> additional column for flux in TgC/region/month
        - 'TM5_flux_gridded'
        - 'TRENDY'
    - TRENDY_variable: str, e.g. 'nbp' ; only needed if model=='TRENDY'
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
    elif model=='TRENDY' or model.startswith('TRE') or model in TrendyModels:
        print('start calculating MSC for TRENDY fluxes')
        Flux_msc = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[[TRENDY_variable]].mean().reset_index()
        Flux_msc_std = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[[TRENDY_variable]].std(ddof=0).reset_index()
        Flux_msc_count = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[[TRENDY_variable]].count().reset_index()
        Flux_msc = Flux_msc.merge(Flux_msc_std, on='Month', suffixes=('', '_std'))
        Flux_msc = Flux_msc.merge(Flux_msc_count, on='Month', suffixes=('', '_count'))
        return Flux_msc


# plotting functions
# NOT FINAL YET!
def plotFluxTimeseries_and_MSC_general(start_month: int = 5, start_year: int=2009, end_year: int=2019,
                               gdf_list: list=[],
                               variable_to_plot_list: list=['CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'Landtot', 'nbptot_TgC_month', 'mean'],
                               model_assimilation_list: list=['TM5-4DVar_IS+ACOS', 'MIP_IS+OCO2_ens', 'TRENDY_ensemble_mean', 'TRENDY_CLASSIC'],
                               what_is_plotted: str='NEE+fire_flux',
                               color_list: list=['dimgrey', 'forestgreen', 'royalblue', 'firebrick'],
                               linestyle_list: list=['dashed', 'solid'],
                               region_name: str='SAT', savepath: str='/home/lartelt/MA/software/', compare_case: int=1):
    '''# Documentation
    Function to plot timeseries of the mean seasonal cycle of CO2 fluxes modelled by TRENDY, TM5, MIP & next to it the mean seasonal cycle
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
        print('Case 1 timeseries')
        for i,model in enumerate(model_assimilation_list):
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
            if i!=len(model_assimilation_list)-1:
                savename = savename + '_' + model_assimilation_list[i] + '_&'
            else:
                savename = savename + '_' + model_assimilation_list[i] + '_timeseries_AND_msc_' + region_name + '.png'

    fig.suptitle(what_is_plotted+' for '+region_name, y=0.92, fontsize=12)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('CO$_2$ flux [TgC/month/region]', color='black')
    ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax1.minorticks_on()
    ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax1.tick_params(bottom=True, left=True, color='gray')
    ax1.legend(framealpha=0.25, facecolor='grey', loc='lower left', fontsize=9.5, ncol=2, columnspacing=0.6)
    #ax1.set_title('TM5-4DVar land fluxes '+region_name+' regional Sourish & gridded '+grid_res, pad=-13)
    #MSC axis
    ax2.set_xlabel('Month')
    ax2.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax2.set_xticks([5,10,15])
    ax2.set_xticklabels([5, 10, 3]) # set the x-axis tick labels
    ax2.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax2.tick_params(bottom=True, color='gray')
    ax2.set_title('MSC '+str(start_year)+'-'+str(end_year), pad=-13)
    
    print('Saving plot...')
    if compare_case==1:
        plt.savefig(savepath + savename, dpi=300, bbox_inches='tight')
    print('DONE plotting!')


def plot_timeseries_of_TRENDY_fluxes(df: pd.DataFrame,
                                     df_TM5: pd.DataFrame=None,
                                     start_month: int=5,
                                     what_is_plotted='nbp',
                                     color_list: list=['dimgrey', 'forestgreen', 'royalblue', 'firebrick'],
                                     region_name='SAT',
                                     savepath='/home/lartelt/MA/software/TRENDY/',
                                     compare_case=1):
    '''# Documentation
    # arguments:
        - df: ONLY use the df containing the ensemble mean, std & count of all models
    '''
    print('Start plotting...')
    fig, ax1 = plt.subplots(1,1, figsize=(8,5))
    first_month = start_month-2
    for i, column in enumerate(df.columns):
        if column=='MonthDate' or column=='Year' or column=='Month' or column=='std' or column=='count':
            print('column = '+column)
            i=i-1
            continue
        else:
            ax1.plot(df.MonthDate, df[column], color = color_list[i], linestyle='dashed', label=(column[17:]), linewidth=1.25, zorder=i+1)
            print(i)
    ax1.plot(df.MonthDate, df['mean'], color = 'black', linestyle='dashed', label='mean TRENDY', linewidth=1.5, zorder=len(df.columns)+2)
    ax1.fill_between(df.MonthDate, df['mean']-df['std'], df['mean']+df['std'],
                     color = 'dimgrey', linestyle='solid', alpha=0.5, linewidth=0, label='std TRENDY', zorder=len(df.columns)+1)
    #ax1.plot(df_TM5.MonthDate, df_TM5['CO2_NEE_flux_monthly_TgC_per_subregion'], color = 'red', linestyle='solid', label='TM5-4DVar/IS+ACOS', linewidth=2, zorder=len(df.columns)+3)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('CO$_2$ flux [TgC/month/region]', color='black')
    ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax1.minorticks_on()
    ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax1.tick_params(bottom=True, left=True, color='gray')
    #ax1.legend(framealpha=0.25, facecolor='grey', loc='upper left', fontsize=9.5, ncol=6, columnspacing=0.6, bbox_to_anchor=(-0.13, 1.3))
    ax1.legend(framealpha=0.25, facecolor='grey', loc='upper left', fontsize=9.5, ncol=6, columnspacing=0.6, bbox_to_anchor=(-0.13, 1.3))
    ax1.set_title('TRENDY '+what_is_plotted+' fluxes & mean '+region_name)#, pad=-13)
    
    #plt.savefig(savepath+'timeseries_all_TRENDY_models_with_mean_WITH_TM5_IS_ACOS_new', dpi=300, bbox_inches='tight')
    plt.savefig(savepath+'timeseries_all_TRENDY_models_with_mean_NEW', dpi=300, bbox_inches='tight')
    print('Done saving figure!')


