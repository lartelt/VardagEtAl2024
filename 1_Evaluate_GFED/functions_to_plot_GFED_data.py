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
from matplotlib import ticker as mticker
from matplotlib import cm
import imageio
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
#sns.set_theme(style="darkgrid")
sns.set(rc={'axes.facecolor':'gainsboro'})


# calculations:
'''def calculateMeanSeasonalCycle_of_fluxes(Flux_per_subregion_monthly_timeseries: pd.DataFrame, model: str='MIP'):
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
    elif model=='GFED':
        print('start calculating MSC for GFED fluxes')
        Flux_msc = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['total_emission']].mean().reset_index()
        Flux_msc_std = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['total_emission']].std(ddof=0).reset_index()
        Flux_msc_count = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['total_emission']].count().reset_index()
        Flux_msc = Flux_msc.merge(Flux_msc_std, on='Month', suffixes=('', '_std'))
        Flux_msc = Flux_msc.merge(Flux_msc_count, on='Month', suffixes=('', '_count'))
        return Flux_msc
'''

def calculateMeanSeasonalCycle_of_fluxes(Flux_per_subregion_monthly_timeseries: pd.DataFrame, model: str='MIP', TRENDY_FLUXCOM_variable: str=None):
    '''## Documentation
    ### input:
    - Flux_per_subregion_monthly_timeseries: df with the fluxes per subregion and month
    - model: 
        - 'MIP'
        - 'TM5_flux_regional_Sourish'
                - df contains CO2 values in Tg_CO2/region/month -> additional column for flux in TgC/region/month
        - 'TM5_flux_gridded'
        - 'TRENDY'
        - 'FLUXCOM'
    - TRENDY_FLUXCOM_variable: str, 
        - for TRENDY: e.g. 'nbp' ; only needed if model=='TRENDY'
        - for FLUXCOM: 'NEE', 'GPP', 'TER'
    ### output:
    - Flux_msc: df with the mean seasonal cycle of the fluxes per subregion
    '''
    if model=='MIP' or model=='MIP_ens':
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
    elif model=='TRENDY' or model.startswith('TRE'):# or model in TrendyModels:
        print('start calculating MSC for TRENDY fluxes')
        Flux_msc = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[[TRENDY_FLUXCOM_variable]].mean().reset_index()
        Flux_msc_std = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[[TRENDY_FLUXCOM_variable]].std(ddof=0).reset_index()
        Flux_msc_count = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[[TRENDY_FLUXCOM_variable]].count().reset_index()
        Flux_msc = Flux_msc.merge(Flux_msc_std, on='Month', suffixes=('', '_std'))
        Flux_msc = Flux_msc.merge(Flux_msc_count, on='Month', suffixes=('', '_count'))
        return Flux_msc
    elif model=='FLUXCOM' or model.startswith('FLU'):
        print('start calculating MSC for FLUXCOM fluxes')
        if TRENDY_FLUXCOM_variable.endswith('tot'):
            Flux_msc = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[[TRENDY_FLUXCOM_variable]].mean().reset_index()
            Flux_msc_std = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[[TRENDY_FLUXCOM_variable]].std(ddof=0).reset_index()
            Flux_msc_count = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[[TRENDY_FLUXCOM_variable]].count().reset_index()
            Flux_msc = Flux_msc.merge(Flux_msc_std, on='Month', suffixes=('', '_std'))
            Flux_msc = Flux_msc.merge(Flux_msc_count, on='Month', suffixes=('', '_count'))
        else:
            Flux_msc = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[[TRENDY_FLUXCOM_variable+'tot']].mean().reset_index()
            Flux_msc_std = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[[TRENDY_FLUXCOM_variable+'tot']].std(ddof=0).reset_index()
            Flux_msc_count = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[[TRENDY_FLUXCOM_variable+'tot']].count().reset_index()
            Flux_msc = Flux_msc.merge(Flux_msc_std, on='Month', suffixes=('', '_std'))
            Flux_msc = Flux_msc.merge(Flux_msc_count, on='Month', suffixes=('', '_count'))
        return Flux_msc
    elif model=='GFED':
        print('start calculating MSC for GFED fluxes')
        Flux_msc = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['total_emission']].mean().reset_index()
        Flux_msc_std = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['total_emission']].std(ddof=0).reset_index()
        Flux_msc_count = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['total_emission']].count().reset_index()
        Flux_msc = Flux_msc.merge(Flux_msc_std, on='Month', suffixes=('', '_std'))
        Flux_msc = Flux_msc.merge(Flux_msc_count, on='Month', suffixes=('', '_count'))
        return Flux_msc



# plotting functions
def plotFluxTimeseries_and_MSC_general(start_month: int = 5, start_year: int=2009, end_year: int=2019, fix_y_scale: bool=False,
                               gdf_list: list=[],
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
            - Case 1: one timeseries
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
            elif model=='GFED':
                print('calc MSC GFED')
                MSC = calculateMeanSeasonalCycle_of_fluxes(gdf_list[i], model=model_assimilation_list[i])
            elif model=='TRENDY' or model=='FLUXCOM':
                print('calc MSC TRENDY or TRENDY')
                MSC = calculateMeanSeasonalCycle_of_fluxes(gdf_list[i], model=model_assimilation_list[i], TRENDY_FLUXCOM_variable=variable_to_plot_list[i])
            #elif model=='FLUXCOM':
            #    print('calc MSC FLUXCOM')
            #    MSC = calculateMeanSeasonalCycle_of_fluxes(gdf_list[i], model=model_assimilation_list[i], TRENDY_FLUXCOM_variable=variable_to_plot_list[i])
            else:
                print('calc MSC '+model[:3])
                MSC = calculateMeanSeasonalCycle_of_fluxes(gdf_list[i], model=model_assimilation_list[i][:3])
            MSC.loc[:first_month,'Month'] += 12
            MSC_new = pd.DataFrame(pd.concat([MSC.loc[first_month+1:, 'Month'], MSC.loc[:first_month, 'Month']], ignore_index=True))
            MSC = pd.merge(MSC_new, MSC, on='Month', how='outer')
            ax2.plot(MSC.Month, MSC[variable_to_plot_list[i]], color = color_list[i], linestyle=linestyle_list[i], label = model, linewidth=1.25)
            ax2.fill_between(MSC.Month,
                             MSC[variable_to_plot_list[i]] - MSC[(variable_to_plot_list[i]+'_std')], 
                             MSC[variable_to_plot_list[i]] + MSC[(variable_to_plot_list[i]+'_std')], 
                             color = color_list[i], linestyle=linestyle_list[i], alpha=0.2)
            #if model=='TRENDY':
            #    add_to_savename = 'TRENDY_'+TRENDY_model_category
            #else:
            add_to_savename = model_assimilation_list[i]
            if i!=len(model_assimilation_list)-1:
                savename = savename + '_' + add_to_savename + '_&'
            else:
                savename = savename + '_' + add_to_savename + '_timeseries_AND_msc_' + region_name + '.png'

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
    ax1.set_title(what_is_plotted.replace('_', ' ')+' '+region_name, pad=-13)
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
    ax2.set_title('MSC '+str(start_year)+'-'+str(end_year), pad=-13)
    if fix_y_scale==True:
        ax2.set_ylim([-450, 500])
    
    
    print('Saving plot...')
    if compare_case==1:
        plt.savefig(savepath + savename, dpi=300, bbox_inches='tight')
    print('DONE plotting!')


# plot maps
def plotMapForMonthDate_GFED(gdf_GFED: geopandas.GeoDataFrame, Year_or_Month_plotted: int=1, savepath: str='/home/lartelt/MA/software/'): 
    '''Documentation
    Function to plot the GFED fire emissions for a specific month or year in SA
    # arguments:
        - gdf_GFED: geopandas gdf with the GFED fire emissions per_gridcell
        - Year_or_Month_plotted: int, year or month that should be plotted. Month=[1,12], Year=[2009,2021]
    '''   
    
    #sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
    
    if Year_or_Month_plotted>12:
        what_is_plotted = 'Year'
        print('grouping GFED by Year...')
        df_GFED_new = gdf_GFED.groupby(['Year', 'Lat', 'Long'])['total_emission', 'emission'].sum().reset_index()
        #gdf_GFED_new = geopandas.GeoDataFrame(df_GFED_new, geometry=geopandas.points_from_xy(df_GFED_new.Long, df_GFED_new.Lat), crs="EPSG:4326")
    else:
        what_is_plotted = 'Month'
        print('grouping GFED by Month...')
        df_GFED_new = gdf_GFED.groupby(['Month', 'Lat', 'Long'])['total_emission', 'emission'].sum().reset_index()
        #gdf_GFED_new = geopandas.GeoDataFrame(df_GFED_new, geometry=geopandas.points_from_xy(df_GFED_new.Long, df_GFED_new.Lat), crs="EPSG:4326")
    savename='GFED_fire_emission_for_'+what_is_plotted+'_'+str(Year_or_Month_plotted)
    
    sdf = df_GFED_new[(df_GFED_new[what_is_plotted]==Year_or_Month_plotted)]
    sdf = sdf.set_index(['Lat', 'Long'])  # set two index: First index = Lat ; Second index = Long
    xsdf = xr.Dataset.from_dataframe(sdf)  # convert pandas df to an xarray dataset
    foo = xr.DataArray(xsdf['total_emission'][:])  # creates a DataArray with the xsdf data: ndarrays with values & labeled dimensions & coordinates
    #print(foo)
    
    plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree()) # cartopy ccrs.PlateCarree() == normal 2D world map
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4, color='white', alpha=0.5, linestyle='-') # creates gridlines of continents
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False
    # for South-America:
    gl.xlocator = mticker.FixedLocator([-30,-40,-50,-60,-70,-80,-90])
    gl.ylocator = mticker.FixedLocator([20,10,0,-10,-20,-30,-40,-50,-60])
    
    #cmap = cm.get_cmap('hot_r', 256)
    cmap = cm.hot_r
    norm = mpl.colors.Normalize(vmin=0, vmax=foo.max())
    foo.plot.contourf(x = 'Long', y = 'Lat', extend = 'max', ax=ax, transform = ccrs.PlateCarree(), levels=256, cmap=cmap, vmin=0, vmax=1,
                      cbar_kwargs=dict(orientation='vertical', norm=norm, shrink=1, pad=0.03, 
                                       ticks=np.arange(0,1.2,0.2),
                                       label='CO$_2$ Emissions [TgC/'+what_is_plotted+'/gridcell]'))
    #cmap.ax.minorticks_off
    ax.coastlines()

    # define region map to plot as contour in background
    CT_Mask = pd.read_pickle('/home/eschoema/masks/CTTranscomMask1x1Borders.pkl')
    SAT=CT_Mask[(CT_Mask.transcom == 'SAT')].to_crs('EPSG:4326')
    SAT.boundary.plot(ax=ax, color='black', linewidth=0.5)
    
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.tick_params(bottom=True, left=True, axis='both', color='slategray')
    ax.set_title('GFED fire emissions for '+what_is_plotted+' '+str(Year_or_Month_plotted), fontsize=18)
    print('done plotting, now saving...')
    plt.savefig(savepath+savename+'.png', dpi=300, bbox_inches='tight')
    
    
# plot GIF of maps
def create_GIF_from_images(image_list: list, savepath_images: str, monthly_or_yearly: str='monthly') -> None:
    '''Creates a GIF from a list of images
    Parameters
    ----------
    image_list : list
        List of images
    savepath_images : str
        Path to images
    monthly_or_yearly : str, optional
        Whether the GIF is a 'monthly' or 'yearly' GIF
    '''
    images = []
    for name in image_list:
        images.append(imageio.imread(savepath_images + name))
    imageio.mimsave(savepath_images + 'GIF_GFED_fire_emissions_measurements_'+monthly_or_yearly+'.gif', images, duration=1000, loop=0)




