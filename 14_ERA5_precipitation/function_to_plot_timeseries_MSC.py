# Author: Lukas Artelt
import pandas as pd
import xarray as xr
#import cartopy.crs as ccrs
#import cartopy.feature as cf
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


# calculations:
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
        - 'ERA5'
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
            TRENDY_FLUXCOM_variable=TRENDY_FLUXCOM_variable[:-3]
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
    elif model=='ERA5':
        print('start calculating MSC for ERA5 precipitation')
        Flux_msc = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['tp']].mean().reset_index()
        Flux_msc_std = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['tp']].std(ddof=0).reset_index()
        Flux_msc_count = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['tp']].count().reset_index()
        Flux_msc = Flux_msc.merge(Flux_msc_std, on='Month', suffixes=('', '_std'))
        Flux_msc = Flux_msc.merge(Flux_msc_count, on='Month', suffixes=('', '_count'))
        return Flux_msc



# NOT FINAL? include FLUXCOM, TRENDY, TM5, MIP nbp fluxes
def plot_timeseries_MSC_of_FLUXCOM_vs_TRENDY_vs_TM5_vs_MIP(df_FLUXCOM: pd.DataFrame,
                                                           df_TRENDY: pd.DataFrame,
                                                           TRENDY_model_category: str='very_good_models',
                                                           plot_TM5_IS_flux: bool=True,
                                                           df_TM5_IS: pd.DataFrame=None,
                                                           plot_TM5_ACOS_flux: bool=True,
                                                           df_TM5_ACOS: pd.DataFrame=None,
                                                           plot_TM5_RT_flux: bool=True,
                                                           df_TM5_RT: pd.DataFrame=None,
                                                           plot_MIP_flux: bool=True,
                                                           df_MIP_ens: pd.DataFrame=None,
                                                           color_list = ['black', 'forestgreen'],
                                                           linestyle_list = ['solid', 'dashed'],
                                                           plot_MSC_only: bool=False,
                                                           start_month: int=5,
                                                           start_year: int=2009,
                                                           end_year: int=2019,
                                                           region_name='SAT',
                                                           savepath='/home/lartelt/MA/software/FLUXCOM/',
                                                           compare_case=1):
    '''# Documentation
    # arguments:
        - df_FLUXCOM: df containing the FLUXCOM fluxes for specific variable
        - df_TRENDY: df containing the MEAN TRENDY fluxes of models from 'TRENDY_model_category' for specific variable
        - df_TM5_IS: df containing the TM5 fluxes assimilating IS
        - df_TM5_ACOS: df containing the TM5 fluxes assimilating IS+ACOS
        - df_TM5_RT: df containing the TM5 fluxes assimilating IS+RT
        - df_MIP_ens: df containing the MIP fluxes of the ensemble mean assimilating IS+OCO2
        - color_list: 0: FLUXCOM, 1: TRENDY, 2: TM5_ACOS, 3: TM5_RT, 4: MIP, 5: TM5/IS
        - compare_case:
            - 1: plot 2009-2019
            - 2: plot from start_year to end_year
    '''
    print('start plotting...')
    if plot_MSC_only==True:
            fig, ax2 = plt.subplots(1,1, figsize=(8,5))
    else:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5), sharey=True, gridspec_kw={'width_ratios':[8,3]})
        fig.subplots_adjust(wspace=0)
    first_month = start_month-2
    if compare_case==1 or compare_case==2:
        if plot_TM5_IS_flux:
            if plot_MSC_only==False:
                ax1.plot(df_TM5_IS.MonthDate, df_TM5_IS.CO2_NEE_fire_flux_monthly_TgC_per_subregion, color = color_list[5], linestyle=linestyle_list[5], label = 'TM5-4DVar/IS', linewidth=1.25)
            MSC_TM5_IS = calculateMeanSeasonalCycle_of_fluxes(df_TM5_IS, model='TM5')
            MSC_TM5_IS.loc[:first_month,'Month'] += 12
            MSC_TM5_IS_new = pd.DataFrame(pd.concat([MSC_TM5_IS.loc[first_month+1:, 'Month'], MSC_TM5_IS.loc[:first_month, 'Month']], ignore_index=True))
            MSC_TM5_IS = pd.merge(MSC_TM5_IS_new, MSC_TM5_IS, on='Month', how='outer')
            ax2.plot(MSC_TM5_IS.Month, MSC_TM5_IS['CO2_NEE_fire_flux_monthly_TgC_per_subregion'], color=color_list[5], linestyle=linestyle_list[5], label = 'TM5-4DVar/IS', linewidth=1.25)
            ax2.fill_between(MSC_TM5_IS.Month,
                            MSC_TM5_IS['CO2_NEE_fire_flux_monthly_TgC_per_subregion'] - MSC_TM5_IS[('CO2_NEE_fire_flux_monthly_TgC_per_subregion'+'_std')], 
                            MSC_TM5_IS['CO2_NEE_fire_flux_monthly_TgC_per_subregion'] + MSC_TM5_IS[('CO2_NEE_fire_flux_monthly_TgC_per_subregion'+'_std')], 
                            color = color_list[5], linestyle=linestyle_list[5], alpha=0.2)
        if plot_TM5_ACOS_flux:
                if plot_MSC_only==False:
                    ax1.plot(df_TM5_ACOS.MonthDate, df_TM5_ACOS['CO2_NEE_fire_flux_monthly_TgC_per_subregion'], color = color_list[2], linestyle=linestyle_list[2], label='TM5-4DVar/IS+ACOS', linewidth=1.25)
                MSC_TM5_ACOS = calculateMeanSeasonalCycle_of_fluxes(df_TM5_ACOS, model='TM5')
                MSC_TM5_ACOS.loc[:first_month,'Month'] += 12
                MSC_TM5_ACOS_new = pd.DataFrame(pd.concat([MSC_TM5_ACOS.loc[first_month+1:, 'Month'], MSC_TM5_ACOS.loc[:first_month, 'Month']], ignore_index=True))
                MSC_TM5_ACOS = pd.merge(MSC_TM5_ACOS_new, MSC_TM5_ACOS, on='Month', how='outer')
                ax2.plot(MSC_TM5_ACOS.Month, MSC_TM5_ACOS['CO2_NEE_fire_flux_monthly_TgC_per_subregion'], color=color_list[2], linestyle=linestyle_list[2], label = 'TM5-4DVar/IS+ACOS', linewidth=1.25)
                ax2.fill_between(MSC_TM5_ACOS.Month,
                                MSC_TM5_ACOS['CO2_NEE_fire_flux_monthly_TgC_per_subregion'] - MSC_TM5_ACOS[('CO2_NEE_fire_flux_monthly_TgC_per_subregion'+'_std')], 
                                MSC_TM5_ACOS['CO2_NEE_fire_flux_monthly_TgC_per_subregion'] + MSC_TM5_ACOS[('CO2_NEE_fire_flux_monthly_TgC_per_subregion'+'_std')], 
                                color = color_list[2], linestyle=linestyle_list[2], alpha=0.2)
        if plot_TM5_RT_flux:
            if plot_MSC_only==False:
                ax1.plot(df_TM5_RT.MonthDate, df_TM5_RT['CO2_NEE_fire_flux_monthly_TgC_per_subregion'], color = color_list[3], linestyle=linestyle_list[3], label='TM5-4DVar/IS+RT', linewidth=1.25)
            MSC_TM5_RT = calculateMeanSeasonalCycle_of_fluxes(df_TM5_RT, model='TM5')
            MSC_TM5_RT.loc[:first_month,'Month'] += 12
            MSC_TM5_RT_new = pd.DataFrame(pd.concat([MSC_TM5_RT.loc[first_month+1:, 'Month'], MSC_TM5_RT.loc[:first_month, 'Month']], ignore_index=True))
            MSC_TM5_RT = pd.merge(MSC_TM5_RT_new, MSC_TM5_RT, on='Month', how='outer')
            ax2.plot(MSC_TM5_RT.Month, MSC_TM5_RT['CO2_NEE_fire_flux_monthly_TgC_per_subregion'], color=color_list[3], linestyle=linestyle_list[3], label = 'TM5-4DVar/IS+RT', linewidth=1.25)
            ax2.fill_between(MSC_TM5_RT.Month,
                            MSC_TM5_RT['CO2_NEE_fire_flux_monthly_TgC_per_subregion'] - MSC_TM5_RT[('CO2_NEE_fire_flux_monthly_TgC_per_subregion'+'_std')], 
                            MSC_TM5_RT['CO2_NEE_fire_flux_monthly_TgC_per_subregion'] + MSC_TM5_RT[('CO2_NEE_fire_flux_monthly_TgC_per_subregion'+'_std')], 
                            color = color_list[3], linestyle=linestyle_list[3], alpha=0.2)
        if plot_MIP_flux:
            if plot_MSC_only==False:
                ax1.plot(df_MIP_ens.MonthDate, df_MIP_ens['Landtot'], color = color_list[4], linestyle=linestyle_list[4], label='MIP_ens/IS+OCO2', linewidth=1.25)
            MSC_MIP = calculateMeanSeasonalCycle_of_fluxes(df_MIP_ens, model='MIP')
            MSC_MIP.loc[:first_month,'Month'] += 12
            MSC_MIP_new = pd.DataFrame(pd.concat([MSC_MIP.loc[first_month+1:, 'Month'], MSC_MIP.loc[:first_month, 'Month']], ignore_index=True))
            MSC_MIP = pd.merge(MSC_MIP_new, MSC_MIP, on='Month', how='outer')
            ax2.plot(MSC_MIP.Month, MSC_MIP['Landtot'], color=color_list[4], linestyle=linestyle_list[4], label = 'TM5-4DVar/IS+RT', linewidth=1.25)
            ax2.fill_between(MSC_MIP.Month,
                            MSC_MIP['Landtot'] - MSC_MIP[('Landtot'+'_std')], 
                            MSC_MIP['Landtot'] + MSC_MIP[('Landtot'+'_std')], 
                            color = color_list[4], linestyle=linestyle_list[4], alpha=0.2)
                
        if plot_MSC_only==False:
            ax1.plot(df_FLUXCOM.MonthDate, df_FLUXCOM['NBPtot'], color = color_list[0], linestyle=linestyle_list[0], label='FLUXCOM', linewidth=1.25)
            ax1.fill_between(df_FLUXCOM.MonthDate,
                             df_FLUXCOM['NBPtot'] - df_FLUXCOM['NBP_madtot'],
                             df_FLUXCOM['NBPtot'] + df_FLUXCOM['NBP_madtot'],
                             color = color_list[0], linestyle=linestyle_list[0], alpha=0.2, linewidth=0.2)
            ax1.plot(df_TRENDY.MonthDate, df_TRENDY['mean'], color = color_list[1], linestyle=linestyle_list[1], label='TRENDY'.replace('_', ' '), linewidth=1.25)
            ax1.fill_between(df_TRENDY.MonthDate, 
                             df_TRENDY['mean']-df_TRENDY['std'], df_TRENDY['mean']+df_TRENDY['std'], 
                             color = color_list[1], linestyle=linestyle_list[1], alpha=0.2, linewidth=0.2)
            
        
        # MSC
        MSC_FLUXCOM = calculateMeanSeasonalCycle_of_fluxes(df_FLUXCOM, model='FLUXCOM', TRENDY_FLUXCOM_variable='NBP')
        MSC_FLUXCOM_std = calculateMeanSeasonalCycle_of_fluxes(df_FLUXCOM, model='FLUXCOM', TRENDY_FLUXCOM_variable='NBP_mad')
        MSC_TRENDY = calculateMeanSeasonalCycle_of_fluxes(df_TRENDY, model='TRENDY', TRENDY_FLUXCOM_variable='mean')
        MSC_TRENDY_std = calculateMeanSeasonalCycle_of_fluxes(df_TRENDY, model='TRENDY', TRENDY_FLUXCOM_variable='std')
        MSC_FLUXCOM.loc[:first_month,'Month'] += 12
        MSC_FLUXCOM_std.loc[:first_month,'Month'] += 12
        MSC_TRENDY.loc[:first_month,'Month'] += 12
        MSC_TRENDY_std.loc[:first_month,'Month'] += 12
        MSC_FLUXCOM_new = pd.DataFrame(pd.concat([MSC_FLUXCOM.loc[first_month+1:, 'Month'], MSC_FLUXCOM.loc[:first_month, 'Month']], ignore_index=True))
        MSC_FLUXCOM_std_new = pd.DataFrame(pd.concat([MSC_FLUXCOM_std.loc[first_month+1:, 'Month'], MSC_FLUXCOM_std.loc[:first_month, 'Month']], ignore_index=True))
        MSC_TRENDY_new = pd.DataFrame(pd.concat([MSC_TRENDY.loc[first_month+1:, 'Month'], MSC_TRENDY.loc[:first_month, 'Month']], ignore_index=True))
        MSC_TRENDY_std_new = pd.DataFrame(pd.concat([MSC_TRENDY_std.loc[first_month+1:, 'Month'], MSC_TRENDY_std.loc[:first_month, 'Month']], ignore_index=True))
        MSC_FLUXCOM = pd.merge(MSC_FLUXCOM_new, MSC_FLUXCOM, on='Month', how='outer')
        MSC_FLUXCOM_std = pd.merge(MSC_FLUXCOM_std_new, MSC_FLUXCOM_std, on='Month', how='outer')
        MSC_TRENDY = pd.merge(MSC_TRENDY_new, MSC_TRENDY, on='Month', how='outer')
        MSC_TRENDY_std = pd.merge(MSC_TRENDY_std_new, MSC_TRENDY_std, on='Month', how='outer')
        ax2.plot(MSC_FLUXCOM.Month, MSC_FLUXCOM['NBPtot'], color=color_list[0], linestyle=linestyle_list[0], label = 'FLUXCOM', linewidth=1.25)
        ax2.plot(MSC_TRENDY.Month, MSC_TRENDY['mean'], color=color_list[1], linestyle=linestyle_list[1], label = 'TRENDY', linewidth=1.25)
        if plot_MSC_only==False:
            ax2.fill_between(MSC_FLUXCOM.Month,
                             MSC_FLUXCOM['NBPtot'] - MSC_FLUXCOM[('NBPtot'+'_std')], MSC_FLUXCOM['NBPtot'] + MSC_FLUXCOM[('NBPtot'+'_std')], 
                             color=color_list[0], linestyle=linestyle_list[0], alpha=0.2)
            ax2.fill_between(MSC_TRENDY.Month,
                             MSC_TRENDY['mean'] - MSC_TRENDY[('mean'+'_std')], MSC_TRENDY['mean'] + MSC_TRENDY[('mean'+'_std')], 
                             color=color_list[1], linestyle=linestyle_list[1], alpha=0.2)
        else:
            ax2.fill_between(MSC_FLUXCOM.Month,
                             MSC_FLUXCOM['NBPtot'] - MSC_FLUXCOM_std[('NBP_madtot')], MSC_FLUXCOM['NBPtot'] + MSC_FLUXCOM_std[('NBP_madtot')], 
                             color=color_list[0], linestyle=linestyle_list[0], alpha=0.2)
            ax2.fill_between(MSC_TRENDY.Month,
                             MSC_TRENDY['mean'] - MSC_TRENDY_std['std'], MSC_TRENDY['mean'] + MSC_TRENDY_std['std'], 
                             color=color_list[1], linestyle=linestyle_list[1], alpha=0.2)
    if plot_MSC_only==False:
        ax1.set_xlabel('Year')
        ax1.set_ylabel('CO$_2$ flux [TgC/month/region]', color='black')
        ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
        ax1.minorticks_on()
        ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
        ax1.tick_params(bottom=True, left=True, color='gray')
        ax1.legend(framealpha=0.25, facecolor='grey', loc='lower left', fontsize=9, ncol=3, columnspacing=0.6)
        ax1.set_title('FLUXCOM & TRENDY '+TRENDY_model_category.replace('_', ' ')+' & TM5 & MIP nbp '+region_name, pad=-13)
        ax2.set_xticks([5,10,15])
        ax2.set_xticklabels([5, 10, 3])
        ax2.set_title('MSC '+str(start_year)+'-'+str(end_year), pad=-13)
        ax2.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    #MSC axis
    ax2.set_xlabel('Month')
    ax2.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax2.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.tick_params(bottom=True, color='gray')
    if plot_MSC_only:
        ax2.set_ylabel('CO$_2$ flux [TgC/month/region]', color='black')
        ax2.set_title('FLUXCOM & TRENDY '+TRENDY_model_category.replace('_', ' ')+' & TM5 & MIP nbp '+region_name+' '+str(start_year)+'-'+str(end_year), pad=-13)
        ax2.legend(framealpha=0.25, facecolor='grey', loc='lower left', fontsize=9.5, ncol=3, columnspacing=0.6)
        ax2.set_xticks([5,6,7,8,9,10,11,12,13,14,15,16])
        ax2.set_xticklabels([5,6,7,8,9,10,11,12,1,2,3,4])
        ax2.grid(visible=True, which='minor', color='white', axis='y', linestyle='-', linewidth='0.4')
    
    if TRENDY_model_category=='vg_models':
        TRENDY_model_category='very_good_models'
    elif TRENDY_model_category=='bla_models':
        TRENDY_model_category='bad_low_ampl_models'
    elif TRENDY_model_category=='bha_models':
        TRENDY_model_category='bad_high_ampl_models'
    if compare_case==1:
        if plot_MSC_only==False:
            plt.savefig(savepath+'timeseries_MSC_FLUXCOM_TRENDY_'+TRENDY_model_category+'_mean_TM5_MIP_nbp_'+region_name+'.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(savepath+'MSC_only_FLUXCOM_TRENDY_'+TRENDY_model_category+'_mean_TM5_MIP_nbp_'+region_name+'.png', dpi=300, bbox_inches='tight')
    elif compare_case==2:
        if plot_MSC_only==False:
            plt.savefig(savepath+'/'+str(start_year)+'-'+str(end_year)+'/timeseries_MSC_FLUXCOM_TRENDY_'+TRENDY_model_category+'_mean_TM5_MIP_nbp_'+region_name+'.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(savepath+'/'+str(start_year)+'-'+str(end_year)+'/MSC_only_FLUXCOM_TRENDY_'+TRENDY_model_category+'_mean_TM5_MIP_nbp_'+region_name+'.png', dpi=300, bbox_inches='tight')



# plot timeseries & MSC GENERAL
def plot_timeseries_MSC_general(df_list: list=[],
                                model_list: list=['FLUXCOM', 'TRENDY'],
                                columns_to_plot: list=['TERtot', 'mean'],
                                columns_to_plot_std: list=['TER_madtot', 'std'],
                                columns_to_plot_label: list=['TER', 'ra+rh'],
                                TRENDY_model_category: list=['very_good_models'],
                                norm_timeseries: bool=False,
                                color_list = ['black', 'forestgreen'],
                                linestyle_list = ['solid', 'dashed'],
                                plot_MSC_only: bool=False,
                                start_month: int=5,
                                start_year: int=2009,
                                end_year: int=2020,
                                region_name='SAT',
                                plot_title='TER and gpp',
                                savepath='/home/lartelt/MA/software/FLUXCOM/',
                                compare_case=1):
    '''# Documentation
    # arguments:
        - df_list = [df_FLUXCOM: df containing the FLUXCOM fluxes for specific variable, df_TRENDY: df containing the MEAN TRENDY fluxes of model category for specific variable]
        - columns_to_plot: columns in df's that should be plotted
    '''
    print('start plotting...')
    if plot_MSC_only==True:
            fig, ax2 = plt.subplots(1,1, figsize=(8,5))
    else:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5), sharey=True, gridspec_kw={'width_ratios':[8,3]})
        fig.subplots_adjust(wspace=0)
    if plot_MSC_only==False:
        if norm_timeseries==True:
            savename='Normed_timeseries_and_MSC_'
        else:
            savename='timeseries_and_MSC_'
    else:
        if norm_timeseries==True:
            savename='Normed_MSC_only_'
        else:
            savename='MSC_only_'
    first_month = start_month-2
    first_trendy=True
    j=0
    if compare_case==1 or compare_case==2:
        for i, df in enumerate(df_list):
            if model_list[i]=='TRENDY' and first_trendy==True and TRENDY_model_category[j]!=None:
                savename=savename+model_list[i]+'_'+TRENDY_model_category[j]+'_'+columns_to_plot[j]+'_&_'
                first_trendy=False
                j=j+1
            else:
                savename=savename+model_list[i]+'_'+columns_to_plot[i]+'_&_'
                
            if plot_MSC_only==False:
                ax1.plot(df.MonthDate, df[columns_to_plot[i]], color = color_list[i], linestyle=linestyle_list[i], label=model_list[i]+' '+columns_to_plot_label[i], linewidth=1.25)
                #include std of the here plotted TRENDY mean
                if columns_to_plot_std[i] in df.columns:
                    print('plot std in timeseries with std column '+columns_to_plot_std[i])
                    ax1.fill_between(df.MonthDate, df[columns_to_plot[i]]-df[columns_to_plot_std[i]], df[columns_to_plot[i]]+df[columns_to_plot_std[i]], 
                                    color = color_list[i], linestyle=linestyle_list[i], alpha=0.2, linewidth=0.3)
            MSC = calculateMeanSeasonalCycle_of_fluxes(df, model=model_list[i], TRENDY_FLUXCOM_variable=columns_to_plot[i])
            MSC.loc[:first_month,'Month'] += 12
            MSC_new = pd.DataFrame(pd.concat([MSC.loc[first_month+1:, 'Month'], MSC.loc[:first_month, 'Month']], ignore_index=True))
            MSC = pd.merge(MSC_new, MSC, on='Month', how='outer')
            if norm_timeseries==True:
                a = MSC[columns_to_plot[i]]-np.mean([np.min(MSC[columns_to_plot[i]]), np.max(MSC[columns_to_plot[i]])])
                ax2.plot(MSC.Month, a/np.max(a), color=color_list[i], linestyle=linestyle_list[i], label=model_list[i]+' '+columns_to_plot_label[i], linewidth=1.25)
            else:
                ax2.plot(MSC.Month, MSC[columns_to_plot[i]], color=color_list[i], linestyle=linestyle_list[i], label=model_list[i]+' '+columns_to_plot_label[i], linewidth=1.25)
            if columns_to_plot_std[i] in df.columns:
                MSC_std = calculateMeanSeasonalCycle_of_fluxes(df, model=model_list[i], TRENDY_FLUXCOM_variable=columns_to_plot_std[i])
                MSC_std.loc[:first_month,'Month'] += 12
                MSC_std_new = pd.DataFrame(pd.concat([MSC_std.loc[first_month+1:, 'Month'], MSC_std.loc[:first_month, 'Month']], ignore_index=True))
                MSC_std = pd.merge(MSC_std_new, MSC_std, on='Month', how='outer')
            if plot_MSC_only==False:
                ax2.fill_between(MSC.Month,
                                    MSC[columns_to_plot[i]] - MSC[(columns_to_plot[i]+'_std')], MSC[columns_to_plot[i]] + MSC[(columns_to_plot[i]+'_std')], 
                                    color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)
            elif plot_MSC_only==True:
                if columns_to_plot_std[i] in df.columns:
                    print('plot std in MSC with std column '+columns_to_plot_std[i])
                    if norm_timeseries==True:
                        a = MSC[columns_to_plot[i]]-np.mean([np.min(MSC[columns_to_plot[i]]), np.max(MSC[columns_to_plot[i]])])
                        ax2.fill_between(MSC.Month,
                                         a/np.max(a) - MSC_std[(columns_to_plot_std[i])]/np.max(a), a/np.max(a) + MSC_std[(columns_to_plot_std[i])]/np.max(a), 
                                         color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)
                    else:
                        ax2.fill_between(MSC.Month,
                                         MSC[columns_to_plot[i]] - MSC_std[columns_to_plot_std[i]], MSC[columns_to_plot[i]] + MSC_std[columns_to_plot_std[i]], 
                                         color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)
                else:
                    if norm_timeseries==True:
                        a = MSC[columns_to_plot[i]]-np.mean([np.min(MSC[columns_to_plot[i]]), np.max(MSC[columns_to_plot[i]])])
                        b = MSC[(columns_to_plot[i]+'_std')]-np.mean([np.min(MSC[(columns_to_plot[i])]), np.max(MSC[(columns_to_plot[i])])])
                        ax2.fill_between(MSC.Month,
                                         a/np.max(a) - b/np.max(a), a/np.max(a) + b/np.max(a), 
                                         color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)
                    else:
                        ax2.fill_between(MSC.Month,
                                        MSC[columns_to_plot[i]] - MSC[(columns_to_plot[i]+'_std')], MSC[columns_to_plot[i]] + MSC[(columns_to_plot[i]+'_std')], 
                                        color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)

    ncol = np.ceil(len(df_list)/2)
    if plot_MSC_only==False:
        ax1.set_xlabel('Year')
        if 'ERA5' in model_list:
            ax1.set_ylabel('Precipitation [m/day]', color='black')
        else:
            ax1.set_ylabel('CO$_2$ flux [TgC/month/region]', color='black')
        ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
        ax1.minorticks_on()
        ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
        ax1.tick_params(bottom=True, left=True, color='gray')
        ax1.legend(framealpha=0.4, facecolor='grey', loc='lower left', fontsize=9.5, ncol=ncol, columnspacing=0.6, handletextpad=0.4)
        ax1.set_title(plot_title+' '+region_name, pad=-13)
        ax2.set_xticks([5,10,15])
        ax2.set_xticklabels([5, 10, 3])
        ax2.set_title('MSC '+str(start_year)+'-'+str(end_year), pad=-13)
        ax2.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    #MSC axis
    ax2.set_xlabel('Month')
    ax2.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax2.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.tick_params(bottom=True, color='gray')
    if plot_MSC_only:
        if 'ERA5' in model_list:
            ax2.set_ylabel('Precipitation [m/day]', color='black')
        else:
            ax2.set_ylabel('CO$_2$ flux [TgC/month/region]', color='black')
        ax2.set_title(plot_title+' '+region_name, pad=-13)
        ax2.legend(framealpha=0.4, facecolor='grey', loc='lower right', fontsize=9.5, ncol=ncol, columnspacing=0.6, handletextpad=0.4)
        ax2.set_xticks([5,6,7,8,9,10,11,12,13,14,15,16])
        ax2.set_xticklabels([5,6,7,8,9,10,11,12,1,2,3,4])
        ax2.grid(visible=True, which='minor', color='white', axis='y', linestyle='-', linewidth='0.4')
    
    if compare_case==1:
        if plot_MSC_only==False:
            plt.savefig(savepath+savename[:-3]+'.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(savepath+savename[:-3]+'.png', dpi=300, bbox_inches='tight')
    elif compare_case==2:
        if plot_MSC_only==False:
            plt.savefig(savepath+'/'+str(start_year)+'-'+str(end_year)+'/'+savename[:-3]+'.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(savepath+'/'+str(start_year)+'-'+str(end_year)+'/'+savename[:-3]+'.png', dpi=300, bbox_inches='tight')
    print('done saving!')



# plot timeseries & MSC GENERAL
def plot_timeseries_general(df_list: list=[],
                            model_list: list=['FLUXCOM', 'TRENDY'],
                            columns_to_plot: list=['TERtot', 'mean'],
                            columns_to_plot_std: list=['TER_madtot', 'std'],
                            columns_to_plot_label: list=['TER', 'ra+rh'],
                            TRENDY_model_category: list=['very_good_models'],
                            color_list = ['black', 'forestgreen'],
                            linestyle_list = ['solid', 'dashed'],
                            start_year: int=2009,
                            end_year: int=2020,
                            region_name='SAT',
                            plot_title='TER and gpp',
                            savepath='/home/lartelt/MA/software/FLUXCOM/',
                            savename_input:str=None,
                            compare_case=1):
    '''# Documentation
    # arguments:
        - df_list = [df_FLUXCOM: df containing the FLUXCOM fluxes for specific variable, df_TRENDY: df containing the MEAN TRENDY fluxes of model category for specific variable]
        - columns_to_plot: columns in df's that should be plotted
    '''
    print('start plotting...')
    fig, ax1 = plt.subplots(1,1, figsize=(8,5))
    first_trendy=True
    if savename_input is None:
        savename='timeseries_'
    j=0
    if compare_case==1 or compare_case==2:
        for i, df in enumerate(df_list):
            if model_list[i]=='TRENDY' and first_trendy==True and TRENDY_model_category[j]!=None and savename_input is None:
                savename=savename+model_list[i]+'_'+TRENDY_model_category[j]+'_'+columns_to_plot[j]+'_&_'
                first_trendy=False
                j=j+1
            elif savename_input is None:
                savename=savename+model_list[i]+'_'+columns_to_plot[i]+'_&_'
            
            ax1.plot(df.MonthDate, df[columns_to_plot[i]], color = color_list[i], linestyle=linestyle_list[i], label=model_list[i]+' '+columns_to_plot_label[i], linewidth=1.25)
            #include std of the here plotted TRENDY mean
            if columns_to_plot_std[i] in df.columns:
                print('plot std in timeseries with std column '+columns_to_plot_std[i])
                ax1.fill_between(df.MonthDate, df[columns_to_plot[i]]-df[columns_to_plot_std[i]], df[columns_to_plot[i]]+df[columns_to_plot_std[i]], 
                                color = color_list[i], linestyle=linestyle_list[i], alpha=0.2, linewidth=0.3)
    
    ncol = 3
    ax1.set_xlabel('Year')
    if 'ERA5' in model_list:
        ax1.set_ylabel('Precipitation [m/day]', color='black')
    else:
        ax1.set_ylabel('CO$_2$ flux [TgC/month/region]', color='black')
    ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax1.minorticks_on()
    ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax1.tick_params(bottom=True, left=True, color='gray')
    ax1.legend(framealpha=0.4, facecolor='grey', loc='lower left', fontsize=9.5, ncol=ncol, columnspacing=0.6, handletextpad=0.4)
    ax1.set_title(plot_title+' '+region_name, pad=-13)

    if compare_case==1:
        if savename_input is None:
            plt.savefig(savepath+savename[:-3]+'.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(savepath+savename_input+'.png', dpi=300, bbox_inches='tight')
    elif compare_case==2:
        if savename_input is None:
            plt.savefig(savepath+'/'+str(start_year)+'-'+str(end_year)+'/'+savename[:-3]+'.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(savepath+'/'+str(start_year)+'-'+str(end_year)+'/'+savename_input+'.png', dpi=300, bbox_inches='tight')
    print('done saving!')



# plot bar-chart of precipitation in different regions
def plotSumOfFluxesBarChart(df_ERA5: list=[],
                            variable_plotted: str='tp_mean',
                            start_year: int=2009,
                            end_year: int=2020,
                            color_list: list=[],
                            label_list: list=[],
                            region_name: str='SAT', 
                            title: str='ERA5 mean precipitation',
                            savename: str='',
                            savepath: str=''):
    '''#Documentation
    Function to plot a bar chart with bars representing the monthly precipitation in different regions
    # arguments
        compare_case:
            1: only plot ERA5 precip
    '''
    
    fig, ax = plt.subplots(figsize=(12,6))
    bar_width = 20
    opacity = 1
    year_list = np.arange(start_year, end_year+1, 1)
    j=0

    for i, column in enumerate(df_ERA5):
        a = ax.bar(column['MonthDate'], (column[variable_plotted[i]])*1000, bar_width, alpha=opacity, color=color_list[i], label=label_list[i])
        #bottom+=
        j=j+1

    ax.minorticks_on()
    #ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    #ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax.tick_params(bottom=True, left=True, color='gray')
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean monthly precipitation [mm]', color='black')
    ax.set_title(title+ ' ' + region_name)
    ax.legend(framealpha=0.5, fontsize=9.5, loc='upper right', ncol=3, columnspacing=0.6)
    plt.savefig(savepath + savename + '_' + region_name + '.png', dpi=300, bbox_inches='tight')
    print('Done saving plot!')
