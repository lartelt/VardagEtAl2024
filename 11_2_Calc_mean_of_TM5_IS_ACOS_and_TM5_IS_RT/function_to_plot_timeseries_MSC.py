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

TrendyModels = ['CABLE-POP', 'CLASSIC', 'DLEM', 'IBIS', 'ISAM', 'JSBACH', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'ORCHIDEE-CNP', 'ORCHIDEEv3', 'CLM5.0',
                'ISBA-CTRIP', 'JULES-ES-1p0', 'LPJ', 'SDGVM', 'VISIT', 'YIBs']

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
    elif model=='TM5_MINUS_GFED':
        print('Calculating MSC for TM5 fluxes MINUS GFED')
        Flux_per_subregion_monthly_timeseries['Month'] = Flux_per_subregion_monthly_timeseries['MonthDate'].apply(lambda x: x.month)
        Flux_msc = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['CO2_NEE_flux_monthly_TgC_per_subregion', 
                                                                             'CO2_fire_flux_monthly_TgC_per_subregion', 
                                                                             'CO2_ocean_flux_monthly_TgC_per_subregion',
                                                                             'CO2_fossil_flux_monthly_TgC_per_subregion',
                                                                             'CO2_NEE_fire_flux_monthly_TgC_per_subregion',
                                                                             'CO2_NEE_fire_MINUS_GFED_flux_monthly_TgC_per_subregion']].mean()
        Flux_msc_std = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['CO2_NEE_flux_monthly_TgC_per_subregion', 
                                                                                 'CO2_fire_flux_monthly_TgC_per_subregion', 
                                                                                 'CO2_ocean_flux_monthly_TgC_per_subregion',
                                                                                 'CO2_fossil_flux_monthly_TgC_per_subregion',
                                                                                 'CO2_NEE_fire_flux_monthly_TgC_per_subregion',
                                                                                 'CO2_NEE_fire_MINUS_GFED_flux_monthly_TgC_per_subregion']].std(ddof=0).reset_index()
        Flux_msc_count = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['CO2_NEE_flux_monthly_TgC_per_subregion', 
                                                                                   'CO2_fire_flux_monthly_TgC_per_subregion', 
                                                                                   'CO2_ocean_flux_monthly_TgC_per_subregion',
                                                                                   'CO2_fossil_flux_monthly_TgC_per_subregion',
                                                                                   'CO2_NEE_fire_flux_monthly_TgC_per_subregion',
                                                                                   'CO2_NEE_fire_MINUS_GFED_flux_monthly_TgC_per_subregion']].count().reset_index()
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
    elif model=='FLUXCOM' or model.startswith('FLU') or model=='FXC':
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


# plotting functions

def plot_timeseries_of_TRENDY_fluxes(df: pd.DataFrame,
                                     what_is_plotted_values: str=None,
                                     df_TM5_ACOS: pd.DataFrame=None,
                                     df_TM5_RT: pd.DataFrame=None,
                                     df_MIP_ens: pd.DataFrame=None,
                                     start_month: int=5,
                                     what_is_plotted='nbp',
                                     color_list: list=['dimgrey', 'forestgreen', 'royalblue', 'firebrick'],
                                     region_name='SAT',
                                     savepath='/home/lartelt/MA/software/TRENDY/',
                                     compare_case=1):
    '''# Documentation
    # arguments:
        - df: ONLY use the df containing the ensemble mean, std & count of all models
              df can also contain the TRENDY mean & std for each model
        - compare_case:
            1: Only plot TRENDY timeseries
            2: plot TRENDY & TM5/IS+ACOS
            3: TRENDY mean w/ std & TM5/IS+ACOS & TM5/IS+RT
    '''
    print('Start plotting...')
    fig, ax1 = plt.subplots(1,1, figsize=(8,5))
    first_month = start_month-2
    if compare_case==1:
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
    elif compare_case==2:
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
        ax1.plot(df_TM5_ACOS.MonthDate, df_TM5_ACOS['CO2_NEE_flux_monthly_TgC_per_subregion'], color = 'red', linestyle='solid', label='TM5-4DVar/IS+ACOS', linewidth=2, zorder=len(df.columns)+3)
    elif compare_case==3:
        ax1.plot(df.MonthDate, df[what_is_plotted_values], color = 'black', linestyle='solid', label='TRENDY '+(what_is_plotted_values[17:])+' '+(what_is_plotted_values[:3]), linewidth=1.25)
        #ax1.fill_between(df.MonthDate, df['mean']-df['std'], df['mean']+df['std'], color = 'dimgrey', linestyle='dashed', alpha=0.4, linewidth=0)
        ax1.plot(df_TM5_ACOS.MonthDate, df_TM5_ACOS['CO2_NEE_flux_monthly_TgC_per_subregion'], color = 'firebrick', linestyle='solid', label='TM5-4DVar/IS+ACOS', linewidth=1.25)
        ax1.plot(df_TM5_RT.MonthDate, df_TM5_RT['CO2_NEE_flux_monthly_TgC_per_subregion'], color = 'coral', linestyle='solid', label='TM5-4DVar/IS+RT', linewidth=1.25)
        ax1.plot(df_MIP_ens.MonthDate, df_MIP_ens['Landtot'], color = 'royalblue', linestyle='solid', label='MIP_ens/IS+OCO2', linewidth=1.25)

    ax1.set_xlabel('Year')
    ax1.set_ylabel('CO$_2$ flux [TgC/month/region]', color='black')
    ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax1.minorticks_on()
    ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax1.tick_params(bottom=True, left=True, color='gray')
    #ax1.legend(framealpha=0.25, facecolor='grey', loc='upper left', fontsize=9.5, ncol=6, columnspacing=0.6, bbox_to_anchor=(-0.13, 1.3))
    if compare_case==1 or compare_case==2:
        ax1.legend(framealpha=0.25, facecolor='grey', loc='upper left', fontsize=9.5, ncol=6, columnspacing=0.6, bbox_to_anchor=(-0.13, 1.3))
        ax1.set_title('TRENDY '+what_is_plotted+' fluxes & mean '+region_name)
        plt.savefig(savepath+'timeseries_all_TRENDY_models_with_mean_NEW', dpi=300, bbox_inches='tight')
        #plt.savefig(savepath+'timeseries_all_TRENDY_models_with_mean_WITH_TM5_IS_ACOS_new', dpi=300, bbox_inches='tight')
    elif compare_case==3:
        ax1.legend(framealpha=0.25, facecolor='grey', loc='lower left', fontsize=9.5, ncol=2, columnspacing=0.6)
        ax1.set_title('TRENDY model '+(what_is_plotted_values[17:])+' '+what_is_plotted_values[:3]+' & TM5 & MIP flux '+region_name, pad=-13)
        #plt.savefig(savepath+'timeseries_TRENDY_'+(what_is_plotted_values)+'_'+what_is_plotted+'_with_TM5_ACOS_and_TM5_RT_and_MIP_ens_NEW', dpi=300, bbox_inches='tight')
        plt.savefig(savepath+'timeseries_TRENDY_'+(what_is_plotted_values)+'_with_TM5_ACOS_and_TM5_RT_and_MIP_ens.png', dpi=300, bbox_inches='tight')
    print('Done saving figure!')
    

def plot_timeseries_MSC_of_TRENDY_vs_TM5_MIP_nbp_fluxes(df: pd.DataFrame,
                                             column_to_plot: str=None,
                                             df_TM5_ACOS: pd.DataFrame=None,
                                             df_TM5_RT: pd.DataFrame=None,
                                             df_MIP_ens: pd.DataFrame=None,
                                             start_month: int=5,
                                             start_year: int=2009,
                                             end_year: int=2020,
                                             region_name='SAT',
                                             savepath='/home/lartelt/MA/software/TRENDY/',
                                             compare_case=1):
    '''# Documentation
    # arguments:
        - df: ONLY use the df containing the ensemble mean, std & count of all models
        - trendy_model_name: 'CLASSIC', 'CABLE-POP', ...
        - compare_case:
            1: TRENDY mean w/ std & TM5/IS+ACOS & TM5/IS+RT
            2: plot timeseries & MSC from start_year - end_year
    '''
    print('Start plotting...')
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5), sharey=True, gridspec_kw={'width_ratios':[8,3]})
    fig.subplots_adjust(wspace=0)
    first_month = start_month-2
    if compare_case==1 or compare_case==2:
        ax1.plot(df.MonthDate, df[column_to_plot], color = 'black', linestyle='solid', label='TRENDY '+(column_to_plot[17:])+' '+(column_to_plot[:3]), linewidth=1.25)
        #ax1.fill_between(df.MonthDate, df['mean']-df['std'], df['mean']+df['std'], color = 'dimgrey', linestyle='dashed', alpha=0.4, linewidth=0)
        ax1.plot(df_TM5_ACOS.MonthDate, df_TM5_ACOS['CO2_NEE_fire_flux_monthly_TgC_per_subregion'], color = 'firebrick', linestyle='solid', label='TM5-4DVar/IS+ACOS', linewidth=1.25)
        ax1.plot(df_TM5_RT.MonthDate, df_TM5_RT['CO2_NEE_fire_flux_monthly_TgC_per_subregion'], color = 'coral', linestyle='solid', label='TM5-4DVar/IS+RT', linewidth=1.25)
        if df_MIP_ens is not None:
            ax1.plot(df_MIP_ens.MonthDate, df_MIP_ens['Landtot'], color = 'royalblue', linestyle='solid', label='MIP_ens/IS+OCO2', linewidth=1.25)
            MSC_names = ['MSC_TM5_ACOS', 'MSC_TM5_RT', 'MSC_MIP_ens', 'MSC_TRENDY']
            MSC_list = [df_TM5_ACOS, df_TM5_RT, df_MIP_ens, df]
        else: 
            MSC_names = ['MSC_TM5_ACOS', 'MSC_TM5_RT', 'MSC_TRENDY']
            MSC_list = [df_TM5_ACOS, df_TM5_RT, df]
        for i, msc in enumerate(MSC_names):
            print(msc[4:7])
            if msc=='MSC_TRENDY':
                MSC = calculateMeanSeasonalCycle_of_fluxes(MSC_list[i], model=msc[4:7], TRENDY_variable=column_to_plot)
            else:
                MSC = calculateMeanSeasonalCycle_of_fluxes(MSC_list[i], model=msc[4:7])
            MSC.loc[:first_month,'Month'] += 12
            MSC_new = pd.DataFrame(pd.concat([MSC.loc[first_month+1:, 'Month'], MSC.loc[:first_month, 'Month']], ignore_index=True))
            MSC = pd.merge(MSC_new, MSC, on='Month', how='outer')
            if msc=='MSC_TRENDY':
                ax2.plot(MSC.Month, MSC[column_to_plot], color = 'black', linestyle='solid', label = msc, linewidth=1.25)
                ax2.fill_between(MSC.Month,
                                 MSC[column_to_plot] - MSC[(column_to_plot+'_std')], MSC[column_to_plot] + MSC[(column_to_plot+'_std')], 
                                 color = 'black', linestyle='solid', alpha=0.2)
            elif msc=='MSC_TM5_ACOS' or msc=='MSC_TM5_RT':
                if msc=='MSC_TM5_ACOS':
                    color_msc='firebrick'
                elif msc=='MSC_TM5_RT':
                    color_msc='coral'
                ax2.plot(MSC.Month, MSC['CO2_NEE_fire_flux_monthly_TgC_per_subregion'], color = color_msc, linestyle='solid', label = msc, linewidth=1.25)
                ax2.fill_between(MSC.Month,
                                 MSC['CO2_NEE_fire_flux_monthly_TgC_per_subregion'] - MSC[('CO2_NEE_fire_flux_monthly_TgC_per_subregion'+'_std')], 
                                 MSC['CO2_NEE_fire_flux_monthly_TgC_per_subregion'] + MSC[('CO2_NEE_fire_flux_monthly_TgC_per_subregion'+'_std')], 
                                 color = color_msc, linestyle='solid', alpha=0.2)
            elif msc=='MSC_MIP_ens':
                ax2.plot(MSC.Month, MSC['Landtot'], color = 'royalblue', linestyle='solid', label = msc, linewidth=1.25)
                ax2.fill_between(MSC.Month,
                                 MSC['Landtot'] - MSC[('Landtot'+'_std')], MSC['Landtot'] + MSC[('Landtot'+'_std')], 
                                 color = 'royalblue', linestyle='solid', alpha=0.2)
        
    ax1.set_xlabel('Year')
    ax1.set_ylabel('CO$_2$ flux [TgC/month/region]', color='black')
    ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax1.minorticks_on()
    ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    if column_to_plot[:3]=='nbp':
        ax1.set_yticks([-300, -200, -100, 0, 100, 200, 300])
        ax1.set_yticklabels([-300, -200, -100, 0, 100, 200, 300])
        ax1.set_ylim([-350, 350])
    ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax1.tick_params(bottom=True, left=True, color='gray')
    ax1.legend(framealpha=0.25, facecolor='grey', loc='lower left', fontsize=9.5, ncol=2, columnspacing=0.6)
    ax1.set_title('TRENDY model '+(column_to_plot[17:])+' '+column_to_plot[:3]+' flux '+region_name, pad=-13)
    #ax1.legend(framealpha=0.25, facecolor='grey', loc='upper left', fontsize=9.5, ncol=6, columnspacing=0.6, bbox_to_anchor=(-0.13, 1.3))
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
    if column_to_plot[:3]=='nbp':
        ax2.set_ylim([-350, 350])
    
    if compare_case==1:
        plt.savefig(savepath+'timeseries_MSC_TRENDY_'+(column_to_plot)+'_with_TM5_ACOS_and_TM5_RT_and_MIP_ens.png', dpi=300, bbox_inches='tight')
    if compare_case==2:
        plt.savefig(savepath+str(start_year)+'-'+str(end_year)+'/timeseries_MSC_'+str(start_year)+'-'+str(end_year)+'_TRENDY_'+(column_to_plot)+'_with_TM5_ACOS_and_TM5_RT_and_MIP_ens.png', dpi=300, bbox_inches='tight')
    print('Done saving figure!')


# MAIN: plot only some TRENDY modelled fluxes for specific variable
def plot_timeseries_MSC_of_TRENDY_variable_fluxes(df_TRENDY: pd.DataFrame,
                                             models_to_plot: list=None,
                                             variable_plotted: str=None,
                                             plot_TM5_ACOS: bool=False,
                                             df_TM5_ACOS: pd.DataFrame=None,
                                             df_TM5_RT: pd.DataFrame=None,
                                             plot_GFED: bool=False,
                                             df_GFED: pd.DataFrame=None,
                                             color_dict: dict=None,
                                             start_month: int=5,
                                             start_year: int=2009,
                                             end_year: int=2019,
                                             region_name='SAT',
                                             savepath='/home/lartelt/MA/software/TRENDY/',
                                             compare_case=1):
    '''# Documentation
    # arguments:
        - df: ONLY use the df containing the ensemble mean, std & count of all models
        - models_to_plot: ['CLASSIC', 'CABLE-POP', ...]
        - compare_case:
            1: TRENDY modelled Variable w/ std & TM5/IS+ACOS
            2: plot timeseries & MSC from start_year - end_year
    '''
    print('Start plotting...')
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5), sharey=True, gridspec_kw={'width_ratios':[8,3]})
    fig.subplots_adjust(wspace=0)
    first_month = start_month-2
    if plot_TM5_ACOS:
        savename = 'TRENDY_and_TM5_ACOS_RT_'+variable_plotted+'_'
    elif plot_GFED:
        savename = 'TRENDY_and_GFED_'+variable_plotted+'_'
    else:
        savename = 'TRENDY_'+variable_plotted+'_'
    if variable_plotted.endswith('fFire'):
        longer_var_name = 2
    elif variable_plotted.endswith('rh') or variable_plotted.endswith('ra'):
        longer_var_name = -1
    elif variable_plotted.endswith('ra+rh+gpp') or variable_plotted.endswith('ra+rh-gpp'):
        longer_var_name = 12
    elif variable_plotted.endswith('nbp-(ra+rh)+gpp'):
        longer_var_name = 21
    else:
        longer_var_name = 0
    if compare_case==1 or compare_case==2:
        # plot TM5/IS+ACOS flux or not
        if plot_TM5_ACOS:
            ax1.plot(df_TM5_ACOS.MonthDate, df_TM5_ACOS['CO2_NEE_fire_flux_monthly_TgC_per_subregion'], color = 'black', linestyle='dashed', label='TM5-4DVar/IS+ACOS', linewidth=1.25)
            MSC_TM5 = calculateMeanSeasonalCycle_of_fluxes(df_TM5_ACOS, model='TM5')
            MSC_TM5.loc[:first_month,'Month'] += 12
            MSC_TM5_new = pd.DataFrame(pd.concat([MSC_TM5.loc[first_month+1:, 'Month'], MSC_TM5.loc[:first_month, 'Month']], ignore_index=True))
            MSC_TM5 = pd.merge(MSC_TM5_new, MSC_TM5, on='Month', how='outer')
            ax2.plot(MSC_TM5.Month, MSC_TM5['CO2_NEE_fire_flux_monthly_TgC_per_subregion'], color = 'black', linestyle='dashed', label = 'TM5-4DVar/IS+ACOS', linewidth=1.25)
            ax2.fill_between(MSC_TM5.Month,
                             MSC_TM5['CO2_NEE_fire_flux_monthly_TgC_per_subregion'] - MSC_TM5[('CO2_NEE_fire_flux_monthly_TgC_per_subregion'+'_std')], 
                             MSC_TM5['CO2_NEE_fire_flux_monthly_TgC_per_subregion'] + MSC_TM5[('CO2_NEE_fire_flux_monthly_TgC_per_subregion'+'_std')], 
                             color = 'black', linestyle='dashed', alpha=0.2)
            #RT
            ax1.plot(df_TM5_RT.MonthDate, df_TM5_RT['CO2_NEE_fire_flux_monthly_TgC_per_subregion'], color = 'dimgrey', linestyle='dashed', label='TM5-4DVar/IS+RT', linewidth=1.25)
            MSC_TM5_RT = calculateMeanSeasonalCycle_of_fluxes(df_TM5_RT, model='TM5')
            MSC_TM5_RT.loc[:first_month,'Month'] += 12
            MSC_TM5_RT_new = pd.DataFrame(pd.concat([MSC_TM5_RT.loc[first_month+1:, 'Month'], MSC_TM5_RT.loc[:first_month, 'Month']], ignore_index=True))
            MSC_TM5_RT = pd.merge(MSC_TM5_RT_new, MSC_TM5_RT, on='Month', how='outer')
            ax2.plot(MSC_TM5_RT.Month, MSC_TM5_RT['CO2_NEE_fire_flux_monthly_TgC_per_subregion'], color = 'dimgrey', linestyle='dashed', label = 'TM5-4DVar/IS+RT', linewidth=1.25)
            ax2.fill_between(MSC_TM5_RT.Month,
                             MSC_TM5_RT['CO2_NEE_fire_flux_monthly_TgC_per_subregion'] - MSC_TM5_RT[('CO2_NEE_fire_flux_monthly_TgC_per_subregion'+'_std')], 
                             MSC_TM5_RT['CO2_NEE_fire_flux_monthly_TgC_per_subregion'] + MSC_TM5_RT[('CO2_NEE_fire_flux_monthly_TgC_per_subregion'+'_std')], 
                             color = 'dimgrey', linestyle='dashed', alpha=0.2)
        elif plot_GFED:
            ax1.plot(df_GFED.MonthDate, df_GFED['total_emission'], color = 'black', linestyle='solid', label='GFED fire', linewidth=1.25)
            MSC_GFED = calculateMeanSeasonalCycle_of_fluxes(df_GFED, model='GFED')
            MSC_GFED.loc[:first_month,'Month'] += 12
            MSC_GFED_new = pd.DataFrame(pd.concat([MSC_GFED.loc[first_month+1:, 'Month'], MSC_GFED.loc[:first_month, 'Month']], ignore_index=True))
            MSC_GFED = pd.merge(MSC_GFED_new, MSC_GFED, on='Month', how='outer')
            ax2.plot(MSC_GFED.Month, MSC_GFED['total_emission'], color = 'black', linestyle='solid', label = 'GFED fire', linewidth=1.25)
            ax2.fill_between(MSC_GFED.Month,
                             MSC_GFED['total_emission'] - MSC_GFED[('total_emission'+'_std')], 
                             MSC_GFED['total_emission'] + MSC_GFED[('total_emission'+'_std')], 
                             color = 'black', linestyle='solid', alpha=0.2)
        
        for i, column in enumerate(df_TRENDY.columns):
            if column[17+longer_var_name:] not in models_to_plot or column in ['MonthDate', 'Year', 'Month', 'mean', 'std', 'count']:
                print('do not use column = '+column)
                #print(len(column))
                continue
            else:
                print('use column = '+column)
                df_TRENDY_modified = df_TRENDY[['MonthDate', 'Month', column]]
                #if variable_plotted.endswith('gpp'):
                #    df_TRENDY_modified[column] = df_TRENDY_modified[column]*-1
                ax1.plot(df_TRENDY_modified.MonthDate, df_TRENDY_modified[column], color=color_dict[column[17+longer_var_name:]], linestyle='solid', label=(column[17+longer_var_name:]), linewidth=1.25)
                
                MSC = calculateMeanSeasonalCycle_of_fluxes(df_TRENDY_modified, model='TRENDY', TRENDY_variable=column)
                MSC.loc[:first_month,'Month'] += 12
                MSC_new = pd.DataFrame(pd.concat([MSC.loc[first_month+1:, 'Month'], MSC.loc[:first_month, 'Month']], ignore_index=True))
                MSC = pd.merge(MSC_new, MSC, on='Month', how='outer')
                ax2.plot(MSC.Month, MSC[column], color=color_dict[column[17+longer_var_name:]], linestyle='solid', linewidth=1.25)
                ax2.fill_between(MSC.Month,
                                 MSC[column] - MSC[(column+'_std')], MSC[column] + MSC[(column+'_std')], 
                                 color=color_dict[column[17+longer_var_name:]], linestyle='solid', alpha=0.2)
                savename=savename+column[17+longer_var_name:]+'_'
            del df_TRENDY_modified
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('CO$_2$ flux [TgC/month/region]', color='black')
    ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax1.minorticks_on()
    ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    #ax1.set_yticks([-300, -200, -100, 0, 100, 200, 300])
    #ax1.set_yticklabels([-300, -200, -100, 0, 100, 200, 300])
    #ax1.set_ylim([-350, 350])
    ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax1.tick_params(bottom=True, left=True, color='gray')
    ax1.legend(framealpha=0.25, facecolor='grey', loc='lower left', fontsize=9.5, ncol=3, columnspacing=0.6)
    ax1.set_title('TRENDY '+variable_plotted+' flux '+region_name, pad=-13)#pad=-10)
    #ax1.legend(framealpha=0.25, facecolor='grey', loc='upper left', fontsize=9.5, ncol=6, columnspacing=0.6, bbox_to_anchor=(-0.13, 1.3))
    #MSC axis
    ax2.set_xlabel('Month')
    ax2.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax2.set_xticks([5,10,15])
    ax2.set_xticklabels([5, 10, 3])
    ax2.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax2.tick_params(bottom=True, color='gray')
    ax2.set_title('MSC '+str(start_year)+'-'+str(end_year), pad=-13)#pad=-10)
    #ax2.set_ylim([-350, 350])
    
    if compare_case==1:
        plt.savefig(savepath+savename+'timeseries_&_MSC.png', dpi=300, bbox_inches='tight')
    elif compare_case==2:
        plt.savefig(savepath+str(start_year)+'-'+str(end_year)+'/'+savename+'timeseries_&_MSC.png', dpi=300, bbox_inches='tight')
    print('Done saving figure!')


# plot only some TRENDY modelled fluxes for SUM of two variables
def plot_timeseries_MSC_of_TRENDY_sum_of_two_variables_fluxes(df_TRENDY_var1: pd.DataFrame,
                                                              df_TRENDY_var2: pd.DataFrame,
                                                              models_to_plot: list=None,
                                                              model_catagory: str='bad_high_amplitude',
                                                              var1: str=None,
                                                              var2: str=None,
                                                              color_dict: dict=None,
                                                              start_month: int=5,
                                                              start_year: int=2009,
                                                              end_year: int=2019,
                                                              region_name='SAT',
                                                              savepath='/home/lartelt/MA/software/TRENDY/',
                                                              compare_case=1):
    '''# Documentation
    # arguments:
        - df: ONLY use the df containing the ensemble mean, std & count of all models
        - models_to_plot: ['CLASSIC', 'CABLE-POP', ...]
        - compare_case:
            1: TRENDY modelled Variable w/ std & TM5/IS+ACOS
    '''
    print('Start plotting...')
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5), sharey=True, gridspec_kw={'width_ratios':[8,3]})
    fig.subplots_adjust(wspace=0)
    first_month = start_month-2
    savename = 'TRENDY_'+model_catagory+'_'+var1+'_'+var2+'_'
    variable_length = len(var1+'tot+'+var2+'tot_TgC_month_')
    if compare_case==1 or compare_case==2:
        df_TRENDY = pd.merge(df_TRENDY_var1, df_TRENDY_var2, on=['MonthDate'], how='outer')
        df_TRENDY_sum = df_TRENDY[['MonthDate']]
        df_TRENDY_sum['Month'] = df_TRENDY_sum.apply(lambda x: x['MonthDate'].month, axis=1)
        for model in models_to_plot:
            df_TRENDY_sum[var1+'tot+'+var2+'tot_TgC_month_'+model] = df_TRENDY[var1+'tot_TgC_month_'+model] + df_TRENDY[var2+'tot_TgC_month_'+model]
        print(df_TRENDY_sum.head())
        for i, column in enumerate(df_TRENDY_sum.columns):
            if column in ['MonthDate', 'Year', 'Month', 'mean', 'std', 'count']:
                print('column = '+column)
                continue
            else:
                df_TRENDY_modified = df_TRENDY_sum[['MonthDate', 'Month', column]]
                if var1=='gpp' or var2=='gpp':
                    df_TRENDY_modified[column] = df_TRENDY_modified[column]*-1
                ax1.plot(df_TRENDY_modified.MonthDate, df_TRENDY_modified[column], color=color_dict[column[variable_length:]], linestyle='solid', label=(column[variable_length:]), linewidth=1.25)
                
                MSC = calculateMeanSeasonalCycle_of_fluxes(df_TRENDY_modified, model='TRENDY', TRENDY_variable=column)
                MSC.loc[:first_month,'Month'] += 12
                MSC_new = pd.DataFrame(pd.concat([MSC.loc[first_month+1:, 'Month'], MSC.loc[:first_month, 'Month']], ignore_index=True))
                MSC = pd.merge(MSC_new, MSC, on='Month', how='outer')
                ax2.plot(MSC.Month, MSC[column], color=color_dict[column[variable_length:]], linestyle='solid', linewidth=1.25)
                ax2.fill_between(MSC.Month,
                                 MSC[column] - MSC[(column+'_std')], MSC[column] + MSC[(column+'_std')], 
                                 color=color_dict[column[variable_length:]], linestyle='solid', alpha=0.2)
                savename=savename+column[variable_length:]+'_'
            del df_TRENDY_modified
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('CO$_2$ flux [TgC/month/region]', color='black')
    ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax1.minorticks_on()
    ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    #ax1.set_yticks([-300, -200, -100, 0, 100, 200, 300])
    #ax1.set_yticklabels([-300, -200, -100, 0, 100, 200, 300])
    #ax1.set_ylim([-350, 350])
    ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax1.tick_params(bottom=True, left=True, color='gray')
    ax1.legend(framealpha=0.25, facecolor='grey', loc='lower left', fontsize=9.5, ncol=3, columnspacing=0.6)
    ax1.set_title('TRENDY '+model_catagory+' '+var1+'+'+var2+' flux '+region_name, pad=-10)#pad=-13)
    #ax1.legend(framealpha=0.25, facecolor='grey', loc='upper left', fontsize=9.5, ncol=6, columnspacing=0.6, bbox_to_anchor=(-0.13, 1.3))
    #MSC axis
    ax2.set_xlabel('Month')
    ax2.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax2.set_xticks([5,10,15])
    ax2.set_xticklabels([5, 10, 3])
    ax2.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    ax2.tick_params(bottom=True, color='gray')
    ax2.set_title('MSC '+str(start_year)+'-'+str(end_year), pad=-10)#pad=-13)
    #ax2.set_ylim([-350, 350])
    
    if compare_case==1:
        plt.savefig(savepath+savename+'timeseries_&_MSC.png', dpi=300, bbox_inches='tight')
    elif compare_case==2:
        plt.savefig(savepath+str(start_year)+'-'+str(end_year)+'/'+savename+'timeseries_&_MSC.png', dpi=300, bbox_inches='tight')
    print('Done saving figure!')


# plot bar chart sum of fluxes
def plotSumOfFluxesBarChart(df_trendy_fluxes: list=[],
                            df_TM5_ACOS_sum_fluxes: pd.DataFrame=None,
                            models_to_plot: list=None,
                            model_catagory: str='bad_high_amplitude',
                            variable_plotted: str='nbp',
                            color_dict: list=[],
                            hatch_list: list=None,
                            compare_case: int=1, region_name: str='SAT', savepath: str=''):
    '''#Documentation
    Function to plot a bar chart with bars representing the total flux in a year
    # arguments
        compare_case:
            1: only plot TRENDY fluxes
            2: plot TRENDY fluxes & TM5/ACOS NEE+fire fluxes
        hatch_list: list of what hatches should be plotted
    '''
    fig, ax = plt.subplots()
    bar_width = 1/(len(models_to_plot)+2)
    opacity = 1
    variable_length=len(variable_plotted)
    savename = 'TRENDY_bar-chart_'+model_catagory+'_'+variable_plotted
    year_list = np.arange(2009, 2020, 1)
    j=0
    if compare_case==1:
        print('Case 1')
        for i, column in enumerate(df_trendy_fluxes.columns):
            if column[14+variable_length:] not in models_to_plot or column in ['MonthDate', 'Year', 'Month', 'mean', 'std', 'count']:
                print('column = '+column)
                continue
            else:
                print(j)
                #length_models_plotted=length_models_plotted+1
                df_trendy_fluxes_sum = df_trendy_fluxes.groupby(['Year'])[column].sum().reset_index()
                #b = ax.bar(df_yearly_sum['Year'] + (i)*bar_width, (df_yearly_sum[column]), bar_width, alpha=opacity, color=color_dict[column[14+variable_length:]], hatch=hatch_list[i], label=column[14+variable_length:])
                b = ax.bar(df_trendy_fluxes_sum['Year'] + (j)*bar_width, (df_trendy_fluxes_sum[column]), bar_width, alpha=opacity, color=color_dict[column[14+variable_length:]], label=column[14+variable_length:])
                savename = savename + '_' + column[14+variable_length:]
                j=j+1
            del df_trendy_fluxes_sum
    if compare_case==2:
        print('Case 2')
        a = ax.bar(df_TM5_ACOS_sum_fluxes['year'] + (j)*bar_width, (df_TM5_ACOS_sum_fluxes['CO2_NEE_fire_flux_monthly_TgC_per_subregion']), bar_width, alpha=opacity, color='black', hatch='///', label='TM5/IS+ACOS')
        savename = savename + '_TM5_IS_ACOS_NEE_fire'
        j=j+1
        for i, column in enumerate(df_trendy_fluxes.columns):
            if column[14+variable_length:] not in models_to_plot or column in ['MonthDate', 'Year', 'Month', 'mean', 'std', 'count']:
                print('column = '+column)
                continue
            else:
                print(j)
                #length_models_plotted=length_models_plotted+1
                df_trendy_fluxes_sum = df_trendy_fluxes.groupby(['Year'])[column].sum().reset_index()
                #b = ax.bar(df_yearly_sum['Year'] + (i)*bar_width, (df_yearly_sum[column]), bar_width, alpha=opacity, color=color_dict[column[14+variable_length:]], hatch=hatch_list[i], label=column[14+variable_length:])
                b = ax.bar(df_trendy_fluxes_sum['Year'] + (j)*bar_width, (df_trendy_fluxes_sum[column]), bar_width, alpha=opacity, color=color_dict[column[14+variable_length:]], label=column[14+variable_length:])
                savename = savename + '_' + column[14+variable_length:]
                j=j+1
            del df_trendy_fluxes_sum
    print(j)
    ax.set_xticks(year_list + ((j-1)/2)*bar_width)
    ax.set_xticklabels(year_list)
    ax.set_xlabel('Year')
    ax.set_ylabel('Carbon flux [TgC/region/year]', color='black')
    ax.set_title('TRENDY '+model_catagory+' yearly sum of '+variable_plotted+' fluxes ' + region_name)
    ax.set_axisbelow(True)
    ax.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth='0.9')
    ax.minorticks_on()
    #ax.grid(which='minor', color='grey', axis='x', linestyle='-', linewidth='0.1')
    ax.grid(which='major', color='grey', axis='x', linestyle='-', linewidth='0.75', alpha=1)
    ax.legend(framealpha=0.5, fontsize=9.5, loc='upper right', ncol=3, columnspacing=0.6)
    plt.savefig(savepath + savename + '_' + region_name + '.png', dpi=300, bbox_inches='tight')
    #plt.savefig('/home/lartelt/MA/software/12_TRENDY_flux/plot_bar-charts/test.png', dpi=300, bbox_inches='tight')
    print('Done saving plot!')


# plot MSC only before & after 2015 in one plot
def plot_MSC_of_two_different_TRENDY_models_but_same_flux(df_TRENDY_1: pd.DataFrame,
                                                          df_TRENDY_2: pd.DataFrame,
                                                          column_to_plot: str='nbptot',
                                                          list_what_input: list=['2009-2014', '2015-2019'],
                                                          start_month: int=5,
                                                          region_name='SAT',
                                                          savepath='/home/lartelt/MA/software/TRENDY/',
                                                          compare_case=1):
    '''Documentation
    # arguments:
        df_TRENDY_1: dataframe containing TRENDY fluxes from 2009-2014
        df_TRENDY_2: dataframe containing TRENDY fluxes from 2015-2019
        compare_case:
            1: plot MSC of both TRENDY models in one plot    
    '''
    fig, ax2 = plt.subplots(1,1, figsize=(8,5))
    first_month = start_month-2
    if column_to_plot.startswith('fFire'):
        longer_var_name = 2
    elif column_to_plot.startswith('rh') or column_to_plot.startswith('ra'):
        longer_var_name = -1
    else:
        longer_var_name = 0
        
    if compare_case==1:
        savename='TRENDY_model_'+column_to_plot+'_MSC_'+list_what_input[0]+'_'+list_what_input[1]
        MSC1 = calculateMeanSeasonalCycle_of_fluxes(df_TRENDY_1, model='TRENDY', TRENDY_variable=column_to_plot)
        MSC2 = calculateMeanSeasonalCycle_of_fluxes(df_TRENDY_2, model='TRENDY', TRENDY_variable=column_to_plot)
        MSC1.loc[:first_month,'Month'] += 12
        MSC2.loc[:first_month,'Month'] += 12
        MSC1_new = pd.DataFrame(pd.concat([MSC1.loc[first_month+1:, 'Month'], MSC1.loc[:first_month, 'Month']], ignore_index=True))
        MSC2_new = pd.DataFrame(pd.concat([MSC2.loc[first_month+1:, 'Month'], MSC2.loc[:first_month, 'Month']], ignore_index=True))
        MSC1 = pd.merge(MSC1_new, MSC1, on='Month', how='outer')
        MSC2 = pd.merge(MSC2_new, MSC2, on='Month', how='outer')
        
        ax2.plot(MSC1.Month, MSC1[column_to_plot], color = 'forestgreen', linestyle='solid', label = list_what_input[0], linewidth=1.25)
        ax2.plot(MSC2.Month, MSC2[column_to_plot], color = 'midnightblue', linestyle='solid', label = list_what_input[1], linewidth=1.25)
        ax2.fill_between(MSC1.Month,
                        MSC1[column_to_plot] - MSC1[(column_to_plot+'_std')], MSC1[column_to_plot] + MSC1[(column_to_plot+'_std')], 
                        color = 'forestgreen', linestyle='solid', alpha=0.2)
        ax2.fill_between(MSC2.Month,
                        MSC2[column_to_plot] - MSC2[(column_to_plot+'_std')], MSC2[column_to_plot] + MSC2[(column_to_plot+'_std')], 
                        color = 'midnightblue', linestyle='solid', alpha=0.2)
    #MSC axis
    ax2.set_xlabel('Month')
    ax2.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax2.set_xticks([5,6,7,8,9,10,11,12,13,14,15,16])
    ax2.set_xticklabels([5,6,7,8,9,10,11,12,1,2,3,4]) # set the x-axis tick labels
    ax2.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.grid(visible=True, which='minor', color='white', axis='y', linestyle='-', linewidth='0.4')
    ax2.tick_params(bottom=True, color='gray')
    ax2.legend(framealpha=0.25, facecolor='grey', loc='lower left', fontsize=9.5, ncol=1)
    ax2.set_title('TRENDY model '+(column_to_plot[17+longer_var_name:])+' '+column_to_plot[:3+longer_var_name]+' '+region_name, pad=-13)
    if compare_case==1:
        plt.savefig(savepath+savename+'.png', dpi=300, bbox_inches='tight')
    print('Done saving figure!')
        
        

# MAIN: plot MEAN of different variables in one plot MSC only
def plot_MSC_of_mean_of_different_variables(TRENDY_list: list=[],
                                            model_list_for_mean: list=['CABLE-POP', 'CLASSIC', 'DLEM'],
                                            model_catagory: str='bad_high_amplitude',
                                            variables_to_plot_list: list=['gpp', 'ra', 'rh', 'ra+rh'],
                                            error_type: str='std_of_models',
                                            set_ax_limit: bool=True,
                                            color_list: list=['forestgreen', 'midnightblue', 'darkorange'],
                                            linestyle_list: list=['solid', 'solid', 'dashed'],
                                            start_month: int=5,
                                            region_name: str='SAT',
                                            compare_case: int=1,
                                            savepath: str='/home/lartelt/MA/software/12_TRENDY_flux/plot_MSC_only/'):
    '''Documentation
    # arguments:
        TRENDY_list: list of TRENDY dataframes, each entry is a df containing all trendy models for one variable in monthly mean values
        model_list_for_mean: list of TRENDY models for which the mean of the variable should be calculated
        variables_to_plot_list: list of variables to calculate the mean & to plot. 
                                The order of the variables in the list has to be the same order of the trendy dataframes in the TRENDY_list
        model_catagory = 'good_models', 'bad_low_amplitude', 'bad_high_amplitude'                            
        error_type = 'std_of_models' --> better, 'std_of_year_variations'
    '''
    
    print('Start plotting...')
    fig, ax2 = plt.subplots(1,1, figsize=(8,5))
    first_month = start_month-2
    savename = 'TRENDY_mean_of_models_'
    savepart = ''
    for i, trendy in enumerate(TRENDY_list):
        df_TRENDY_modified = trendy[['MonthDate', 'Month']]
        column_name_list = []
        if i!=len(TRENDY_list)-1:
            savepart = savepart+variables_to_plot_list[i]+'_'
        else:
            savepart = savepart+variables_to_plot_list[i]
        # calculate mean of specific columns only
        for model in model_list_for_mean:
            if variables_to_plot_list[i] == 'ra+rh':
                variable = 'ratot+rh'
            else:
                variable = variables_to_plot_list[i]
            column_name_list.append(variable + 'tot_TgC_month_' + model)
            if i==0:
                savename = savename+model+'_'
        df_TRENDY_modified['mean'] = trendy[column_name_list].mean(axis=1)
        df_TRENDY_modified['std'] = trendy[column_name_list].std(ddof=0, axis=1)
        if compare_case==1:
            MSC_mean = calculateMeanSeasonalCycle_of_fluxes(df_TRENDY_modified, model='TRENDY', TRENDY_variable='mean')
            MSC_std = calculateMeanSeasonalCycle_of_fluxes(df_TRENDY_modified, model='TRENDY', TRENDY_variable='std')
            #print(MSC_std)
            MSC_mean.loc[:first_month,'Month'] += 12
            MSC_std.loc[:first_month,'Month'] += 12
            MSC_mean_new = pd.DataFrame(pd.concat([MSC_mean.loc[first_month+1:, 'Month'], MSC_mean.loc[:first_month, 'Month']], ignore_index=True))
            MSC_std_new = pd.DataFrame(pd.concat([MSC_std.loc[first_month+1:, 'Month'], MSC_std.loc[:first_month, 'Month']], ignore_index=True))
            MSC_mean = pd.merge(MSC_mean_new, MSC_mean, on='Month', how='outer')
            MSC_std = pd.merge(MSC_std_new, MSC_std, on='Month', how='outer')
            ax2.plot(MSC_mean.Month, MSC_mean['mean'], color=color_list[i], linestyle=linestyle_list[i], label=variables_to_plot_list[i], linewidth=1.25)
            # plot std of the mean of the MSC/Yearly differences
            if error_type=='std_of_year_variations':
                ax2.fill_between(MSC_mean.Month,
                                 MSC_mean['mean'] - MSC_mean[('mean'+'_std')], MSC_mean['mean'] + MSC_mean[('mean'+'_std')], 
                                 color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)
            # plot std of the models
            elif error_type=='std_of_models':
                ax2.fill_between(MSC_mean.Month,
                                 MSC_mean['mean'] - MSC_std['std'], MSC_mean['mean'] + MSC_std['std'], 
                                 color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)
        del df_TRENDY_modified
    #MSC axis
    ax2.set_xlabel('Month')
    ax2.set_ylabel('CO$_2$ flux [TgC/month/region]', color='black')
    ax2.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax2.set_xticks([5,6,7,8,9,10,11,12,13,14,15,16])
    ax2.set_xticklabels([5,6,7,8,9,10,11,12,1,2,3,4]) # set the x-axis tick labels
    ax2.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.grid(visible=True, which='minor', color='white', axis='y', linestyle='-', linewidth='0.4')
    ax2.tick_params(bottom=True, color='gray')
    ax2.legend(framealpha=0.25, facecolor='grey', loc='lower left', fontsize=9.5, ncol=2, columnspacing=0.6)
    #ax2.set_title('TRENDY mean of '+(savename[22:len(savename)-1]).replace('_', ' & ')+' '+region_name, pad=-13)
    if set_ax_limit:
        ax2.set_yticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800])
        ax2.set_yticklabels([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800])
        ax2.set_ylim([-50, 1850])
    if error_type=='std_of_year_variations':
        ax2.set_title('TRENDY mean of '+(model_catagory).replace('_', ' ')+' '+region_name+' - std over years', pad=-13)
    elif error_type=='std_of_models':
        ax2.set_title('TRENDY mean of '+(model_catagory).replace('_', ' ')+' '+region_name+' - std over models', pad=-13)
    if compare_case==1:
        if error_type=='std_of_year_variations':
            plt.savefig(savepath+'Variables_'+savepart+'/'+savename+savepart+'.png', dpi=300, bbox_inches='tight')
        elif error_type=='std_of_models':
            plt.savefig(savepath+'Variables_'+savepart+'/'+'TRENDY_mean_of_models_'+model_catagory+'_'+savepart+'_std_of_models.png', dpi=300, bbox_inches='tight')
    print('Done saving figure!')


# plot timeseries & MSC for the MEAN of specific TRENDY models & plot all of them in one plot
def plot_timeseries_MSC_of_mean_of_different_model_categories(df_TRENDY: pd.DataFrame,
                                                  good_model_list: list=['CLASSIC'],
                                                  bad_high_ampl_model_list: list=['SDGVM'],
                                                  bad_low_ampl_model_list: list=['LPJ'],
                                                  variable_to_plot: str='nbp',
                                                  plot_MSC_only: bool=False,
                                                  plot_TM5_ACOS_flux: bool=False,
                                                  df_TM5_ACOS: pd.DataFrame=None,
                                                  set_ax_limit: bool=False,
                                                  error_type: str='std_of_models',
                                                  color_list: list=['forestgreen', 'midnightblue', 'darkorange'],
                                                  linestyle_list: list=['solid', 'solid', 'solid'],
                                                  start_month: int=5,
                                                  start_year: int=2009,
                                                  end_year: int=2019,
                                                  region_name='SAT',
                                                  savepath='/home/lartelt/MA/software/TRENDY/',
                                                  compare_case=1):
    '''# Documentation
    # arguments:
        - df: ONLY use the df containing the ensemble mean, std & count of all models
        - good_model_list: ['CLASSIC', 'ORCHIDEE', ...]
        - error_type: if plot_MSC_only:
                          'std_of_models' --> to plot the std based on the deviations of the models from which the mean was taken, 
                      elif also timeseries plotted:
                          'std_of_year_variations' --> to plot the std based on the deviations of the mean of the yearly variations
        - compare_case:
            1: X
    '''
    print('Start plotting...')
    if plot_MSC_only:
        fig, ax2 = plt.subplots(1,1, figsize=(8,5))
    else:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5), sharey=True, gridspec_kw={'width_ratios':[8,3]})
        fig.subplots_adjust(wspace=0)
    first_month = start_month-2
    
    column_name_list_good = []
    column_name_list_bad_low = []
    column_name_list_bad_high = []
    df_TRENDY_modified_good = df_TRENDY[['MonthDate', 'Month']]
    df_TRENDY_modified_bad_low = df_TRENDY[['MonthDate', 'Month']]
    df_TRENDY_modified_bad_high = df_TRENDY[['MonthDate', 'Month']]
    
    if variable_to_plot == 'ra+rh':
        variable = 'ratot+rh'
    else:
        variable = variable_to_plot
    for model in good_model_list:
        column_name_list_good.append(variable + 'tot_TgC_month_' + model)
    for model in bad_low_ampl_model_list:
        column_name_list_bad_low.append(variable + 'tot_TgC_month_' + model)
    for model in bad_high_ampl_model_list:
        column_name_list_bad_high.append(variable + 'tot_TgC_month_' + model)
    
    df_TRENDY_modified_good['mean'] = df_TRENDY[column_name_list_good].mean(axis=1)
    df_TRENDY_modified_good['std'] = df_TRENDY[column_name_list_good].std(ddof=0, axis=1)
    df_TRENDY_modified_bad_low['mean'] = df_TRENDY[column_name_list_bad_low].mean(axis=1)
    df_TRENDY_modified_bad_low['std'] = df_TRENDY[column_name_list_bad_low].std(ddof=0, axis=1)
    df_TRENDY_modified_bad_high['mean'] = df_TRENDY[column_name_list_bad_high].mean(axis=1)
    df_TRENDY_modified_bad_high['std'] = df_TRENDY[column_name_list_bad_high].std(ddof=0, axis=1)

    error_type='std_of_models'
    if compare_case==1:
        if plot_TM5_ACOS_flux:
            if plot_MSC_only==False:
                ax1.plot(df_TM5_ACOS.MonthDate, df_TM5_ACOS['CO2_NEE_fire_flux_monthly_TgC_per_subregion'], color = 'black', linestyle='dashed', label='TM5-4DVar/IS+ACOS', linewidth=1.25)
            MSC_TM5 = calculateMeanSeasonalCycle_of_fluxes(df_TM5_ACOS, model='TM5')
            MSC_TM5.loc[:first_month,'Month'] += 12
            MSC_TM5_new = pd.DataFrame(pd.concat([MSC_TM5.loc[first_month+1:, 'Month'], MSC_TM5.loc[:first_month, 'Month']], ignore_index=True))
            MSC_TM5 = pd.merge(MSC_TM5_new, MSC_TM5, on='Month', how='outer')
            ax2.plot(MSC_TM5.Month, MSC_TM5['CO2_NEE_fire_flux_monthly_TgC_per_subregion'], color='black', linestyle='dashed', label = 'TM5-4DVar/IS+ACOS', linewidth=1.25)
            ax2.fill_between(MSC_TM5.Month,
                             MSC_TM5['CO2_NEE_fire_flux_monthly_TgC_per_subregion'] - MSC_TM5[('CO2_NEE_fire_flux_monthly_TgC_per_subregion'+'_std')], 
                             MSC_TM5['CO2_NEE_fire_flux_monthly_TgC_per_subregion'] + MSC_TM5[('CO2_NEE_fire_flux_monthly_TgC_per_subregion'+'_std')], 
                             color = 'black', linestyle='dashed', alpha=0.2)
        if plot_MSC_only==False:
            error_type='std_of_year_variations'
            ax1.plot(df_TRENDY_modified_good.MonthDate, df_TRENDY_modified_good['mean'], color=color_list[0], linestyle=linestyle_list[0], label='mean good models', linewidth=1.25)
            ax1.plot(df_TRENDY_modified_bad_low.MonthDate, df_TRENDY_modified_bad_low['mean'], color=color_list[1], linestyle=linestyle_list[1], label='mean bad low ampl', linewidth=1.25)
            ax1.plot(df_TRENDY_modified_bad_high.MonthDate, df_TRENDY_modified_bad_high['mean'], color=color_list[2], linestyle=linestyle_list[2], label='mean bad high ampl', linewidth=1.25)
        
        MSC_mean_good = calculateMeanSeasonalCycle_of_fluxes(df_TRENDY_modified_good, model='TRENDY', TRENDY_variable='mean')
        MSC_std_good = calculateMeanSeasonalCycle_of_fluxes(df_TRENDY_modified_good, model='TRENDY', TRENDY_variable='std')
        MSC_mean_bad_low = calculateMeanSeasonalCycle_of_fluxes(df_TRENDY_modified_bad_low, model='TRENDY', TRENDY_variable='mean')
        MSC_std_bad_low = calculateMeanSeasonalCycle_of_fluxes(df_TRENDY_modified_bad_low, model='TRENDY', TRENDY_variable='std')
        MSC_mean_bad_high = calculateMeanSeasonalCycle_of_fluxes(df_TRENDY_modified_bad_high, model='TRENDY', TRENDY_variable='mean')
        MSC_std_bad_high = calculateMeanSeasonalCycle_of_fluxes(df_TRENDY_modified_bad_high, model='TRENDY', TRENDY_variable='std')
        
        MSC_mean_good.loc[:first_month,'Month'] += 12
        MSC_std_good.loc[:first_month,'Month'] += 12
        MSC_mean_bad_low.loc[:first_month,'Month'] += 12
        MSC_std_bad_low.loc[:first_month,'Month'] += 12
        MSC_mean_bad_high.loc[:first_month,'Month'] += 12
        MSC_std_bad_high.loc[:first_month,'Month'] += 12
        
        MSC_mean_new_good = pd.DataFrame(pd.concat([MSC_mean_good.loc[first_month+1:, 'Month'], MSC_mean_good.loc[:first_month, 'Month']], ignore_index=True))
        MSC_std_new_good = pd.DataFrame(pd.concat([MSC_std_good.loc[first_month+1:, 'Month'], MSC_std_good.loc[:first_month, 'Month']], ignore_index=True))
        MSC_mean_new_bad_low = pd.DataFrame(pd.concat([MSC_mean_bad_low.loc[first_month+1:, 'Month'], MSC_mean_bad_low.loc[:first_month, 'Month']], ignore_index=True))
        MSC_std_new_bad_low = pd.DataFrame(pd.concat([MSC_std_bad_low.loc[first_month+1:, 'Month'], MSC_std_bad_low.loc[:first_month, 'Month']], ignore_index=True))
        MSC_mean_new_bad_high = pd.DataFrame(pd.concat([MSC_mean_bad_high.loc[first_month+1:, 'Month'], MSC_mean_bad_high.loc[:first_month, 'Month']], ignore_index=True))
        MSC_std_new_bad_high = pd.DataFrame(pd.concat([MSC_std_bad_high.loc[first_month+1:, 'Month'], MSC_std_bad_high.loc[:first_month, 'Month']], ignore_index=True))
        
        MSC_mean_good = pd.merge(MSC_mean_new_good, MSC_mean_good, on='Month', how='outer')
        MSC_std_good = pd.merge(MSC_std_new_good, MSC_std_good, on='Month', how='outer')
        MSC_mean_bad_low = pd.merge(MSC_mean_new_bad_low, MSC_mean_bad_low, on='Month', how='outer')
        MSC_std_bad_low = pd.merge(MSC_std_new_bad_low, MSC_std_bad_low, on='Month', how='outer')
        MSC_mean_bad_high = pd.merge(MSC_mean_new_bad_high, MSC_mean_bad_high, on='Month', how='outer')
        MSC_std_bad_high = pd.merge(MSC_std_new_bad_high, MSC_std_bad_high, on='Month', how='outer')
        
        ax2.plot(MSC_mean_good.Month, MSC_mean_good['mean'], color=color_list[0], linestyle=linestyle_list[0], label='mean good models', linewidth=1.25)
        ax2.plot(MSC_mean_bad_low.Month, MSC_mean_bad_low['mean'], color=color_list[1], linestyle=linestyle_list[1], label='mean bad low ampl', linewidth=1.25)
        ax2.plot(MSC_mean_bad_high.Month, MSC_mean_bad_high['mean'], color=color_list[2], linestyle=linestyle_list[2], label='mean bad high ampl', linewidth=1.25)
        
        # plot std of the mean of the MSC/Yearly differences
        if error_type=='std_of_year_variations':
            ax2.fill_between(MSC_mean_good.Month,
                                MSC_mean_good['mean'] - MSC_mean_good[('mean'+'_std')], MSC_mean_good['mean'] + MSC_mean_good[('mean'+'_std')], 
                                color=color_list[0], linestyle=linestyle_list[0], alpha=0.2)
            ax2.fill_between(MSC_mean_bad_low.Month,
                                MSC_mean_bad_low['mean'] - MSC_mean_bad_low[('mean'+'_std')], MSC_mean_bad_low['mean'] + MSC_mean_bad_low[('mean'+'_std')], 
                                color=color_list[1], linestyle=linestyle_list[1], alpha=0.2)
            ax2.fill_between(MSC_mean_bad_high.Month,
                                MSC_mean_bad_high['mean'] - MSC_mean_bad_high[('mean'+'_std')], MSC_mean_bad_high['mean'] + MSC_mean_bad_high[('mean'+'_std')], 
                                color=color_list[2], linestyle=linestyle_list[2], alpha=0.2)
        # plot std of the models
        elif error_type=='std_of_models':
            ax2.fill_between(MSC_mean_good.Month,
                                MSC_mean_good['mean'] - MSC_std_good['std'], MSC_mean_good['mean'] + MSC_std_good['std'], 
                                color=color_list[0], linestyle=linestyle_list[0], alpha=0.2)
            ax2.fill_between(MSC_mean_bad_low.Month,
                                MSC_mean_bad_low['mean'] - MSC_std_bad_low['std'], MSC_mean_bad_low['mean'] + MSC_std_bad_low['std'], 
                                color=color_list[1], linestyle=linestyle_list[1], alpha=0.2)
            ax2.fill_between(MSC_mean_bad_high.Month,
                                MSC_mean_bad_high['mean'] - MSC_std_bad_high['std'], MSC_mean_bad_high['mean'] + MSC_std_bad_high['std'], 
                                color=color_list[2], linestyle=linestyle_list[2], alpha=0.2)

    if plot_MSC_only==False:
        ax1.set_xlabel('Year')
        ax1.set_ylabel('CO$_2$ flux [TgC/month/region]', color='black')
        ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
        ax1.minorticks_on()
        ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        if set_ax_limit:
            ax1.set_yticks([-300, -200, -100, 0, 100, 200, 300])
            ax1.set_yticklabels([-300, -200, -100, 0, 100, 200, 300])
            ax1.set_ylim([-350, 350])
        ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
        ax1.tick_params(bottom=True, left=True, color='gray')
        ax1.legend(framealpha=0.25, facecolor='grey', loc='lower left', fontsize=9.5, ncol=2, columnspacing=0.6)
        ax1.set_title('TRENDY '+variable_to_plot+' mean flux of good & bad models '+region_name, pad=-13)
        #ax1.legend(framealpha=0.25, facecolor='grey', loc='upper left', fontsize=9.5, ncol=6, columnspacing=0.6, bbox_to_anchor=(-0.13, 1.3))
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
        ax2.set_title('TRENDY '+variable_to_plot+' mean flux of good & bad models '+region_name, pad=-13)
        ax2.legend(framealpha=0.25, facecolor='grey', loc='lower left', fontsize=9.5, ncol=2, columnspacing=0.6)
        ax2.set_xticks([5,6,7,8,9,10,11,12,13,14,15,16])
        ax2.set_xticklabels([5,6,7,8,9,10,11,12,1,2,3,4])
        ax2.grid(visible=True, which='minor', color='white', axis='y', linestyle='-', linewidth='0.4')
    if set_ax_limit:
        ax2.set_ylim([-350, 350])
    
    if compare_case==1:
        if plot_MSC_only:
            if plot_TM5_ACOS_flux:
                plt.savefig(savepath+'TRENDY_MSC_only_'+variable_to_plot+'_mean_of_good_bad_low_bad_high_model_flux_AND_TM5_IS+ACOS.png', dpi=300, bbox_inches='tight')
            else:
                plt.savefig(savepath+'TRENDY_MSC_only_'+variable_to_plot+'_mean_of_good_bad_low_bad_high_model_flux.png', dpi=300, bbox_inches='tight')
        else:
            if plot_TM5_ACOS_flux:
                plt.savefig(savepath+'TRENDY_timeseries_MSC_'+variable_to_plot+'_mean_of_good_bad_low_bad_high_model_flux_AND_TM5_IS+ACOS.png', dpi=300, bbox_inches='tight')
            else:
                plt.savefig(savepath+'TRENDY_timeseries_MSC_'+variable_to_plot+'_mean_of_good_bad_low_bad_high_model_flux.png', dpi=300, bbox_inches='tight')
    print('Done saving figure!')
        
        

# MAIN: FOR NEW MODEL CATEGORIES: plot timeseries & MSC for the MEAN of specific TRENDY models & plot all of them in one plot
def plot_timeseries_MSC_of_mean_of_different_model_categories_NEW(df_TRENDY: pd.DataFrame,
                                                  very_good_model_list: list=['ORCHIDEE'],
                                                  ok_model_list: list=['CLASSIC'],
                                                  bad_high_ampl_model_list: list=['SDGVM'],
                                                  bad_low_ampl_model_list: list=['LPJ'],
                                                  variable_to_plot: str='nbp',
                                                  plot_MSC_only: bool=False,
                                                  plot_TM5_ACOS_flux: bool=False,
                                                  df_TM5_ACOS: pd.DataFrame=None,
                                                  df_TM5_RT: pd.DataFrame=None,
                                                  set_ax_limit: bool=False,
                                                  error_type: str='std_of_models',
                                                  color_list: list=['forestgreen', 'midnightblue', 'darkorange'],
                                                  linestyle_list: list=['solid', 'solid', 'solid'],
                                                  start_month: int=5,
                                                  start_year: int=2009,
                                                  end_year: int=2019,
                                                  region_name='SAT',
                                                  savepath='/home/lartelt/MA/software/TRENDY/',
                                                  compare_case=1):
    '''# Documentation
    # arguments:
        - df: ONLY use the df containing the ensemble mean, std & count of all models
        - very_good_model_list: ['OCN', 'ORCHIDEE', ...]
        - error_type: if plot_MSC_only:
                          'std_of_models' --> to plot the std based on the deviations of the models from which the mean was taken, 
                      elif also timeseries plotted:
                          'std_of_year_variations' --> to plot the std based on the deviations of the mean of the yearly variations
        - compare_case:
            1: 2009-2019
            2: sart_year-end_year other than 2009-2019
    '''
    print('Start plotting...')
    if plot_MSC_only:
        fig, ax2 = plt.subplots(1,1, figsize=(8,5))
    else:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5), sharey=True, gridspec_kw={'width_ratios':[8,3]})
        fig.subplots_adjust(wspace=0)
    first_month = start_month-2
    
    column_name_list_very_good = []
    column_name_list_ok = []
    column_name_list_bad_low = []
    column_name_list_bad_high = []
    df_TRENDY_modified_very_good = df_TRENDY[['MonthDate', 'Month']]
    df_TRENDY_modified_ok = df_TRENDY[['MonthDate', 'Month']]
    df_TRENDY_modified_bad_low = df_TRENDY[['MonthDate', 'Month']]
    df_TRENDY_modified_bad_high = df_TRENDY[['MonthDate', 'Month']]
    
    if variable_to_plot == 'ra+rh':
        variable = 'ratot+rh'
    elif variable_to_plot == 'ra+rh+gpp':
        variable = 'ratot+rhtot+gpp'
    elif variable_to_plot == 'ra+rh-gpp':
        variable = 'ratot+rhtot-gpp'
    elif variable_to_plot == 'nbp-(ra+rh)+gpp':
        variable = 'nbptot-(ratot+rhtot)+gpp'
    else:
        variable = variable_to_plot
    
    for model in very_good_model_list:
        column_name_list_very_good.append(variable + 'tot_TgC_month_' + model)
    for model in ok_model_list:
        column_name_list_ok.append(variable + 'tot_TgC_month_' + model)
    for model in bad_low_ampl_model_list:
        column_name_list_bad_low.append(variable + 'tot_TgC_month_' + model)
    for model in bad_high_ampl_model_list:
        column_name_list_bad_high.append(variable + 'tot_TgC_month_' + model)
    
    df_TRENDY_modified_very_good['mean'] = df_TRENDY[column_name_list_very_good].mean(axis=1)
    df_TRENDY_modified_very_good['std'] = df_TRENDY[column_name_list_very_good].std(ddof=0, axis=1)
    df_TRENDY_modified_ok['mean'] = df_TRENDY[column_name_list_ok].mean(axis=1)
    df_TRENDY_modified_ok['std'] = df_TRENDY[column_name_list_ok].std(ddof=0, axis=1)
    df_TRENDY_modified_bad_low['mean'] = df_TRENDY[column_name_list_bad_low].mean(axis=1)
    df_TRENDY_modified_bad_low['std'] = df_TRENDY[column_name_list_bad_low].std(ddof=0, axis=1)
    df_TRENDY_modified_bad_high['mean'] = df_TRENDY[column_name_list_bad_high].mean(axis=1)
    df_TRENDY_modified_bad_high['std'] = df_TRENDY[column_name_list_bad_high].std(ddof=0, axis=1)

    error_type='std_of_models'
    if compare_case==1 or compare_case==2:
        if plot_TM5_ACOS_flux:
            if plot_MSC_only==False:
                ax1.plot(df_TM5_ACOS.MonthDate, df_TM5_ACOS['CO2_NEE_fire_flux_monthly_TgC_per_subregion'], color = 'black', linestyle='dashed', label='TM5-4DVar/IS+ACOS', linewidth=1.25)
                ax1.plot(df_TM5_RT.MonthDate, df_TM5_RT['CO2_NEE_fire_flux_monthly_TgC_per_subregion'], color = 'dimgrey', linestyle='dashed', label='TM5-4DVar/IS+RT', linewidth=1.25)
            MSC_TM5 = calculateMeanSeasonalCycle_of_fluxes(df_TM5_ACOS, model='TM5')
            MSC_TM5.loc[:first_month,'Month'] += 12
            MSC_TM5_new = pd.DataFrame(pd.concat([MSC_TM5.loc[first_month+1:, 'Month'], MSC_TM5.loc[:first_month, 'Month']], ignore_index=True))
            MSC_TM5 = pd.merge(MSC_TM5_new, MSC_TM5, on='Month', how='outer')
            ax2.plot(MSC_TM5.Month, MSC_TM5['CO2_NEE_fire_flux_monthly_TgC_per_subregion'], color='black', linestyle='dashed', label = 'TM5-4DVar/IS+ACOS', linewidth=1.25)
            ax2.fill_between(MSC_TM5.Month,
                             MSC_TM5['CO2_NEE_fire_flux_monthly_TgC_per_subregion'] - MSC_TM5[('CO2_NEE_fire_flux_monthly_TgC_per_subregion'+'_std')], 
                             MSC_TM5['CO2_NEE_fire_flux_monthly_TgC_per_subregion'] + MSC_TM5[('CO2_NEE_fire_flux_monthly_TgC_per_subregion'+'_std')], 
                             color = 'black', linestyle='dashed', alpha=0.2)
            MSC_TM5_RT = calculateMeanSeasonalCycle_of_fluxes(df_TM5_RT, model='TM5')
            MSC_TM5_RT.loc[:first_month,'Month'] += 12
            MSC_TM5_RT_new = pd.DataFrame(pd.concat([MSC_TM5_RT.loc[first_month+1:, 'Month'], MSC_TM5_RT.loc[:first_month, 'Month']], ignore_index=True))
            MSC_TM5_RT = pd.merge(MSC_TM5_RT_new, MSC_TM5_RT, on='Month', how='outer')
            ax2.plot(MSC_TM5_RT.Month, MSC_TM5_RT['CO2_NEE_fire_flux_monthly_TgC_per_subregion'], color='dimgrey', linestyle='dashed', label = 'TM5-4DVar/IS+RT', linewidth=1.25)
            ax2.fill_between(MSC_TM5_RT.Month,
                             MSC_TM5_RT['CO2_NEE_fire_flux_monthly_TgC_per_subregion'] - MSC_TM5_RT[('CO2_NEE_fire_flux_monthly_TgC_per_subregion'+'_std')], 
                             MSC_TM5_RT['CO2_NEE_fire_flux_monthly_TgC_per_subregion'] + MSC_TM5_RT[('CO2_NEE_fire_flux_monthly_TgC_per_subregion'+'_std')], 
                             color = 'dimgrey', linestyle='dashed', alpha=0.2)
        if plot_MSC_only==False:
            error_type='std_of_year_variations'
            ax1.plot(df_TRENDY_modified_very_good.MonthDate, df_TRENDY_modified_very_good['mean'], color=color_list[0], linestyle=linestyle_list[0], label='mean very good models', linewidth=1.25)
            ax1.plot(df_TRENDY_modified_ok.MonthDate, df_TRENDY_modified_ok['mean'], color=color_list[1], linestyle=linestyle_list[1], label='mean ok models', linewidth=1.25)
            ax1.plot(df_TRENDY_modified_bad_low.MonthDate, df_TRENDY_modified_bad_low['mean'], color=color_list[2], linestyle=linestyle_list[2], label='mean bad low ampl', linewidth=1.25)
            ax1.plot(df_TRENDY_modified_bad_high.MonthDate, df_TRENDY_modified_bad_high['mean'], color=color_list[3], linestyle=linestyle_list[3], label='mean bad high ampl', linewidth=1.25)
        
        MSC_mean_very_good = calculateMeanSeasonalCycle_of_fluxes(df_TRENDY_modified_very_good, model='TRENDY', TRENDY_variable='mean')
        MSC_std_very_good = calculateMeanSeasonalCycle_of_fluxes(df_TRENDY_modified_very_good, model='TRENDY', TRENDY_variable='std')
        MSC_mean_ok = calculateMeanSeasonalCycle_of_fluxes(df_TRENDY_modified_ok, model='TRENDY', TRENDY_variable='mean')
        MSC_std_ok = calculateMeanSeasonalCycle_of_fluxes(df_TRENDY_modified_ok, model='TRENDY', TRENDY_variable='std')
        MSC_mean_bad_low = calculateMeanSeasonalCycle_of_fluxes(df_TRENDY_modified_bad_low, model='TRENDY', TRENDY_variable='mean')
        MSC_std_bad_low = calculateMeanSeasonalCycle_of_fluxes(df_TRENDY_modified_bad_low, model='TRENDY', TRENDY_variable='std')
        MSC_mean_bad_high = calculateMeanSeasonalCycle_of_fluxes(df_TRENDY_modified_bad_high, model='TRENDY', TRENDY_variable='mean')
        MSC_std_bad_high = calculateMeanSeasonalCycle_of_fluxes(df_TRENDY_modified_bad_high, model='TRENDY', TRENDY_variable='std')
        
        MSC_mean_very_good.loc[:first_month,'Month'] += 12
        MSC_std_very_good.loc[:first_month,'Month'] += 12
        MSC_mean_ok.loc[:first_month,'Month'] += 12
        MSC_std_ok.loc[:first_month,'Month'] += 12
        MSC_mean_bad_low.loc[:first_month,'Month'] += 12
        MSC_std_bad_low.loc[:first_month,'Month'] += 12
        MSC_mean_bad_high.loc[:first_month,'Month'] += 12
        MSC_std_bad_high.loc[:first_month,'Month'] += 12
        
        MSC_mean_new_very_good = pd.DataFrame(pd.concat([MSC_mean_very_good.loc[first_month+1:, 'Month'], MSC_mean_very_good.loc[:first_month, 'Month']], ignore_index=True))
        MSC_std_new_very_good = pd.DataFrame(pd.concat([MSC_std_very_good.loc[first_month+1:, 'Month'], MSC_std_very_good.loc[:first_month, 'Month']], ignore_index=True))
        MSC_mean_new_ok = pd.DataFrame(pd.concat([MSC_mean_ok.loc[first_month+1:, 'Month'], MSC_mean_ok.loc[:first_month, 'Month']], ignore_index=True))
        MSC_std_new_ok = pd.DataFrame(pd.concat([MSC_std_ok.loc[first_month+1:, 'Month'], MSC_std_ok.loc[:first_month, 'Month']], ignore_index=True))
        MSC_mean_new_bad_low = pd.DataFrame(pd.concat([MSC_mean_bad_low.loc[first_month+1:, 'Month'], MSC_mean_bad_low.loc[:first_month, 'Month']], ignore_index=True))
        MSC_std_new_bad_low = pd.DataFrame(pd.concat([MSC_std_bad_low.loc[first_month+1:, 'Month'], MSC_std_bad_low.loc[:first_month, 'Month']], ignore_index=True))
        MSC_mean_new_bad_high = pd.DataFrame(pd.concat([MSC_mean_bad_high.loc[first_month+1:, 'Month'], MSC_mean_bad_high.loc[:first_month, 'Month']], ignore_index=True))
        MSC_std_new_bad_high = pd.DataFrame(pd.concat([MSC_std_bad_high.loc[first_month+1:, 'Month'], MSC_std_bad_high.loc[:first_month, 'Month']], ignore_index=True))
        
        MSC_mean_very_good = pd.merge(MSC_mean_new_very_good, MSC_mean_very_good, on='Month', how='outer')
        MSC_std_very_good = pd.merge(MSC_std_new_very_good, MSC_std_very_good, on='Month', how='outer')
        MSC_mean_ok = pd.merge(MSC_mean_new_ok, MSC_mean_ok, on='Month', how='outer')
        MSC_std_ok = pd.merge(MSC_std_new_ok, MSC_std_ok, on='Month', how='outer')
        MSC_mean_bad_low = pd.merge(MSC_mean_new_bad_low, MSC_mean_bad_low, on='Month', how='outer')
        MSC_std_bad_low = pd.merge(MSC_std_new_bad_low, MSC_std_bad_low, on='Month', how='outer')
        MSC_mean_bad_high = pd.merge(MSC_mean_new_bad_high, MSC_mean_bad_high, on='Month', how='outer')
        MSC_std_bad_high = pd.merge(MSC_std_new_bad_high, MSC_std_bad_high, on='Month', how='outer')
        
        ax2.plot(MSC_mean_very_good.Month, MSC_mean_very_good['mean'], color=color_list[0], linestyle=linestyle_list[0], label='mean very good models', linewidth=1.25)
        ax2.plot(MSC_mean_ok.Month, MSC_mean_ok['mean'], color=color_list[1], linestyle=linestyle_list[1], label='mean ok models', linewidth=1.25)
        ax2.plot(MSC_mean_bad_low.Month, MSC_mean_bad_low['mean'], color=color_list[2], linestyle=linestyle_list[2], label='mean bad low ampl', linewidth=1.25)
        ax2.plot(MSC_mean_bad_high.Month, MSC_mean_bad_high['mean'], color=color_list[3], linestyle=linestyle_list[3], label='mean bad high ampl', linewidth=1.25)
        
        # plot std of the mean of the MSC/Yearly differences
        if error_type=='std_of_year_variations':
            ax2.fill_between(MSC_mean_very_good.Month,
                                MSC_mean_very_good['mean'] - MSC_mean_very_good[('mean'+'_std')], MSC_mean_very_good['mean'] + MSC_mean_very_good[('mean'+'_std')], 
                                color=color_list[0], linestyle=linestyle_list[0], alpha=0.2)
            ax2.fill_between(MSC_mean_ok.Month,
                                MSC_mean_ok['mean'] - MSC_mean_ok[('mean'+'_std')], MSC_mean_ok['mean'] + MSC_mean_ok[('mean'+'_std')], 
                                color=color_list[1], linestyle=linestyle_list[1], alpha=0.2)
            ax2.fill_between(MSC_mean_bad_low.Month,
                                MSC_mean_bad_low['mean'] - MSC_mean_bad_low[('mean'+'_std')], MSC_mean_bad_low['mean'] + MSC_mean_bad_low[('mean'+'_std')], 
                                color=color_list[2], linestyle=linestyle_list[2], alpha=0.2)
            ax2.fill_between(MSC_mean_bad_high.Month,
                                MSC_mean_bad_high['mean'] - MSC_mean_bad_high[('mean'+'_std')], MSC_mean_bad_high['mean'] + MSC_mean_bad_high[('mean'+'_std')], 
                                color=color_list[3], linestyle=linestyle_list[3], alpha=0.2)
        # plot std of the models
        elif error_type=='std_of_models':
            ax2.fill_between(MSC_mean_very_good.Month,
                                MSC_mean_very_good['mean'] - MSC_std_very_good['std'], MSC_mean_very_good['mean'] + MSC_std_very_good['std'], 
                                color=color_list[0], linestyle=linestyle_list[0], alpha=0.2)
            ax2.fill_between(MSC_mean_ok.Month,
                                MSC_mean_ok['mean'] - MSC_std_ok['std'], MSC_mean_ok['mean'] + MSC_std_ok['std'], 
                                color=color_list[1], linestyle=linestyle_list[1], alpha=0.2)
            ax2.fill_between(MSC_mean_bad_low.Month,
                                MSC_mean_bad_low['mean'] - MSC_std_bad_low['std'], MSC_mean_bad_low['mean'] + MSC_std_bad_low['std'], 
                                color=color_list[2], linestyle=linestyle_list[2], alpha=0.2)
            ax2.fill_between(MSC_mean_bad_high.Month,
                                MSC_mean_bad_high['mean'] - MSC_std_bad_high['std'], MSC_mean_bad_high['mean'] + MSC_std_bad_high['std'], 
                                color=color_list[3], linestyle=linestyle_list[3], alpha=0.2)

    if plot_MSC_only==False:
        ax1.set_xlabel('Year')
        ax1.set_ylabel('CO$_2$ flux [TgC/month/region]', color='black')
        ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
        ax1.minorticks_on()
        ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        if set_ax_limit:
            ax1.set_yticks([-300, -200, -100, 0, 100, 200, 300])
            ax1.set_yticklabels([-300, -200, -100, 0, 100, 200, 300])
            ax1.set_ylim([-350, 350])
        ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
        ax1.tick_params(bottom=True, left=True, color='gray')
        ax1.legend(framealpha=0.25, facecolor='grey', loc='lower left', fontsize=9.5, ncol=2, columnspacing=0.6)
        ax1.set_title('TRENDY '+variable_to_plot+' mean flux of good & bad models '+region_name, pad=-13)
        #ax1.legend(framealpha=0.25, facecolor='grey', loc='upper left', fontsize=9.5, ncol=6, columnspacing=0.6, bbox_to_anchor=(-0.13, 1.3))
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
        ax2.set_title('TRENDY '+variable_to_plot+' mean flux of good & bad models '+region_name, pad=-13)
        ax2.legend(framealpha=0.25, facecolor='grey', loc='lower left', fontsize=9.5, ncol=2, columnspacing=0.6)
        ax2.set_xticks([5,6,7,8,9,10,11,12,13,14,15,16])
        ax2.set_xticklabels([5,6,7,8,9,10,11,12,1,2,3,4])
        ax2.grid(visible=True, which='minor', color='white', axis='y', linestyle='-', linewidth='0.4')
    if set_ax_limit:
        ax2.set_ylim([-350, 350])
    
    if compare_case==1:
        if plot_MSC_only:
            if plot_TM5_ACOS_flux:
                plt.savefig(savepath+'TRENDY_MSC_only_'+variable_to_plot+'_mean_of_very_good_ok_bad_low_bad_new_high_new_model_flux_AND_TM5_IS+ACOS_RT.png', dpi=300, bbox_inches='tight')
            else:
                plt.savefig(savepath+'TRENDY_MSC_only_'+variable_to_plot+'_mean_of_very_good_ok_bad_low_bad_new_high_new_model_flux.png', dpi=300, bbox_inches='tight')
        else:
            if plot_TM5_ACOS_flux:
                plt.savefig(savepath+'TRENDY_timeseries_MSC_'+variable_to_plot+'_mean_of_very_good_ok_bad_low_new_bad_high_new_model_flux_AND_TM5_IS+ACOS_RT.png', dpi=300, bbox_inches='tight')
            else:
                plt.savefig(savepath+'TRENDY_timeseries_MSC_'+variable_to_plot+'_mean_of_very_good_ok_bad_low_new_bad_high_new_model_flux.png', dpi=300, bbox_inches='tight')
    elif compare_case==2:
        if plot_MSC_only:
            if plot_TM5_ACOS_flux:
                plt.savefig(savepath+'/'+str(start_year)+'-'+str(end_year)+'/TRENDY_MSC_only_'+variable_to_plot+'_mean_of_very_good_ok_bad_low_bad_new_high_new_model_flux_AND_TM5_IS+ACOS_RT.png', dpi=300, bbox_inches='tight')
            else:
                plt.savefig(savepath+'/'+str(start_year)+'-'+str(end_year)+'/TRENDY_MSC_only_'+variable_to_plot+'_mean_of_very_good_ok_bad_low_bad_new_high_new_model_flux.png', dpi=300, bbox_inches='tight')
        else:
            if plot_TM5_ACOS_flux:
                plt.savefig(savepath+'/'+str(start_year)+'-'+str(end_year)+'/TRENDY_timeseries_MSC_'+variable_to_plot+'_mean_of_very_good_ok_bad_low_new_bad_high_new_model_flux_AND_TM5_IS+ACOS_RT.png', dpi=300, bbox_inches='tight')
            else:
                plt.savefig(savepath+'/'+str(start_year)+'-'+str(end_year)+'/TRENDY_timeseries_MSC_'+variable_to_plot+'_mean_of_very_good_ok_bad_low_new_bad_high_new_model_flux.png', dpi=300, bbox_inches='tight')
    print('Done saving figure!')





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
                                location_of_legend:str='lower left',
                                legend_columns:int=None,
                                savepath='/home/lartelt/MA/software/12_TRENDY/',
                                savename_given:str=None,
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
                savename=savename+model_list[i]+'_'+TRENDY_model_category[j]+'_'+columns_to_plot_label[j].replace(' ', '_')+'_&_'
                first_trendy=False
                j=j+1
            elif model_list[i]=='TM5_MINUS_GFED':
                savename=savename+'TM5_'+columns_to_plot_label[i].replace('/','_').replace(' ', '_')+'_&_'
            else:
                savename=savename+model_list[i]+'_'+columns_to_plot_label[i].replace('/','_').replace(' ', '_')+'_&_'
            
            if plot_MSC_only==False:
                if model_list[i]=='TM5_MINUS_GFED':
                    ax1.plot(df.MonthDate, df[columns_to_plot[i]], color = color_list[i], linestyle=linestyle_list[i], label='TM5/'+columns_to_plot_label[i], linewidth=1.25)
                else:
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
                        #ax2.fill_between(MSC.Month,
                        #                 a/np.max(a) - MSC_std[(columns_to_plot_std[i])]/np.max(a), a/np.max(a) + MSC_std[(columns_to_plot_std[i])]/np.max(a), 
                        #                 color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)
                    else:
                        ax2.fill_between(MSC.Month,
                                         MSC[columns_to_plot[i]] - MSC_std[columns_to_plot_std[i]], MSC[columns_to_plot[i]] + MSC_std[columns_to_plot_std[i]], 
                                         color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)
                else:
                    if norm_timeseries==True:
                        a = MSC[columns_to_plot[i]]-np.mean([np.min(MSC[columns_to_plot[i]]), np.max(MSC[columns_to_plot[i]])])
                        b = MSC[(columns_to_plot[i]+'_std')]-np.mean([np.min(MSC[(columns_to_plot[i])]), np.max(MSC[(columns_to_plot[i])])])
                        #ax2.fill_between(MSC.Month,
                        #                 a/np.max(a) - b/np.max(a), a/np.max(a) + b/np.max(a), 
                        #                 color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)
                    else:
                        ax2.fill_between(MSC.Month,
                                        MSC[columns_to_plot[i]] - MSC[(columns_to_plot[i]+'_std')], MSC[columns_to_plot[i]] + MSC[(columns_to_plot[i]+'_std')], 
                                        color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)

    if legend_columns is None:
        ncol = np.ceil(len(df_list)/2)-1
    else:
        ncol = legend_columns
    if plot_MSC_only==False:
        ax1.set_xlabel('Year')
        ax1.set_ylabel('CO$_2$ flux [TgC/month/region]', color='black')
        ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
        ax1.minorticks_on()
        ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
        ax1.tick_params(bottom=True, left=True, color='gray')
        ax1.legend(framealpha=0.4, facecolor='grey', loc=location_of_legend, fontsize=9, ncol=ncol, columnspacing=0.6, handletextpad=0.4)#, bbox_to_anchor=(1.38, 0.2))
        ax1.set_title(plot_title+' '+region_name.replace('_',' '))#, pad=-13)
        ax2.set_xticks([5,10,15])
        ax2.set_xticklabels([5, 10, 3])
        ax2.set_title('MSC '+str(start_year)+'-'+str(end_year))#, pad=-13)
        ax2.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    #MSC axis
    ax2.set_xlabel('Month')
    ax2.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax2.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.tick_params(bottom=True, color='gray')
    if plot_MSC_only:
        ax2.set_ylabel('CO$_2$ flux [TgC/month/region]', color='black')
        if norm_timeseries:
            ax2.set_title('Normed '+plot_title+' '+region_name.replace('_',' '))#, pad=-13)
        else:
            ax2.set_title(plot_title+' '+region_name.replace('_',' '))#, pad=-13)
        ax2.legend(framealpha=0.4, facecolor='grey', loc=location_of_legend, fontsize=9.5, ncol=ncol, columnspacing=0.6, handletextpad=0.4)
        ax2.set_xticks([5,6,7,8,9,10,11,12,13,14,15,16])
        ax2.set_xticklabels([5,6,7,8,9,10,11,12,1,2,3,4])
        ax2.grid(visible=True, which='minor', color='white', axis='y', linestyle='-', linewidth='0.4')
    
    if compare_case==1:
        if savename_given is None:
            plt.savefig(savepath+savename[:-3].replace('+','_')+'_'+region_name+'.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(savepath+savename_given.replace('+','_')+'_'+region_name+'.png', dpi=300, bbox_inches='tight')
    elif compare_case==2:
        if savename_given is None:
            plt.savefig(savepath+'/'+str(start_year)+'-'+str(end_year)+'/'+savename[:-3].replace('+','_')+'_'+region_name+'.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(savepath+'/'+str(start_year)+'-'+str(end_year)+'/'+savename_given.replace('+','_')+'_'+region_name+'.png', dpi=300, bbox_inches='tight')



