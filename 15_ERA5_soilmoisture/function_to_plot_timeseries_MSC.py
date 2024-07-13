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
def calculateMeanSeasonalCycle_of_fluxes(Flux_per_subregion_monthly_timeseries: pd.DataFrame, model: str='MIP', TRENDY_FLUXCOM_variable: str=None, 
                                         ERA5_swc_level: str='swvl1'):
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
    elif model=='ERA5_precip' or model=='ERA5_precipitation':
        print('start calculating MSC for ERA5 precipitation')
        Flux_msc = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['tp']].mean().reset_index()
        Flux_msc_std = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['tp']].std(ddof=0).reset_index()
        Flux_msc_count = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[['tp']].count().reset_index()
        Flux_msc = Flux_msc.merge(Flux_msc_std, on='Month', suffixes=('', '_std'))
        Flux_msc = Flux_msc.merge(Flux_msc_count, on='Month', suffixes=('', '_count'))
        return Flux_msc
    elif model=='ERA5_soil' or model=='ERA5_soilmoisture':
        print('start calculating MSC for ERA5 soilmoisture level '+ERA5_swc_level)
        Flux_msc = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[[ERA5_swc_level]].mean().reset_index()
        Flux_msc_std = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[[ERA5_swc_level]].std(ddof=0).reset_index()
        Flux_msc_count = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[[ERA5_swc_level]].count().reset_index()
        Flux_msc = Flux_msc.merge(Flux_msc_std, on='Month', suffixes=('', '_std'))
        Flux_msc = Flux_msc.merge(Flux_msc_count, on='Month', suffixes=('', '_count'))
        return Flux_msc


#----------------------------------------
#plotting
#----------------------------------------

#plot ERA5 timeseries precipitation & ERA5 soilmoisture in one plot with two y-axis
def plot_timeseries_ERA5_precip_soilmoisture(df_list: list=[],
                                             model_list: list=['ERA5_precip', 'ERA5_soil'],
                                             columns_to_plot: list=['tp_mean', 'swvl1_mean'],
                                             columns_to_plot_std: list=['tp_std', 'swvl1_std'],
                                             columns_to_plot_label: list=['precipitation', 'soilmoisture'],
                                             color_precip = 'midnightblue',
                                             color_soil = 'forestgreen',
                                             linestyle_list = ['solid', 'solid'],
                                             start_month: int=5,
                                             start_year: int=2009,
                                             end_year: int=2020,
                                             region_name='SAT',
                                             plot_title='ERA5 precip and soilmoisture',
                                             legend_columns:int=None,
                                             savepath: str=None,
                                             compare_case: int=1):
    print('start plotting...')
    fig, ax1 = plt.subplots(1,1, figsize=(8,5))
    ax3 = ax1.twinx()
    color_list=[color_precip, color_soil]
    savename='timeseries_only_'
    if compare_case==1 or compare_case==2:
        for i, df in enumerate(df_list):
            savename=savename+model_list[i]+'_'+columns_to_plot_label[i].replace('/','_').replace(' ', '_')+'_&_'            
            if model_list[i]=='ERA5_precip':
                ax1.plot(df.MonthDate, df[columns_to_plot[i]]*1000, color = color_list[i], linestyle=linestyle_list[i], label=model_list[i]+' '+columns_to_plot_label[i], linewidth=1.25)
                if columns_to_plot_std[i] in df.columns:
                    print('plot std in timeseries with std column '+columns_to_plot_std[i])
                    ax1.fill_between(df.MonthDate, (df[columns_to_plot[i]]-df[columns_to_plot_std[i]])*1000, (df[columns_to_plot[i]]+df[columns_to_plot_std[i]])*1000, 
                                     color = color_list[i], linestyle=linestyle_list[i], alpha=0.2, linewidth=0.3)
            elif model_list[i]=='ERA5_soil':
                ax3.plot(df.MonthDate, df[columns_to_plot[i]], color = color_list[i], linestyle=linestyle_list[i], label=model_list[i]+' '+columns_to_plot_label[i], linewidth=1.25)
                if columns_to_plot_std[i] in df.columns:
                    print('plot std in timeseries with std column '+columns_to_plot_std[i])
                    ax3.fill_between(df.MonthDate, df[columns_to_plot[i]]-df[columns_to_plot_std[i]], df[columns_to_plot[i]]+df[columns_to_plot_std[i]], 
                                    color = color_list[i], linestyle=linestyle_list[i], alpha=0.2, linewidth=0.3)
            else:
                raise NameError('ERROR: model name '+model_list[i]+' not known!')
    if legend_columns is None:
        ncol = np.ceil(len(df_list)/2)-1
    else:
        ncol = legend_columns
    #MSC axis
    ax1.set_xlabel('Year')
    ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.tick_params(bottom=True, color='gray')
    ax1.set_ylabel('precipitation [mm]', color=color_precip)
    ax1.tick_params(axis='y', labelcolor=color_precip)
    ax1.set_title(plot_title+' '+region_name.replace('_',' '))#, pad=-13)
    #ax1.set_xticks([5,6,7,8,9,10,11,12,13,14,15,16])
    #ax1.set_xticklabels([5,6,7,8,9,10,11,12,1,2,3,4])
    ax1.grid(visible=True, which='minor', color='white', axis='y', linestyle='-', linewidth='0.4')
    ax1.legend(framealpha=0.4, facecolor='grey', loc='lower left', fontsize=9.5, ncol=ncol, columnspacing=0.6, handletextpad=0.4)
    
    ax3.set_ylabel('soilmoisture [m$^3$/m$^3$]', color=color_soil)
    ax3.tick_params(axis='y', labelcolor=color_soil)
    ax3.legend(framealpha=0.4, facecolor='grey', loc='lower right', fontsize=9.5, ncol=ncol, columnspacing=0.6, handletextpad=0.4)
    ax3.grid(visible=False)#, which='minor', color='white', axis='y', linestyle='-', linewidth='0.4')
    
    if compare_case==1:
        plt.savefig(savepath+savename[:-3].replace('+','_')+'_'+region_name+'.png', dpi=300, bbox_inches='tight')
    elif compare_case==2:
        plt.savefig(savepath+'/'+str(start_year)+'-'+str(end_year)+'/'+savename[:-3].replace('+','_')+'_'+region_name+'.png', dpi=300, bbox_inches='tight')


#plot ERA5 MSC precipitation & ERA5 soilmoisture in one plot with two y-axis
def plot_MSC_ERA5_precip_soilmoisture(df_list: list=[],
                                  model_list: list=['ERA5_precip', 'ERA5_soil'],
                                  columns_to_plot: list=['tp_mean', 'swvl1_mean'],
                                  columns_to_plot_std: list=['tp_std', 'swvl1_std'],
                                  columns_to_plot_label: list=['precipitation', 'soilmoisture'],
                                  #color_list = ['midnightblue', 'forestgreen'],
                                  color_precip = 'midnightblue',
                                  color_soil = 'forestgreen',
                                  linestyle_list = ['solid', 'solid'],
                                  plot_std_in_swc_MSC=True,
                                  start_month: int=5,
                                  start_year: int=2009,
                                  end_year: int=2020,
                                  region_name='SAT',
                                  plot_title='ERA5 precip and soilmoisture',
                                  legend_columns:int=None,
                                  savepath: str=None,
                                  compare_case: int=1):
    print('start plotting...')
    fig, ax2 = plt.subplots(1,1, figsize=(8,5))
    ax3 = ax2.twinx()
    #color_list=[color_precip, color_soil]
    color_list=[]
    for model in model_list:
        if model=='ERA5_precip' or model=='ERA5_precipitation':
            color_list.append(color_precip)
        elif model=='ERA5_soil' or model=='ERA5_soilmoisture':
            color_list.append(color_soil)
    savename='MSC_only_'
    first_month = start_month-2
    if compare_case==1 or compare_case==2:
        for i, df in enumerate(df_list):
            savename=savename+model_list[i]+'_'+columns_to_plot_label[i].replace('/','_').replace(' ', '_')+'_&_'            
            MSC = calculateMeanSeasonalCycle_of_fluxes(df, model=model_list[i], TRENDY_FLUXCOM_variable=columns_to_plot[i], ERA5_swc_level=columns_to_plot[i])
            MSC.loc[:first_month,'Month'] += 12
            MSC_new = pd.DataFrame(pd.concat([MSC.loc[first_month+1:, 'Month'], MSC.loc[:first_month, 'Month']], ignore_index=True))
            MSC = pd.merge(MSC_new, MSC, on='Month', how='outer')
            if model_list[i]=='ERA5_precip':
                ax2.plot(MSC.Month, MSC[columns_to_plot[i]]*1000, color=color_list[i], linestyle=linestyle_list[i], label='ERA5 '+columns_to_plot_label[i], linewidth=1.25)
            elif model_list[i]=='ERA5_soil':
                ax3.plot(MSC.Month, MSC[columns_to_plot[i]], color=color_list[i], linestyle=linestyle_list[i], label='ERA5 '+columns_to_plot_label[i], linewidth=1.25)
            else:
                raise NameError('ERROR: model name '+model_list[i]+' not known!')
            if columns_to_plot_std[i] in df.columns:
                MSC_std = calculateMeanSeasonalCycle_of_fluxes(df, model=model_list[i], TRENDY_FLUXCOM_variable=columns_to_plot_std[i], ERA5_swc_level=columns_to_plot[i])
                MSC_std.loc[:first_month,'Month'] += 12
                MSC_std_new = pd.DataFrame(pd.concat([MSC_std.loc[first_month+1:, 'Month'], MSC_std.loc[:first_month, 'Month']], ignore_index=True))
                MSC_std = pd.merge(MSC_std_new, MSC_std, on='Month', how='outer')
                print('plot std in MSC with std column '+columns_to_plot_std[i])
                if model_list[i]=='ERA5_precip':
                    ax2.fill_between(MSC.Month,
                                     (MSC[columns_to_plot[i]] - MSC_std[columns_to_plot_std[i]])*1000, (MSC[columns_to_plot[i]] + MSC_std[columns_to_plot_std[i]])*1000, 
                                     color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)
                if model_list[i]=='ERA5_soil' and plot_std_in_swc_MSC==True:
                    ax3.fill_between(MSC.Month,
                                     MSC[columns_to_plot[i]] - MSC_std[columns_to_plot_std[i]], MSC[columns_to_plot[i]] + MSC_std[columns_to_plot_std[i]], 
                                     color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)
            else:
                if model_list[i]=='ERA5_precip':
                    ax2.fill_between(MSC.Month,
                                     (MSC[columns_to_plot[i]] - MSC[(columns_to_plot[i]+'_std')])*1000, (MSC[columns_to_plot[i]] + MSC[(columns_to_plot[i]+'_std')])*1000, 
                                     color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)
                if model_list[i]=='ERA5_soil' and plot_std_in_swc_MSC==True:
                    ax3.fill_between(MSC.Month,
                                     MSC[columns_to_plot[i]] - MSC[(columns_to_plot[i]+'_std')], MSC[columns_to_plot[i]] + MSC[(columns_to_plot[i]+'_std')], 
                                     color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)

    if legend_columns is None:
        ncol = np.ceil(len(df_list)/2)-1
    else:
        ncol = legend_columns
    #MSC axis
    ax2.set_xlabel('Month')
    ax2.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax2.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.tick_params(bottom=True, color='gray')
    ax2.set_ylabel('precipitation [mm]', color=color_precip)
    ax2.tick_params(axis='y', labelcolor=color_precip)
    ax2.set_title(plot_title+' '+region_name.replace('_',' '))#, pad=-13)
    ax2.set_xticks([5,6,7,8,9,10,11,12,13,14,15,16])
    ax2.set_xticklabels([5,6,7,8,9,10,11,12,1,2,3,4])
    ax2.grid(visible=True, which='minor', color='white', axis='y', linestyle='-', linewidth='0.4')
    ax2.legend(framealpha=0.4, facecolor='grey', loc='lower left', fontsize=9.5, ncol=ncol, columnspacing=0.6, handletextpad=0.4)
    
    ax3.set_ylabel('soilmoisture [m$^3$/m$^3$]', color=color_soil)
    ax3.tick_params(axis='y', labelcolor=color_soil)
    ax3.legend(framealpha=0.4, facecolor='grey', loc='lower right', fontsize=9.5, ncol=ncol, columnspacing=0.6, handletextpad=0.4)
    ax3.grid(visible=False)#, which='minor', color='white', axis='y', linestyle='-', linewidth='0.4')
    
    if compare_case==1:
        plt.savefig(savepath+savename[:-3].replace('+','_')+'_'+region_name+'.png', dpi=300, bbox_inches='tight')
    elif compare_case==2:
        plt.savefig(savepath+'/'+str(start_year)+'-'+str(end_year)+'/'+savename[:-3].replace('+','_')+'_'+region_name+'.png', dpi=300, bbox_inches='tight')
    



def plot_timeseries_MSC_ERA5_precip_soilmoisture(df_list: list=[],
                                  model_list: list=['ERA5_precip', 'ERA5_soil'],
                                  columns_to_plot: list=['tp_mean', 'swvl1_mean'],
                                  columns_to_plot_std: list=['tp_std', 'swvl1_std'],
                                  columns_to_plot_label: list=['precipitation', 'soilmoisture'],
                                  plot_std_in_swc_MSC: bool=True,
                                  #color_list = ['midnightblue', 'forestgreen'],
                                  color_precip = 'royalblue',
                                  color_soil = 'darkorange',
                                  linestyle_list = ['solid', 'solid'],
                                  start_month: int=5,
                                  start_year: int=2009,
                                  end_year: int=2020,
                                  region_name='SAT',
                                  plot_title='ERA5 precip and soilmoisture',
                                  legend_columns:int=None,
                                  savepath: str=None,
                                  compare_case: int=1):
    print('start plotting...')
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5), sharey=True, gridspec_kw={'width_ratios':[8,3]})
    fig.subplots_adjust(wspace=0)
    ax3 = ax1.twinx()
    ax4 = ax2.twinx()
    ax3.get_shared_y_axes().join(ax3, ax4)
    #color_list=[color_precip, color_soil]
    color_list=[]
    for model in model_list:
        if model=='ERA5_precip' or model=='ERA5_precipitation':
            color_list.append(color_precip)
        elif model=='ERA5_soil' or model=='ERA5_soilmoisture':
            color_list.append(color_soil)
    savename='timeseries_MSC_'
    first_month = start_month-2
    if compare_case==1 or compare_case==2:
        for i, df in enumerate(df_list):
            savename=savename+model_list[i]+'_'+columns_to_plot_label[i].replace('/','_').replace(' ', '_')+'_&_'            
            MSC = calculateMeanSeasonalCycle_of_fluxes(df, model=model_list[i], TRENDY_FLUXCOM_variable=columns_to_plot[i], ERA5_swc_level=columns_to_plot[i])
            MSC.loc[:first_month,'Month'] += 12
            MSC_new = pd.DataFrame(pd.concat([MSC.loc[first_month+1:, 'Month'], MSC.loc[:first_month, 'Month']], ignore_index=True))
            MSC = pd.merge(MSC_new, MSC, on='Month', how='outer')
            if model_list[i]=='ERA5_precip':
                ax1.plot(df.MonthDate, df[columns_to_plot[i]]*1000, color = color_list[i], linestyle=linestyle_list[i], label=model_list[i]+' '+columns_to_plot_label[i], linewidth=1.25)
                if columns_to_plot_std[i] in df.columns:
                    print('plot std in timeseries with std column '+columns_to_plot_std[i])
                    ax1.fill_between(df.MonthDate, (df[columns_to_plot[i]]-df[columns_to_plot_std[i]])*1000, (df[columns_to_plot[i]]+df[columns_to_plot_std[i]])*1000, 
                                     color = color_list[i], linestyle=linestyle_list[i], alpha=0.2, linewidth=0.3)
                ax2.plot(MSC.Month, MSC[columns_to_plot[i]]*1000, color=color_list[i], linestyle=linestyle_list[i], label='ERA5 '+columns_to_plot_label[i], linewidth=1.25)
            elif model_list[i]=='ERA5_soil':
                ax3.plot(df.MonthDate, df[columns_to_plot[i]], color = color_list[i], linestyle=linestyle_list[i], label=columns_to_plot_label[i], linewidth=1.25)
                if columns_to_plot_std[i] in df.columns:
                    print('plot std in timeseries with std column '+columns_to_plot_std[i])
                    ax3.fill_between(df.MonthDate, df[columns_to_plot[i]]-df[columns_to_plot_std[i]], df[columns_to_plot[i]]+df[columns_to_plot_std[i]], 
                                    color = color_list[i], linestyle=linestyle_list[i], alpha=0.2, linewidth=0.3)
                ax4.plot(MSC.Month, MSC[columns_to_plot[i]], color=color_list[i], linestyle=linestyle_list[i], label='ERA5 '+columns_to_plot_label[i], linewidth=1.25)
            else:
                raise NameError('ERROR: model name '+model_list[i]+' not known!')
            
            if columns_to_plot_std[i] in df.columns:
                MSC_std = calculateMeanSeasonalCycle_of_fluxes(df, model=model_list[i], TRENDY_FLUXCOM_variable=columns_to_plot_std[i], ERA5_swc_level=columns_to_plot_std[i])
                MSC_std.loc[:first_month,'Month'] += 12
                MSC_std_new = pd.DataFrame(pd.concat([MSC_std.loc[first_month+1:, 'Month'], MSC_std.loc[:first_month, 'Month']], ignore_index=True))
                MSC_std = pd.merge(MSC_std_new, MSC_std, on='Month', how='outer')
                print('plot std in MSC with std column '+columns_to_plot_std[i])
                if model_list[i]=='ERA5_precip':
                    ax2.fill_between(MSC.Month,
                                    (MSC[columns_to_plot[i]] - MSC_std[columns_to_plot_std[i]])*1000, (MSC[columns_to_plot[i]] + MSC_std[columns_to_plot_std[i]])*1000, 
                                    color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)
                if model_list[i]=='ERA5_soil' and plot_std_in_swc_MSC:
                    ax4.fill_between(MSC.Month,
                                    MSC[columns_to_plot[i]] - MSC_std[columns_to_plot_std[i]], MSC[columns_to_plot[i]] + MSC_std[columns_to_plot_std[i]], 
                                    color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)
            else:
                if model_list[i]=='ERA5_precip':
                    ax2.fill_between(MSC.Month,
                                    (MSC[columns_to_plot[i]] - MSC[(columns_to_plot[i]+'_std')])*1000, (MSC[columns_to_plot[i]] + MSC[(columns_to_plot[i]+'_std')])*1000, 
                                    color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)
                if model_list[i]=='ERA5_soil' and plot_std_in_swc_MSC:
                    ax4.fill_between(MSC.Month,
                                    MSC[columns_to_plot[i]] - MSC[(columns_to_plot[i]+'_std')], MSC[columns_to_plot[i]] + MSC[(columns_to_plot[i]+'_std')], 
                                    color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)

    if legend_columns is None:
        ncol = np.ceil(len(df_list)/2)-1
    else:
        ncol = legend_columns
    #MSC axis
    ax1.set_xlabel('Year')
    ax1.set_ylabel('precipitation [mm]', color=color_precip)
    ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax1.minorticks_on()
    ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    #ax1.tick_params(bottom=True, left=True, right=False, color='gray')
    ax1.tick_params(axis='y', labelcolor=color_precip)
    ax1.tick_params(bottom=True, color='black')#, labelcolor='black')
    #ax1.tick_params(left=True, color='black', labelcolor=color_precip)
    #ax1.tick_params(bottom=True, color='gray')
    ax1.set_title(plot_title+' '+region_name.replace('_',' '), fontsize=10)#, pad=-13)
    ax2.set_xticks([5,10,15])
    ax2.set_xticklabels([5, 10, 3])
    ax2.set_title('MSC '+str(start_year)+'-'+str(end_year), fontsize=10)#, pad=-13)
    ax2.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    #MSC axis
    ax2.set_xlabel('Month')
    ax2.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax2.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.tick_params(bottom=True, left=False, color='black')
    ax2.tick_params(which='minor', left=False)
    
    #ax3.set_ylabel('soilmoisture [m$^3$/m$^3$]', color=color_soil)
    ax3.tick_params(left=False, right=False, labelright=False)
    ax3.grid(visible=False)
    ax3.legend(framealpha=0.4, facecolor='grey', loc='upper right', fontsize=9.5, ncol=2, columnspacing=0.6, handletextpad=0.4)
    #ax3.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    
    ax4.set_ylabel('soilmoisture [m$^3$/m$^3$]', color=color_soil)
    ax4.tick_params(axis='y', labelcolor=color_soil)
    ax4.grid(visible=False)
    #ax4.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    
    if compare_case==1:
        plt.savefig(savepath+savename[:-3].replace('+','_')+'_'+region_name+'.png', dpi=300, bbox_inches='tight')
    elif compare_case==2:
        plt.savefig(savepath+'/'+str(start_year)+'-'+str(end_year)+'/'+savename[:-3].replace('+','_')+'_'+region_name+'.png', dpi=300, bbox_inches='tight')
    



#!!!NOT FINAL YET!!!
'''#plot timeseries ERA5 precipitation & ERA5 soilmoisture in one plot with two y-axis
def plot_timeseries_ERA5_precip_soilmoisture(df_list: list=[],
                                  model_list: list=['ERA5_precip', 'ERA5_soil'],
                                  columns_to_plot: list=['tp_mean', 'swvl1_mean'],
                                  columns_to_plot_std: list=['tp_std', 'swvl1_std'],
                                  columns_to_plot_label: list=['precipitation', 'soilmoisture'],
                                  color_list = ['midnightblue', 'forestgreen'],
                                  linestyle_list = ['solid', 'solid'],
                                  start_month: int=5,
                                  start_year: int=2009,
                                  end_year: int=2020,
                                  region_name='SAT',
                                  plot_title='ERA5 precip and soilmoisture',
                                  legend_columns:int=None,
                                  compare_case: int=1):
    print('start plotting...')
    if plot_MSC_only==True:
            fig, ax2 = plt.subplots(1,1, figsize=(8,5))
            ax3 = ax2.twinx()
    else:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5), sharey=True, gridspec_kw={'width_ratios':[8,3]})
        fig.subplots_adjust(wspace=0)
        ax3 = ax1.twinx()
    if plot_MSC_only==False:
        savename='timeseries_and_MSC_'
    else:
        savename='MSC_only_'
    first_month = start_month-2
    first_trendy=True
    j=0
    if compare_case==1 or compare_case==2:
        for i, df in enumerate(df_list):
            savename=savename+model_list[i]+'_'+columns_to_plot_label[i].replace('/','_').replace(' ', '_')+'_&_'            
            if plot_MSC_only==False:
                if model_list[i]=='ERA5_precip':
                    ax1.plot(df.MonthDate, df[columns_to_plot[i]], color = color_list[i], linestyle=linestyle_list[i], label=model_list[i]+' '+columns_to_plot_label[i], linewidth=1.25)
                    #include std of the here plotted TRENDY mean
                    if columns_to_plot_std[i] in df.columns:
                        print('plot std in timeseries with std column '+columns_to_plot_std[i])
                        ax1.fill_between(df.MonthDate, df[columns_to_plot[i]]-df[columns_to_plot_std[i]], df[columns_to_plot[i]]+df[columns_to_plot_std[i]], 
                                        color = color_list[i], linestyle=linestyle_list[i], alpha=0.2, linewidth=0.3)
                elif model_list[i]=='ERA5_soil':
                    ax3.plot(df.MonthDate, df[columns_to_plot[i]], color = color_list[i], linestyle=linestyle_list[i], label=model_list[i]+' '+columns_to_plot_label[i], linewidth=1.25)
                    #include std of the here plotted TRENDY mean
                    if columns_to_plot_std[i] in df.columns:
                        print('plot std in timeseries with std column '+columns_to_plot_std[i])
                        ax3.fill_between(df.MonthDate, df[columns_to_plot[i]]-df[columns_to_plot_std[i]], df[columns_to_plot[i]]+df[columns_to_plot_std[i]], 
                                        color = color_list[i], linestyle=linestyle_list[i], alpha=0.2, linewidth=0.3)
            MSC = calculateMeanSeasonalCycle_of_fluxes(df, model=model_list[i], TRENDY_FLUXCOM_variable=columns_to_plot[i])
            MSC.loc[:first_month,'Month'] += 12
            MSC_new = pd.DataFrame(pd.concat([MSC.loc[first_month+1:, 'Month'], MSC.loc[:first_month, 'Month']], ignore_index=True))
            MSC = pd.merge(MSC_new, MSC, on='Month', how='outer')
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
                    ax2.fill_between(MSC.Month,
                                     MSC[columns_to_plot[i]] - MSC_std[columns_to_plot_std[i]], MSC[columns_to_plot[i]] + MSC_std[columns_to_plot_std[i]], 
                                     color=color_list[i], linestyle=linestyle_list[i], alpha=0.2)
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
        ax1.legend(framealpha=0.4, facecolor='grey', loc='lower left', fontsize=9, ncol=ncol, columnspacing=0.6, handletextpad=0.4)
        ax1.set_title(plot_title+' '+region_name.replace('_',' '), fontsize=10)#, pad=-13)
        ax2.set_xticks([5,10,15])
        ax2.set_xticklabels([5, 10, 3])
        ax2.set_title('MSC '+str(start_year)+'-'+str(end_year), fontsize=10)#, pad=-13)
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
        if legend_next_to_plot:
            ax2.legend(framealpha=0.4, facecolor='grey', loc=location_of_legend, fontsize=9.5, ncol=ncol, columnspacing=0.6, handletextpad=0.4, bbox_to_anchor=(1, 0.2))
        else:
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
'''



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
                                legend_next_to_plot:bool=False,
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
                if model_list[i]=='TM5_MINUS_GFED' or model_list[i]=='TM5':
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
                if model_list[i]=='TM5_MINUS_GFED' or model_list[i]=='TM5':
                    ax2.plot(MSC.Month, MSC[columns_to_plot[i]], color=color_list[i], linestyle=linestyle_list[i], label='TM5/'+columns_to_plot_label[i], linewidth=1.25)
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
        if legend_next_to_plot:
            ax1.legend(framealpha=0.4, facecolor='grey', loc=location_of_legend, fontsize=9, ncol=ncol, columnspacing=0.6, handletextpad=0.4, bbox_to_anchor=(1.38, 0.2))
        else:
            ax1.legend(framealpha=0.4, facecolor='grey', loc=location_of_legend, fontsize=9, ncol=ncol, columnspacing=0.6, handletextpad=0.4)
        ax1.set_title(plot_title+' '+region_name.replace('_',' '), fontsize=10)#, pad=-13)
        ax2.set_xticks([5,10,15])
        ax2.set_xticklabels([5, 10, 3])
        ax2.set_title('MSC '+str(start_year)+'-'+str(end_year), fontsize=10)#, pad=-13)
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
        if legend_next_to_plot:
            ax2.legend(framealpha=0.4, facecolor='grey', loc=location_of_legend, fontsize=9.5, ncol=ncol, columnspacing=0.6, handletextpad=0.4, bbox_to_anchor=(1, 0.2))
        else:
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



