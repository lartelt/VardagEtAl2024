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
    elif model=='FLUXNET':
        print('start calculating MSC for FLUXNET fluxes')
        Flux_msc = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[[TRENDY_FLUXCOM_variable]].mean().reset_index()
        Flux_msc_std = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[[TRENDY_FLUXCOM_variable]].std(ddof=0).reset_index()
        Flux_msc_count = Flux_per_subregion_monthly_timeseries.groupby(['Month'])[[TRENDY_FLUXCOM_variable]].count().reset_index()
        Flux_msc = Flux_msc.merge(Flux_msc_std, on='Month', suffixes=('', '_std'))
        Flux_msc = Flux_msc.merge(Flux_msc_count, on='Month', suffixes=('', '_count'))
        return Flux_msc


#----------------------------------------------------------------------------------------------------------------------------
# plotting functions

#plot timeseries precipitation & NEE in one plot with two y-axis
def plot_timeseries_FLUXNET_precip_fluxes(df: pd.DataFrame,
                                          year_to_plot:int=2014,
                                          columns_to_plot: list=['P_ERA', 'NEE_VUT_REF'],
                                          columns_to_plot_label: list=['precipitation', 'NEE'],
                                          color_precip: str='royalblue',
                                          color_list_fluxes: list=['forestgreen', 'olive'],
                                          linestyle_list: list=['solid', 'solid'],
                                          daily_from_HH_nighttime_values_only: bool=False,
                                          region_name: str='FLUXNET_BR-CST',
                                          plot_title: str='FLUXNET BR-CST precip and fluxes',
                                          savepath: str='/home/lartelt/MA/software/16_FLUXNET_flux/timeseries/',
                                          compare_case: int=1):
    print('start plotting...')
    fig, ax1 = plt.subplots(1,1, figsize=(8,5))
    ax3 = ax1.twinx()
    if 'SWC_F_MDS_1' in columns_to_plot:
        ax4 = ax1.twinx()
        ax4.spines.right.set_position(("axes", 1.11))
    #color_list=color_list_fluxes.append(color_precip)
    savename='timeseries_FLUXNET_'+str(year_to_plot)+'_'
    if compare_case==1:
        j=0
        for i, column in enumerate(columns_to_plot):
            savename=savename+columns_to_plot_label[i].replace('/','_').replace(' ', '_')+'_&_'            
            if column=='P_ERA':
                ax3.plot(df.DOY, df[columns_to_plot[i]], color=color_precip, linestyle=linestyle_list[i], label=columns_to_plot_label[i], linewidth=1.25)
            elif column=='SWC_F_MDS_1':
                ax4.plot(df.DOY, df[columns_to_plot[i]], color=color_list_fluxes[j], linestyle=linestyle_list[i], label=columns_to_plot_label[i], linewidth=1.25)
                j=j+1
            else:
                ax1.plot(df.DOY, df[columns_to_plot[i]], color=color_list_fluxes[j], linestyle=linestyle_list[i], label=columns_to_plot_label[i].replace('_', ' '), linewidth=1.25)
                j=j+1
    if compare_case==2:
        j=0
        for i, column in enumerate(columns_to_plot):
            savename=savename+columns_to_plot_label[i].replace('/','_').replace(' ', '_')+'_&_'            
            if column=='P_ERA':
                ax3.plot(df.Days_total, df[columns_to_plot[i]], color=color_precip, linestyle=linestyle_list[i], label=columns_to_plot_label[i], linewidth=1.25)
            elif column=='SWC_F_MDS_1':
                ax4.plot(df.Days_total, df[columns_to_plot[i]], color=color_list_fluxes[j], linestyle=linestyle_list[i], label=columns_to_plot_label[i], linewidth=1.25)
                j=j+1
            else:
                ax1.plot(df.Days_total, df[columns_to_plot[i]], color=color_list_fluxes[j], linestyle=linestyle_list[i], label=columns_to_plot_label[i], linewidth=1.25)
                j=j+1
        # vertical line at DOY=0 in year=2015. Because x-axis is in days starting from 0, DOY=0 for year 2015 is at x=205 because we start at DOY=160 in year 2014 which has DOY_max=365
        #ax1.vlines(205, ymin=-6, ymax=7.5, color='grey', linestyle='-', linewidth=0.8)
        ax1.axvline(x=205, color='black', linewidth=0.9)
    ncol = np.ceil(len(columns_to_plot)/2)-1
    #MSC axis
    ax1.set_xlabel('DOY')
    ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax1.minorticks_on()
    ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    #ax1.grid(visible=True, which='minor', color='white', axis='y', linestyle='-', linewidth='0.4')
    ax1.tick_params(bottom=True, left=True, color='black')
    if compare_case==2:
        print('compare_case=2')
        ax1.set_xticks(np.arange(0,416,20))
        ax1.set_xticklabels(np.arange(160,576,20)%365, rotation=90)
        #ax1.set_xticks([     0,   50,100,150,200,250,300,350,400])
        #ax1.set_xticklabels([160,210,260,310,360, 45, 95,145,195])
        #ax1.set_xticks([       0,100,200,300,400])
        #ax1.set_xticklabels([160,260,360, 60,160])
    
    ax1.set_title(plot_title+' '+region_name.replace('_',' ')+' '+str(year_to_plot))#, pad=-13)
    if daily_from_HH_nighttime_values_only:
        ax1.set_ylabel('Flux [$\mu$molCO$_2$/$m^2$/s]', color='black')
    else:
        ax1.set_ylabel('Flux [gC/$m^2$/day]', color='black')
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax3.get_legend_handles_labels()
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    if 'SWC_F_MDS_1' in columns_to_plot:
        lines_3, labels_3 = ax4.get_legend_handles_labels()
        lines = lines_1 + lines_2 + lines_3
        labels = labels_1 + labels_2 + labels_3
    if year_to_plot==2014:
        ax1.legend(lines, labels, framealpha=0.4, facecolor='grey', loc='upper left', fontsize=9.5, ncol=2, columnspacing=0.6, handletextpad=0.4)
    elif year_to_plot=='2014 DOY 310-340':
        ax1.legend(lines, labels, framealpha=0.4, facecolor='grey', loc='upper left', fontsize=9.5, ncol=1, columnspacing=0.6, handletextpad=0.4)
    else:
        ax1.legend(lines, labels, framealpha=0.4, facecolor='grey', loc='upper right', fontsize=9.5, ncol=2, columnspacing=0.6, handletextpad=0.4)
    
    ax3.set_ylabel('precipitation [mm/day]', color=color_precip)
    ax3.tick_params(axis='y', labelcolor=color_precip)
    #ax3.legend(framealpha=0.4, facecolor='grey', loc='upper right', fontsize=9.5, ncol=ncol, columnspacing=0.6, handletextpad=0.4)
    ax3.grid(visible=False)#, which='minor', color='white', axis='y', linestyle='-', linewidth='0.4')
    
    if 'SWC_F_MDS_1' in columns_to_plot:
        ax4.set_ylabel('Soil water content [%]', color='sienna')
        ax4.tick_params(axis='y', colors='sienna')
        ax4.yaxis.label.set_color('sienna')
        ax4.set_ylim(0, 40)
        ax4.spines['right'].set_visible(True)
        ax4.spines['right'].set_color('dimgrey')
        ax4.grid(visible=False)
    
    if compare_case==1 or compare_case==2:
        plt.savefig(savepath+savename[:-3]+'_'+region_name+'.png', dpi=300, bbox_inches='tight')
    

# SCATTER-plot timeseries precipitation & NEE in one plot with two y-axis
def plot_scatterplot_FLUXNET_precip_fluxes_NT_NEE_with_daily_precip(df: pd.DataFrame,
                                          year_to_plot:int=2014,
                                          columns_to_plot: list=['P_ERA', 'NEE_VUT_REF'],
                                          columns_to_plot_label: list=['precipitation', 'NEE'],
                                          color_precip: str='royalblue',
                                          color_list_fluxes: list=['forestgreen', 'olive'],
                                          linestyle_list: list=['solid', 'solid'],
                                          daily_from_HH_nighttime_values_only: bool=False,
                                          region_name: str='FLUXNET_BR-CST',
                                          plot_title: str='FLUXNET BR-CST precip and fluxes',
                                          savepath: str='/home/lartelt/MA/software/16_FLUXNET_flux/timeseries/',
                                          compare_case: int=1):
    print('start plotting...')
    fig, ax1 = plt.subplots(1,1, figsize=(8,5))
    ax3 = ax1.twinx()
    if 'SWC_F_MDS_1' in columns_to_plot:
        ax4 = ax1.twinx()
        ax4.spines.right.set_position(("axes", 1.11))
    #color_list=color_list_fluxes.append(color_precip)
    savename='Scatterplot_FLUXNET_'+str(year_to_plot)+'_'
    if compare_case==1:
        j=0
        for i, column in enumerate(columns_to_plot):
            savename=savename+columns_to_plot_label[i].replace('/','_').replace(' ', '_')+'_&_'            
            if column=='P_ERA':
                ax3.scatter(df.DOY, df[columns_to_plot[i]], c=color_precip, marker=linestyle_list[i], label=columns_to_plot_label[i], s=25)
            elif column=='SWC_F_MDS_1':
                ax4.scatter(df.DOY, df[columns_to_plot[i]], c=color_list_fluxes[j], marker=linestyle_list[i], label=columns_to_plot_label[i], s=25)
                j=j+1
            else:
                ax1.scatter(df.DOY, df[columns_to_plot[i]], c=color_list_fluxes[j], marker=linestyle_list[i], label=columns_to_plot_label[i].replace('_', ' '), s=25)
                j=j+1
    
    ncol = np.ceil(len(columns_to_plot)/2)-1
    #MSC axis
    ax1.set_xlabel('DOY')
    ax1.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.8)
    ax1.minorticks_on()
    ax1.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.grid(visible=True, which='minor', color='white', axis='both', linestyle='-', linewidth='0.4')
    #ax1.grid(visible=True, which='minor', color='white', axis='y', linestyle='-', linewidth='0.4')
    ax1.tick_params(bottom=True, left=True, labelcolor='black', color='black')
    #ax1.tick_params(axis='y', labelcolor=color_list_fluxes[0])
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_title(plot_title+' '+region_name.replace('_',' ')+' '+str(year_to_plot))#, pad=-13)
    if daily_from_HH_nighttime_values_only:
        #ax1.set_ylabel('Flux [$\mu$molCO$_2$/$m^2$/s]', color=color_list_fluxes[0])
        ax1.set_ylabel('Flux [$\mu$molCO$_2$/$m^2$/s]', color='black')
    else:
        ax1.set_ylabel('Flux [gC/$m^2$/day]', color='black')
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax3.get_legend_handles_labels()
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    if 'SWC_F_MDS_1' in columns_to_plot:
        lines_3, labels_3 = ax4.get_legend_handles_labels()
        lines = lines_1 + lines_2 + lines_3
        labels = labels_1 + labels_2 + labels_3
    if year_to_plot==2014:
        ax1.legend(lines, labels, framealpha=0.4, facecolor='grey', loc='upper left', fontsize=9.5, ncol=2, columnspacing=0.6, handletextpad=0.4)
    elif year_to_plot=='2014 DOY 310-340':
        ax1.legend(lines, labels, framealpha=0.4, facecolor='grey', loc='upper left', fontsize=9.5, ncol=1, columnspacing=0.6, handletextpad=0.4)
    else:
        ax1.legend(lines, labels, framealpha=0.4, facecolor='grey', loc='upper right', fontsize=9.5, ncol=2, columnspacing=0.6, handletextpad=0.4)
    
    ax3.set_ylabel('precipitation [mm/day]', color=color_precip)
    ax3.tick_params(axis='y', labelcolor=color_precip)
    #ax3.legend(framealpha=0.4, facecolor='grey', loc='upper right', fontsize=9.5, ncol=ncol, columnspacing=0.6, handletextpad=0.4)
    ax3.grid(visible=False)#, which='minor', color='white', axis='y', linestyle='-', linewidth='0.4')
    
    if 'SWC_F_MDS_1' in columns_to_plot:
        ax4.set_ylabel('Soil water content [%]', color='sienna')
        ax4.tick_params(axis='y', colors='sienna')
        ax4.yaxis.label.set_color('sienna')
        ax4.set_ylim(0, 40)
        ax4.spines['right'].set_visible(True)
        ax4.spines['right'].set_color('dimgrey')
        ax4.grid(visible=False)
    
    if compare_case==1 or compare_case==2:
        plt.savefig(savepath+savename[:-3]+'_'+region_name+'.png', dpi=300, bbox_inches='tight')
    



