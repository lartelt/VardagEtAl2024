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
import function_to_plot_timeseries_MSC as plot_func


'''# Load data & save as pkl
filepath='/mnt/data/users/lartelt/MA/FLUXNET/AmeriFlux_FLUXNET_FULLSET/'
station_list=['AMF_BR-CST_FLUXNET_FULLSET_2014-2015_3-5/', 'AMF_BR-Npw_FLUXNET_FULLSET_2013-2017_3-5/']
filename_list=['AMF_BR-CST_FLUXNET_FULLSET_DD_2014-2015_3-5', 'AMF_BR-Npw_FLUXNET_FULLSET_DD_2013-2017_3-5']
variable_list=[['TIMESTAMP', 'TA_ERA', 'P_ERA', 'CO2_F_MDS', 'SWC_F_MDS_1', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'GPP_DT_VUT_REF'],
               ['TIMESTAMP', 'TA_ERA', 'P_ERA', 'CO2_F_MDS', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'GPP_DT_VUT_REF']]

for i, station in enumerate(station_list):
    print('Start analysing station: '+station)
    ds = pd.read_csv(filepath+station+filename_list[i]+'.csv', sep=',', header=0, parse_dates=True)#, index_col=0)
    ds = ds[variable_list[i]]
    ds['year'] = ds['TIMESTAMP'].apply(lambda x: int(str(x)[:4]))
    ds['month'] = ds['TIMESTAMP'].apply(lambda x: int(str(x)[4:6]))
    ds['day'] = ds['TIMESTAMP'].apply(lambda x: int(str(x)[6:8]))
    ds['date'] = pd.to_datetime(ds[['year', 'month', 'day']])
    print(ds.columns)
    for column in ds.columns:
        ds[column] = ds[column].apply(lambda x: np.nan if x==-9999 else x)
    # save dataframe as pkl
    print('Done, now saving...')
    ds.to_pickle(filepath+filename_list[i][:-4]+'.pkl')
'''    

'''# plot timeseries for 2014 or 2015
['TIMESTAMP', 'TA_ERA', 'P_ERA', 'CO2_F_MDS', 'SWC_F_MDS_1', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'GPP_DT_VUT_REF', 'year', 'month', 'day', 'date']
#columns_to_plot = ['P_ERA', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'GPP_DT_VUT_REF']
#columns_to_plot_BR_CST_NT = ['P_ERA', 'NEE_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'SWC_F_MDS_1']
columns_to_plot_BR_CST_DT = ['P_ERA', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'GPP_DT_VUT_REF', 'SWC_F_MDS_1']
#columns_to_plot_BR_Npw_NT = ['P_ERA', 'NEE_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF']
#columns_to_plot_BR_Npw_DT = ['P_ERA', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'GPP_DT_VUT_REF']

for year in [2014, 2015]:
    df = pd.read_pickle('/mnt/data/users/lartelt/MA/FLUXNET/AmeriFlux_FLUXNET_FULLSET/AMF_BR-CST_FLUXNET_FULLSET_DD_2014-2015.pkl')
    #df = pd.read_pickle('/mnt/data/users/lartelt/MA/FLUXNET/AmeriFlux_FLUXNET_FULLSET/AMF_BR-Npw_FLUXNET_FULLSET_DD_2013-2017.pkl')
    df = df[df['year']==year].reset_index(drop=True)
    df['DOY'] = df.index.values+1
    plot_func.plot_timeseries_FLUXNET_precip_fluxes(df = df,
                                                    year_to_plot = year,
                                                    columns_to_plot = columns_to_plot_BR_CST_DT,
                                                    columns_to_plot_label = columns_to_plot_BR_CST_DT,
                                                    color_precip='royalblue',
                                                    #color_list_fluxes=['darkgreen', 'sienna', 'darkorange', 'limegreen', 'olive', 'black'],
                                                    color_list_fluxes=['darkgreen', 'darkorange', 'limegreen', 'sienna'],
                                                    #linestyle_list=['solid', 'solid', 'solid', 'dashed', 'solid', 'dashed'],
                                                    linestyle_list=['solid', 'solid', 'solid', 'solid', 'solid'],
                                                    region_name='BR-CST',
                                                    plot_title='FLUXNET precip and fluxes',
                                                    savepath='/home/lartelt/MA/software/16_FLUXNET_flux/timeseries/',
                                                    compare_case=1)
'''



'''# plot timeseries for ONE seasonal cycle from 2014 DOY=160 to 2015 DOY=210
#['TIMESTAMP', 'TA_ERA', 'P_ERA', 'CO2_F_MDS', 'SWC_F_MDS_1', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'GPP_DT_VUT_REF', 'year', 'month', 'day', 'date']
#columns_to_plot = ['P_ERA', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'GPP_DT_VUT_REF']
#columns_to_plot_BR_CST_NT = ['P_ERA', 'NEE_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'SWC_F_MDS_1']
#columns_to_plot_BR_CST_DT = ['P_ERA', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'GPP_DT_VUT_REF', 'SWC_F_MDS_1']
#columns_to_plot_BR_Npw_NT = ['P_ERA', 'NEE_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF']
columns_to_plot_BR_Npw_DT = ['P_ERA', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'GPP_DT_VUT_REF']

#df = pd.read_pickle('/mnt/data/users/lartelt/MA/FLUXNET/AmeriFlux_FLUXNET_FULLSET/AMF_BR-CST_FLUXNET_FULLSET_DD_2014-2015.pkl')
df = pd.read_pickle('/mnt/data/users/lartelt/MA/FLUXNET/AmeriFlux_FLUXNET_FULLSET/AMF_BR-Npw_FLUXNET_FULLSET_DD_2013-2017.pkl')
df_2014 = df[df['year']==2014].reset_index(drop=True)
df_2014['DOY'] = df_2014.index.values+1
df_2014 = df_2014[df_2014['DOY']>=160].reset_index(drop=True)

df_2015 = df[df['year']==2015].reset_index(drop=True)
df_2015['DOY'] = df_2015.index.values+1
df_2015 = df_2015[df_2015['DOY']<=210].reset_index(drop=True)

df_2014_2015 = pd.concat([df_2014, df_2015], axis=0).reset_index()
df_2014_2015['Days_total'] = df_2014_2015.index.values

plot_func.plot_timeseries_FLUXNET_precip_fluxes(df=df_2014_2015,
                                                year_to_plot = '2014-2015',
                                                columns_to_plot = columns_to_plot_BR_Npw_DT,
                                                columns_to_plot_label = columns_to_plot_BR_Npw_DT,
                                                color_precip='royalblue',
                                                #color_list_fluxes=['darkgreen', 'sienna', 'darkorange', 'limegreen', 'olive', 'black'],
                                                color_list_fluxes=['darkgreen', 'darkorange', 'limegreen', 'sienna'],
                                                #linestyle_list=['solid', 'solid', 'solid', 'dashed', 'solid', 'dashed'],
                                                linestyle_list=['solid', 'solid', 'solid', 'solid', 'solid'],
                                                region_name='BR-Npw',
                                                plot_title='FLUXNET precip and fluxes',
                                                savepath='/home/lartelt/MA/software/16_FLUXNET_flux/timeseries/Plot_one_seasonal_cycle_2014-2015/',
                                                compare_case=2)
'''


'''# plot timeseries for specific days in 2014
['TIMESTAMP', 'TA_ERA', 'P_ERA', 'CO2_F_MDS', 'SWC_F_MDS_1', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'GPP_DT_VUT_REF', 'year', 'month', 'day', 'date']
#columns_to_plot = ['P_ERA', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'GPP_DT_VUT_REF']
#columns_to_plot_BR_CST_NT = ['P_ERA', 'NEE_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'SWC_F_MDS_1']
#columns_to_plot_BR_CST_DT = ['P_ERA', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'GPP_DT_VUT_REF', 'SWC_F_MDS_1']
columns_to_plot_BR_CST_DT_NT = ['P_ERA', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'GPP_DT_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'SWC_F_MDS_1']
#columns_to_plot_BR_Npw_NT = ['P_ERA', 'NEE_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF']
#columns_to_plot_BR_Npw_DT = ['P_ERA', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'GPP_DT_VUT_REF']

df = pd.read_pickle('/mnt/data/users/lartelt/MA/FLUXNET/AmeriFlux_FLUXNET_FULLSET/AMF_BR-CST_FLUXNET_FULLSET_DD_2014-2015.pkl')
#df = pd.read_pickle('/mnt/data/users/lartelt/MA/FLUXNET/AmeriFlux_FLUXNET_FULLSET/AMF_BR-Npw_FLUXNET_FULLSET_DD_2013-2017.pkl')
df_2014 = df[df['year']==2014].reset_index(drop=True)
df_2014['DOY'] = df_2014.index.values+1
df_2014 = df_2014[df_2014['DOY']>=160].reset_index(drop=True)

df_2014_specific_days_1 = df_2014[df_2014['DOY']>=310].reset_index(drop=True)
df_2014_specific_days = df_2014_specific_days_1[df_2014_specific_days_1['DOY']<=340].reset_index(drop=True)

plot_func.plot_timeseries_FLUXNET_precip_fluxes(df=df_2014_specific_days,
                                                year_to_plot = '2014 DOY 310-340',
                                                columns_to_plot = columns_to_plot_BR_CST_DT_NT,
                                                columns_to_plot_label = columns_to_plot_BR_CST_DT_NT,
                                                color_precip='royalblue',
                                                #color_list_fluxes=['darkgreen', 'sienna', 'darkorange', 'limegreen', 'olive', 'black'],
                                                color_list_fluxes=['darkgreen', 'darkorange', 'limegreen', 'darkorange', 'limegreen', 'sienna'],
                                                #linestyle_list=['solid', 'solid', 'solid', 'dashed', 'solid', 'dashed'],
                                                linestyle_list=['solid', 'solid', 'solid', 'solid', 'dashed', 'dashed', 'solid'],
                                                region_name='BR-CST',
                                                plot_title='FLUXNET precip and fluxes',
                                                savepath='/home/lartelt/MA/software/16_FLUXNET_flux/timeseries/Plot_specific_days_only/',
                                                compare_case=1)
'''


'''# SEE "test_handle_FLUXNET_HH_data.ipynb" for data generation of daily values from HH nighttime values only
# plot daily timeseries from HH nighttime values only
['TIMESTAMP', 'TA_ERA', 'P_ERA', 'CO2_F_MDS', 'SWC_F_MDS_1', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'GPP_DT_VUT_REF', 'year', 'month', 'day', 'date']
#columns_to_plot = ['P_ERA', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'GPP_DT_VUT_REF']
#columns_to_plot_BR_CST_NT = ['P_ERA', 'NEE_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'SWC_F_MDS_1']
#columns_to_plot_BR_CST_DT = ['P_ERA', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'GPP_DT_VUT_REF', 'SWC_F_MDS_1']
columns_to_plot_BR_CST_DT_NT = ['P_ERA', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'GPP_DT_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'SWC_F_MDS_1']
#columns_to_plot_BR_Npw_NT = ['P_ERA', 'NEE_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF']
#columns_to_plot_BR_Npw_DT = ['P_ERA', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'GPP_DT_VUT_REF']

columns_to_plot_BR_CST_nighttime_data_only = ['P_ERA', 'NEE_VUT_REF']
columns_to_plot_BR_CST_nighttime_data_only_label = ['Precipitation', 'NEE_nighttime']

df = pd.read_pickle('/mnt/data/users/lartelt/MA/FLUXNET/AmeriFlux_FLUXNET_FULLSET/AMF_BR-CST_FLUXNET_FULLSET_daily_values_from_only_nighttime_HH_2014_DOY_310_340.pkl')
#df = pd.read_pickle('/mnt/data/users/lartelt/MA/FLUXNET/AmeriFlux_FLUXNET_FULLSET/AMF_BR-Npw_FLUXNET_FULLSET_DD_2013-2017.pkl')

df['DOY'] = df.index.values+305
df = df[df['DOY']>=310].reset_index(drop=True)

plot_func.plot_timeseries_FLUXNET_precip_fluxes(df=df,
                                                year_to_plot = '2014 DOY 310-340',
                                                columns_to_plot = columns_to_plot_BR_CST_nighttime_data_only,
                                                columns_to_plot_label = columns_to_plot_BR_CST_nighttime_data_only_label,
                                                color_precip='royalblue',
                                                #color_list_fluxes=['darkgreen', 'sienna', 'darkorange', 'limegreen', 'olive', 'black'],
                                                color_list_fluxes=['darkgreen', 'darkorange', 'limegreen', 'darkorange', 'limegreen', 'sienna'],
                                                #linestyle_list=['solid', 'solid', 'solid', 'dashed', 'solid', 'dashed'],
                                                linestyle_list=['solid', 'solid', 'solid', 'solid', 'dashed', 'dashed', 'solid'],
                                                daily_from_HH_nighttime_values_only=True,
                                                region_name='BR-CST',
                                                plot_title='FLUXNET nighttime precip and fluxes',
                                                savepath='/home/lartelt/MA/software/16_FLUXNET_flux/Plots_from_nighttime_HH_values_only/BR-CST/',
                                                compare_case=1)
'''



'''# plot daily timeseries from HH nighttime NEE values only WITH PRECIPITATION that includes also daytime values
['TIMESTAMP', 'TA_ERA', 'P_ERA', 'CO2_F_MDS', 'SWC_F_MDS_1', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'GPP_DT_VUT_REF', 'year', 'month', 'day', 'date']
#columns_to_plot = ['P_ERA', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'GPP_DT_VUT_REF']
#columns_to_plot_BR_CST_NT = ['P_ERA', 'NEE_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'SWC_F_MDS_1']
#columns_to_plot_BR_CST_DT = ['P_ERA', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'GPP_DT_VUT_REF', 'SWC_F_MDS_1']
columns_to_plot_BR_CST_DT_NT = ['P_ERA', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'GPP_DT_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'SWC_F_MDS_1']
#columns_to_plot_BR_Npw_NT = ['P_ERA', 'NEE_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF']
#columns_to_plot_BR_Npw_DT = ['P_ERA', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'GPP_DT_VUT_REF']

df_NT_only = pd.read_pickle('/mnt/data/users/lartelt/MA/FLUXNET/AmeriFlux_FLUXNET_FULLSET/AMF_BR-CST_FLUXNET_FULLSET_daily_values_from_only_nighttime_HH_2014_DOY_310_340.pkl')
columns_to_plot_BR_CST_nighttime_data_only = ['P_ERA', 'NEE_VUT_REF']
columns_to_plot_BR_CST_nighttime_data_only_label = ['Precipitation', 'NEE_nighttime']

df_all = pd.read_pickle('/mnt/data/users/lartelt/MA/FLUXNET/AmeriFlux_FLUXNET_FULLSET/AMF_BR-CST_FLUXNET_FULLSET_DD_2014-2015.pkl')
#df = pd.read_pickle('/mnt/data/users/lartelt/MA/FLUXNET/AmeriFlux_FLUXNET_FULLSET/AMF_BR-Npw_FLUXNET_FULLSET_DD_2013-2017.pkl')
df_all_2014 = df_all[df_all['year']==2014].reset_index(drop=True)
df_all_2014['DOY'] = df_all_2014.index.values+1
df_all_2014 = df_all_2014[df_all_2014['DOY']>=160].reset_index(drop=True)
df_all_2014_specific_days_1 = df_all_2014[df_all_2014['DOY']>=310].reset_index(drop=True)
df_all_2014_specific_days = df_all_2014_specific_days_1[df_all_2014_specific_days_1['DOY']<=340].reset_index(drop=True)
df_all_2014_specific_days = df_all_2014_specific_days[['DOY', 'date', 'P_ERA']]

df_NT_only['DOY'] = df_NT_only.index.values+305
df_NT_only = df_NT_only[df_NT_only['DOY']>=310].reset_index(drop=True)
df_NT_only = df_NT_only[['DOY', 'date', 'NEE_VUT_REF']]
#print(df_NT_only.head())
#print(df_all_2014_specific_days.head())

df = df_NT_only.merge(df_all_2014_specific_days, on=['DOY', 'date'])

plot_func.plot_scatterplot_FLUXNET_precip_fluxes_NT_NEE_with_daily_precip(df=df,
                                                year_to_plot = '2014 DOY 310-340',
                                                columns_to_plot = columns_to_plot_BR_CST_nighttime_data_only,
                                                columns_to_plot_label = columns_to_plot_BR_CST_nighttime_data_only_label,
                                                color_precip='royalblue',
                                                color_list_fluxes=['darkgreen'],
                                                linestyle_list=['o', 'o'],
                                                daily_from_HH_nighttime_values_only=True,
                                                region_name='BR-CST',
                                                plot_title='FLUXNET nighttime NEE & daily precip',
                                                savepath='/home/lartelt/MA/software/16_FLUXNET_flux/Plots_from_nighttime_HH_values_only/BR-CST/',
                                                compare_case=1)
'''


# NEW plot timeseries for ONE seasonal cycle from 2014 DOY=160 to 2015 DOY=210 only NEE and precip

columns_to_plot_BR_CST_DT_new = ['P_ERA', 'NEE_VUT_REF']
#df = pd.read_pickle('/mnt/data/users/lartelt/MA/FLUXNET/AmeriFlux_FLUXNET_FULLSET/AMF_BR-CST_FLUXNET_FULLSET_DD_2014-2015.pkl')
columns_to_plot_BR_Npw_DT_new = ['P_ERA', 'NEE_VUT_REF']
df = pd.read_pickle('/mnt/data/users/lartelt/MA/FLUXNET/AmeriFlux_FLUXNET_FULLSET/AMF_BR-Npw_FLUXNET_FULLSET_DD_2013-2017.pkl')

df_2014 = df[df['year']==2014].reset_index(drop=True)
df_2014['DOY'] = df_2014.index.values+1
df_2014 = df_2014[df_2014['DOY']>=160].reset_index(drop=True)

df_2015 = df[df['year']==2015].reset_index(drop=True)
df_2015['DOY'] = df_2015.index.values+1
df_2015 = df_2015[df_2015['DOY']<=210].reset_index(drop=True)

df_2014_2015 = pd.concat([df_2014, df_2015], axis=0).reset_index()
df_2014_2015['Days_total'] = df_2014_2015.index.values

plot_func.plot_timeseries_FLUXNET_precip_fluxes(df=df_2014_2015,
                                                year_to_plot = '2014-2015',
                                                columns_to_plot = columns_to_plot_BR_CST_DT_new,
                                                columns_to_plot_label = columns_to_plot_BR_CST_DT_new,
                                                color_precip='royalblue',
                                                #color_list_fluxes=['darkgreen', 'sienna', 'darkorange', 'limegreen', 'olive', 'black'],
                                                color_list_fluxes=['darkgreen', 'darkorange', 'limegreen', 'sienna'],
                                                #linestyle_list=['solid', 'solid', 'solid', 'dashed', 'solid', 'dashed'],
                                                linestyle_list=['solid', 'solid', 'solid', 'solid', 'solid'],
                                                region_name='BR-Npw',
                                                plot_title='FLUXNET precip and NEE',
                                                savepath='/home/lartelt/MA/software/16_FLUXNET_flux/timeseries/Plot_one_seasonal_cycle_2014-2015/supplemental/',
                                                compare_case=2)




'''# NEW plot daily timeseries from HH nighttime NEE values only WITH PRECIPITATION that includes also daytime values WITH DAILY NEE 
['TIMESTAMP', 'TA_ERA', 'P_ERA', 'CO2_F_MDS', 'SWC_F_MDS_1', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'GPP_DT_VUT_REF', 'year', 'month', 'day', 'date']
#columns_to_plot = ['P_ERA', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'GPP_DT_VUT_REF']
#columns_to_plot_BR_CST_NT = ['P_ERA', 'NEE_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'SWC_F_MDS_1']
#columns_to_plot_BR_CST_DT = ['P_ERA', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'GPP_DT_VUT_REF', 'SWC_F_MDS_1']
columns_to_plot_BR_CST_DT_NT = ['P_ERA', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'GPP_DT_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF', 'SWC_F_MDS_1']
#columns_to_plot_BR_Npw_NT = ['P_ERA', 'NEE_VUT_REF', 'RECO_NT_VUT_REF', 'GPP_NT_VUT_REF']
#columns_to_plot_BR_Npw_DT = ['P_ERA', 'NEE_VUT_REF', 'RECO_DT_VUT_REF', 'GPP_DT_VUT_REF']

df_NT_only = pd.read_pickle('/mnt/data/users/lartelt/MA/FLUXNET/AmeriFlux_FLUXNET_FULLSET/AMF_BR-CST_FLUXNET_FULLSET_daily_values_from_only_nighttime_HH_2014_DOY_310_340.pkl')
columns_to_plot_BR_CST_nighttime_data_only = ['P_ERA', 'NEE_VUT_REF_nighttime', 'NEE_VUT_REF']
columns_to_plot_BR_CST_nighttime_data_only_label = ['Precipitation', 'NEE_nighttime', 'NEE_daily']

df_all = pd.read_pickle('/mnt/data/users/lartelt/MA/FLUXNET/AmeriFlux_FLUXNET_FULLSET/AMF_BR-CST_FLUXNET_FULLSET_DD_2014-2015.pkl')
#df = pd.read_pickle('/mnt/data/users/lartelt/MA/FLUXNET/AmeriFlux_FLUXNET_FULLSET/AMF_BR-Npw_FLUXNET_FULLSET_DD_2013-2017.pkl')
df_all_2014 = df_all[df_all['year']==2014].reset_index(drop=True)
df_all_2014['DOY'] = df_all_2014.index.values+1
df_all_2014 = df_all_2014[df_all_2014['DOY']>=160].reset_index(drop=True)
df_all_2014_specific_days_1 = df_all_2014[df_all_2014['DOY']>=310].reset_index(drop=True)
df_all_2014_specific_days = df_all_2014_specific_days_1[df_all_2014_specific_days_1['DOY']<=340].reset_index(drop=True)
df_all_2014_specific_days = df_all_2014_specific_days[['DOY', 'date', 'P_ERA', 'NEE_VUT_REF']]

df_NT_only['DOY'] = df_NT_only.index.values+305
df_NT_only = df_NT_only[df_NT_only['DOY']>=310].reset_index(drop=True)
df_NT_only = df_NT_only[['DOY', 'date', 'NEE_VUT_REF']]
df_NT_only.rename(columns={'NEE_VUT_REF': 'NEE_VUT_REF_nighttime'}, inplace=True)
#print(df_NT_only.head())
#print(df_all_2014_specific_days.head())

df = df_NT_only.merge(df_all_2014_specific_days, on=['DOY', 'date'])

plot_func.plot_scatterplot_FLUXNET_precip_fluxes_NT_NEE_with_daily_precip(df=df,
                                                year_to_plot = '2014 DOY 310-340',
                                                columns_to_plot = columns_to_plot_BR_CST_nighttime_data_only,
                                                columns_to_plot_label = columns_to_plot_BR_CST_nighttime_data_only_label,
                                                color_precip='royalblue',
                                                color_list_fluxes=['darkgoldenrod', 'darkgreen'],
                                                linestyle_list=['o', 'o', 'o'],
                                                daily_from_HH_nighttime_values_only=True,
                                                region_name='BR-CST',
                                                plot_title='FLUXNET nighttime NEE & daily precip',
                                                savepath='/home/lartelt/MA/software/16_FLUXNET_flux/Plots_from_nighttime_HH_values_only/BR-CST/supplemental/',
                                                compare_case=1)
'''


