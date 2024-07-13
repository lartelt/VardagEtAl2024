# By Lukas Artelt
# import packages
import numpy as np
import xarray as xr
import pandas as pd
import geopandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import os
import datetime
import seaborn as sns
sns.set(rc={'axes.facecolor':'gainsboro'})
import functions_to_load_datasets as load_datasets
import plotting_map_functions as plot_map_func
import function_to_plot_timeseries_MSC as plot_timeseries_func


#!!!NOT updated to include temperature!!!
'''#plot timeseries, MSC of ERA5 precip & soilmoisture layer 1 only
start_year = 2009
end_year = 2019
region_list = ['arid', 'humid', 'arid_west', 'arid_east']

for boundary in ['strict_limit', 'loose_limit']:
    for region in region_list:
        ERA5_precip = load_datasets.load_ERA5_precip_gdf(region='SAT_'+boundary+'_'+region, unit='per_subregion', start_year=start_year, end_year=end_year)
        ERA5_precip.rename(columns={'tp_mean': 'tp'}, inplace=True)
        ERA5_soil = load_datasets.load_ERA5_soilmoisture_gdf(region='SAT_'+boundary.replace('_','')+'_'+region, unit='per_subregion', start_year=start_year, end_year=end_year)
        ERA5_soil.rename(columns={'swvl1_mean': 'swvl1'}, inplace=True)
        plot_timeseries_func.plot_MSC_ERA5_precip_soilmoisture(df_list=[ERA5_precip, ERA5_soil],
                                                            model_list=['ERA5_precip', 'ERA5_soil'],
                                                            columns_to_plot=['tp', 'swvl1'],
                                                            columns_to_plot_std=['tp_std', 'swvl1_std'],
                                                            columns_to_plot_label=['precipitation', 'soilmoisture'],
                                                            color_precip='royalblue',
                                                            color_soil='darkorange',
                                                            start_year=start_year,
                                                            end_year=end_year,
                                                            region_name='SAT '+boundary.replace('_','')+' '+region,
                                                            plot_title='ERA5 precipitation & soilmoisture',
                                                            legend_columns=1,
                                                            savepath='/home/lartelt/MA/software/15_ERA5_soilmoisture/MSCs/')
        
        plot_timeseries_func.plot_timeseries_ERA5_precip_soilmoisture(df_list=[ERA5_precip, ERA5_soil],
                                                                      model_list=['ERA5_precip', 'ERA5_soil'],
                                                                      columns_to_plot=['tp', 'swvl1'],
                                                                      columns_to_plot_std=['tp_std', 'swvl1_std'],
                                                                      columns_to_plot_label=['precipitation', 'soilmoisture'],
                                                                      color_precip='royalblue',
                                                                      color_soil='darkorange',
                                                                      start_year=start_year,
                                                                      end_year=end_year,
                                                                      region_name='SAT '+boundary.replace('_','')+' '+region,
                                                                      plot_title='ERA5 precipitation & soilmoisture',
                                                                      legend_columns=1,
                                                                      savepath='/home/lartelt/MA/software/15_ERA5_soilmoisture/timeseries/')
        plot_timeseries_func.plot_timeseries_MSC_ERA5_precip_soilmoisture(df_list=[ERA5_precip, ERA5_soil],
                                                                      model_list=['ERA5_precip', 'ERA5_soil'],
                                                                      columns_to_plot=['tp', 'swvl1'],
                                                                      columns_to_plot_std=['tp_std', 'swvl1_std'],
                                                                      columns_to_plot_label=['precipitation', 'soilmoisture'],
                                                                      color_precip='royalblue',
                                                                      color_soil='darkorange',
                                                                      start_year=start_year,
                                                                      end_year=end_year,
                                                                      region_name='SAT '+boundary.replace('_','')+' '+region,
                                                                      plot_title='ERA5 precipitation & soilmoisture',
                                                                      legend_columns=1,
                                                                      savepath='/home/lartelt/MA/software/15_ERA5_soilmoisture/timeseries_AND_MSC/')
'''


#plot timeseries, MSC of ERA5 precip & soilmoisture ALL layers (1,2,3,4)
start_year = 2009
end_year = 2019
region_list = ['arid', 'humid', 'arid_west', 'arid_east']

for boundary in ['strict_limit', 'loose_limit']:
    for region in region_list:
        ERA5_precip = load_datasets.load_ERA5_precip_gdf(region='SAT_'+boundary+'_'+region, unit='per_subregion', start_year=start_year, end_year=end_year)
        ERA5_precip.rename(columns={'tp_mean': 'tp'}, inplace=True)
        
        ERA5_temp = load_datasets.load_ERA5_temperature_gdf(region='SAT_'+boundary.replace('_','')+'_'+region, unit='per_subregion', start_year=start_year, end_year=end_year)
        ERA5_temp.rename(columns={'t2m_mean': 't2m'}, inplace=True)
        
        ERA5_soil_1 = load_datasets.load_ERA5_soilmoisture_gdf(layer='VSWL1', region='SAT_'+boundary.replace('_','')+'_'+region, unit='per_subregion', start_year=start_year, end_year=end_year)
        ERA5_soil_2 = load_datasets.load_ERA5_soilmoisture_gdf(layer='VSWL2', region='SAT_'+boundary.replace('_','')+'_'+region, unit='per_subregion', start_year=start_year, end_year=end_year)
        ERA5_soil_3 = load_datasets.load_ERA5_soilmoisture_gdf(layer='VSWL3', region='SAT_'+boundary.replace('_','')+'_'+region, unit='per_subregion', start_year=start_year, end_year=end_year)
        ERA5_soil_4 = load_datasets.load_ERA5_soilmoisture_gdf(layer='VSWL4', region='SAT_'+boundary.replace('_','')+'_'+region, unit='per_subregion', start_year=start_year, end_year=end_year)
        ERA5_soil_1.rename(columns={'swvl1_mean': 'swvl1'}, inplace=True)
        ERA5_soil_2.rename(columns={'swvl2_mean': 'swvl2'}, inplace=True)
        ERA5_soil_3.rename(columns={'swvl3_mean': 'swvl3'}, inplace=True)
        ERA5_soil_4.rename(columns={'swvl4_mean': 'swvl4'}, inplace=True)
        
        plot_timeseries_func.plot_MSC_ERA5_precip_soilmoisture_temp(
                                df_list=[ERA5_precip, ERA5_soil_1, ERA5_soil_2, ERA5_soil_3, ERA5_soil_4, ERA5_temp],
                                model_list=['ERA5_precip', 'ERA5_soil', 'ERA5_soil', 'ERA5_soil', 'ERA5_soil', 'ERA5_temp'],
                                columns_to_plot=['tp', 'swvl1', 'swvl2', 'swvl3', 'swvl4', 't2m'],
                                columns_to_plot_std=['tp_std', '', '', '', '', 't2m_std'],
                                columns_to_plot_label=['precip', 'swc1', 'swc2', 'swc3', 'swc4', 'temp'],
                                plot_std_in_swc_MSC=True,
                                color_precip='royalblue',
                                color_soil='darkorange',
                                color_temp='firebrick',
                                linestyle_list=['solid', 'solid', 'dashed', 'dotted', 'dashdot', 'solid'],
                                start_year=start_year,
                                end_year=end_year,
                                region_name='SAT '+boundary.replace('_','')+' '+region,
                                plot_title='ERA5 precip & soilmoist & temp',
                                legend_columns=1,
                                savepath='/home/lartelt/MA/software/17_ERA5_temperature/MSCs/')


