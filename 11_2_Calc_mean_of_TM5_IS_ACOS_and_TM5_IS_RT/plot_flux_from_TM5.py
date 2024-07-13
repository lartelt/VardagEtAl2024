# By Lukas Artelt
# import packages
import numpy as np
import xarray as xr
import pandas as pd
import geopandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
sns.set(rc={'axes.facecolor':'gainsboro'})
import functions_to_load_datasets as load_datasets
import function_to_plot_timeseries_MSC as plot_flux_func


# general plot function
start_year = 2009
end_year   = 2019

for aridity_limit in ['strict', 'loose']:
    for aridity in ['arid', 'humid']:
        column_TM5 = 'CO2_NEE_fire_flux_monthly_TgC_per_subregion'
        #column_FXC = 'NBPtot'

        df_TM5_IS_GOSAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='mean_of_IS_ACOS_and_IS_RT', region=aridity_limit+'limit_'+aridity, unit='per_subregion',  start_year=start_year, end_year=end_year)
        df_TM5_IS_ACOS = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_ACOS', region=aridity_limit+'limit_'+aridity, unit='per_subregion',  start_year=start_year, end_year=end_year)
        df_TM5_IS_RT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_RT', region=aridity_limit+'limit_'+aridity, unit='per_subregion',  start_year=start_year, end_year=end_year)
        #df_FXC = load_datasets.load_FLUXCOM_model_gdf(variable='NBP', unit='per_subregion', region=aridity_limit+'limit_'+aridity, start_year=start_year, end_year=end_year)

        plot_flux_func.plot_timeseries_MSC_general(df_list = [df_TM5_IS_GOSAT, df_TM5_IS_ACOS, df_TM5_IS_RT],
                                                model_list = ['TM5', 'TM5', 'TM5'],
                                                columns_to_plot = [column_TM5, column_TM5, column_TM5],
                                                columns_to_plot_std = [column_TM5+'_std', column_TM5+'_std', column_TM5+'_std'],
                                                columns_to_plot_label = ['/IS+GOSAT', '/IS+ACOS', '/IS+RT'],
                                                TRENDY_model_category = ['NONE'],
                                                norm_timeseries = False,
                                                color_list = ['black', 'firebrick', 'coral'],
                                                linestyle_list = ['solid', 'solid', 'solid'],
                                                plot_MSC_only = False,
                                                start_year = start_year,
                                                end_year = end_year,
                                                #location_of_legend='center left', #'lower right',
                                                legend_columns = 3,
                                                region_name='SAT_'+aridity_limit+'_limit_'+aridity,
                                                plot_title = 'TM5 nbp fluxes',
                                                savepath = '/home/lartelt/MA/software/11_2_Calc_mean_of_TM5_IS_ACOS_and_TM5_IS_RT/timeseries_and_MSC/'+aridity_limit+'_limit/',
                                                compare_case = 1)

