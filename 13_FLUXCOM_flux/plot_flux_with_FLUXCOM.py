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
import function_to_plot_timeseries_MSC as plot_flux_func
import plotting_map_functions as plot_map_func


# plot FLUXCOM and TRENDY data in one plot
Variables = ['GPP', 'TER', 'NEE']

model_categories = ['very_good_models', 'ok_models', 'bad_low_ampl_models', 'bad_high_ampl_models']
TRENDY_colors = {'very_good_models': 'limegreen', 'ok_models': 'darkolivegreen', 'bad_low_ampl_models': 'orangered', 'bad_high_ampl_models': 'darkmagenta'}


'''# plot FLUXCOM & TRENDY NEE or "Var_FLUXCOM" and "Var_TRENDY" in one plot
Var_FLUXCOM= 'TER' # 'NEE', 'TER', 'GPP'
Var_TRENDY = 'ra+rh' # 'ra+rh-gpp', 'ra+rh', 'gpp'

df_FLUXCOM = load_datasets.load_FLUXCOM_model_gdf(variable=Var_FLUXCOM, unit='per_subregion')

for model_category in model_categories:
    print(model_category)
    df_TRENDY = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY, unit='mean_of_variable_model_category', model_category=model_category)

    plot_flux_func.plot_timeseries_MSC_of_FLUXCOM_compared_to_TRENDY(df_FLUXCOM = df_FLUXCOM,
                                                                     df_TRENDY = df_TRENDY,
                                                                     what_is_plotted = Var_FLUXCOM,
                                                                     model_category=model_category,
                                                                     plot_MSC_only=False,
                                                                     color_list=['black', TRENDY_colors[model_category]],
                                                                     linestyle_list=['solid', 'solid'],
                                                                     savepath='/home/lartelt/MA/software/13_FLUXCOM_flux/',
                                                                     compare_case=1)
'''

'''# plot FLUXCOM & TRENDY & TM5 & MIP nbp in one plot
compare_case=2
if compare_case==1:
    start_year = 2009
    end_year = 2019 #2019
elif compare_case==2:
    start_year = 2009 #2009, 2015
    end_year = 2018   #2014, 2018

Var='nbp'
df_TM5_IS = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
df_TM5_IS_ACOS = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_ACOS', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
df_TM5_IS_RT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_RT', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
df_MIP_ens = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_ens', assimilated='IS_OCO2', end_year=end_year, region='SAT', unit='per_subregion')

TRENDY_model_category = 'bad_high_ampl_models'
TRENDY_model_category_short = 'bha_models'
df_TRENDY = load_datasets.load_TRENDY_model_gdf(variable='nbp', unit='mean_of_variable_model_category', model_category=TRENDY_model_category, start_year=start_year, end_year=end_year)
df_FLUXCOM = load_datasets.load_FLUXCOM_model_gdf(variable='NBP', unit='per_subregion', start_year=start_year, end_year=end_year)

plot_flux_func.plot_timeseries_MSC_of_FLUXCOM_vs_TRENDY_vs_TM5_vs_MIP(df_FLUXCOM = df_FLUXCOM,
                                                                      df_TRENDY=df_TRENDY,
                                                                      df_TM5_IS=df_TM5_IS,
                                                                      df_TM5_ACOS=df_TM5_IS_ACOS,
                                                                      df_TM5_RT=df_TM5_IS_RT,
                                                                      df_MIP_ens=df_MIP_ens,
                                                                      TRENDY_model_category=TRENDY_model_category_short,
                                                                      plot_MSC_only=True,
                                                                      start_year=start_year,
                                                                      end_year=end_year,
                                                                      color_list=['black', TRENDY_colors[TRENDY_model_category], 'firebrick', 'coral', 'royalblue', 'indianred'],
                                                                      linestyle_list=['solid', 'solid', 'solid', 'solid', 'solid', 'dashed'],
                                                                      savepath='/home/lartelt/MA/software/13_FLUXCOM_flux/Compare_FLUXCOM_nbp/',
                                                                      compare_case=compare_case)
plot_flux_func.plot_timeseries_MSC_of_FLUXCOM_vs_TRENDY_vs_TM5_vs_MIP(df_FLUXCOM = df_FLUXCOM,
                                                                      df_TRENDY=df_TRENDY,
                                                                      df_TM5_IS=df_TM5_IS,
                                                                      df_TM5_ACOS=df_TM5_IS_ACOS,
                                                                      df_TM5_RT=df_TM5_IS_RT,
                                                                      df_MIP_ens=df_MIP_ens,
                                                                      TRENDY_model_category=TRENDY_model_category_short,
                                                                      plot_MSC_only=False,
                                                                      start_year=start_year,
                                                                      end_year=end_year,
                                                                      color_list=['black', TRENDY_colors[TRENDY_model_category], 'firebrick', 'coral', 'royalblue', 'indianred'],
                                                                      linestyle_list=['solid', 'solid', 'solid', 'solid', 'solid', 'dashed'],
                                                                      savepath='/home/lartelt/MA/software/13_FLUXCOM_flux/Compare_FLUXCOM_nbp/',
                                                                      compare_case=compare_case)
'''



'''# plot FLUXCOM gpp, ter & TRENDY gpp, ra+rh ; including nee below
start_year = 2009
end_year   = 2019

TRENDY_model_category = 'ok_models'
TRENDY_model_category_short = 'ok_models'
Var_TRENDY1='gpp'
Var_TRENDY2='ra+rh'
Var_FLUXCOM1='GPPtot'
Var_FLUXCOM2='TERtot'

df_TRENDY_1 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY1, unit='mean_of_variable_model_category', model_category=TRENDY_model_category, start_year=start_year, end_year=end_year)
df_TRENDY_2 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY2, unit='mean_of_variable_model_category', model_category=TRENDY_model_category, start_year=start_year, end_year=end_year)
df_FLUXCOM_1 = load_datasets.load_FLUXCOM_model_gdf(variable=Var_FLUXCOM1, unit='per_subregion', start_year=start_year, end_year=end_year)
df_FLUXCOM_2 = load_datasets.load_FLUXCOM_model_gdf(variable=Var_FLUXCOM2, unit='per_subregion', start_year=start_year, end_year=end_year)

# TRENDY color always forestgreen
# FLUXCOM color always sierra
# GPP alsways solid
# TER always dashed

plot_flux_func.plot_timeseries_MSC_general(df_list=[df_FLUXCOM_1, df_FLUXCOM_2, df_TRENDY_1, df_TRENDY_2],
                                           model_list=['FLUXCOM', 'FLUXCOM', 'TRENDY', 'TRENDY'],
                                           columns_to_plot=[Var_FLUXCOM1, Var_FLUXCOM2, 'mean', 'mean'],
                                           columns_to_plot_std=['GPP_madtot', 'TER_madtot', 'std', 'std'],
                                           columns_to_plot_label=[Var_FLUXCOM1[:-3], Var_FLUXCOM2[:-3], Var_TRENDY1, Var_TRENDY2],
                                           TRENDY_model_category=[TRENDY_model_category],
                                           color_list=['sienna', 'sienna', 'forestgreen', 'forestgreen'],
                                           linestyle_list=['solid', 'dashed', 'solid', 'dashed'],
                                           plot_MSC_only=False,
                                           plot_title='FLUXCOM vs. TRENDY '+TRENDY_model_category_short.replace('_',' ')+' GPP and TER',
                                           start_year=start_year,
                                           end_year=end_year,
                                           savepath='/home/lartelt/MA/software/13_FLUXCOM_flux/plot_timeseries_and_MSC/FLUXCOM_TRENDY_gpp_respiration/',
                                           compare_case=1)

plot_flux_func.plot_timeseries_MSC_general(df_list=[df_FLUXCOM_1, df_FLUXCOM_2, df_TRENDY_1, df_TRENDY_2],
                                           model_list=['FLUXCOM', 'FLUXCOM', 'TRENDY', 'TRENDY'],
                                           columns_to_plot=[Var_FLUXCOM1, Var_FLUXCOM2, 'mean', 'mean'],
                                           columns_to_plot_std=['GPP_madtot', 'TER_madtot', 'std', 'std'],
                                           columns_to_plot_label=[Var_FLUXCOM1[:-3], Var_FLUXCOM2[:-3], Var_TRENDY1, Var_TRENDY2],
                                           TRENDY_model_category=[TRENDY_model_category],
                                           color_list=['sienna', 'sienna', 'forestgreen', 'forestgreen'],
                                           linestyle_list=['solid', 'dashed', 'solid', 'dashed'],
                                           norm_timeseries=True,
                                           plot_MSC_only=True,
                                           plot_title='FLUXCOM vs. TRENDY '+TRENDY_model_category_short.replace('_',' ')+' GPP and TER',
                                           start_year=start_year,
                                           end_year=end_year,
                                           savepath='/home/lartelt/MA/software/13_FLUXCOM_flux/plot_timeseries_and_MSC/FLUXCOM_TRENDY_gpp_respiration/',
                                           compare_case=1)
'''


'''# plot FLUXCOM gpp, ter & TRENDY gpp, ra+rh including nbp for arid/humid subregions
start_year = 2009
end_year   = 2019

#TRENDY_model_category = 'very_good_models'
#TRENDY_model_category_short = 'vg_models'
TRENDY_model_categories = ['very_good_models', 'ok_models', 'bad_low_amplitude_models', 'bad_high_amplitude_models']
TRENDY_model_categories_short = ['vg_models', 'ok_models', 'bla_models', 'bha_models']
Var_TRENDY1='gpp'
Var_TRENDY2='ra+rh'
Var_TRENDY3='nbp'
Var_FLUXCOM1='GPPtot'
Var_FLUXCOM2='TERtot'
Var_FLUXCOM3='NBPtot'

for i, category in enumerate(TRENDY_model_categories):
    for arid_type in ['strictlimit_arid_east', 'strictlimit_arid_west', 'strictlimit_arid', 'strictlimit_humid', 'looselimit_arid', 'looselimit_arid_east', 'looselimit_arid_west', 'looselimit_humid']:
        df_TRENDY_1 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY1, unit='mean_of_variable_model_category', model_category=category, arid_type=arid_type, start_year=start_year, end_year=end_year)
        df_TRENDY_2 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY2, unit='mean_of_variable_model_category', model_category=category, arid_type=arid_type, start_year=start_year, end_year=end_year)
        df_TRENDY_3 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY3, unit='mean_of_variable_model_category', model_category=category, arid_type=arid_type, start_year=start_year, end_year=end_year)
        df_FLUXCOM_1 = load_datasets.load_FLUXCOM_model_gdf(variable=Var_FLUXCOM1, unit='per_subregion', region=arid_type, start_year=start_year, end_year=end_year)
        df_FLUXCOM_2 = load_datasets.load_FLUXCOM_model_gdf(variable=Var_FLUXCOM2, unit='per_subregion', region=arid_type, start_year=start_year, end_year=end_year)
        df_FLUXCOM_3 = load_datasets.load_FLUXCOM_model_gdf(variable=Var_FLUXCOM3, unit='per_subregion', region=arid_type, start_year=start_year, end_year=end_year)

        plot_flux_func.plot_timeseries_MSC_general(df_list=[df_FLUXCOM_1, df_FLUXCOM_2, df_FLUXCOM_3, df_TRENDY_1, df_TRENDY_2, df_TRENDY_3],
                                                model_list=['FLUXCOM', 'FLUXCOM', 'FLUXCOM', 'TRENDY', 'TRENDY', 'TRENDY'],
                                                columns_to_plot=[Var_FLUXCOM1, Var_FLUXCOM2, Var_FLUXCOM3, 'mean', 'mean', 'mean'],
                                                columns_to_plot_std=['GPP_madtot', 'TER_madtot', 'NBP_madtot', 'std', 'std', 'std'],
                                                columns_to_plot_label=['FXC gpp', 'FXC ter', 'FLX NEE+GFED', 'TRENDY gpp', 'TRENDY ra+rh', 'TRENDY nbp'],
                                                TRENDY_model_category=[category],
                                                color_list=['sienna', 'sienna', 'sienna', 'forestgreen', 'forestgreen', 'forestgreen'],
                                                linestyle_list=['solid', 'dashed', 'dotted', 'solid', 'dashed', 'dotted'],
                                                plot_MSC_only=True,
                                                plot_title='FXC vs. TRENDY '+TRENDY_model_categories_short[i].replace('_',' '),
                                                start_year=start_year,
                                                end_year=end_year,
                                                label_columns=2,
                                                region_name=arid_type,
                                                savepath='/home/lartelt/MA/software/13_FLUXCOM_flux/plot_timeseries_and_MSC/FLUXCOM_TRENDY_gpp_respiration_nbp/MSCs/supplemental/',
                                                compare_case=1)
'''


'''# plot FLUXCOM nbp & TRENDY vg,ok nbp & TM5/IS+ACOS & TM5/IS+RT & GFED fire flux
start_year = 2009
end_year   = 2018

TRENDY_model_category1 = 'very_good_models'
TRENDY_model_category1_short = 'vg_models'
TRENDY_model_category2 = 'ok_models'
TRENDY_model_category2_short = 'ok_models'
Var_FLUXCOM='NBPtot'
Var_TRENDY1='nbp'
Var_TRENDY2='nbp'


df_FLUXCOM = load_datasets.load_FLUXCOM_model_gdf(variable=Var_FLUXCOM, unit='per_subregion', start_year=start_year, end_year=end_year)
df_TRENDY_1 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY1, unit='mean_of_variable_model_category', model_category=TRENDY_model_category1, start_year=start_year, end_year=end_year)
df_TRENDY_2 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY2, unit='mean_of_variable_model_category', model_category=TRENDY_model_category2, start_year=start_year, end_year=end_year)

#df_TM5_IS = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
df_TM5_IS_ACOS = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_ACOS', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
df_TM5_IS_RT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_RT', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
df_MIP_ens = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_ens', assimilated='IS_OCO2', end_year=end_year, region='SAT', unit='per_subregion')

df_GFED = load_datasets.load_GFED_model_gdf(region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)

##with MIP, but use labels and so on from below without MIP
#plot_flux_func.plot_timeseries_MSC_general(df_list=[df_FLUXCOM, df_TRENDY_1, df_TRENDY_2, df_TM5_IS_ACOS, df_TM5_IS_RT, df_MIP_ens, df_GFED],
#                                           model_list=['FLUXCOM', 'TRENDY', 'TRENDY', 'TM5', 'TM5', 'MIP', 'GFED'],
#                                           columns_to_plot=[Var_FLUXCOM, 'mean', 'mean', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'Landtot', 'total_emission'],
#                                           columns_to_plot_std=['NBP_madtot', 'std', 'std', '', '', '', ''],
#                                           columns_to_plot_label=[Var_FLUXCOM[:-3], Var_TRENDY1, Var_TRENDY2, 'IS+ACOS', 'IS+RT', 'IS+OCO2', ''],
#                                           TRENDY_model_category=None,
#                                           color_list=['sienna', 'limegreen', 'darkolivegreen', 'firebrick', 'coral', 'royalblue', 'black'],
#                                           linestyle_list=['solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid'],
#                                           plot_MSC_only=False,
#                                           plot_title='FLUXCOM & TRENDY & TM5 & MIP nbp and GFED',
#                                           start_year=start_year,
#                                           end_year=end_year,
#                                           savepath='/home/lartelt/MA/software/13_FLUXCOM_flux/plot_timeseries_and_MSC/FLUXCOM_TRENDY_TM5_MIP_GFED/',
#                                           compare_case=1)

plot_flux_func.plot_timeseries_MSC_general(df_list=[df_FLUXCOM, df_TRENDY_1, df_TRENDY_2, df_TM5_IS_ACOS, df_TM5_IS_RT, df_GFED],
                                           model_list=['FLUXCOM', 'TRENDY', 'TRENDY', 'TM5', 'TM5', 'GFED'],
                                           columns_to_plot=[Var_FLUXCOM, 'mean', 'mean', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'total_emission'],
                                           columns_to_plot_std=['NBP_madtot', 'std', 'std', '', '', ''],
                                           columns_to_plot_label=['', TRENDY_model_category1_short.replace('_', ' '), TRENDY_model_category2_short.replace('_', ' '), 'IS+ACOS', 'IS+RT', ''],
                                           TRENDY_model_category=[TRENDY_model_category1_short, TRENDY_model_category2_short],
                                           norm_timeseries=False,
                                           color_list=['sienna', 'limegreen', 'darkolivegreen', 'firebrick', 'coral', 'black'],
                                           linestyle_list=['solid', 'solid', 'solid', 'solid', 'solid', 'solid'],
                                           plot_MSC_only=False,
                                           plot_title='FLUXCOM & TRENDY & TM5 nbp and GFED',
                                           start_year=start_year,
                                           end_year=end_year,
                                           savepath='/home/lartelt/MA/software/13_FLUXCOM_flux/plot_timeseries_and_MSC/FLUXCOM_TRENDY_TM5_GFED/',
                                           compare_case=1)
plot_flux_func.plot_timeseries_MSC_general(df_list=[df_FLUXCOM, df_TRENDY_1, df_TRENDY_2, df_TM5_IS_ACOS, df_TM5_IS_RT, df_GFED],
                                           model_list=['FLUXCOM', 'TRENDY', 'TRENDY', 'TM5', 'TM5', 'GFED'],
                                           columns_to_plot=[Var_FLUXCOM, 'mean', 'mean', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'total_emission'],
                                           columns_to_plot_std=['NBP_madtot', 'std', 'std', '', '', ''],
                                           columns_to_plot_label=['', TRENDY_model_category1_short.replace('_', ' '), TRENDY_model_category2_short.replace('_', ' '), 'IS+ACOS', 'IS+RT', ''],
                                           TRENDY_model_category=[TRENDY_model_category1_short, TRENDY_model_category2_short],
                                           norm_timeseries=False,
                                           color_list=['sienna', 'limegreen', 'darkolivegreen', 'firebrick', 'coral', 'black'],
                                           linestyle_list=['solid', 'solid', 'solid', 'solid', 'solid', 'solid'],
                                           plot_MSC_only=True,
                                           plot_title='FLUXCOM & TRENDY & TM5 nbp and GFED',
                                           start_year=start_year,
                                           end_year=end_year,
                                           savepath='/home/lartelt/MA/software/13_FLUXCOM_flux/plot_timeseries_and_MSC/FLUXCOM_TRENDY_TM5_GFED/',
                                           compare_case=1)
'''


'''# plot NORMED msc for FLUXCOM nbp & TRENDY vg,ok nbp & TM5/IS+ACOS & TM5/IS+RT nbp flux
start_year = 2009
end_year   = 2018

TRENDY_model_category1 = 'very_good_models'
TRENDY_model_category1_short = 'vg_models'
TRENDY_model_category2 = 'ok_models'
TRENDY_model_category2_short = 'ok_models'
Var_FLUXCOM='NBPtot'
Var_TRENDY1='nbp'
Var_TRENDY2='nbp'


df_FLUXCOM = load_datasets.load_FLUXCOM_model_gdf(variable=Var_FLUXCOM, unit='per_subregion', start_year=start_year, end_year=end_year)
df_TRENDY_1 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY1, unit='mean_of_variable_model_category', model_category=TRENDY_model_category1, start_year=start_year, end_year=end_year)
df_TRENDY_2 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY2, unit='mean_of_variable_model_category', model_category=TRENDY_model_category2, start_year=start_year, end_year=end_year)

#df_TM5_IS = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
df_TM5_IS_ACOS = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_ACOS', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
df_TM5_IS_RT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_RT', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
df_MIP_ens = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_ens', assimilated='IS_OCO2', end_year=end_year, region='SAT', unit='per_subregion')

df_GFED = load_datasets.load_GFED_model_gdf(region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)

plot_flux_func.plot_timeseries_MSC_general(df_list=[df_FLUXCOM, df_TRENDY_1, df_TRENDY_2, df_TM5_IS_ACOS, df_TM5_IS_RT],
                                           model_list=['FLUXCOM', 'TRENDY', 'TRENDY', 'TM5', 'TM5'],
                                           columns_to_plot=[Var_FLUXCOM, 'mean', 'mean', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion'],
                                           columns_to_plot_std=['NBP_madtot', 'std', 'std', '', ''],
                                           columns_to_plot_label=['', TRENDY_model_category1_short.replace('_', ' '), TRENDY_model_category2_short.replace('_', ' '), 'IS+ACOS', 'IS+RT'],
                                           TRENDY_model_category=[TRENDY_model_category1_short, TRENDY_model_category2_short],
                                           norm_timeseries=True,
                                           color_list=['sienna', 'limegreen', 'darkolivegreen', 'firebrick', 'coral'],
                                           linestyle_list=['solid', 'solid', 'solid', 'dashed', 'dashed'],
                                           plot_MSC_only=True,
                                           plot_title='FLUXCOM & TRENDY & TM5 nbp',
                                           start_year=start_year,
                                           end_year=end_year,
                                           savepath='/home/lartelt/MA/software/13_FLUXCOM_flux/plot_timeseries_and_MSC/FLUXCOM_TRENDY_TM5/',
                                           compare_case=1)
'''


'''# plot FXC & TRENDY vg_model_mean & TM5/IS+GOSAT & MIP nbp & GFED flux
start_year = 2009 #2009, 2015
end_year = 2018   #2014, 2018

Var='nbp'
df_TM5_IS = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
df_TM5_IS_GOSAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='mean_of_IS_ACOS_and_IS_RT', region='SAT', unit='per_subregion',  start_year=start_year, end_year=end_year)
df_MIP_ens = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_ens', assimilated='IS_OCO2', end_year=end_year, region='SAT', unit='per_subregion')

TRENDY_model_category = 'very_good_models'
TRENDY_model_category_short = 'vg_models'
df_TRENDY = load_datasets.load_TRENDY_model_gdf(variable='nbp', unit='mean_of_variable_model_category', model_category=TRENDY_model_category, start_year=start_year, end_year=end_year)
df_FLUXCOM = load_datasets.load_FLUXCOM_model_gdf(variable='NBP', unit='per_subregion', start_year=start_year, end_year=end_year)

df_GFED = load_datasets.load_GFED_model_gdf(region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)

plot_flux_func.plot_timeseries_MSC_general(df_list=[df_FLUXCOM, df_TRENDY, df_TM5_IS, df_TM5_IS_GOSAT, df_MIP_ens, df_GFED],
                                           model_list=['FLUXCOM', 'TRENDY', 'TM5', 'TM5', 'MIP', 'GFED'],
                                           columns_to_plot=['NBPtot', 'mean', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'Landtot', 'total_emission'],
                                           columns_to_plot_std=['NBP_madtot', 'std', '', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion_std', '', ''],
                                           columns_to_plot_label=['FXC NEE+GFED', 'TRENDY vg models', 'TM5-4DVar/IS', 'TM5-4DVar/IS+GOSAT', 'MIP/IS+OCO2', 'GFED'],
                                           TRENDY_model_category=['very_good_models'],
                                           color_list=['purple', 'limegreen', 'dimgrey', 'firebrick', 'royalblue', 'black'],
                                           linestyle_list=['solid', 'solid', 'dashed', 'solid', 'solid', 'solid'],
                                           plot_MSC_only=False,
                                           plot_title='FLUXCOM & TRENDY & TM5 & MIP nbp',
                                           start_year=start_year,
                                           end_year=end_year,
                                           label_columns=2,
                                           savepath='/home/lartelt/MA/software/13_FLUXCOM_flux/Compare_FLUXCOM_nbp/2009-2018/improved_plots/',
                                           compare_case=1)
'''



'''# plot FLUXCOM gpp, ter & TRENDY gpp, ra+rh including nbp for SAT
start_year = 2009
end_year   = 2019

#TRENDY_model_category = 'very_good_models'
#TRENDY_model_category_short = 'vg_models'
TRENDY_model_categories = ['very_good_models', 'ok_models', 'bad_low_amplitude_models', 'bad_high_amplitude_models']
TRENDY_model_categories_short = ['vg_models', 'ok_models', 'bla_models', 'bha_models']
Var_TRENDY1='gpp'
Var_TRENDY2='ra+rh'
Var_TRENDY3='nbp'
Var_FLUXCOM1='GPPtot'
Var_FLUXCOM2='TERtot'
Var_FLUXCOM3='NBPtot'

for i, category in enumerate(TRENDY_model_categories):
    df_TRENDY_1 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY1, unit='mean_of_variable_model_category', model_category=category, start_year=start_year, end_year=end_year)
    df_TRENDY_2 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY2, unit='mean_of_variable_model_category', model_category=category, start_year=start_year, end_year=end_year)
    df_TRENDY_3 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY3, unit='mean_of_variable_model_category', model_category=category, start_year=start_year, end_year=end_year)
    df_FLUXCOM_1 = load_datasets.load_FLUXCOM_model_gdf(variable=Var_FLUXCOM1, unit='per_subregion', region='SAT', start_year=start_year, end_year=end_year)
    df_FLUXCOM_2 = load_datasets.load_FLUXCOM_model_gdf(variable=Var_FLUXCOM2, unit='per_subregion', region='SAT', start_year=start_year, end_year=end_year)
    df_FLUXCOM_3 = load_datasets.load_FLUXCOM_model_gdf(variable=Var_FLUXCOM3, unit='per_subregion', region='SAT', start_year=start_year, end_year=end_year)

    plot_flux_func.plot_timeseries_MSC_general(df_list=[df_FLUXCOM_1, df_FLUXCOM_2, df_FLUXCOM_3, df_TRENDY_1, df_TRENDY_2, df_TRENDY_3],
                                            model_list=['FLUXCOM', 'FLUXCOM', 'FLUXCOM', 'TRENDY', 'TRENDY', 'TRENDY'],
                                            columns_to_plot=[Var_FLUXCOM1, Var_FLUXCOM2, Var_FLUXCOM3, 'mean', 'mean', 'mean'],
                                            columns_to_plot_std=['GPP_madtot', 'TER_madtot', 'NBP_madtot', 'std', 'std', 'std'],
                                            columns_to_plot_label=['FXC gpp', 'FXC ter', 'FLX nee+GFED', 'TRENDY gpp', 'TRENDY ra+rh', 'TRENDY nbp'],
                                            TRENDY_model_category=[category],
                                            color_list=['sienna', 'sienna', 'sienna', 'forestgreen', 'forestgreen', 'forestgreen'],
                                            linestyle_list=['solid', 'dashed', 'dotted', 'solid', 'dashed', 'dotted'],
                                            plot_MSC_only=True,
                                            plot_title='FXC vs. TRENDY '+TRENDY_model_categories_short[i].replace('_',' '),
                                            start_year=start_year,
                                            end_year=end_year,
                                            label_columns=2,
                                            region_name='SAT',
                                            savepath='/home/lartelt/MA/software/13_FLUXCOM_flux/plot_timeseries_and_MSC/FLUXCOM_TRENDY_gpp_respiration_nbp/MSCs/supplemental/',
                                            compare_case=1)
'''



# plot 2 MSCs for FXC & TRENDY vg_model_mean & TM5/IS+GOSAT & MIP nbp & GFED flux
start_year = 2009 #2009, 2015
end_year = 2018   #2014, 2018


Var='nbp'
df_TM5_IS = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
df_TM5_IS_GOSAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='mean_of_IS_ACOS_and_IS_RT', region='SAT', unit='per_subregion',  start_year=start_year, end_year=end_year)
df_MIP_ens = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_ens', assimilated='IS_OCO2', end_year=end_year, region='SAT', unit='per_subregion')

TRENDY_model_category = 'very_good_models'
TRENDY_model_category_short = 'vg_models'
df_TRENDY = load_datasets.load_TRENDY_model_gdf(variable='nbp', unit='mean_of_variable_model_category', model_category=TRENDY_model_category, start_year=start_year, end_year=end_year)
df_FLUXCOM = load_datasets.load_FLUXCOM_model_gdf(variable='NBP', unit='per_subregion', start_year=start_year, end_year=end_year)
print(df_FLUXCOM.columns)
df_GFED = load_datasets.load_GFED_model_gdf(region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)

plot_flux_func.plot_timeseries_MSC_general_2_MSCs(df_list=[df_FLUXCOM, df_TRENDY, df_TM5_IS, df_TM5_IS_GOSAT, df_MIP_ens, df_GFED],
                                           model_list=['FLUXCOM', 'TRENDY', 'TM5', 'TM5', 'MIP', 'GFED'],
                                           columns_to_plot=['NBPtot', 'mean', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'Landtot', 'total_emission'],
                                           columns_to_plot_std=['NBP_madtot', 'std', '', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion_std', '', ''],
                                           columns_to_plot_label=['FXC NEE+GFED', 'TRENDY selected strict', 'TM5-4DVar/IS', 'TM5-4DVar/IS+GOSAT', 'MIP/IS+OCO2', 'GFED'],
                                           TRENDY_model_category=['very_good_models'],
                                           color_list=['purple', 'limegreen', 'dimgrey', 'firebrick', 'royalblue', 'black'],
                                           linestyle_list=['solid', 'solid', 'dashed', 'solid', 'solid', 'solid'],
                                           plot_title='FLUXCOM & TRENDY & TM5 & MIP nbp',
                                           start_year=start_year,
                                           end_year=end_year,
                                           plot_second_MSCs=True,
                                           second_MSC_start_year=2015,
                                           second_MSC_end_year=2018,
                                           label_columns=2,
                                           savepath='/home/lartelt/MA/software/13_FLUXCOM_flux/Compare_FLUXCOM_nbp/2009-2018/improved_plots/NEW_',
                                           compare_case=1)

