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
import plotting_map_functions as plot_map_func

TrendyModels = ['CABLE-POP', 'CLASSIC', 'DLEM', 'IBIS', 'ISAM', 'JSBACH', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'ORCHIDEE-CNP', 'ORCHIDEEv3', 'CLM5.0',
                'ISBA-CTRIP', 'JULES-ES-1p0', 'LPJ', 'SDGVM', 'VISIT', 'YIBs']
Trendy_model_colors = {'CABLE-POP':'r', 'CLASSIC':'g', 'DLEM':'b', 'IBIS':'c', 'ISAM':'m', 'JSBACH':'gold', 'LPX-Bern':'orange', 'OCN':'purple', 
                       'ORCHIDEE':'brown', 'ORCHIDEE-CNP':'pink', 'ORCHIDEEv3':'olive', 'CLM5.0':'cyan', 'ISBA-CTRIP':'springgreen', 'JULES-ES-1p0':'teal',
                       'LPJ':'darkred', 'SDGVM':'darkgreen', 'VISIT':'darkblue', 'YIBs':'darkcyan'}

# NEW model categories
models_very_good_list = ['CLASSIC', 'OCN']
models_ok_list = ['ISBA-CTRIP', 'ISAM', 'YIBs', 'ORCHIDEE']
models_bad_low_amplitude_new = ['ORCHIDEE-CNP', 'LPJ', 'JULES-ES-1p0', 'JSBACH', 'IBIS']
models_bad_low_amplitude_new_for_ra_rh_mean = ['LPJ', 'JSBACH', 'IBIS'] # all models containing respiration & without ORCHIDEE-CNP because it is very bad for respiration
models_bad_high_amplitude_new = ['SDGVM', 'VISIT', 'LPX-Bern', 'CLM5.0', 'ORCHIDEEv3']
models_bad_high_amplitude_new_for_npp_mean = ['SDGVM', 'LPX-Bern', 'CLM5.0', 'ORCHIDEEv3'] # all models except VISIT because this models does not have npp



'''# general plot function for single TRENDY models and one single variable: nbp, gpp, ra+rh-gpp
start_year = 2009
end_year   = 2019

#TRENDY_model = 'CLASSIC' # 'ISBA-CTRIP', 'ISAM', 'YIBs', 'ORCHIDEE'
model_list=['CLASSIC', 'OCN', 'ISBA-CTRIP', 'ISAM', 'YIBs', 'ORCHIDEE']
#Var='nbp'
#Var_long = 'nbp'#'ratot+rh'
#aridity_limit = 'strict'
for aridity_limit in ['strict', 'loose']:
    for Var in ['ra+rh-gpp']:#['nbp', 'gpp', 'ra', 'rh']:
        Var_long='ratot+rhtot-gpp'
        for TRENDY_model in model_list:
            column = Var_long+'tot_TgC_month_'+TRENDY_model

            df_TRENDY_1 = load_datasets.load_TRENDY_model_gdf(variable=Var, unit='mean_of_variable', start_year=start_year, end_year=end_year)
            df_TRENDY_2 = load_datasets.load_TRENDY_model_gdf(variable=Var, unit='mean_of_variable', arid_type=aridity_limit+'limit_arid', start_year=start_year, end_year=end_year)
            df_TRENDY_3 = load_datasets.load_TRENDY_model_gdf(variable=Var, unit='mean_of_variable', arid_type=aridity_limit+'limit_humid', start_year=start_year, end_year=end_year)

            plot_flux_func.plot_timeseries_MSC_general(df_list = [df_TRENDY_1, df_TRENDY_2, df_TRENDY_3],
                                                    model_list = ['TRENDY', 'TRENDY', 'TRENDY'],
                                                    columns_to_plot = [column, column, column],
                                                    columns_to_plot_std = ['', '', ''],
                                                    columns_to_plot_label = ['SAT', 'arid', 'humid'],
                                                    TRENDY_model_category = [TRENDY_model],
                                                    norm_timeseries = True,
                                                    color_list = ['black', 'coral', 'cornflowerblue'],
                                                    linestyle_list = ['solid', 'solid', 'solid'],
                                                    plot_MSC_only = True,
                                                    start_year = start_year,
                                                    end_year = end_year,
                                                    #location_of_legend='lower right',
                                                    legend_columns = 3,
                                                    region_name='SAT_'+aridity_limit+'_arid_humid',
                                                    plot_title = 'TRENDY '+TRENDY_model+' '+Var+' fluxes',
                                                    savepath = '/home/lartelt/MA/software/12_TRENDY_flux/plot_arid_humid_region/'+aridity_limit+'_limit/Variable_'+Var+'/',
                                                    compare_case = 1)
'''


'''# general plot function for single TRENDY models and multiple variables: nbp & gpp & ra+rh & ...
start_year = 2009
end_year   = 2019

#TRENDY_model = 'CLASSIC' # 'ISBA-CTRIP', 'ISAM', 'YIBs', 'ORCHIDEE'
#good_model_list=['CLASSIC', 'OCN', 'ISBA-CTRIP', 'ISAM', 'YIBs', 'ORCHIDEE']
all_model_list=['CLASSIC', 'OCN', 'ISBA-CTRIP', 'ISAM', 'YIBs', 'ORCHIDEE', 'LPJ', 'JSBACH', 'IBIS', 'ORCHIDEE-CNP', 'SDGVM', 'VISIT', 'LPX-Bern', 'CLM5.0', 'ORCHIDEEv3']

Var1='nbp'
Var1_long=Var1
Var2='gpp'
Var2_long=Var2
Var3='ra+rh'
Var3_long='ratot+rh'
Var4='ra'
Var4_long=Var4
Var5='rh'
Var5_long=Var5
Var6='ra+rh-gpp'
Var6_long='ratot+rhtot-gpp'

for aridity_limit in ['strict', 'loose']:
    for aridity in ['arid', 'humid']:
        for TRENDY_model in all_model_list:
            column1 = Var1_long+'tot_TgC_month_'+TRENDY_model
            column2 = Var2_long+'tot_TgC_month_'+TRENDY_model
            column3 = Var3_long+'tot_TgC_month_'+TRENDY_model
            column4 = Var4_long+'tot_TgC_month_'+TRENDY_model
            column5 = Var5_long+'tot_TgC_month_'+TRENDY_model
            column6 = Var6_long+'tot_TgC_month_'+TRENDY_model
            column_TM5 = 'CO2_NEE_fire_flux_monthly_TgC_per_subregion'
            column_FXC = 'NBPtot'

            df_TRENDY_1 = load_datasets.load_TRENDY_model_gdf(variable=Var1, unit='mean_of_variable', arid_type=aridity_limit+'limit_'+aridity, start_year=start_year, end_year=end_year)
            df_TRENDY_2 = load_datasets.load_TRENDY_model_gdf(variable=Var2, unit='mean_of_variable', arid_type=aridity_limit+'limit_'+aridity, start_year=start_year, end_year=end_year)
            df_TRENDY_3 = load_datasets.load_TRENDY_model_gdf(variable=Var3, unit='mean_of_variable', arid_type=aridity_limit+'limit_'+aridity, start_year=start_year, end_year=end_year)
            df_TRENDY_4 = load_datasets.load_TRENDY_model_gdf(variable=Var4, unit='mean_of_variable', arid_type=aridity_limit+'limit_'+aridity, start_year=start_year, end_year=end_year)
            df_TRENDY_5 = load_datasets.load_TRENDY_model_gdf(variable=Var5, unit='mean_of_variable', arid_type=aridity_limit+'limit_'+aridity, start_year=start_year, end_year=end_year)
            df_TRENDY_6 = load_datasets.load_TRENDY_model_gdf(variable=Var6, unit='mean_of_variable', arid_type=aridity_limit+'limit_'+aridity, start_year=start_year, end_year=end_year)
            df_TM5_IS_GOSAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='mean_of_IS_ACOS_and_IS_RT', region=aridity_limit+'limit_'+aridity, unit='per_subregion',  start_year=start_year, end_year=end_year)
            df_FXC = load_datasets.load_FLUXCOM_model_gdf(variable='NBP', unit='per_subregion', region=aridity_limit+'limit_'+aridity, start_year=start_year, end_year=end_year)

            plot_flux_func.plot_timeseries_MSC_general(df_list = [df_TRENDY_1, df_TRENDY_2, df_TRENDY_3, df_TRENDY_4, df_TRENDY_5, df_TRENDY_6, 
                                                                  df_TM5_IS_GOSAT, df_FXC],
                                                    model_list = ['TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TM5', 'FXC'],
                                                    columns_to_plot = [column1, column2, column3, column4, column5, column6, column_TM5, column_FXC],
                                                    columns_to_plot_std = ['', '', '', '', '', '', column_TM5+'_std', 'NBP_madtot'],
                                                    columns_to_plot_label = [Var1, Var2, Var3, Var4, Var5, Var6, 'IS+GOSAT nbp', 'nbp'],
                                                    TRENDY_model_category = [TRENDY_model],
                                                    norm_timeseries = False,
                                                    color_list = ['black', 'forestgreen', 'sienna', 'olivedrab', 'rosybrown', 'dimgrey', 'firebrick', 'purple'],
                                                    linestyle_list = ['solid', 'solid', 'solid', 'dashed', 'dashed', 'dotted', 'dashed', 'dashed'],
                                                    plot_MSC_only = False,
                                                    start_year = start_year,
                                                    end_year = end_year,
                                                    location_of_legend='center left', #'lower right',
                                                    legend_columns = 1,
                                                    region_name=aridity_limit+'_limit_'+aridity+'_SAT',
                                                    plot_title = 'TRENDY '+TRENDY_model+' & TM5 & FXC',
                                                    savepath = '/home/lartelt/MA/software/12_TRENDY_flux/plot_arid_humid_region/'+aridity_limit+'_limit/Variable_nbp_AND_gpp_AND_ra_AND_rh_AND_ra+rh_AND_ra+rh-gpp_AND_TM5_mean_AND_FXC/',
                                                    compare_case = 1)
'''


'''# general plot function for mean of TRENDY model category and multiple variables: nbp & gpp & ra+rh
start_year = 2009
end_year   = 2019

Var1='nbp'
Var1_long=Var1
Var2='gpp'
Var2_long=Var2
Var3='ra+rh'
Var3_long='ratot+rh'
Var4='ra'
Var4_long=Var4
Var5='rh'
Var5_long=Var5
Var6='ra+rh-gpp'
Var6_long='ratot+rhtot-gpp'

TRENDY_model_categories = ['very_good_models', 'ok_models', 'bad_low_amplitude_models', 'bad_high_amplitude_models']
TRENDY_model_categories_short = ['vg_models', 'ok_models', 'bla_models', 'bha_models']

for i,model_category in enumerate(TRENDY_model_categories):
    for aridity_limit in ['strict', 'loose']:
        for aridity in ['arid', 'humid']:
            df_TRENDY_1 = load_datasets.load_TRENDY_model_gdf(variable=Var1, unit='mean_of_variable_model_category', arid_type=aridity_limit+'limit_'+aridity, model_category=model_category, start_year=start_year, end_year=end_year)
            df_TRENDY_2 = load_datasets.load_TRENDY_model_gdf(variable=Var2, unit='mean_of_variable_model_category', arid_type=aridity_limit+'limit_'+aridity, model_category=model_category, start_year=start_year, end_year=end_year)
            df_TRENDY_3 = load_datasets.load_TRENDY_model_gdf(variable=Var3, unit='mean_of_variable_model_category', arid_type=aridity_limit+'limit_'+aridity, model_category=model_category, start_year=start_year, end_year=end_year)
            df_TRENDY_4 = load_datasets.load_TRENDY_model_gdf(variable=Var4, unit='mean_of_variable_model_category', arid_type=aridity_limit+'limit_'+aridity, model_category=model_category, start_year=start_year, end_year=end_year)
            df_TRENDY_5 = load_datasets.load_TRENDY_model_gdf(variable=Var5, unit='mean_of_variable_model_category', arid_type=aridity_limit+'limit_'+aridity, model_category=model_category, start_year=start_year, end_year=end_year)
            df_TRENDY_6 = load_datasets.load_TRENDY_model_gdf(variable=Var6, unit='mean_of_variable_model_category', arid_type=aridity_limit+'limit_'+aridity, model_category=model_category, start_year=start_year, end_year=end_year)
            
            column_TM5 = 'CO2_NEE_fire_flux_monthly_TgC_per_subregion'
            column_FXC = 'NBPtot'
            df_TM5_IS_GOSAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='mean_of_IS_ACOS_and_IS_RT', region=aridity_limit+'limit_'+aridity, unit='per_subregion',  start_year=start_year, end_year=end_year)
            df_FXC = load_datasets.load_FLUXCOM_model_gdf(variable='NBP', unit='per_subregion', region=aridity_limit+'limit_'+aridity, start_year=start_year, end_year=end_year)

            plot_flux_func.plot_timeseries_MSC_general(df_list = [df_TRENDY_1, df_TRENDY_2, df_TRENDY_3, df_TRENDY_4, df_TRENDY_5, df_TRENDY_6, 
                                                                  df_TM5_IS_GOSAT, df_FXC],
                                                    model_list = ['TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TM5', 'FXC'],
                                                    columns_to_plot = ['mean', 'mean', 'mean', 'mean', 'mean', 'mean', column_TM5, column_FXC],
                                                    columns_to_plot_std = ['std', 'std', 'std', 'std', 'std', 'std', column_TM5+'_std', 'NBP_madtot'],
                                                    columns_to_plot_label = ['TRENDY '+Var1, 'TRENDY '+Var2, 'TRENDY '+Var3, 'TRENDY '+Var4, 'TRENDY '+Var5, 'TRENDY '+Var6, 'TM5/IS+GOSAT nbp', 'FXC NEE+GFED'],
                                                    TRENDY_model_category = [TRENDY_model_categories_short[i]],
                                                    norm_timeseries = False,
                                                    color_list = ['black', 'forestgreen', 'sienna', 'olivedrab', 'rosybrown', 'dimgrey', 'firebrick', 'purple'],
                                                    linestyle_list = ['solid', 'solid', 'solid', 'dashed', 'dashed', 'dotted', 'dashed', 'dashed'],
                                                    plot_MSC_only = True,
                                                    start_year = start_year,
                                                    end_year = end_year,
                                                    location_of_legend='center left', #'lower right',
                                                    legend_columns = 1,
                                                    legend_next_to_plot=True,
                                                    region_name=aridity_limit+'_limit_'+aridity+'_SAT',
                                                    plot_title = 'TRENDY '+TRENDY_model_categories_short[i].replace('_',' ')+' & TM5 & FXC',
                                                    savepath = '/home/lartelt/MA/software/12_TRENDY_flux/plot_arid_humid_region/'+aridity_limit+'_limit/Variable_nbp_AND_gpp_AND_ra_AND_rh_AND_ra+rh_AND_ra+rh-gpp_AND_TM5_mean_AND_FXC/plot_mean_model_categories/supplemental/',
                                                    compare_case = 1)
'''



# plot function for precip BAR-PLOT as SUBPLOT of mean of TRENDY model category and multiple variables: nbp & gpp & ra+rh
start_year = 2009
end_year   = 2019

Var1='nbp'
Var1_long=Var1
Var2='gpp'
Var2_long=Var2
Var3='ra+rh'
Var3_long='ratot+rh'
Var4='ra'
Var4_long=Var4
Var5='rh'
Var5_long=Var5
Var6='ra+rh-gpp'
Var6_long='ratot+rhtot-gpp'

TRENDY_model_categories = ['very_good_models', 'ok_models', 'bad_low_amplitude_models', 'bad_high_amplitude_models']
TRENDY_model_categories_short = ['vg_models', 'ok_models', 'bla_models', 'bha_models']

for i,model_category in enumerate(TRENDY_model_categories):
    for aridity_limit in ['strict', 'loose']:
        for aridity in ['arid', 'humid']:
            df_TRENDY_1 = load_datasets.load_TRENDY_model_gdf(variable=Var1, unit='mean_of_variable_model_category', arid_type=aridity_limit+'limit_'+aridity, model_category=model_category, start_year=start_year, end_year=end_year)
            df_TRENDY_2 = load_datasets.load_TRENDY_model_gdf(variable=Var2, unit='mean_of_variable_model_category', arid_type=aridity_limit+'limit_'+aridity, model_category=model_category, start_year=start_year, end_year=end_year)
            df_TRENDY_3 = load_datasets.load_TRENDY_model_gdf(variable=Var3, unit='mean_of_variable_model_category', arid_type=aridity_limit+'limit_'+aridity, model_category=model_category, start_year=start_year, end_year=end_year)
            df_TRENDY_4 = load_datasets.load_TRENDY_model_gdf(variable=Var4, unit='mean_of_variable_model_category', arid_type=aridity_limit+'limit_'+aridity, model_category=model_category, start_year=start_year, end_year=end_year)
            df_TRENDY_5 = load_datasets.load_TRENDY_model_gdf(variable=Var5, unit='mean_of_variable_model_category', arid_type=aridity_limit+'limit_'+aridity, model_category=model_category, start_year=start_year, end_year=end_year)
            df_TRENDY_6 = load_datasets.load_TRENDY_model_gdf(variable=Var6, unit='mean_of_variable_model_category', arid_type=aridity_limit+'limit_'+aridity, model_category=model_category, start_year=start_year, end_year=end_year)
            
            column_TM5 = 'CO2_NEE_fire_flux_monthly_TgC_per_subregion'
            column_FXC = 'NBPtot'
            df_TM5_IS_GOSAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='mean_of_IS_ACOS_and_IS_RT', region=aridity_limit+'limit_'+aridity, unit='per_subregion',  start_year=start_year, end_year=end_year)
            df_FXC = load_datasets.load_FLUXCOM_model_gdf(variable='NBP', unit='per_subregion', region=aridity_limit+'limit_'+aridity, start_year=start_year, end_year=end_year)

            df_ERA5_precip = load_datasets.load_ERA5_precip_gdf(region='SAT_'+aridity_limit+'_limit_'+aridity, unit='per_subregion', resolution=1., start_year=start_year, end_year=end_year)

            plot_flux_func.plot_MSC_with_bar_subplot(df_list = [df_TRENDY_1, df_TRENDY_2, df_TRENDY_3, df_TRENDY_4, df_TRENDY_5, df_TRENDY_6, 
                                                                  df_TM5_IS_GOSAT, df_FXC],
                                                    model_list = ['TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TM5', 'FXC'],
                                                    columns_to_plot = ['mean', 'mean', 'mean', 'mean', 'mean', 'mean', column_TM5, column_FXC],
                                                    columns_to_plot_std = ['std', 'std', 'std', 'std', 'std', 'std', column_TM5+'_std', 'NBP_madtot'],
                                                    columns_to_plot_label = ['TRENDY '+Var1, 'TRENDY '+Var2, 'TRENDY '+Var3, 'TRENDY '+Var4, 'TRENDY '+Var5, 'TRENDY '+Var6, 'TM5-4DVar/IS+GOSAT nbp', 'FXC NEE+GFED'],
                                                    TRENDY_model_category = [TRENDY_model_categories_short[i]],
                                                    color_list = ['black', 'forestgreen', 'sienna', 'olivedrab', 'rosybrown', 'dimgrey', 'firebrick', 'purple'],
                                                    linestyle_list = ['solid', 'solid', 'solid', 'dashed', 'dashed', 'dotted', 'dashed', 'dashed'],
                                                    df_for_bar_plot=[df_ERA5_precip],
                                                    column_to_plot_bar_plot=['tp_mean'],
                                                    columns_to_plot_label_bar_plot=['ERA5 precip'],
                                                    start_year = start_year,
                                                    end_year = end_year,
                                                    location_of_legend='upper left', #'center left',
                                                    legend_columns = 2,
                                                    legend_next_to_plot=False,
                                                    region_name=aridity_limit+'_limit_'+aridity+'_SAT',
                                                    plot_title = 'TRENDY '+TRENDY_model_categories_short[i].replace('_',' ')+' & TM5 & FXC',
                                                    savepath = '/home/lartelt/MA/software/12_TRENDY_flux/plot_arid_humid_region/'+aridity_limit+'_limit/Variable_nbp_AND_gpp_AND_ra_AND_rh_AND_ra+rh_AND_ra+rh-gpp_AND_TM5_mean_AND_FXC/plot_mean_model_categories/supplemental/',
                                                    compare_case = 1)


