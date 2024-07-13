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

TrendyModels = ['CABLE-POP', 'CLASSIC', 'DLEM', 'IBIS', 'ISAM', 'JSBACH', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'ORCHIDEE-CNP', 'ORCHIDEEv3', 'CLM5.0',
                'ISBA-CTRIP', 'JULES-ES-1p0', 'LPJ', 'SDGVM', 'VISIT', 'YIBs']
Trendy_model_colors = {'CABLE-POP':'r', 'CLASSIC':'g', 'DLEM':'b', 'IBIS':'c', 'ISAM':'m', 'JSBACH':'gold', 'LPX-Bern':'orange', 'OCN':'purple', 
                       'ORCHIDEE':'brown', 'ORCHIDEE-CNP':'pink', 'ORCHIDEEv3':'olive', 'CLM5.0':'cyan', 'ISBA-CTRIP':'springgreen', 'JULES-ES-1p0':'teal',
                       'LPJ':'crimson', 'SDGVM':'darkgreen', 'VISIT':'darkblue', 'YIBs':'darkcyan'}

# rewrite the dictionary "Trendy_model_colors" to have the colors ['limegreen', 'darkolivegreen', 'orangered', 'mediumvioletred'] for the model categories
Trendy_model_colors_per_category = {'OCN':'limegreen', 'CLASSIC':'limegreen', 
                       'ISBA-CTRIP':'darkolivegreen', 'ISAM':'darkolivegreen', 'YIBs':'darkolivegreen', 'ORCHIDEE':'darkolivegreen',
                       'ORCHIDEE-CNP':'orangered', 'LPJ':'orangered', 'JULES-ES-1p0':'orangered', 'JSBACH':'orangered', 'IBIS':'orangered',
                       'SDGVM':'sienna', 'LPX-Bern':'sienna', 'CLM5.0':'sienna', 'ORCHIDEEv3':'sienna', 'VISIT':'sienna'}

Trendy_model_linestyle_per_category = {'OCN':'solid', 'CLASSIC':'dashed',
                                       'ISBA-CTRIP':'solid', 'ISAM':'dashed', 'YIBs':'dotted', 'ORCHIDEE':'dashdot',
                                        'ORCHIDEE-CNP':'solid', 'LPJ':'dashed', 'JULES-ES-1p0':'dotted', 'JSBACH':'dashdot', 'IBIS':'dashdot',
                                        'SDGVM':'solid', 'LPX-Bern':'dashed', 'CLM5.0':'dotted', 'ORCHIDEEv3':'dashdot', 'VISIT':'dashdot'}

# NEW model categories
models_very_good_list = ['CLASSIC', 'OCN']
models_ok_list = ['ISBA-CTRIP', 'ISAM', 'YIBs', 'ORCHIDEE']
models_bad_low_amplitude_new = ['ORCHIDEE-CNP', 'LPJ', 'JULES-ES-1p0', 'JSBACH', 'IBIS']
models_bad_low_amplitude_new_for_ra_rh_mean = ['LPJ', 'JSBACH', 'IBIS'] # all models containing respiration & without ORCHIDEE-CNP because it is very bad for respiration
models_bad_high_amplitude_new = ['SDGVM', 'VISIT', 'LPX-Bern', 'CLM5.0', 'ORCHIDEEv3']
models_bad_high_amplitude_new_for_npp_mean = ['SDGVM', 'LPX-Bern', 'CLM5.0', 'ORCHIDEEv3'] # all models except VISIT because this models does not have npp

'''# OLD model categories
models_good_list = ['ORCHIDEE', 'OCN', 'ISBA-CTRIP', 'ISAM', 'CLASSIC']
#models_bad_list = ['SDGVM', 'ORCHIDEE-CNP', 'LPJ', 'JULES-ES-1p0', 'JSBACH', 'IBIS']
#models_bad_high_amplitude = ['SDGVM', 'LPX-Bern', 'CLM5.0']
models_bad_high_amplitude = ['SDGVM', 'YIBs', 'VISIT', 'ORCHIDEEv3', 'LPX-Bern', 'CLM5.0']
models_bad_high_amplitude_for_mean_npp = ['SDGVM', 'YIBs', 'ORCHIDEEv3', 'LPX-Bern', 'CLM5.0'] # all models except VISIT because this models does not have npp
#models_bad_low_amplitude = ['ORCHIDEE-CNP', 'LPJ', 'JULES-ES-1p0', 'JSBACH', 'IBIS']
models_bad_low_amplitude = ['ORCHIDEE-CNP', 'LPJ', 'JULES-ES-1p0', 'JSBACH', 'IBIS']
models_bad_low_amplitude_ra_rh = ['ORCHIDEE-CNP', 'LPJ', 'JSBACH', 'IBIS'] # all models containing respiration
models_bad_low_amplitude_for_mean = ['LPJ', 'JSBACH', 'IBIS'] # all models containing GOOD quality respiration
#models_to_plot_for_2016_peak = ['YIBs', 'VISIT', 'ORCHIDEEv3']
'''

#Variables = ['gpp', 'nbp', 'npp', 'ra', 'rh', 'fFire']
#Var = 'nbp' # gpp, respiration, fFire
Var = 'nbp'
#Var = 'ra+rh-gpp'
#Var = 'nbp-(ra+rh)+gpp'
#Var = 'fFire'

'''
gdf_list=[]
variable_to_plot_list=[]
for model in TrendyModels:
    try:
        gdf = load_datasets.load_TRENDY_model_gdf(model=model, variable=Var, unit='TgC_per_gridcell_per_month')
        gdf_list.append(gdf)
        variable_to_plot_list.append(Var+'tot_TgC_month')
    except:
        print('Model '+model+' does not have this variable!')
        continue

gdf_list.append(load_datasets.load_TRENDY_model_gdf(model='CABLE-POP', variable=Var, unit='mean_of_variable'))
variable_to_plot_list.append('mean')
plot_flux_func.plotFluxTimeseries_and_MSC_general(gdf_list=gdf_list,
                                                  variable_to_plot_list=variable_to_plot_list,
                                                  model_assimilation_list=TrendyModels,
                                                  what_is_plotted='TRENDY_ens_mean_')
'''
'''# plot only timeseries
df_TM5_IS_ACOS = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_ACOS', region='SAT', unit='per_subregion')
df_TM5_IS_RT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_RT', region='SAT', unit='per_subregion')
df_MIP_ens = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_ens', assimilated='IS_OCO2', end_year=2019, region='SAT', unit='per_subregion')
df_TM5_IS_GOSAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='mean_of_IS_ACOS_and_IS_RT', region='SAT', unit='per_subregion', end_year=2018)

# plot all TRENDY models in one plot incl. mean & std
plot_flux_func.plot_timeseries_of_TRENDY_fluxes(df=load_datasets.load_TRENDY_model_gdf(model='CABLE-POP', variable=Var, unit='mean_of_variable', start_year=2009, end_year=2018),
                                                df_TM5_ACOS=df_TM5_IS_GOSAT,
                                                df_TM5_RT=df_TM5_IS_RT,
                                                df_MIP_ens=df_MIP_ens,
                                                what_is_plotted_values='mean',
                                                what_is_plotted='nbp', 
                                                color_list=None, 
                                                color_dict=Trendy_model_colors_per_category,
                                                linestyle_dict=Trendy_model_linestyle_per_category,
                                                region_name='SAT', 
                                                savepath='/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_only/NEW_',
                                                compare_case=4)
'''

'''# plot one TRENDY model's nbp together with TM5/IS+ACOS & TM5/IS+RT
df_TM5_IS_ACOS = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_ACOS', region='SAT', unit='per_subregion')
df_TM5_IS_RT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_RT', region='SAT', unit='per_subregion')
df_MIP_ens = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_ens', assimilated='IS_OCO2', end_year=2019, region='SAT', unit='per_subregion')

df_TRENDY = load_datasets.load_TRENDY_model_gdf(model='CABLE-POP', variable=Var, unit='mean_of_variable', start_year=2009)

for i, column in enumerate(df_TRENDY.columns):
            if column=='MonthDate' or column=='Year' or column=='Month' or column=='mean' or column=='std' or column=='count':
                print('column = '+column)
                continue
            else:
                df_TRENDY_modified = df_TRENDY[['MonthDate', 'Month', column]]
                plot_flux_func.plot_timeseries_of_TRENDY_fluxes(df=df_TRENDY_modified,
                                                                what_is_plotted_values=column,
                                                                what_is_plotted=column[:3],
                                                                df_TM5_ACOS=df_TM5_IS_ACOS,
                                                                df_TM5_RT=df_TM5_IS_RT,
                                                                df_MIP_ens=df_MIP_ens,
                                                                color_list=color_list, region_name='SAT', 
                                                                savepath='/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_only/single_TRENDY_model_compared_to_TM5_&_MIP/',
                                                                compare_case=3)
                del df_TRENDY_modified
'''

'''# plot one TRENDY model's nbp together with TM5/IS+ACOS & TM5/IS+RT timeseries & MSC
start_year = 2009 #2009, 2015
end_year = 2018   #2014, 2019
df_TM5_IS_ACOS = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_ACOS', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
df_TM5_IS_RT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_RT', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
df_MIP_ens = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_ens', assimilated='IS_OCO2', end_year=end_year, region='SAT', unit='per_subregion')

df_TRENDY = load_datasets.load_TRENDY_model_gdf(model='CABLE-POP', variable=Var, unit='mean_of_variable', start_year=start_year, end_year=end_year)

for i, column in enumerate(df_TRENDY.columns):
    if column=='MonthDate' or column=='Year' or column=='Month' or column=='mean' or column=='std' or column=='count':
        print('column = '+column)
        continue
    else:
        df_TRENDY_modified = df_TRENDY[['MonthDate', 'Month', column]]
        #print(df_TRENDY_modified)
        print(column)
        plot_flux_func.plot_timeseries_MSC_of_TRENDY_vs_TM5_MIP_nbp_fluxes(df=df_TRENDY_modified,
                                                            column_to_plot=column,
                                                            df_TM5_ACOS=df_TM5_IS_ACOS,
                                                            df_TM5_RT=df_TM5_IS_RT,
                                                            df_MIP_ens=df_MIP_ens,
                                                            start_year=start_year, end_year=end_year,
                                                            region_name='SAT', 
                                                            savepath='/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_and_MSC/Variable_'+Var+'/',
                                                            compare_case=2)
        del df_TRENDY_modified
'''


'''# MAIN: plot only some TRENDY modelled nbp/gpp/... timeseries & MSC
compare_case=1
if compare_case==1:
    start_year = 2009
    end_year = 2019 #2019
elif compare_case==2:
    start_year = 2009 #2009, 2015
    end_year = 2018   #2014, 2019

df_TM5_IS_ACOS = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_ACOS', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
df_TM5_IS_RT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_RT', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
#df_MIP_ens = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_ens', assimilated='IS_OCO2', end_year=2019, region='SAT', unit='per_subregion')

df_TRENDY = load_datasets.load_TRENDY_model_gdf(variable=Var, unit='mean_of_variable', start_year=start_year, end_year=end_year)
df_GFED = load_datasets.load_GFED_model_gdf(region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)

plot_flux_func.plot_timeseries_MSC_of_TRENDY_variable_fluxes(df_TRENDY=df_TRENDY,
                                                    models_to_plot=models_bad_high_amplitude_new,
                                                    variable_plotted='bad_high_amplitude_new_'+Var,
                                                    plot_TM5_ACOS=False,
                                                    df_TM5_ACOS=df_TM5_IS_ACOS,
                                                    df_TM5_RT=df_TM5_IS_RT,
                                                    plot_GFED=False,
                                                    df_GFED=df_GFED,
                                                    color_dict=Trendy_model_colors,
                                                    start_year=start_year, end_year=end_year,
                                                    region_name='SAT', 
                                                    savepath='/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_and_MSC/Variable_'+Var+'/',
                                                    compare_case=compare_case)

plot_flux_func.plot_timeseries_MSC_of_TRENDY_variable_fluxes(df_TRENDY=df_TRENDY,
                                                    models_to_plot=models_bad_low_amplitude_new,
                                                    variable_plotted='bad_low_amplitude_new_'+Var,
                                                    plot_TM5_ACOS=False,
                                                    df_TM5_ACOS=df_TM5_IS_ACOS,
                                                    df_TM5_RT=df_TM5_IS_RT,
                                                    plot_GFED=False,
                                                    df_GFED=df_GFED,
                                                    color_dict=Trendy_model_colors,
                                                    start_year=start_year, end_year=end_year,
                                                    region_name='SAT', 
                                                    savepath='/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_and_MSC/Variable_'+Var+'/',
                                                    compare_case=compare_case)

plot_flux_func.plot_timeseries_MSC_of_TRENDY_variable_fluxes(df_TRENDY=df_TRENDY,
                                                    models_to_plot=models_ok_list,
                                                    variable_plotted='ok_models_'+Var,
                                                    plot_TM5_ACOS=False,
                                                    df_TM5_ACOS=df_TM5_IS_ACOS,
                                                    df_TM5_RT=df_TM5_IS_RT,
                                                    plot_GFED=False,
                                                    df_GFED=df_GFED,
                                                    color_dict=Trendy_model_colors,
                                                    start_year=start_year, end_year=end_year,
                                                    region_name='SAT', 
                                                    savepath='/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_and_MSC/Variable_'+Var+'/',
                                                    compare_case=compare_case)

plot_flux_func.plot_timeseries_MSC_of_TRENDY_variable_fluxes(df_TRENDY=df_TRENDY,
                                                    models_to_plot=models_very_good_list,
                                                    variable_plotted='very_good_models_'+Var,
                                                    plot_TM5_ACOS=False,
                                                    df_TM5_ACOS=df_TM5_IS_ACOS,
                                                    df_TM5_RT=df_TM5_IS_RT,
                                                    plot_GFED=False,
                                                    df_GFED=df_GFED,
                                                    color_dict=Trendy_model_colors,
                                                    start_year=start_year, end_year=end_year,
                                                    region_name='SAT', 
                                                    savepath='/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_and_MSC/Variable_'+Var+'/',
                                                    compare_case=compare_case)
'''


'''# plot sum of specified TRENDY variables ; timeseries & MSC
Var1='ra'
Var2='rh'
start_year = 2015 #2009, 2015
end_year = 2019   #2014, 2019
compare_case = 2

df_TRENDY_var1 = load_datasets.load_TRENDY_model_gdf(variable=Var1, unit='mean_of_variable', start_year=start_year, end_year=end_year)
df_TRENDY_var2 = load_datasets.load_TRENDY_model_gdf(variable=Var2, unit='mean_of_variable', start_year=start_year, end_year=end_year)

plot_flux_func.plot_timeseries_MSC_of_TRENDY_sum_of_two_variables_fluxes(df_TRENDY_var1=df_TRENDY_var1,
                                                                         df_TRENDY_var2=df_TRENDY_var2,
                                                                         models_to_plot=models_good_list,
                                                                         model_catagory='good_models',
                                                                         var1=Var1,
                                                                         var2=Var2,
                                                                         color_dict=Trendy_model_colors,
                                                                         start_year=start_year, end_year=end_year,
                                                                         savepath='/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_and_MSC/Variable_'+Var1+'+'+Var2+'/',
                                                                         compare_case=compare_case)

plot_flux_func.plot_timeseries_MSC_of_TRENDY_sum_of_two_variables_fluxes(df_TRENDY_var1=df_TRENDY_var1,
                                                                         df_TRENDY_var2=df_TRENDY_var2,
                                                                         models_to_plot=models_bad_high_amplitude,
                                                                         model_catagory='bad_high_amplitude',
                                                                         var1=Var1,
                                                                         var2=Var2,
                                                                         color_dict=Trendy_model_colors,
                                                                         start_year=start_year, end_year=end_year,
                                                                         savepath='/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_and_MSC/Variable_'+Var1+'+'+Var2+'/',
                                                                         compare_case=compare_case)

plot_flux_func.plot_timeseries_MSC_of_TRENDY_sum_of_two_variables_fluxes(df_TRENDY_var1=df_TRENDY_var1,
                                                                         df_TRENDY_var2=df_TRENDY_var2,
                                                                         models_to_plot=models_bad_low_amplitude_ra_rh,
                                                                         model_catagory='bad_low_amplitude',
                                                                         var1=Var1,
                                                                         var2=Var2,
                                                                         color_dict=Trendy_model_colors,
                                                                         start_year=start_year, end_year=end_year,
                                                                         savepath='/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_and_MSC/Variable_'+Var1+'+'+Var2+'/',
                                                                         compare_case=compare_case)
'''


'''# plot bar chart of sum of yearly fluxes for specified TRENDY variables
Var1='nbp'
df_trendy_fluxes = load_datasets.load_TRENDY_model_gdf(variable=Var1, unit='mean_of_variable', start_year=2009)
#df_trendy_fluxes['year'] = df_trendy_fluxes.apply(lambda x: x['MonthDate'].year, axis=1)
Sum_Fluxes_ACOS_IS_region = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_ACOS', start_year=2009, region='SAT', unit='per_subregion')
Sum_Fluxes_ACOS_IS_region = Sum_Fluxes_ACOS_IS_region.groupby(['year'])[['CO2_NEE_fire_flux_monthly_TgC_per_subregion']].sum().reset_index()

plot_flux_func.plotSumOfFluxesBarChart(df_trendy_fluxes = df_trendy_fluxes,
                                       df_TM5_ACOS_sum_fluxes=Sum_Fluxes_ACOS_IS_region,
                                       models_to_plot = models_bad_low_amplitude,
                                       model_catagory = 'bad_low_amplitude',
                                       variable_plotted = Var1,
                                       color_dict = Trendy_model_colors,
                                       compare_case=1, region_name='SAT', 
                                       savepath='/home/lartelt/MA/software/12_TRENDY_flux/plot_bar-charts/')
'''


'''# plot MSC only before & after 2015 in one plot
df_TRENDY_1 = load_datasets.load_TRENDY_model_gdf(model='CABLE-POP', variable=Var, unit='mean_of_variable', start_year=2009, end_year=2014)
df_TRENDY_2 = load_datasets.load_TRENDY_model_gdf(model='CABLE-POP', variable=Var, unit='mean_of_variable', start_year=2015, end_year=2019)

for i, column in enumerate(df_TRENDY_1.columns):
            if column=='MonthDate' or column=='Year' or column=='Month' or column=='mean' or column=='std' or column=='count':
                print('column = '+column)
                continue
            else:
                df_TRENDY_modified_1 = df_TRENDY_1[['MonthDate', 'Month', column]]
                df_TRENDY_modified_2 = df_TRENDY_2[['MonthDate', 'Month', column]]
                print(column)
                plot_flux_func.plot_MSC_of_two_different_TRENDY_models_but_same_flux(df_TRENDY_1 = df_TRENDY_modified_1,
                                                                                     df_TRENDY_2 = df_TRENDY_modified_2,
                                                                                     column_to_plot = column,
                                                                                     list_what_input = ['2009-2014', '2015-2019'],
                                                                                     savepath='/home/lartelt/MA/software/12_TRENDY_flux/plot_MSC_only/Variable_'+Var+'/',
                                                                                     compare_case=1)
'''


'''# MAIN: plot MSC for the MEAN of specific TRENDY models & plot the means of different variables in one plot
Var1 = 'gpp'
Var2 = 'ra+rh'
Var3 = 'ra'
Var4 = 'rh'

df_TRENDY_1 = load_datasets.load_TRENDY_model_gdf(model='CABLE-POP', variable=Var1, unit='mean_of_variable', start_year=2009, end_year=2019)
df_TRENDY_2 = load_datasets.load_TRENDY_model_gdf(model='CABLE-POP', variable=Var2, unit='mean_of_variable', start_year=2009, end_year=2019)
df_TRENDY_3 = load_datasets.load_TRENDY_model_gdf(model='CABLE-POP', variable=Var3, unit='mean_of_variable', start_year=2009, end_year=2019)
df_TRENDY_4 = load_datasets.load_TRENDY_model_gdf(model='CABLE-POP', variable=Var4, unit='mean_of_variable', start_year=2009, end_year=2019)

#for model in models_bad_high_amplitude:
plot_flux_func.plot_MSC_of_mean_of_different_variables(TRENDY_list=[df_TRENDY_1, df_TRENDY_2, df_TRENDY_3, df_TRENDY_4],
                                                    model_list_for_mean=models_ok_list,
                                                    model_catagory='models_ok',
                                                    variables_to_plot_list=[Var1, Var2, Var3, Var4],
                                                    set_ax_limit=True,
                                                    error_type='std_of_models',
                                                    color_list=['forestgreen', 'dimgrey', 'olivedrab', 'saddlebrown'],
                                                    linestyle_list=['solid', 'dashed', 'solid', 'solid'],
                                                    compare_case=1,
                                                    savepath='/home/lartelt/MA/software/12_TRENDY_flux/plot_MSC_only/')
'''


'''# MAIN: plot timeseries & MSC for the MEAN of specific TRENDY models & plot the means of different variables in one plot
#Var1 = 'nbp'

compare_case=2
if compare_case==1:
    start_year = 2009
    end_year = 2019 #2019
elif compare_case==2:
    start_year = 2009 #2009, 2015
    end_year = 2018   #2014, 2019

#df_TM5_IS_ACOS = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_ACOS', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
#df_TM5_IS_RT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_RT', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
df_TM5_IS_GOSAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='mean_of_IS_ACOS_and_IS_RT', region='SAT', unit='per_subregion',  start_year=start_year, end_year=end_year)

df_FXC = load_datasets.load_FLUXCOM_model_gdf(variable='NBP', unit='per_subregion', region='SAT', start_year=start_year, end_year=end_year)

for var in ['nbp']: # use normal model lists & set plot_TM5_ACOS=True
#for var in ['gpp']: # use normal model lists
#for var in ['npp']:  # use models_bad_high_amplitude_new_for_npp_mean but normal lists for other catagories
#for var in ['ra', 'rh', 'ra+rh', 'ra+rh-gpp', 'ra+rh+gpp', 'nbp-(ra+rh)+gpp']:  # use models_bad_low_amplitude_new_for_ra_rh_mean but normal lists for other catagories
    df_TRENDY_1 = load_datasets.load_TRENDY_model_gdf(model='CABLE-POP', variable=var, unit='mean_of_variable', start_year=start_year, end_year=end_year)
    print(var)
    plot_flux_func.plot_timeseries_MSC_of_mean_of_different_model_categories_NEW(
                                                        df_TRENDY=df_TRENDY_1,
                                                        very_good_model_list=models_very_good_list,
                                                        ok_model_list=models_ok_list,
                                                        bad_high_ampl_model_list=models_bad_high_amplitude_new,
                                                        bad_low_ampl_model_list=models_bad_low_amplitude_new,
                                                        variable_to_plot=var,
                                                        plot_MSC_only=False,
                                                        plot_TM5_ACOS_flux=True,
                                                        #df_TM5_ACOS=df_TM5_IS_ACOS,
                                                        #df_TM5_RT=df_TM5_IS_RT,
                                                        df_TM5_IS_GOSAT=df_TM5_IS_GOSAT,
                                                        plot_FXC=False,
                                                        df_FXC=df_FXC,
                                                        set_ax_limit=False,
                                                        color_list=['limegreen', 'darkolivegreen', 'orangered', 'mediumvioletred'],#darkmagenta
                                                        linestyle_list=['solid', 'solid', 'solid', 'solid'],
                                                        start_year=start_year, end_year=end_year,
                                                        savepath='/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_and_MSC/Mean_of_good_and_bad_models_with_MSC_only/',
                                                        compare_case=compare_case)
'''


'''# general plot function for mean of TRENDY model category: gpp or ra+rh
start_year = 2009
end_year   = 2018

TRENDY_model_category1 = 'very_good_models'
TRENDY_model_category1_short = 'vg_models'
TRENDY_model_category2 = 'ok_models'
TRENDY_model_category2_short = 'ok_models'
TRENDY_model_category3 = 'bad_low_ampl_models'
TRENDY_model_category3_short = 'bla_models'
TRENDY_model_category4 = 'bad_high_ampl_models'
TRENDY_model_category4_short = 'bha_models'

Var_TRENDY='rh'

df_TRENDY_1 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY, unit='mean_of_variable_model_category', model_category=TRENDY_model_category1, start_year=start_year, end_year=end_year)
df_TRENDY_2 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY, unit='mean_of_variable_model_category', model_category=TRENDY_model_category2, start_year=start_year, end_year=end_year)
df_TRENDY_3 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY, unit='mean_of_variable_model_category', model_category=TRENDY_model_category3, start_year=start_year, end_year=end_year)
df_TRENDY_4 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY, unit='mean_of_variable_model_category', model_category=TRENDY_model_category4, start_year=start_year, end_year=end_year)

plot_flux_func.plot_timeseries_MSC_general(df_list = [df_TRENDY_1, df_TRENDY_2, df_TRENDY_3, df_TRENDY_4],
                                           model_list = ['TRENDY', 'TRENDY', 'TRENDY', 'TRENDY'],
                                           columns_to_plot = ['mean', 'mean', 'mean', 'mean'],
                                           columns_to_plot_std = ['std', 'std', 'std', 'std'],
                                           columns_to_plot_label = ['mean '+Var_TRENDY+' vg models', 'mean '+Var_TRENDY+' ok models', 'mean '+Var_TRENDY+' bad low ampl', 'mean '+Var_TRENDY+' bad high ampl'],
                                           TRENDY_model_category = [TRENDY_model_category1_short, TRENDY_model_category2_short, TRENDY_model_category3_short, TRENDY_model_category4_short],
                                           norm_timeseries = True,
                                           color_list = ['limegreen', 'darkgreen', 'orangered', 'purple'],
                                           linestyle_list = ['solid', 'solid', 'solid', 'solid'],
                                           plot_MSC_only = True,
                                           start_year = start_year,
                                           end_year = end_year,
                                           plot_title = 'TRENDY '+Var_TRENDY+' mean flux of good & bad models',
                                           savepath = '/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_and_MSC/Mean_of_good_and_bad_models_with_MSC_only/',
                                           compare_case = 2)
'''


'''# general plot function for mean of TRENDY model category: nbp, ra+rh-gpp, fFire, ra+rh-gpp+fFire, nbp-(ra+rh-gpp+fFire)
start_year = 2009
end_year   = 2019

TRENDY_model_category = 'very_good_models'
TRENDY_model_category_short = 'vg_models'
Var_TRENDY1='nbp'
Var_TRENDY2='ra+rh-gpp' # = NEP=NEE
Var_TRENDY3='fFire'
Var_TRENDY4='ra+rh-gpp+fFire' # = nbp_calc
Var_TRENDY5='nbp-(ra+rh-gpp+fFire)' # = rest

df_TRENDY_1 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY1, unit='mean_of_variable_model_category', model_category=TRENDY_model_category, start_year=start_year, end_year=end_year)
df_TRENDY_2 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY2, unit='mean_of_variable_model_category', model_category=TRENDY_model_category, start_year=start_year, end_year=end_year)
df_TRENDY_3 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY3, unit='mean_of_variable_model_category', model_category=TRENDY_model_category, start_year=start_year, end_year=end_year)
df_TRENDY_4 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY4, unit='mean_of_variable_model_category', model_category=TRENDY_model_category, start_year=start_year, end_year=end_year)
df_TRENDY_5 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY5, unit='mean_of_variable_model_category', model_category=TRENDY_model_category, start_year=start_year, end_year=end_year)

plot_flux_func.plot_timeseries_MSC_general(df_list = [df_TRENDY_1, df_TRENDY_2, df_TRENDY_3, df_TRENDY_4, df_TRENDY_5],
                                           model_list = ['TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TRENDY'],
                                           columns_to_plot = ['mean', 'mean', 'mean', 'mean', 'mean'],
                                           columns_to_plot_std = ['std', 'std', 'std', 'std', 'std'],
                                           columns_to_plot_label = [Var_TRENDY1, Var_TRENDY2, Var_TRENDY3, Var_TRENDY4, Var_TRENDY5],
                                           TRENDY_model_category = [TRENDY_model_category],
                                           norm_timeseries = False,
                                           color_list = ['forestgreen', 'slateblue', 'firebrick', 'darkolivegreen', 'black'],
                                           linestyle_list = ['solid', 'solid', 'solid', 'dashed', 'dashed'],
                                           plot_MSC_only = True,
                                           start_year = start_year,
                                           end_year = end_year,
                                           plot_title = 'TRENDY '+TRENDY_model_category_short+' fluxes',
                                           savepath = '/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_and_MSC/Variables_nbp_AND_ra+rh-gpp_AND_fFire_AND_ra+rh-gpp+fFire_AND_nbp-(ra+rh-gpp+fFire)/',
                                           compare_case = 1)
'''


'''# general plot function for single TRENDY models: nbp, ra+rh-gpp, fFire, ra+rh-gpp+fFire, nbp-(ra+rh-gpp+fFire)
start_year = 2009
end_year   = 2019

TRENDY_model = 'ORCHIDEE' # 'ISBA-CTRIP', 'ISAM', 'YIBs', 'ORCHIDEE'
Var_TRENDY1='nbp'
Var_TRENDY2='ra+rh-gpp' # = NEP=NEE
Var_TRENDY3='fFire'
Var_TRENDY4='ra+rh-gpp+fFire' # = nbp_calc
Var_TRENDY5='nbp-(ra+rh-gpp+fFire)' # = rest

column1 = 'nbptot_TgC_month_'+TRENDY_model
column2 = 'ratot+rhtot-gpptot_TgC_month_'+TRENDY_model
column3 = 'fFiretot_TgC_month_'+TRENDY_model
column4 = 'ratot+rhtot-gpptot+fFiretot_TgC_month_'+TRENDY_model
column5 = 'nbptot-(ratot+rhtot-gpptot+fFiretot)_TgC_month_'+TRENDY_model

df_TRENDY_1 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY1, unit='mean_of_variable', start_year=start_year, end_year=end_year)
df_TRENDY_2 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY2, unit='mean_of_variable', start_year=start_year, end_year=end_year)
df_TRENDY_3 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY3, unit='mean_of_variable', start_year=start_year, end_year=end_year)
df_TRENDY_4 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY4, unit='mean_of_variable', start_year=start_year, end_year=end_year)
df_TRENDY_5 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY5, unit='mean_of_variable', start_year=start_year, end_year=end_year)

plot_flux_func.plot_timeseries_MSC_general(df_list = [df_TRENDY_1, df_TRENDY_2, df_TRENDY_3, df_TRENDY_4, df_TRENDY_5],
                                           model_list = ['TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TRENDY'],
                                           columns_to_plot = [column1, column2, column3, column4, column5],
                                           columns_to_plot_std = ['', '', '', '', ''],
                                           columns_to_plot_label = [Var_TRENDY1, Var_TRENDY2, Var_TRENDY3, Var_TRENDY4, Var_TRENDY5],
                                           TRENDY_model_category = [TRENDY_model],
                                           norm_timeseries = False,
                                           color_list = ['forestgreen', 'slateblue', 'firebrick', 'darkolivegreen', 'black'],
                                           linestyle_list = ['solid', 'solid', 'solid', 'dashed', 'dashed'],
                                           plot_MSC_only = True,
                                           start_year = start_year,
                                           end_year = end_year,
                                           plot_title = 'TRENDY '+TRENDY_model+' fluxes',
                                           savepath = '/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_and_MSC/Variables_nbp_AND_ra+rh-gpp_AND_fFire_AND_ra+rh-gpp+fFire_AND_nbp-(ra+rh-gpp+fFire)/',
                                           compare_case = 1)
'''


'''# general plot function for single TRENDY models: ra+rh-gpp, ra+rh-gpp+GFED, ra+rh-gpp+fFire, ACOS nbp, ACOS nbp-GFED
start_year = 2009
end_year   = 2018

#TRENDY_models = ['CLASSIC', 'ISBA-CTRIP', 'ISAM', 'YIBs', 'ORCHIDEE']
#TRENDY_models = ['CLASSIC', 'ISBA-CTRIP'] # models with fFire data
TRENDY_models = ['OCN', 'ISAM', 'YIBs', 'ORCHIDEE'] # models without fFire

#resp - gpp, resp - gpp + feuer, und ACOS nbp, und ACOS nbp - gfed
Var_TRENDY1='ra+rh-gpp' # = NEP=NEE
Var_TRENDY2='ra+rh-gpp+GFED'
#Var_TRENDY3='ra+rh-gpp+fFire' # = nbp_calc

for TRENDY_model in TRENDY_models:
    print(TRENDY_model)
    column1 = 'ratot+rhtot-gpptot_TgC_month_'+TRENDY_model
    column2 = 'ratot+rhtot-gpptot+GFED_TgC_month_'+TRENDY_model
    #column3 = 'ratot+rhtot-gpptot+fFiretot_TgC_month_'+TRENDY_model

    df_TRENDY_1 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY1, unit='mean_of_variable', start_year=start_year, end_year=end_year)
    #df_TRENDY_2 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY2, unit='mean_of_variable', start_year=start_year, end_year=end_year)
    #df_TRENDY_3 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY3, unit='mean_of_variable', start_year=start_year, end_year=end_year)

    df_TM5_IS_ACOS = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_ACOS', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
    df_TM5_IS_ACOS_MINUS_GFED = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_ACOS_MINUS_GFED', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
    #df_TM5_IS_RT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_RT', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
    #df_MIP_ens = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_ens', assimilated='IS_OCO2', end_year=end_year, region='SAT', unit='per_subregion')

    plot_flux_func.plot_timeseries_MSC_general(df_list = [df_TRENDY_1, df_TM5_IS_ACOS, df_TM5_IS_ACOS_MINUS_GFED],
                                               model_list = ['TRENDY', 'TM5', 'TM5_MINUS_GFED'],
                                               columns_to_plot = [column1,
                                                                  'CO2_NEE_fire_flux_monthly_TgC_per_subregion', 
                                                                  'CO2_NEE_fire_MINUS_GFED_flux_monthly_TgC_per_subregion'],
                                               columns_to_plot_std = ['', '', '', '', ''],
                                               columns_to_plot_label = [Var_TRENDY1, 'IS+ACOS nbp', 'nbp-GFED'],
                                               TRENDY_model_category = [TRENDY_model],
                                               norm_timeseries = False,
                                               color_list = ['green', 'black', 'royalblue'],
                                               linestyle_list = ['solid', 'solid', 'dashed'],
                                               plot_MSC_only = False,
                                               start_year = start_year,
                                               end_year = end_year,
                                               plot_title = 'TRENDY '+TRENDY_model+' fluxes & TM5/IS+ACOS',
                                               savepath = '/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_and_MSC/Variables_ra+rh-gpp_AND_ra+rh-gpp+fFire_AND_TM5_IS_ACOS_minus_GFED/',
                                               compare_case = 1)
'''



'''# general plot function for mean of TRENDY models: ra+rh-gpp or nbp with ACOS nbp-GFED or ACOS nbp
start_year = 2009
end_year   = 2018

#resp - gpp, resp - gpp + feuer, und ACOS nbp, und ACOS nbp - gfed
Var_TRENDY='ra+rh-gpp' # = NEP
#Var_TRENDY='nbp'

TRENDY_model_category1 = 'very_good_models'
TRENDY_model_category1_short = 'vg_models'
TRENDY_model_category2 = 'ok_models'
TRENDY_model_category2_short = 'ok_models'

df_TRENDY_1 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY, unit='mean_of_variable_model_category', model_category=TRENDY_model_category1, start_year=start_year, end_year=end_year)
df_TRENDY_2 = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY, unit='mean_of_variable_model_category', model_category=TRENDY_model_category2, start_year=start_year, end_year=end_year)

df_TM5_IS_ACOS = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_ACOS', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
df_TM5_IS_ACOS_MINUS_GFED = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_ACOS_MINUS_GFED', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)

plot_flux_func.plot_timeseries_MSC_general(df_list = [df_TRENDY_1, df_TRENDY_2, df_TM5_IS_ACOS_MINUS_GFED],
                                           model_list = ['TRENDY', 'TRENDY', 'TM5_MINUS_GFED'],
                                           columns_to_plot = ['mean', 'mean',
                                                              'CO2_NEE_fire_MINUS_GFED_flux_monthly_TgC_per_subregion'],
                                           columns_to_plot_std = ['std', 'std', ''],
                                           columns_to_plot_label = ['mean '+Var_TRENDY+' vg models', 'mean '+Var_TRENDY+' ok models', 'IS+ACOS nbp-GFED'],
                                           TRENDY_model_category = [TRENDY_model_category1_short, TRENDY_model_category2_short],
                                           norm_timeseries = True,
                                           color_list = ['limegreen', 'darkgreen', 'black'],
                                           linestyle_list = ['solid', 'solid', 'dashed'],
                                           plot_MSC_only = True,
                                           start_year = start_year,
                                           end_year = end_year,
                                           plot_title = 'TRENDY '+Var_TRENDY+' mean flux & TM5/IS+ACOS',
                                           savepath = '/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_and_MSC/Variables_ra+rh-gpp_AND_TM5_IS_ACOS_(minus_GFED)/',
                                           compare_case = 1)
'''


'''# general plot function for single TRENDY models and multiple variables: nbp & gpp & ra+rh
start_year = 2009
end_year   = 2019

#TRENDY_model = 'CLASSIC' # 'ISBA-CTRIP', 'ISAM', 'YIBs', 'ORCHIDEE'
model_list=['CLASSIC', 'OCN', 'ISBA-CTRIP', 'ISAM', 'YIBs', 'ORCHIDEE']

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
        for TRENDY_model in model_list:
            column1 = Var1_long+'tot_TgC_month_'+TRENDY_model
            column2 = Var2_long+'tot_TgC_month_'+TRENDY_model
            column3 = Var3_long+'tot_TgC_month_'+TRENDY_model
            column4 = Var4_long+'tot_TgC_month_'+TRENDY_model
            column5 = Var5_long+'tot_TgC_month_'+TRENDY_model
            column6 = Var6_long+'tot_TgC_month_'+TRENDY_model

            df_TRENDY_1 = load_datasets.load_TRENDY_model_gdf(variable=Var1, unit='mean_of_variable', arid_type=aridity_limit+'limit_'+aridity, start_year=start_year, end_year=end_year)
            df_TRENDY_2 = load_datasets.load_TRENDY_model_gdf(variable=Var2, unit='mean_of_variable', arid_type=aridity_limit+'limit_'+aridity, start_year=start_year, end_year=end_year)
            df_TRENDY_3 = load_datasets.load_TRENDY_model_gdf(variable=Var3, unit='mean_of_variable', arid_type=aridity_limit+'limit_'+aridity, start_year=start_year, end_year=end_year)
            df_TRENDY_4 = load_datasets.load_TRENDY_model_gdf(variable=Var4, unit='mean_of_variable', arid_type=aridity_limit+'limit_'+aridity, start_year=start_year, end_year=end_year)
            df_TRENDY_5 = load_datasets.load_TRENDY_model_gdf(variable=Var5, unit='mean_of_variable', arid_type=aridity_limit+'limit_'+aridity, start_year=start_year, end_year=end_year)
            df_TRENDY_6 = load_datasets.load_TRENDY_model_gdf(variable=Var6, unit='mean_of_variable', arid_type=aridity_limit+'limit_'+aridity, start_year=start_year, end_year=end_year)

            plot_flux_func.plot_timeseries_MSC_general(df_list = [df_TRENDY_1, df_TRENDY_2, df_TRENDY_3, df_TRENDY_4, df_TRENDY_5, df_TRENDY_6],
                                                    model_list = ['TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TRENDY'],
                                                    columns_to_plot = [column1, column2, column3, column4, column5, column6],
                                                    columns_to_plot_std = ['', '', '', '', '', ''],
                                                    columns_to_plot_label = [Var1, Var2, Var3, Var4, Var5, Var6],
                                                    TRENDY_model_category = [TRENDY_model],
                                                    norm_timeseries = False,
                                                    color_list = ['black', 'forestgreen', 'sienna', 'olivedrab', 'rosybrown', 'dimgrey'],
                                                    linestyle_list = ['solid', 'solid', 'solid', 'dashed', 'dashed', 'dotted'],
                                                    plot_MSC_only = False,
                                                    start_year = start_year,
                                                    end_year = end_year,
                                                    #location_of_legend='lower right',
                                                    legend_columns = 3,
                                                    region_name='SAT_'+aridity_limit+'_limit_'+aridity,
                                                    plot_title = 'TRENDY '+TRENDY_model+' fluxes',
                                                    savepath = '/home/lartelt/MA/software/12_TRENDY_flux/plot_arid_humid_region/'+aridity_limit+'_limit/Variable_nbp_AND_gpp_AND_ra_AND_rh_AND_ra+rh_AND_ra+rh-gpp/',
                                                    compare_case = 1)
'''



'''# general plot function for multiple TRENDY models: TRENDY nbp, TM5/IS+GOSAT mean nbp
start_year = 2009
end_year   = 2018

TRENDY_models = models_very_good_list
#TRENDY_models = models_ok_list
#TRENDY_models = models_bad_low_amplitude_new
#TRENDY_models = models_bad_high_amplitude_new
Var_TRENDY1='nbp'

columns=[]
for i,model in enumerate(TRENDY_models):
    #if i ==0:
    #    columns=[Var_TRENDY1+'tot_TgC_month_'+TRENDY_models[i]]
    #else:
    columns.append(Var_TRENDY1+'tot_TgC_month_'+model)
print(columns)

df_TRENDY = load_datasets.load_TRENDY_model_gdf(variable=Var_TRENDY1, unit='mean_of_variable', start_year=start_year, end_year=end_year)
df_TM5_IS_GOSAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='mean_of_IS_ACOS_and_IS_RT', region='SAT', unit='per_subregion',  start_year=start_year, end_year=end_year)

#for the 5 models in each bad category
plot_flux_func.plot_timeseries_MSC_general(df_list = [df_TRENDY, df_TRENDY, df_TRENDY, df_TRENDY, df_TRENDY, df_TM5_IS_GOSAT],
                                            model_list = ['TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TM5'],
                                            columns_to_plot = [columns[0], columns[1], columns[2], columns[3], columns[4],
                                                               'CO2_NEE_fire_flux_monthly_TgC_per_subregion'],
                                            columns_to_plot_std = ['', '', '', '', '', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion_std'],
                                            columns_to_plot_label = [TRENDY_models[0], TRENDY_models[1], TRENDY_models[2], TRENDY_models[3], 
                                                                     TRENDY_models[4], 'TM5-4DVar/IS+GOSAT'],
                                            TRENDY_model_category = ['bad_low_ampl_models'],
                                            norm_timeseries = False,
                                            color_list = [Trendy_model_colors[TRENDY_models[0]], Trendy_model_colors[TRENDY_models[1]],
                                                          Trendy_model_colors[TRENDY_models[2]], Trendy_model_colors[TRENDY_models[3]],
                                                          Trendy_model_colors[TRENDY_models[4]], 'firebrick'],
                                            linestyle_list = ['solid', 'solid', 'solid', 'solid', 'solid', 'dashed'],
                                            plot_MSC_only = False,
                                            start_year = start_year,
                                            end_year = end_year,
                                            plot_title = 'TRENDY bad low amplitude models & TM5/IS+GOSAT nbp',
                                            savepath = '/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_and_MSC/Variable_nbp/2009-2018/new_plots/NEW_',
                                            compare_case = 1)
'''
'''#for the ok model category
plot_flux_func.plot_timeseries_MSC_general(df_list = [df_TRENDY, df_TRENDY, df_TRENDY, df_TRENDY, df_TM5_IS_GOSAT],
                                            model_list = ['TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TM5'],
                                            columns_to_plot = [columns[0], columns[1], columns[2], columns[3],
                                                               'CO2_NEE_fire_flux_monthly_TgC_per_subregion'],
                                            columns_to_plot_std = ['', '', '', '', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion_std'],
                                            columns_to_plot_label = [TRENDY_models[0], TRENDY_models[1], TRENDY_models[2], TRENDY_models[3], 
                                                                     'TM5-4DVar/IS+GOSAT'],
                                            TRENDY_model_category = ['ok_models'],
                                            norm_timeseries = False,
                                            color_list = [Trendy_model_colors[TRENDY_models[0]], Trendy_model_colors[TRENDY_models[1]],
                                                          Trendy_model_colors[TRENDY_models[2]], Trendy_model_colors[TRENDY_models[3]],
                                                          'firebrick'],
                                            linestyle_list = ['solid', 'solid', 'solid', 'solid', 'dashed'],
                                            plot_MSC_only = False,
                                            start_year = start_year,
                                            end_year = end_year,
                                            plot_title = 'TRENDY ok models & TM5/IS+GOSAT nbp',
                                            savepath = '/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_and_MSC/Variable_nbp/2009-2018/new_plots/NEW_',
                                            compare_case = 1)
'''
'''#for the vg model category
plot_flux_func.plot_timeseries_MSC_general(df_list = [df_TRENDY, df_TRENDY, df_TM5_IS_GOSAT],
                                            model_list = ['TRENDY', 'TRENDY', 'TM5'],
                                            columns_to_plot = [columns[0], columns[1],
                                                               'CO2_NEE_fire_flux_monthly_TgC_per_subregion'],
                                            columns_to_plot_std = ['', '', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion_std'],
                                            columns_to_plot_label = [TRENDY_models[0], TRENDY_models[1], 
                                                                     'TM5-4DVar/IS+GOSAT'],
                                            TRENDY_model_category = ['vg_models'],
                                            norm_timeseries = False,
                                            color_list = [Trendy_model_colors[TRENDY_models[0]], Trendy_model_colors[TRENDY_models[1]],
                                                          'firebrick'],
                                            linestyle_list = ['solid', 'solid', 'dashed'],
                                            plot_MSC_only = False,
                                            start_year = start_year,
                                            end_year = end_year,
                                            plot_title = 'TRENDY vg models & TM5/IS+GOSAT nbp',
                                            savepath = '/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_and_MSC/Variable_nbp/2009-2018/new_plots/NEW_',
                                            compare_case = 1)
'''

'''# general plot function for mean of TRENDY model category and multiple variables: nbp & gpp & ra+rh & nee
start_year = 2009
end_year   = 2018

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
    df_TRENDY_1 = load_datasets.load_TRENDY_model_gdf(variable=Var1, unit='mean_of_variable_model_category', model_category=model_category, start_year=start_year, end_year=end_year)
    df_TRENDY_2 = load_datasets.load_TRENDY_model_gdf(variable=Var2, unit='mean_of_variable_model_category', model_category=model_category, start_year=start_year, end_year=end_year)
    df_TRENDY_3 = load_datasets.load_TRENDY_model_gdf(variable=Var3, unit='mean_of_variable_model_category', model_category=model_category, start_year=start_year, end_year=end_year)
    df_TRENDY_4 = load_datasets.load_TRENDY_model_gdf(variable=Var4, unit='mean_of_variable_model_category', model_category=model_category, start_year=start_year, end_year=end_year)
    df_TRENDY_5 = load_datasets.load_TRENDY_model_gdf(variable=Var5, unit='mean_of_variable_model_category', model_category=model_category, start_year=start_year, end_year=end_year)
    df_TRENDY_6 = load_datasets.load_TRENDY_model_gdf(variable=Var6, unit='mean_of_variable_model_category', model_category=model_category, start_year=start_year, end_year=end_year)
    
    column_TM5 = 'CO2_NEE_fire_flux_monthly_TgC_per_subregion'
    column_FXC = 'NBPtot'
    df_TM5_IS_GOSAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='mean_of_IS_ACOS_and_IS_RT', region='SAT', unit='per_subregion',  start_year=start_year, end_year=end_year)
    df_FXC = load_datasets.load_FLUXCOM_model_gdf(variable='NBP', unit='per_subregion', region='SAT', start_year=start_year, end_year=end_year)
    
    plot_flux_func.plot_timeseries_MSC_general(df_list = [df_TRENDY_1, df_TRENDY_2, df_TRENDY_3, df_TRENDY_4, df_TRENDY_5, df_TRENDY_6, df_TM5_IS_GOSAT, df_FXC],
                                            model_list = ['TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TM5', 'FLUXCOM'],
                                            columns_to_plot = ['mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'NBPtot'],
                                            columns_to_plot_std = ['std', 'std', 'std', 'std', 'std', 'std', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion_std', 'NBP_madtot'],
                                            columns_to_plot_label = [Var1, Var2, Var3, Var4, Var5, Var6, 'TM5-4DVar/IS+GOSAT nbp', 'FXC NEE+GFED'],
                                            TRENDY_model_category = [TRENDY_model_categories_short[i]],
                                            norm_timeseries = False,
                                            color_list = ['black', 'forestgreen', 'sienna', 'olivedrab', 'rosybrown', 'dimgrey', 'firebrick', 'purple'],#, 'firebrick', 'purple'],
                                            linestyle_list = ['solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed', 'solid', 'solid'],#, 'dashed', 'dashed'],
                                            plot_MSC_only = True,
                                            start_year = start_year,
                                            end_year = end_year,
                                            location_of_legend='upper left', #'center left',
                                            legend_columns = 2,
                                            legend_next_to_plot=False,
                                            region_name='SAT',
                                            plot_title = 'TRENDY '+TRENDY_model_categories[i].replace('_',' ')+' & TM5 & FXC',
                                            savepath = '/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_and_MSC/Mean_of_good_and_bad_models_with_MSC_only/2009-2018/supplemental/',
                                            compare_case = 1)
'''



# plot bar chart of sum of yearly fluxes for mean TRENDY model categories
Var1='nbp'
start_year=2010
end_year=2018

TRENDY_model_categories = ['very_good_models', 'ok_models', 'bad_low_amplitude_models', 'bad_high_amplitude_models']
TRENDY_model_categories_short = ['vg_models', 'ok_models', 'bla_models', 'bha_models']

df_TRENDY_vg = load_datasets.load_TRENDY_model_gdf(variable=Var1, unit='mean_of_variable_model_category', model_category=TRENDY_model_categories[0], start_year=start_year, end_year=end_year)
df_TRENDY_ok = load_datasets.load_TRENDY_model_gdf(variable=Var1, unit='mean_of_variable_model_category', model_category=TRENDY_model_categories[1], start_year=start_year, end_year=end_year)
df_TRENDY_bla = load_datasets.load_TRENDY_model_gdf(variable=Var1, unit='mean_of_variable_model_category', model_category=TRENDY_model_categories[2], start_year=start_year, end_year=end_year)
df_TRENDY_bha = load_datasets.load_TRENDY_model_gdf(variable=Var1, unit='mean_of_variable_model_category', model_category=TRENDY_model_categories[3], start_year=start_year, end_year=end_year)
#df_trendy_fluxes['year'] = df_trendy_fluxes.apply(lambda x: x['MonthDate'].year, axis=1)

df_TM5_IS_GOSAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='mean_of_IS_ACOS_and_IS_RT', region='SAT', unit='per_subregion',  start_year=start_year, end_year=end_year)
Sum_Fluxes_IS_GOSAT = df_TM5_IS_GOSAT.groupby(['year'])[['CO2_NEE_fire_flux_monthly_TgC_per_subregion']].sum().reset_index()
df_FLUXCOM = load_datasets.load_FLUXCOM_model_gdf(variable='NBP', unit='per_subregion', start_year=start_year, end_year=end_year)
#print(df_FLUXCOM.head())
Sum_Fluxes_FLUXCOM = df_FLUXCOM.groupby(['Year'])[['NBPtot']].sum().reset_index()

plot_flux_func.plotSumOfFluxesBarChart_general(df_trendy_list = [df_TRENDY_vg, df_TRENDY_ok, df_TRENDY_bla, df_TRENDY_bha],
                                               df_TM5_IS_GOSAT_sum_fluxes=Sum_Fluxes_IS_GOSAT,
                                               df_FLUXCOM_sum_fluxes = Sum_Fluxes_FLUXCOM,
                                               variable_plotted = 'nbp',
                                               color_list = ['limegreen', 'darkolivegreen', 'orangered', 'sienna'],
                                               label_list = ['selected strict', 'selected loose', 'other low amplitude', 'other high amplitude'],
                                               compare_case=1, region_name='SAT', 
                                               savepath='/home/lartelt/MA/software/12_TRENDY_flux/plot_bar-charts/supplemental/NEW_')




'''# general plot function for Precip bar plot as subplot of mean of TRENDY model category and multiple variables: nbp & gpp & ra+rh & nee
start_year = 2009
end_year   = 2018

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
    df_TRENDY_1 = load_datasets.load_TRENDY_model_gdf(variable=Var1, unit='mean_of_variable_model_category', model_category=model_category, start_year=start_year, end_year=end_year)
    df_TRENDY_2 = load_datasets.load_TRENDY_model_gdf(variable=Var2, unit='mean_of_variable_model_category', model_category=model_category, start_year=start_year, end_year=end_year)
    df_TRENDY_3 = load_datasets.load_TRENDY_model_gdf(variable=Var3, unit='mean_of_variable_model_category', model_category=model_category, start_year=start_year, end_year=end_year)
    df_TRENDY_4 = load_datasets.load_TRENDY_model_gdf(variable=Var4, unit='mean_of_variable_model_category', model_category=model_category, start_year=start_year, end_year=end_year)
    df_TRENDY_5 = load_datasets.load_TRENDY_model_gdf(variable=Var5, unit='mean_of_variable_model_category', model_category=model_category, start_year=start_year, end_year=end_year)
    df_TRENDY_6 = load_datasets.load_TRENDY_model_gdf(variable=Var6, unit='mean_of_variable_model_category', model_category=model_category, start_year=start_year, end_year=end_year)
    
    column_TM5 = 'CO2_NEE_fire_flux_monthly_TgC_per_subregion'
    column_FXC = 'NBPtot'
    df_TM5_IS_GOSAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='mean_of_IS_ACOS_and_IS_RT', region='SAT', unit='per_subregion',  start_year=start_year, end_year=end_year)
    df_FXC = load_datasets.load_FLUXCOM_model_gdf(variable='NBP', unit='per_subregion', region='SAT', start_year=start_year, end_year=end_year)
    
    df_ERA5_precip = load_datasets.load_ERA5_precip_gdf(region='SAT', unit='per_subregion', resolution=1., start_year=start_year, end_year=end_year)
    #print(df_ERA5_precip.head())
    
    plot_flux_func.plot_MSC_with_bar_subplot(df_list = [df_TRENDY_1, df_TRENDY_2, df_TRENDY_3, df_TRENDY_4, df_TRENDY_5, df_TRENDY_6, df_TM5_IS_GOSAT, df_FXC],
                                            model_list = ['TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TM5', 'FLUXCOM'],
                                            columns_to_plot = ['mean', 'mean', 'mean', 'mean', 'mean', 'mean', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'NBPtot'],
                                            columns_to_plot_std = ['std', 'std', 'std', 'std', 'std', 'std', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion_std', 'NBP_madtot'],
                                            columns_to_plot_label = [Var1, Var2, Var3, Var4, Var5, Var6, 'TM5-4DVar/IS+GOSAT nbp', 'FXC NEE+GFED'],
                                            TRENDY_model_category = [TRENDY_model_categories_short[i]],
                                            color_list = ['black', 'forestgreen', 'sienna', 'olivedrab', 'rosybrown', 'dimgrey', 'firebrick', 'purple'],#, 'firebrick', 'purple'],
                                            linestyle_list = ['solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed', 'solid', 'solid'],#, 'dashed', 'dashed'],
                                            df_for_bar_plot=[df_ERA5_precip],
                                            column_to_plot_bar_plot=['tp_mean'],
                                            columns_to_plot_label_bar_plot=['ERA5 precip'],
                                            start_year = start_year,
                                            end_year = end_year,
                                            location_of_legend='upper left', #'center left',
                                            legend_columns = 2,
                                            legend_next_to_plot=False,
                                            region_name='SAT',
                                            plot_title = 'TRENDY '+TRENDY_model_categories[i].replace('_',' ')+' & TM5 & FXC',
                                            savepath = '/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_and_MSC/Mean_of_good_and_bad_models_with_MSC_only/2009-2018/supplemental/',
                                            compare_case = 1)
'''




'''# general plot function for mean of TRENDY model category and ONLY NBP
start_year = 2009
end_year   = 2018

Var1='nbp'
Var1_long=Var1

TRENDY_model_categories = ['very_good_models', 'ok_models', 'bad_low_amplitude_models', 'bad_high_amplitude_models']
TRENDY_model_categories_short = ['vg_models', 'ok_models', 'bla_models', 'bha_models']

#for i,model_category in enumerate(TRENDY_model_categories):
df_TRENDY_vg = load_datasets.load_TRENDY_model_gdf(variable=Var1, unit='mean_of_variable_model_category', model_category=TRENDY_model_categories[0], start_year=start_year, end_year=end_year)
df_TRENDY_ok = load_datasets.load_TRENDY_model_gdf(variable=Var1, unit='mean_of_variable_model_category', model_category=TRENDY_model_categories[1], start_year=start_year, end_year=end_year)
df_TRENDY_bla = load_datasets.load_TRENDY_model_gdf(variable=Var1, unit='mean_of_variable_model_category', model_category=TRENDY_model_categories[2], start_year=start_year, end_year=end_year)
df_TRENDY_bha = load_datasets.load_TRENDY_model_gdf(variable=Var1, unit='mean_of_variable_model_category', model_category=TRENDY_model_categories[3], start_year=start_year, end_year=end_year)

column_TM5 = 'CO2_NEE_fire_flux_monthly_TgC_per_subregion'
df_TM5_IS_GOSAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='mean_of_IS_ACOS_and_IS_RT', region='SAT', unit='per_subregion',  start_year=start_year, end_year=end_year)
#column_FXC = 'NBPtot'
#df_FXC = load_datasets.load_FLUXCOM_model_gdf(variable='NBP', unit='per_subregion', region='SAT', start_year=start_year, end_year=end_year)

plot_flux_func.plot_timeseries_MSC_general(df_list = [df_TRENDY_vg, df_TRENDY_ok, df_TRENDY_bla, df_TRENDY_bha, df_TM5_IS_GOSAT],
                                        model_list = ['TRENDY', 'TRENDY', 'TRENDY', 'TRENDY', 'TM5'],
                                        columns_to_plot = ['mean', 'mean', 'mean', 'mean', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion'],
                                        columns_to_plot_std = ['std', 'std', 'std', 'std', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion_std'],
                                        columns_to_plot_label = ['selected strict', 'selected loose', 'other low ampl', 'other high ampl', 'TM5-4DVar/IS+GOSAT'],
                                        TRENDY_model_category = TRENDY_model_categories_short,
                                        norm_timeseries = False,
                                        color_list = ['limegreen', 'darkolivegreen', 'orangered', 'sienna', 'firebrick'],
                                        linestyle_list = ['solid', 'solid', 'solid', 'solid', 'dashed'],
                                        plot_MSC_only = False,
                                        start_year = start_year,
                                        end_year = end_year,
                                        location_of_legend='lower left', #'center left',
                                        legend_columns = 2,
                                        legend_next_to_plot=False,
                                        region_name='SAT',
                                        plot_title = 'TRENDY mean model categories & TM5 nbp',
                                        savepath = '/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_and_MSC/Mean_of_good_and_bad_models_with_MSC_only/2009-2018/supplemental/',
                                        compare_case = 1)
'''
