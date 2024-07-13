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

color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink', 'olive', 'cyan', 'lime', 'teal', 'darkred', 'darkgreen', 'darkblue', 'darkcyan', 'darkmagenta']

Variables = ['gpp', 'nbp', 'npp', 'ra', 'rh', 'fFire']
Var = 'nbp'
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

# plot only timeseries
df_TM5_IS_ACOS = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_ACOS', region='SAT', unit='per_subregion')

plot_flux_func.plot_timeseries_of_TRENDY_fluxes(df=load_datasets.load_TRENDY_model_gdf(model='CABLE-POP', variable=Var, unit='mean_of_variable', start_year=2009),
                                                df_TM5=None,
                                                what_is_plotted='nbp', color_list=color_list, region_name='SAT', 
                                                savepath='/home/lartelt/MA/software/12_TRENDY_flux/plot_timeseries_only/')
