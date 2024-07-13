# Author: Lukas Artelt
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cf
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
import functions_to_plot_GFED_data as plot_func
import functions_to_load_datasets as load_datasets

start_year = 2009
end_year = 2019
region = 'SAT'

df_GFED = load_datasets.load_GFED_model_gdf(region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)

df_TM5_IS = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
df_TM5_IS_ACOS = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_ACOS', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
df_TM5_IS_RT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask('IS_RT', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
df_MIP_ens = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_ens', assimilated='IS_OCO2', end_year=end_year, region='SAT', unit='per_subregion')

#df_TRENDY_vg = load_datasets.load_TRENDY_model_gdf(model='CABLE-POP', variable='nbp', unit='mean_of_variable_model_category', model_category='very_good_models', start_year=start_year, end_year=end_year)
df_TRENDY_ok = load_datasets.load_TRENDY_model_gdf(model='CABLE-POP', variable='nbp', unit='mean_of_variable_model_category', model_category='ok_models', start_year=start_year, end_year=end_year)
df_FLUXCOM = load_datasets.load_FLUXCOM_model_gdf(variable='NBP', region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)


'''# only plot GFED flux
region = 'SAT'
ds = pd.read_pickle('/mnt/data/users/lartelt/MA/GFED4/GFED_cut_with_CT_Mask/GFED_CT_Mask_'+region+'_TgC_per_month_per_subregion.pkl')
ds['Year'] = ds.apply(lambda x: x['MonthDate'].year, axis=1)
ds['Month'] = ds.apply(lambda x: x['MonthDate'].month, axis=1)
# drop entries where column 'Year' is larger than 2019
ds = ds.drop(ds[ds.Year > 2019].index)

plot_func.plotFluxTimeseries_and_MSC_general(gdf_list=[ds], start_year=2009, end_year=2019,
                                             variable_to_plot_list=['total_emission'],
                                             model_assimilation_list=['GFED'],
                                             what_is_plotted='GFED_fire_flux',
                                             color_list=['firebrick'],
                                             linestyle_list=['solid'],
                                             region_name=region,
                                             savepath='/home/lartelt/MA/software/1_Feuerdaten_auswerten/timeseries_and_MSC/')
'''

'''# plot TM5 & MIP nbp & GFED fire flux
plot_func.plotFluxTimeseries_and_MSC_general(gdf_list=[df_GFED, df_TM5_IS, df_TM5_IS_ACOS, df_TM5_IS_RT, df_MIP_ens], 
                                             start_year=start_year, end_year=end_year,
                                             variable_to_plot_list=['total_emission', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion',
                                                                    'CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion',
                                                                    'Landtot'],
                                             model_assimilation_list=['GFED', 'TM5_IS', 'TM5_IS_ACOS', 'TM5_IS_RT', 'MIP_ens'],
                                             what_is_plotted='GFED_fire_and_TM5_MIP_nbp',
                                             color_list=['black', 'indianred', 'firebrick', 'coral', 'royalblue'],
                                             linestyle_list=['solid', 'dashed', 'solid', 'solid', 'solid'],
                                             region_name=region,
                                             savepath='/home/lartelt/MA/software/1_Feuerdaten_auswerten/timeseries_and_MSC/')
'''

# plot FLUXCOM nbp & TRENDY nbp & GFED fire flux
plot_func.plotFluxTimeseries_and_MSC_general(gdf_list=[df_GFED, df_TRENDY_ok, df_FLUXCOM], 
                                             start_year=start_year, end_year=end_year,
                                             variable_to_plot_list=['total_emission', 'mean', 'NBPtot'],
                                             model_assimilation_list=['GFED', 'TRENDY', 'FLUXCOM'],                                            
                                             what_is_plotted='GFED_fire_and_TRENDY_ok_models_and_FLUXCOM_nbp',
                                             color_list=['black', 'forestgreen', 'violet'],
                                             linestyle_list=['solid', 'solid', 'solid'],
                                             region_name=region,
                                             savepath='/home/lartelt/MA/software/1_Feuerdaten_auswerten/timeseries_and_MSC/')



'''# plot maps of GFED flux with SAT border
df_GFED_gridcell = load_datasets.load_GFED_model_gdf(region='SAT_SATr', unit='per_gridcell')#, start_year=start_year, end_year=end_year)
#print(df_GFED_gridcell.head())

#for year in range(start_year, end_year+3):
#for year in range(2010, 2011):
for month in range(1, 13):
    #print(year)
    print(month)
    #plot_func.plotMapForMonthDate_GFED(gdf_GFED=df_GFED_gridcell, Year_or_Month_plotted=year, savepath='/home/lartelt/MA/software/1_Feuerdaten_auswerten/maps/')
    plot_func.plotMapForMonthDate_GFED(gdf_GFED=df_GFED_gridcell, Year_or_Month_plotted=month, savepath='/home/lartelt/MA/software/1_Feuerdaten_auswerten/maps/')
'''

'''# plot maps in gif
plot_func.create_GIF_from_images(image_list=['GFED_fire_emission_for_Year_'+str(year)+'.png' for year in range(2009,2022)],
                                 savepath_images='/home/lartelt/MA/software/1_Feuerdaten_auswerten/maps/',
                                 monthly_or_yearly='yearly')
'''
