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


'''# plot map of ERA5 tp in SA
start_year = 2009
end_year = 2019

print('load dataset')
#gdf_SA = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5/gdf_MonthlyPrecipitation_SA.pkl')
gdf_SA = load_datasets.load_ERA5_precip_gdf(region='SAT_SATr', unit='per_gridcell', start_year=start_year, end_year=end_year)

for month in np.arange(1,13,1):
    print(month)
    plot_map_func.plotMapForERA5_tp(gdf_ERA5=gdf_SA, Year_or_Month_plotted=month, savepath='/home/lartelt/MA/software/14_ERA5_precipitation/maps/')


# generate GIF from map plots
pic_list = ['ERA5_precipitation_for_Month_'+str(month)+'.png' for month in np.arange(1,13,1)]
print(pic_list)
plot_map_func.create_GIF_from_images(pic_list, savepath_images='/home/lartelt/MA/software/14_ERA5_precipitation/maps/', monthly_or_yearly='monthly')
print('done')
'''

'''# plot map of ERA5 months with low precip in SA
start_year = 2009
end_year = 2020

print('load dataset')
#gdf_SA = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5/gdf_MonthlyPrecipitation_SA.pkl')
threshold = 0.001
#gdf_SA = load_datasets.load_ERA5_precip_gdf(region='SAT_SATr', unit='YearlyLowPrecipitation', threshold=threshold, start_year=start_year, end_year=end_year)
gdf_SA = load_datasets.load_ERA5_precip_gdf(region='SAT_SATr', unit='YearlyMEANLowPrecipitation', threshold=threshold, resolution=1., start_year=start_year, end_year=end_year)

column_to_plot = 'N_months_with_low_precip'
plot_map_func.plotMapForERA_low_precip(gdf_ERA5=gdf_SA, Year_plotted='', column_to_plot=column_to_plot, plot_mean=True, threshold=threshold,
                                           savepath='/home/lartelt/MA/software/14_ERA5_precipitation/results_from_1x1_grid/maps/mean_over_all_years/new_try/')
'''

'''#column_to_plot = 'Month_of_last_low_precip'
for year in np.arange(2009,2020,1):
    print(year)
    #plot_map_func.plotMapForERA_low_precip(gdf_ERA5=gdf_SA, Year_plotted=year, column_to_plot='N_months_with_low_precip', threshold=threshold,
    #                                       savepath='/home/lartelt/MA/software/14_ERA5_precipitation/maps/low_precip_threshold_'+str(threshold).replace('.','_')+'/')
    plot_map_func.plotMapForERA_low_precip(gdf_ERA5=gdf_SA, Year_plotted=year, column_to_plot=column_to_plot, plot_mean=False, threshold=threshold,
                                           savepath='/home/lartelt/MA/software/14_ERA5_precipitation/maps/mean_over_all_years/low_precip_threshold_'+str(threshold).replace('.','_')+'/')
    print('done')
'''

'''# plot only the region with dry_months=6
start_year = 2009
end_year = 2019

print('load dataset')
#gdf_SA = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5/gdf_MonthlyPrecipitation_SA.pkl')
threshold = 0.0015
#gdf_SA = load_datasets.load_ERA5_precip_gdf(region='SAT_SATr', unit='YearlyLowPrecipitation', threshold=threshold, start_year=start_year, end_year=end_year)
gdf_SA = load_datasets.load_ERA5_precip_gdf(region='SAT_SATr', unit='YearlyMEANLowPrecipitation', threshold=threshold, start_year=start_year, end_year=end_year)
gdf_SA_months = gdf_SA[gdf_SA['N_months_with_low_precip'] >= 6]

column_to_plot = 'N_months_with_low_precip'
plot_map_func.plotMapForERA_low_precip(gdf_ERA5=gdf_SA_months, Year_plotted='', column_to_plot=column_to_plot, plot_mean=True, threshold=threshold,
                                           savepath='/home/lartelt/MA/software/14_ERA5_precipitation/maps/mean_over_all_years/plot_gdf_with_6_dry_months_')
'''



'''# generate GIF from map plots
pic_list = ['ERA5_low_precipitation_'+column_to_plot+'_for_Year_'+str(year)+'.png' for year in np.arange(2009,2020,1)]
print(pic_list)
plot_map_func.create_GIF_from_images(pic_list, what_is_plotted=column_to_plot, monthly_or_yearly='yearly',
                                     savepath_images='/home/lartelt/MA/software/14_ERA5_precipitation/maps/mean_over_all_years/low_precip_threshold_'+str(threshold).replace('.','_')+'/')
print('done')
'''


'''# plot timeseries & MSC of ERA5 precipitation in SAT & subregions
start_year = 2009
end_year = 2019

print('load dataset')
gdf_SAT = load_datasets.load_ERA5_precip_gdf(region='SAT', unit='per_subregion', resolution=1., start_year=start_year, end_year=end_year)
print(gdf_SAT)
gdf_nSAT = load_datasets.load_ERA5_precip_gdf(region='west_SAT', unit='per_subregion', resolution=1., start_year=start_year, end_year=end_year)
gdf_mSAT = load_datasets.load_ERA5_precip_gdf(region='mid_long_SAT', unit='per_subregion', resolution=1., start_year=start_year, end_year=end_year)
gdf_sSAT = load_datasets.load_ERA5_precip_gdf(region='east_SAT', unit='per_subregion', resolution=1., start_year=start_year, end_year=end_year)
print('done loading datasets')
plot_timeseries_func.plot_timeseries_MSC_general(df_list = [gdf_SAT, gdf_nSAT, gdf_mSAT, gdf_sSAT],
                                                 model_list=['ERA5', 'ERA5', 'ERA5', 'ERA5'],
                                                 columns_to_plot=['tp', 'tp', 'tp', 'tp'],
                                                 columns_to_plot_std=['', '', '', ''],
                                                 columns_to_plot_label=['SAT', 'wSAT', 'mSAT', 'eSAT'],
                                                 norm_timeseries=False,
                                                 color_list=['dimgrey', 'royalblue', 'firebrick', 'forestgreen'],
                                                 linestyle_list=['--', '-', '-', '-'],
                                                 plot_MSC_only=False,
                                                 plot_title='ERA5 precipitation per subregion',
                                                 savepath='/home/lartelt/MA/software/14_ERA5_precipitation/results_from_1x1_grid/timeseries_and_MSC/east_to_west/',
                                                 compare_case=1)
'''

'''# include SATr
start_year = 2009
end_year = 2019

print('load dataset')
gdf_SAT = load_datasets.load_ERA5_precip_gdf(region='SAT', unit='per_subregion', start_year=start_year, end_year=end_year)
gdf_SATr = load_datasets.load_ERA5_precip_gdf(region='SATr', unit='per_subregion', start_year=start_year, end_year=end_year)
print('done loading datasets')
plot_timeseries_func.plot_timeseries_MSC_general(df_list = [gdf_SAT, gdf_SATr],
                                                 model_list=['ERA5', 'ERA5'],
                                                 columns_to_plot=['tp', 'tp'],
                                                 columns_to_plot_std=['', ''],
                                                 columns_to_plot_label=['SAT', 'SATr'],
                                                 norm_timeseries=False,
                                                 color_list=['dimgrey', 'mediumvioletred'],
                                                 linestyle_list=['-', '-'],
                                                 plot_MSC_only=False,
                                                 plot_title='ERA5 precipitation per subregion SATr &',
                                                 savepath='/home/lartelt/MA/software/14_ERA5_precipitation/timeseries_and_MSC/with_SATr/',
                                                 compare_case=1)
'''


'''# plot timeseries of ERA5 SAT & arid region
start_year = 2009
end_year = 2020

print('load dataset')
gdf_SAT = load_datasets.load_ERA5_precip_gdf(region='SAT', unit='per_subregion', resolution=1., start_year=start_year, end_year=end_year)
gdf_SAT_strict_arid = load_datasets.load_ERA5_precip_gdf(region='SAT_strict_limit_arid', unit='per_subregion', resolution=1., start_year=start_year, end_year=end_year)
gdf_SAT_strict_humid = load_datasets.load_ERA5_precip_gdf(region='SAT_strict_limit_humid', unit='per_subregion', resolution=1., start_year=start_year, end_year=end_year)
print('done loading datasets')
plot_timeseries_func.plot_timeseries_general(df_list = [gdf_SAT, gdf_SAT_strict_arid, gdf_SAT_strict_humid],
                                             model_list=['ERA5', 'ERA5', 'ERA5'],
                                             columns_to_plot=['tp_mean', 'tp_mean', 'tp_mean'],
                                             columns_to_plot_std=['', 'tp_std', 'tp_std'],
                                             columns_to_plot_label=['SAT', 'strict_arid', 'strict_humid'],
                                             color_list=['dimgrey', 'coral', 'cornflowerblue'],
                                             linestyle_list=['--', '-', '-'],
                                             plot_title='ERA5 precipitation arid & humid ',
                                             savename_input='timeseries_ERA5_precip_arid_and_humid_and_both_mean_and_std_SAT',
                                             savepath='/home/lartelt/MA/software/14_ERA5_precipitation/results_from_1x1_grid/timeseries_only/',
                                             compare_case=1)
'''


'''# plot bar-chart of ERA5 precipitation in SAT & arid region
start_year = 2009
end_year = 2020
gdf_SAT = load_datasets.load_ERA5_precip_gdf(region='SAT', unit='per_subregion', resolution=1., start_year=start_year, end_year=end_year)

strict_or_loose = 'loose'
gdf_SAT_arid = load_datasets.load_ERA5_precip_gdf(region='SAT_'+strict_or_loose+'_limit_arid', unit='per_subregion', resolution=1., start_year=start_year, end_year=end_year)

plot_timeseries_func.plotSumOfFluxesBarChart(df_ERA5=[gdf_SAT, gdf_SAT_arid],
                                             variable_plotted=['tp_mean', 'tp_mean'],
                                             start_year = start_year,
                                             end_year = end_year,
                                             color_list = ['dimgrey', 'coral'],
                                             label_list = ['SAT', 'SAT_'+strict_or_loose+'_arid'],
                                             region_name = 'SAT', 
                                             title = 'ERA5 mean precipitation total SAT & '+strict_or_loose+' arid region',
                                             savename = 'ERA5_precipitation_total_SAT_and_'+strict_or_loose+'_arid_region',
                                             savepath = '/home/lartelt/MA/software/14_ERA5_precipitation/results_from_1x1_grid/bar_plots/')
'''


# NEW plot bar-chart of ERA5 precipitation in SAT & arid region
start_year = 2009
end_year = 2020
gdf_SAT = load_datasets.load_ERA5_precip_gdf(region='SAT', unit='per_subregion', resolution=1., start_year=start_year, end_year=end_year)

strict_or_loose = 'loose'
gdf_SAT_arid = load_datasets.load_ERA5_precip_gdf(region='SAT_'+strict_or_loose+'_limit_arid', unit='per_subregion', resolution=1., start_year=start_year, end_year=end_year)

plot_timeseries_func.plotSumOfFluxesBarChart(df_ERA5=[gdf_SAT, gdf_SAT_arid],
                                             variable_plotted=['tp_mean', 'tp_mean'],
                                             start_year = start_year,
                                             end_year = end_year,
                                             color_list = ['dimgrey', 'coral'],
                                             label_list = ['SAT', 'SAT arid'],
                                             region_name = 'SAT', 
                                             title = 'ERA5 mean precipitation total SAT & '+strict_or_loose+' arid region',
                                             savename = 'NEW_ERA5_precipitation_total_SAT_and_'+strict_or_loose+'_arid_region',
                                             savepath = '/home/lartelt/MA/software/14_ERA5_precipitation/results_from_1x1_grid/bar_plots/')
