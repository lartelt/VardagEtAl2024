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
import function_to_load_datasets as load_datasets
import functions_for_Flux_plotting as plot_flux_func
#import plotting_map_functions as plot_map_func


#print('Start loading datasets for region: ', region)
#Fluxes_prior_region = load_datasets.dataframe_from_name_TM5_per_region(assimilated='prior', region_name=region, start_year=2009)
#Fluxes_prior_region_gridded = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='prior', start_year=2009, region=region, unit='per_subregion')
#Fluxes_prior_region_gridded = load_datasets.return_dataframe_from_name_TM5_gridded_1x1_CT_Mask(assimilated='prior', start_year=2009, region=region, unit='per_subregion')
#Fluxes_ACOS_IS_region = load_datasets.dataframe_from_name_TM5_per_region(assimilated='ACOSIS', region_name=region, start_year=2009)
#Fluxes_ACOS_IS_region_gridded = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_ACOS', start_year=2009, region=region, unit='per_subregion')
#Fluxes_ACOS_IS_region_gridded = load_datasets.return_dataframe_from_name_TM5_gridded_1x1_CT_Mask(assimilated='IS_ACOS', start_year=2009, region=region, unit='per_subregion')
#Fluxes_IS_region = load_datasets.dataframe_from_name_TM5_per_region(assimilated='IS', region_name=region, start_year=2009)
#Fluxes_IS_region_gridded = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS', start_year=2009, region=region, unit='per_subregion')
#Fluxes_IS_region_gridded = load_datasets.return_dataframe_from_name_TM5_gridded_1x1_CT_Mask(assimilated='IS', start_year=2009, region=region, unit='per_subregion')
#Fluxes_RT_IS_loc_region = load_datasets.dataframe_from_name_TM5_per_region(assimilated='RemoTeCISloc', region_name=region, start_year=2009)
#Fluxes_RT_IS_loc_region_gridded = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_RT', start_year=2009, region=region, unit='per_subregion')
#Fluxes_RT_IS_loc_region_gridded = load_datasets.return_dataframe_from_name_TM5_gridded_1x1_CT_Mask(assimilated='IS_RT', start_year=2009, region=region, unit='per_subregion')
        
        
if __name__=='__main__':
    region_list = ['SAT_SATr', 'SAT', 'SATr']
    region_list_short = ['SAT']
    subregion_list = ['north_SAT', 'mid_SAT', 'south_SAT', 'west_SAT', 'mid_long_SAT', 'east_SAT']
    
    # 1 plot TM5_flux_regional_sourish & TM5_flux_gridded
    for region in region_list_short:
        print('Start plotting for region: ', region)
        
        # 0 plot fluxes compare gridded and regional datasets
        '''
        plot_flux_func.plotFluxTimeseries_and_MSC_compare_gridded_and_regional(start_month = 5,
                                             Fluxes_prior_region = Fluxes_prior_region,
                                             Fluxes_prior_region_gridded = Fluxes_prior_region_gridded,
                                             Fluxes_ACOS_IS_region = Fluxes_ACOS_IS_region,
                                             Fluxes_ACOS_IS_region_gridded = Fluxes_ACOS_IS_region_gridded,
                                             Fluxes_IS_region = Fluxes_IS_region,
                                             Fluxes_IS_region_gridded = Fluxes_IS_region_gridded,
                                             Fluxes_RT_IS_loc_region = Fluxes_RT_IS_loc_region,
                                             Fluxes_RT_IS_loc_region_gridded = Fluxes_RT_IS_loc_region_gridded,
                                             region_name=region, compare_case=1, grid_res='3x2',
                                             savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/flux_plots_comparing_regional_and_gridded_fluxes/')
        '''
        
        # 1 plot fluxes in GENERAL function, here for gridded TM5 fluxes
        '''
        plot_flux_func.plotFluxTimeseries_and_MSC_general(start_month = 5,start_year=2009,
                                                  gdf_list=[Fluxes_prior_region_gridded, Fluxes_ACOS_IS_region_gridded, Fluxes_IS_region_gridded, 
                                                            Fluxes_RT_IS_loc_region_gridded],
                                                  variable_to_plot_list=['CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion', 
                                                                         'CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion'],
                                                  model_assimilation_list=['TM5-4DVar_prior', 'TM5-4DVar_IS+ACOS', 'TM5-4DVar_IS', 'TM5-4DVar_IS+RT'],
                                                  what_is_plotted='NEE+fire_flux',
                                                  color_list=['dimgrey', 'royalblue', 'forestgreen', 'firebrick'],
                                                  linestyle_list=['solid', 'solid', 'solid', 'solid'],
                                                  region_name=region,
                                                  savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/flux_plots_comparing_regional_and_gridded_fluxes/test/',
                                                  compare_case=1)
        '''
        
        # 2 plot fluxes from different assimilations in one subregion
        '''
        Fluxes_prior_region_gridded = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='prior', start_year=2015, region=region, unit='per_subregion')
        Fluxes_ACOS_IS_region_gridded = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_ACOS', start_year=2015, region=region, unit='per_subregion')
        Fluxes_IS_region_gridded = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS', start_year=2015, region=region, unit='per_subregion')
        Fluxes_RT_IS_loc_region_gridded = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_RT', start_year=2015, region=region, unit='per_subregion')
        #Fluxes_MIP_IS_region_gridded = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_ens', assimilated='IS', end_year=2019, region=region, unit='per_subregion')
        #Fluxes_MIP_IS_OCO2_region_gridded = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_ens', assimilated='IS_OCO2', end_year=2019, region=region, unit='per_subregion')
        Fluxes_OCO2_IS_region_gridded_mid_long_SAT = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_TM5', assimilated='IS_OCO2', end_year=2019, region='mid_long_SAT', unit='per_subregion')
        plot_flux_func.plotTimeseries_MSC_start_May_subregion(start_month = 5, include_MIP=True, fix_y_scale=False,
                                                    Fluxes_prior_region_gridded = Fluxes_prior_region_gridded,
                                                    Fluxes_ACOS_IS_region_gridded = Fluxes_ACOS_IS_region_gridded,
                                                    Fluxes_IS_region_gridded = Fluxes_IS_region_gridded,
                                                    Fluxes_RT_IS_loc_region_gridded = Fluxes_RT_IS_loc_region_gridded,
                                                    Fluxes_MIP_IS_region_gridded = Fluxes_MIP_IS_region_gridded,
                                                    Fluxes_MIP_IS_OCO2_region_gridded = Fluxes_MIP_IS_OCO2_region_gridded,
                                                    region = region, savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/flux_plots_comparing_one_subregion/',
                                                    compare_case=1)
        '''
        
        # 3 plot fluxes in GENERAL function, here for gridded MIP_ens & MIP_TM5 fluxes
        '''
        Fluxes_MIP_IS_region_gridded = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_ens', assimilated='IS', end_year=2020, region=region, unit='per_subregion')
        Fluxes_MIP_IS_OCO2_region_gridded = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_ens', assimilated='IS_OCO2', end_year=2020, region=region, unit='per_subregion')
        Fluxes_MIP_TM5_IS_OCO2_region_gridded = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_TM5', assimilated='IS_OCO2', end_year=2020, region=region, unit='per_subregion')
        Fluxes_prior_region_gridded = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='prior', start_year=2015, region=region, unit='per_subregion')
        Fluxes_ACOS_IS_region_gridded = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_ACOS', start_year=2015, region=region, unit='per_subregion')
        plot_flux_func.plotFluxTimeseries_and_MSC_general(start_month = 5,start_year=2009,
                                                          gdf_list=[Fluxes_MIP_IS_region_gridded, Fluxes_MIP_IS_OCO2_region_gridded, 
                                                                    Fluxes_MIP_TM5_IS_OCO2_region_gridded, Fluxes_prior_region_gridded, Fluxes_ACOS_IS_region_gridded],
                                                          variable_to_plot_list=['Landtot', 'Landtot', 'Landtot', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion'],
                                                          model_assimilation_list=['MIP_IS_ens', 'MIP_IS+OCO2_ens', 'TM5-4DVar_IS+OCO2', 'TM5-4DVar_prior', 'TM5-4DVar_IS+ACOS'],
                                                          what_is_plotted='Land_flux',
                                                          color_list=['royalblue', 'royalblue', 'forestgreen', 'dimgrey', 'firebrick'],
                                                          linestyle_list=['dashed', 'solid', 'solid', 'solid', 'solid'],
                                                          region_name=region,
                                                          savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/flux_plots_comparing_one_subregion/',
                                                          compare_case=1)
        '''
        
    # 4 plot fluxes compare different subregions
    '''
    Fluxes_ACOS_IS_region_gridded_SAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_ACOS', start_year=2009, region='SAT', unit='per_subregion')
    #Fluxes_ACOS_IS_region_gridded_nSAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_ACOS', start_year=2009, region='west_SAT', unit='per_subregion')
    Fluxes_ACOS_IS_region_gridded_nSAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_ACOS', start_year=2009, region='north_SAT', unit='per_subregion')
    #Fluxes_ACOS_IS_region_gridded_mSAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_ACOS', start_year=2009, region='mid_long_SAT', unit='per_subregion')
    Fluxes_ACOS_IS_region_gridded_mSAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_ACOS', start_year=2009, region='mid_SAT', unit='per_subregion')
    #Fluxes_ACOS_IS_region_gridded_sSAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_ACOS', start_year=2009, region='east_SAT', unit='per_subregion')
    Fluxes_ACOS_IS_region_gridded_sSAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_ACOS', start_year=2009, region='south_SAT', unit='per_subregion')
    plot_flux_func.plotTimeseries_MSC_start_May_subregion_comparison(is_MIP=False, fix_y_scale=True,
                                                                     Fluxes_ACOS_IS_SAT = Fluxes_ACOS_IS_region_gridded_SAT,
                                                                     Fluxes_ACOS_IS_north_SAT=Fluxes_ACOS_IS_region_gridded_nSAT,
                                                                     Fluxes_ACOS_IS_mid_SAT=Fluxes_ACOS_IS_region_gridded_mSAT,
                                                                     Fluxes_ACOS_IS_south_SAT=Fluxes_ACOS_IS_region_gridded_sSAT,
                                                                     assimilated='IS+ACOS', 
                                                                     savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/flux_plots_comparing_multiple_subregions/supplemental/',
                                                                     compare_case=1)
    '''
    
    # 5 from GENRAL function, plot fluxes for ONE subregions
    '''
    Fluxes_TM5_prior_mid_long_SAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='prior', start_year=2015, region='east_SAT', unit='per_subregion')
    Fluxes_TM5_IS_mid_long_SAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS', start_year=2015, region='east_SAT', unit='per_subregion')
    Fluxes_TM5_IS_ACOS_mid_long_SAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_ACOS', start_year=2015, region='east_SAT', unit='per_subregion')
    Fluxes_TM5_IS_RT_mid_long_SAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_RT', start_year=2015, region='east_SAT', unit='per_subregion')
    Fluxes_TM5_IS_OCO2_mid_long_SAT = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_TM5', assimilated='IS_OCO2', end_year=2020, region='east_SAT', unit='per_subregion')
    #Fluxes_MIP_IS_OCO2_mid_long_SAT = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_ens', assimilated='IS_OCO2', end_year=2020, region='mid_long_SAT', unit='per_subregion')
    plot_flux_func.plotFluxTimeseries_and_MSC_general(start_month = 5,start_year=2015,end_year=2020,
                                                        gdf_list=[Fluxes_TM5_prior_mid_long_SAT, Fluxes_TM5_IS_mid_long_SAT, Fluxes_TM5_IS_ACOS_mid_long_SAT,
                                                                  Fluxes_TM5_IS_RT_mid_long_SAT, Fluxes_TM5_IS_OCO2_mid_long_SAT],
                                                        variable_to_plot_list=['CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion',
                                                                               'CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion',
                                                                               'Landtot'],
                                                        model_assimilation_list=['TM5-4DVar_prior', 'TM5-4DVar_IS', 'TM5-4DVar_IS+ACOS',
                                                                                 'TM5-4DVar_IS+RT', 'TM5-4DVar_IS+OCO2'],
                                                        what_is_plotted='Land_flux',
                                                        color_list=['dimgrey', 'indianred', 'firebrick', 'coral', 'royalblue'],
                                                        linestyle_list=['dashed', 'dashed', 'solid', 'solid', 'solid'],
                                                        region_name='east_SAT',
                                                        savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/flux_plots_comparing_one_subregion/',
                                                        compare_case=1)
    '''
    
    
    # 6 plot fluxes in GENERAL function, here for gridded TM5 fluxes ONLY prior & IS
    
    region = 'SAT'
    #Fluxes_prior_region_gridded = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='prior', start_year=2009, region=region, unit='per_subregion')
    #Fluxes_IS_region_gridded = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS', start_year=2009, region=region, unit='per_subregion')
    #Fluxes_ACOS_IS_region_gridded = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_ACOS', start_year=2009, region=region, unit='per_subregion')
    #Fluxes_RT_IS_loc_region_gridded = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_RT', start_year=2009, region=region, unit='per_subregion')
    '''
    plot_flux_func.plotFluxTimeseries_and_MSC_general(start_month = 5,start_year=2009, fix_y_scale=True,
                                                gdf_list=[Fluxes_prior_region_gridded, Fluxes_IS_region_gridded],
                                                variable_to_plot_list=['CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion'],
                                                model_assimilation_list=['TM5-4DVar_prior', 'TM5-4DVar_IS'],
                                                what_is_plotted='NEE+fire_flux',
                                                color_list=['dimgrey', 'indianred'],
                                                linestyle_list=['dashed', 'dashed'],
                                                region_name=region,
                                                savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/flux_plots_comparing_one_subregion/',
                                                compare_case=1)
    '''
    '''
    plot_flux_func.plotFluxTimeseries_and_MSC_general(start_month = 5,start_year=2009, fix_y_scale=True,
                                                gdf_list=[Fluxes_ACOS_IS_region_gridded, Fluxes_RT_IS_loc_region_gridded],
                                                variable_to_plot_list=['CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion'],
                                                model_assimilation_list=['TM5-4DVar_IS+ACOS', 'TM5-4DVar_IS+RT'],
                                                what_is_plotted='NEE+fire_flux',
                                                color_list=['firebrick', 'coral'],
                                                linestyle_list=['solid', 'solid'],
                                                region_name=region,
                                                savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/flux_plots_comparing_one_subregion/',
                                                compare_case=1)
   ''' 
    
    
    # 7 plot fluxes for SAT region ALL TM5 fluxes + MIP_ens fluxes
    '''Fluxes_TM5_prior_SAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='prior', start_year=2015, region='SAT', unit='per_subregion')
    #Fluxes_TM5_IS_SAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS', start_year=2015, region='SAT', unit='per_subregion')
    Fluxes_TM5_IS_ACOS_SAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_ACOS', start_year=2015, region='SAT', unit='per_subregion')
    Fluxes_TM5_IS_RT_SAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_RT', start_year=2015, region='SAT', unit='per_subregion')
    Fluxes_TM5_IS_OCO2_SAT = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_TM5', assimilated='IS_OCO2', end_year=2020, region='SAT', unit='per_subregion')
    Fluxes_MIP_ens_IS_SAT = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_ens', assimilated='IS', end_year=2020, region='SAT', unit='per_subregion')
    Fluxes_MIP_ens_IS_OCO2_SAT = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_ens', assimilated='IS_OCO2', end_year=2020, region='SAT', unit='per_subregion')
    plot_flux_func.plotFluxTimeseries_and_MSC_general(start_month = 5,start_year=2015,end_year=2020,
                                                      gdf_list=[Fluxes_TM5_prior_SAT, Fluxes_TM5_IS_ACOS_SAT, Fluxes_TM5_IS_RT_SAT, 
                                                                Fluxes_TM5_IS_OCO2_SAT, Fluxes_MIP_ens_IS_SAT, Fluxes_MIP_ens_IS_OCO2_SAT],
                                                      variable_to_plot_list=['CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion',
                                                                              'CO2_NEE_fire_flux_monthly_TgC_per_subregion',
                                                                              'Landtot', 'Landtot', 'Landtot'],
                                                      model_assimilation_list=['TM5-4DVar_prior', 'TM5-4DVar_IS+ACOS', 'TM5-4DVar_IS+RT',
                                                                               'TM5-4DVar_IS+OCO2', 'MIP_ens_IS', 'MIP_ens_IS+OCO2'],
                                                      #plot_second_MSCs = True,
                                                      #second_MSC_start_year = 2015,
                                                      #second_MSC_end_year = 2018,
                                                      what_is_plotted='Land_flux',
                                                      color_list=['dimgrey', 'firebrick', 'coral', 'forestgreen', 'royalblue', 'royalblue'],
                                                      linestyle_list=['dashed', 'solid',     'solid', 'solid',        'dashed',    'solid'],
                                                      region_name='SAT',
                                                      savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/flux_plots_comparing_one_subregion/supplemental/',
                                                      compare_case=1)
    '''
    '''plot_flux_func.plotFluxTimeseries_and_MSC_general(start_month = 5,start_year=2015,end_year=2020,
                                                        gdf_list=[Fluxes_TM5_prior_SAT, Fluxes_TM5_IS_SAT, Fluxes_TM5_IS_ACOS_SAT, Fluxes_TM5_IS_RT_SAT, 
                                                                  Fluxes_TM5_IS_OCO2_SAT],
                                                        variable_to_plot_list=['CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion',
                                                                               'CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion',
                                                                               'Landtot'],
                                                        model_assimilation_list=['TM5-4DVar_prior', 'TM5-4DVar_IS', 'TM5-4DVar_IS+ACOS', 'TM5-4DVar_IS+RT',
                                                                                 'TM5-4DVar_IS+OCO2'],
                                                        what_is_plotted='Land_flux',
                                                        color_list=['dimgrey', 'firebrick', 'firebrick', 'coral', 'forestgreen'],
                                                        linestyle_list=['dashed', 'dashed', 'solid',     'solid', 'solid'],
                                                        region_name='SAT',
                                                        savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/flux_plots_comparing_one_subregion/',
                                                        compare_case=1)'''
    '''
    plot_flux_func.plotFluxTimeseries_and_MSC_general(start_month = 5,start_year=2015,end_year=2020,
                                                        gdf_list=[Fluxes_TM5_IS_SAT, Fluxes_TM5_IS_OCO2_SAT, Fluxes_MIP_ens_IS_SAT, Fluxes_MIP_ens_IS_OCO2_SAT],
                                                        variable_to_plot_list=['CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'Landtot', 'Landtot', 'Landtot'],
                                                        model_assimilation_list=['TM5-4DVar_IS', 'TM5-4DVar_IS+OCO2', 'MIP_ens_IS', 'MIP_ens_IS+OCO2'],
                                                        what_is_plotted='Land_flux',
                                                        color_list=['firebrick', 'forestgreen', 'royalblue', 'royalblue'],
                                                        linestyle_list=['dashed', 'solid',        'dashed',    'solid'],
                                                        region_name='SAT',
                                                        savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/flux_plots_comparing_one_subregion/',
                                                        compare_case=1)
    '''
    
    
    # 7.1 plot fluxes WITH 2 MSCs for SAT region ALL TM5 fluxes + MIP_ens fluxes
    '''
    Fluxes_TM5_prior_SAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='prior', start_year=2015, region='SAT', unit='per_subregion')
    #Fluxes_TM5_IS_SAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS', start_year=2015, region='SAT', unit='per_subregion')
    Fluxes_TM5_IS_ACOS_SAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_ACOS', start_year=2015, region='SAT', unit='per_subregion')
    Fluxes_TM5_IS_RT_SAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_RT', start_year=2015, region='SAT', unit='per_subregion')
    Fluxes_TM5_IS_OCO2_SAT = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_TM5', assimilated='IS_OCO2', end_year=2020, region='SAT', unit='per_subregion')
    Fluxes_MIP_ens_IS_SAT = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_ens', assimilated='IS', end_year=2020, region='SAT', unit='per_subregion')
    Fluxes_MIP_ens_IS_OCO2_SAT = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model='MIP_ens', assimilated='IS_OCO2', end_year=2020, region='SAT', unit='per_subregion')
    plot_flux_func.plotFluxTimeseries_and_MSC_general_with_second_MSC_before_main_MSC(
                                                      start_month = 5,start_year=2015,end_year=2020,
                                                      gdf_list=[Fluxes_TM5_prior_SAT, Fluxes_TM5_IS_ACOS_SAT, Fluxes_TM5_IS_RT_SAT, 
                                                                Fluxes_TM5_IS_OCO2_SAT, Fluxes_MIP_ens_IS_SAT, Fluxes_MIP_ens_IS_OCO2_SAT],
                                                      variable_to_plot_list=['CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion',
                                                                              'CO2_NEE_fire_flux_monthly_TgC_per_subregion',
                                                                              'Landtot', 'Landtot', 'Landtot'],
                                                      model_assimilation_list=['TM5-4DVar_prior', 'TM5-4DVar_IS+ACOS', 'TM5-4DVar_IS+RT',
                                                                               'TM5-4DVar_IS+OCO2', 'MIP_ens_IS', 'MIP_ens_IS+OCO2'],
                                                      plot_second_MSCs = True,
                                                      second_MSC_start_year = 2015,
                                                      second_MSC_end_year = 2018,
                                                      what_is_plotted='Land_flux_NEW_MSCs',
                                                      color_list=['dimgrey', 'firebrick', 'coral', 'forestgreen', 'royalblue', 'royalblue'],
                                                      linestyle_list=['dashed', 'solid',     'solid', 'solid',        'dashed',    'solid'],
                                                      region_name='SAT',
                                                      savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/flux_plots_comparing_one_subregion/supplemental/',
                                                      compare_case=1)
    '''
    
    
    # 8 from GENRAL function, plot TM5 fluxes for SAT
    '''
    Fluxes_TM5_prior_SAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='prior', start_year=2009, region='SAT', unit='per_subregion')
    Fluxes_TM5_IS_SAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS', start_year=2009, region='SAT', unit='per_subregion')
    Fluxes_TM5_IS_ACOS_SAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_ACOS', start_year=2009, region='SAT', unit='per_subregion')
    Fluxes_TM5_IS_RT_SAT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_RT', start_year=2009, region='SAT', unit='per_subregion')
    
    plot_flux_func.plotFluxTimeseries_and_MSC_general(start_month = 5,start_year=2009,end_year=2019,
                                                        gdf_list=[Fluxes_TM5_prior_SAT, Fluxes_TM5_IS_SAT, Fluxes_TM5_IS_ACOS_SAT, Fluxes_TM5_IS_RT_SAT],
                                                        variable_to_plot_list=['CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion',
                                                                               'CO2_NEE_fire_flux_monthly_TgC_per_subregion', 'CO2_NEE_fire_flux_monthly_TgC_per_subregion'],
                                                        model_assimilation_list=['TM5-4DVar_prior', 'TM5-4DVar_IS', 'TM5-4DVar_IS+ACOS', 'TM5-4DVar_IS+RT'],
                                                        label_list=['TM5-4DVar prior', 'TM5-4DVar/IS', 'TM5-4DVar/IS+ACOS', 'TM5-4DVar/IS+RT'],
                                                        what_is_plotted='Land_flux',
                                                        color_list=['dimgrey', 'indianred', 'firebrick', 'coral'],
                                                        linestyle_list=['dashed', 'dashed', 'solid', 'solid'],
                                                        region_name='SAT',
                                                        savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/flux_plots_comparing_one_subregion/supplemental/',
                                                        compare_case=1)
    '''
    
    