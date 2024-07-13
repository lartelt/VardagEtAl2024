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
import functions_for_xCO2_plotting as plot_xCO2_func
#import plotting_map_functions as plot_map_func


if __name__=='__main__':
    annual_xco2_growth_rate_starting_from_2009 = [1.58, 2.42, 1.67, 2.42, 2.45, 2.04, 2.94, 2.84, 2.14, 2.38, 2.55]
    start_co2_ppm_2009_SAT = 384.8
    date_min = datetime.datetime(2009,5,1)
    date_max = datetime.datetime(2020,1,1)
    background_xco2_time_series_SAT = plot_xCO2_func.background_xco2_time_series(annual_xco2_growth_rate_starting_from_2009, start_co2_ppm_2009_SAT, date_min.year, date_max.year)
    
    region_list = ['SAT_SATr', 'SAT', 'SATr']
    region_list_short = ['SAT']
    subregion_list_long = ['north_SAT', 'mid_SAT', 'south_SAT', 'west_SAT', 'mid_long_SAT', 'east_SAT']
    assimilation_list = ['apri', 'IS', 'IS_ACOS', 'IS_RT']
    assimilation_list_not_cos = ['IS', 'IS_ACOS', 'IS_RT']
    
    # 1 plot TM5_flux_regional_sourish & TM5_flux_gridded
    for region in region_list_short:
        #print('Start plotting for region: ', region)

        # 0 plot TM5 xCO2 compare Evas and CT Mask datasets
        '''
        TM5_apri_cos_RT_xCO2_SAT_Eva = load_datasets.load_gdf_TM5_xCO2_cos_RT_Evas_mask(assimilated='apri', region_name=region)
        TM5_apri_cos_RT_xCO2_SAT_CT = load_datasets.load_gdf_TM5_xCO2_cos_RT_CT_Mask(assimilated='apri', region_name=region)
        TM5_IS_cos_RT_xCO2_SAT_Eva = load_datasets.load_gdf_TM5_xCO2_cos_RT_Evas_mask(assimilated='IS', region_name=region)
        TM5_IS_cos_RT_xCO2_SAT_CT = load_datasets.load_gdf_TM5_xCO2_cos_RT_CT_Mask(assimilated='IS', region_name=region)
        TM5_IS_ACOS_cos_RT_xCO2_SAT_Eva = load_datasets.load_gdf_TM5_xCO2_cos_RT_Evas_mask(assimilated='IS_ACOS', region_name=region)
        TM5_IS_ACOS_cos_RT_xCO2_SAT_CT = load_datasets.load_gdf_TM5_xCO2_cos_RT_CT_Mask(assimilated='IS_ACOS', region_name=region)
        TM5_IS_RT_cos_RT_xCO2_SAT_Eva = load_datasets.load_gdf_TM5_xCO2_cos_RT_Evas_mask(assimilated='IS_RT', region_name=region)
        TM5_IS_RT_cos_RT_xCO2_SAT_CT = load_datasets.load_gdf_TM5_xCO2_cos_RT_CT_Mask(assimilated='IS_RT', region_name=region)
        plot_xCO2_func.plot_xCO2_Timeseries_and_MSC_general(start_month=5,start_year=2009, end_year=2019, background_xCO2_timeseries=background_xco2_time_series_SAT,
                            gdf_list=[TM5_apri_cos_RT_xCO2_SAT_Eva, TM5_apri_cos_RT_xCO2_SAT_CT, TM5_IS_cos_RT_xCO2_SAT_Eva, TM5_IS_cos_RT_xCO2_SAT_CT, 
                                      TM5_IS_ACOS_cos_RT_xCO2_SAT_Eva, TM5_IS_ACOS_cos_RT_xCO2_SAT_CT, TM5_IS_RT_cos_RT_xCO2_SAT_Eva, TM5_IS_RT_cos_RT_xCO2_SAT_CT],
                            cosampled_list=['True', 'True', 'True', 'True', 'True', 'True', 'True', 'True'],
                            model_ass_cos_list=['TM5_apri_cos_RT_old_mask', 'TM5_apri_cos_RT_CT_mask', 'TM5_IS_cos_RT_old_mask', 'TM5_IS_cos_RT_CT_mask',
                                                'TM5_IS_ACOS_cos_RT_old_mask', 'TM5_IS_ACOS_cos_RT_CT_mask', 'TM5_IS_RT_cos_RT_old_mask', 'TM5_IS_RT_cos_RT_CT_mask'],
                            what_is_plotted='Modelled_xCO2',
                            color_list=['grey', 'grey', 'black', 'black', 'firebrick', 'firebrick', 'coral', 'coral'],
                            linestyle_list=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed'],
                            region_name=region,
                            savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/xCO2_plots_compare_CT_Mask_Sourish_Mask/',
                            compare_case=1)
        '''
        # 00 plot measured xCO2 compare Evas and CT Mask datasets
        '''
        GOSAT_ACOS_measure_Eva = load_datasets.load_sat_measurements_Evas_mask(satellite='GOSAT_ACOS', region=region)
        GOSAT_ACOS_measure_CT = load_datasets.load_sat_measurements_CT_Mask(satellite='GOSAT_ACOS', region=region)
        GOSAT_RT_measure_Eva = load_datasets.load_sat_measurements_Evas_mask(satellite='GOSAT_RT', region=region)
        GOSAT_RT_measure_CT = load_datasets.load_sat_measurements_CT_Mask(satellite='GOSAT_RT', region=region)
        OCO2_measure_Eva = load_datasets.load_sat_measurements_Evas_mask(satellite='OCO2', region=region)
        OCO2_measure_CT = load_datasets.load_sat_measurements_CT_Mask(satellite='OCO2', region=region)
        
        plot_xCO2_func.plot_measured_xCO2_Timeseries_and_MSC_general(start_month=5,start_year=2009, end_year=2019, 
                            background_xco2_timeseries=background_xco2_time_series_SAT,
                            gdf_list=[GOSAT_ACOS_measure_Eva, GOSAT_ACOS_measure_CT, GOSAT_RT_measure_Eva, GOSAT_RT_measure_CT, OCO2_measure_Eva, OCO2_measure_CT],
                            columns_plotted=['xco2_monthly_mean', 'xco2_monthly_mean', 'xco2_monthly_mean', 'xco2_monthly_mean', 'xco2_monthly_mean', 'xco2_monthly_mean'],
                            satellite_list=['GOSAT_ACOS_Eva_mask', 'GOSAT_ACOS_CT_mask', 'GOSAT_RT_Eva_mask', 'GOSAT_RT_CT_mask', 'OCO2_Eva_mask', 'OCO2_CT_mask'],
                            color_list=['firebrick', 'firebrick', 'coral', 'coral', 'royalblue', 'royalblue'],
                            linestyle_list=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed'],
                            region_name=region,
                            savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/xCO2_plots_compare_CT_Mask_Sourish_Mask/',
                            compare_case=1)
        '''
        
        
        # 1 plot xCO2 timeseries for ONE region
        '''
        #TM5_xCO2_prior_subregion = load_datasets.load_gdf_TM5_xCO2_cos_RT_CT_Mask(assimilated='apri', region_name=region)
        #TM5_xCO2_IS_subregion = load_datasets.load_gdf_TM5_xCO2_cos_RT_CT_Mask(assimilated='IS', region_name=region)
        TM5_xCO2_IS_subregion = load_datasets.load_gdf_TM5_xCO2_NOT_cos_CT_Mask(assimilated='IS', region_name=region)
        #TM5_xCO2_IS_ACOS_subregion = load_datasets.load_gdf_TM5_xCO2_cos_RT_CT_Mask(assimilated='IS_ACOS', region_name=region)
        TM5_xCO2_IS_ACOS_subregion = load_datasets.load_gdf_TM5_xCO2_NOT_cos_CT_Mask(assimilated='IS_ACOS', region_name=region)
        #TM5_xCO2_IS_RT_loc_subregion = load_datasets.load_gdf_TM5_xCO2_cos_RT_CT_Mask(assimilated='IS_RT', region_name=region)
        TM5_xCO2_IS_RT_loc_subregion = load_datasets.load_gdf_TM5_xCO2_NOT_cos_CT_Mask(assimilated='IS_RT', region_name=region)
        plot_xCO2_func.plotTimeseries_MSC_start_May_subregion(start_month=5,
                                                              TM5_xCO2_prior_subregion = None,
                                                              TM5_xCO2_IS_subregion = TM5_xCO2_IS_subregion,
                                                              TM5_xCO2_IS_ACOS_subregion = TM5_xCO2_IS_ACOS_subregion,
                                                              TM5_xCO2_IS_RT_loc_subregion = TM5_xCO2_IS_RT_loc_subregion,
                                                              background_xco2_timeseries = background_xco2_time_series_SAT,
                                                              cosampled_on='', region=region, 
                                                              savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/xCO2_plots_comparing_one_subregion/', 
                                                              compare_case=2)
        '''
        
    
    # 2 plot xCO2 timeseries for three subregions in one plot
    '''
    for assimilated in assimilation_list_not_cos:
        TM5_xCO2_SAT = load_datasets.load_gdf_TM5_xCO2_NOT_cos_CT_Mask(assimilated=assimilated, region_name='SAT')
        TM5_xCO2_north_SAT = load_datasets.load_gdf_TM5_xCO2_NOT_cos_CT_Mask(assimilated=assimilated, region_name='north_SAT')
        TM5_xCO2_mid_SAT = load_datasets.load_gdf_TM5_xCO2_NOT_cos_CT_Mask(assimilated=assimilated, region_name='mid_SAT')
        TM5_xCO2_south_SAT = load_datasets.load_gdf_TM5_xCO2_NOT_cos_CT_Mask(assimilated=assimilated, region_name='south_SAT')
        
        plot_xCO2_func.plotTimeseries_MSC_start_May_subregion_comparison(start_month=5,
                                                                            TM5_xCO2_SAT = TM5_xCO2_SAT,
                                                                            TM5_xCO2_north_SAT = TM5_xCO2_north_SAT,
                                                                            TM5_xCO2_mid_SAT = TM5_xCO2_mid_SAT,
                                                                            TM5_xCO2_south_SAT = TM5_xCO2_south_SAT,
                                                                            background_xco2_timeseries = background_xco2_time_series_SAT,
                                                                            cosampled_on = 'RT',
                                                                            assimilated=assimilated, 
                                                                            savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/xCO2_plots_comparing_multiple_subregions/',
                                                                            compare_case=11)
    '''
    
    # 3 plot measured xCO2 with CT Mask from 2009 to 2020
    '''
    start_year=2009
    end_year=2019
    GOSAT_ACOS_measure_CT = load_datasets.load_sat_measurements_CT_Mask(satellite='GOSAT_ACOS', region='SAT', start_year=start_year, end_year=end_year)
    GOSAT_RT_measure_CT = load_datasets.load_sat_measurements_CT_Mask(satellite='GOSAT_RT', region='SAT', start_year=start_year, end_year=end_year)
    OCO2_measure_CT = load_datasets.load_sat_measurements_CT_Mask(satellite='OCO2', region='SAT', start_year=start_year, end_year=end_year)
    
    # OCO2 is not plotted further than 2020 because the background_xco2_time_series only has data until 2020!
    plot_xCO2_func.plot_measured_xCO2_Timeseries_and_MSC_general(start_month=5,start_year=start_year, end_year=end_year, 
                        background_xco2_timeseries=background_xco2_time_series_SAT,
                        gdf_list=[GOSAT_ACOS_measure_CT, GOSAT_RT_measure_CT, OCO2_measure_CT],
                        columns_plotted=['xco2_monthly_mean', 'xco2_monthly_mean', 'xco2_monthly_mean'],
                        satellite_list=['GOSAT_ACOS', 'GOSAT_RT', 'OCO2'],
                        color_list=['firebrick', 'coral', 'royalblue'],
                        linestyle_list=['solid', 'solid', 'solid'],
                        region_name='SAT',
                        savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/xCO2_plots_supplemental/',
                        compare_case=1)
    '''
    
    
    # 4 plot TM5/IS+ACOS xCO2 cosampled & not cosampled 
    '''
    region='SAT'
    TM5_IS_ACOS_cos_RT_xCO2_SAT_CT = load_datasets.load_gdf_TM5_xCO2_cos_RT_CT_Mask(assimilated='IS_ACOS', region_name=region)
    TM5_IS_RT_cos_RT_xCO2_SAT_CT = load_datasets.load_gdf_TM5_xCO2_cos_RT_CT_Mask(assimilated='IS_RT', region_name=region)
    TM5_IS_ACOS_NOT_cos_xCO2_SAT_CT = load_datasets.load_gdf_TM5_xCO2_NOT_cos_CT_Mask(assimilated='IS_ACOS', region_name=region)
    TM5_IS_RT_NOT_cos_xCO2_SAT_CT = load_datasets.load_gdf_TM5_xCO2_NOT_cos_CT_Mask(assimilated='IS_RT', region_name=region)
    
    plot_xCO2_func.plot_xCO2_Timeseries_and_MSC_general(start_month=5,start_year=2009, end_year=2019, 
                                                        background_xCO2_timeseries=background_xco2_time_series_SAT,
                        gdf_list=[TM5_IS_ACOS_cos_RT_xCO2_SAT_CT, TM5_IS_RT_cos_RT_xCO2_SAT_CT, TM5_IS_ACOS_NOT_cos_xCO2_SAT_CT],
                        cosampled_list=['True', 'True', 'False'],
                        model_ass_cos_list=['TM5_IS_ACOS_cos_RT', 'TM5_IS_RT_cos_RT', 'TM5_IS_ACOS_NOT_cos'],
                        what_is_plotted='Modelled_xCO2',
                        color_list=['firebrick', 'dimgrey', 'midnightblue'],
                        linestyle_list=['solid', 'solid', 'solid'],
                        region_name=region,
                        savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/',
                        compare_case=1)
    
    
    plot_xCO2_func.plot_xCO2_Timeseries_and_MSC_general(start_month=5,start_year=2009, end_year=2019, 
                                                        background_xCO2_timeseries=background_xco2_time_series_SAT,
                        gdf_list=[TM5_IS_RT_NOT_cos_xCO2_SAT_CT, TM5_IS_ACOS_NOT_cos_xCO2_SAT_CT],
                        cosampled_list=['False', 'False'],
                        model_ass_cos_list=['TM5_IS_RT_NOT_cos', 'TM5_IS_ACOS_NOT_cos'],
                        what_is_plotted='Modelled_xCO2',
                        color_list=['firebrick', 'coral'],
                        linestyle_list=['solid', 'solid'],
                        region_name=region,
                        savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/',
                        compare_case=1)
    '''
    
    # 5 plot measured xCO2 with CT Mask with 2 or three MSCs
    '''
    start_year=2009
    end_year=2018
    GOSAT_ACOS_measure_CT = load_datasets.load_sat_measurements_CT_Mask(satellite='GOSAT_ACOS', region='SAT', start_year=start_year, end_year=end_year)
    GOSAT_RT_measure_CT = load_datasets.load_sat_measurements_CT_Mask(satellite='GOSAT_RT', region='SAT', start_year=start_year, end_year=end_year)
    OCO2_measure_CT = load_datasets.load_sat_measurements_CT_Mask(satellite='OCO2', region='SAT', start_year=start_year, end_year=end_year)
    
    # OCO2 is not plotted further than 2020 because the background_xco2_time_series only has data until 2020!
    plot_xCO2_func.plot_measured_xCO2_Timeseries_and_MSC_general(start_month=5,start_year=start_year, end_year=end_year, 
                        background_xco2_timeseries=background_xco2_time_series_SAT,
                        gdf_list=[GOSAT_ACOS_measure_CT, GOSAT_RT_measure_CT, OCO2_measure_CT],
                        plot_second_MSCs=True,
                        second_MSC_start_year=2009,
                        second_MSC_end_year=2014,
                        plot_third_MSCs=True,
                        third_MSC_start_year=2015,
                        third_MSC_end_year=2018,
                        columns_plotted=['xco2_monthly_mean', 'xco2_monthly_mean', 'xco2_monthly_mean'],
                        satellite_list=['GOSAT_ACOS', 'GOSAT_RT', 'OCO2'],
                        color_list=['firebrick', 'coral', 'royalblue'],
                        linestyle_list=['solid', 'solid', 'solid'],
                        region_name='SAT',
                        savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/xCO2_plots_supplemental/',
                        compare_case=1)
    '''
    
    
    # 6 plot modelled xCO2 cosampled RT
    '''
    region='SAT'
    TM5_apri_cos_RT_xCO2_SAT_CT = load_datasets.load_gdf_TM5_xCO2_cos_RT_CT_Mask(assimilated='apri', region_name=region)
    TM5_IS_cos_RT_xCO2_SAT_CT = load_datasets.load_gdf_TM5_xCO2_cos_RT_CT_Mask(assimilated='IS', region_name=region)
    TM5_IS_ACOS_cos_RT_xCO2_SAT_CT = load_datasets.load_gdf_TM5_xCO2_cos_RT_CT_Mask(assimilated='IS_ACOS', region_name=region)
    TM5_IS_RT_cos_RT_xCO2_SAT_CT = load_datasets.load_gdf_TM5_xCO2_cos_RT_CT_Mask(assimilated='IS_RT', region_name=region)
    plot_xCO2_func.plot_xCO2_Timeseries_and_MSC_general(start_month=5,start_year=2009, end_year=2019, 
                        background_xCO2_timeseries=background_xco2_time_series_SAT,
                        gdf_list=[TM5_apri_cos_RT_xCO2_SAT_CT, TM5_IS_cos_RT_xCO2_SAT_CT, 
                                  TM5_IS_ACOS_cos_RT_xCO2_SAT_CT, TM5_IS_RT_cos_RT_xCO2_SAT_CT],
                        cosampled_list=['True', 'True', 'True', 'True'],
                        model_ass_cos_list=['TM5_apri_cos_RT_CT_mask', 'TM5_IS_cos_RT_CT_mask',
                                            'TM5_IS_ACOS_cos_RT_CT_mask', 'TM5_IS_RT_cos_RT_CT_mask'],
                        label_list=['TM5-4DVar apri cos. RT', 'TM5-4DVar/IS cos. RT', 'TM5-4DVar/IS+ACOS cos. RT', 'TM5-4DVar/IS+RT cos. RT'],
                        what_is_plotted='Modelled_xCO2',
                        color_list=['dimgrey', 'indianred', 'firebrick', 'coral'],
                        linestyle_list=['dashed', 'dashed', 'solid', 'solid'],
                        region_name=region,
                        savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/xCO2_plots_comparing_one_subregion/supplemental/',
                        compare_case=1)
    '''
    
    
    # 7 plot modelled xCO2 NOT cosampled
    
    TM5_IS_NOT_cos_xCO2_SAT_CT = load_datasets.load_gdf_TM5_xCO2_NOT_cos_CT_Mask(assimilated='IS', region_name='SAT')
    TM5_IS_ACOS_NOT_cos_xCO2_SAT_CT = load_datasets.load_gdf_TM5_xCO2_NOT_cos_CT_Mask(assimilated='IS_ACOS', region_name='SAT')
    TM5_IS_RT_NOT_cos_xCO2_SAT_CT = load_datasets.load_gdf_TM5_xCO2_NOT_cos_CT_Mask(assimilated='IS_RT', region_name='SAT')
    
    plot_xCO2_func.plot_xCO2_Timeseries_and_MSC_general(start_month=5,start_year=2009, end_year=2019, 
                        background_xCO2_timeseries=background_xco2_time_series_SAT,
                        gdf_list=[TM5_IS_NOT_cos_xCO2_SAT_CT, TM5_IS_ACOS_NOT_cos_xCO2_SAT_CT, TM5_IS_RT_NOT_cos_xCO2_SAT_CT],
                        cosampled_list=['False', 'False', 'False'],
                        model_ass_cos_list=['TM5_IS_NOT_cos_RT_CT_mask', 'TM5_IS_ACOS_NOT_cos_RT_CT_mask', 'TM5_IS_RT_NOT_cos_RT_CT_mask'],
                        label_list=['TM5-4DVar/IS not cos.', 'TM5-4DVar/IS+ACOS not cos.', 'TM5-4DVar/IS+RT not cos.'],
                        what_is_plotted='Modelled_xCO2',
                        color_list=['indianred', 'firebrick', 'coral'],
                        linestyle_list=['dashed', 'solid', 'solid'],
                        region_name=region,
                        savepath='/home/lartelt/MA/software/11_Cut_GDF_with_CT_Mask/xCO2_plots_comparing_one_subregion/supplemental/',
                        compare_case=1)
    