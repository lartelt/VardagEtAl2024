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
import plotting_map_functions as plot_map_func

def cut_gridded_gdf_into_region_from_CT_Mask(gdf: geopandas.GeoDataFrame=None, CT_Mask: pd.DataFrame=None, region_name: str='SAT', savepath: str=None, savename: str=None) -> geopandas.GeoDataFrame:
    '''Documentation
    Function to cut a gridded gdf into a region from the CT Mask. Returned is a gdf with gridded cells that are inside the region.
    ### arguments:
        - gdf: geopandas.GeoDataFrame
        - region_name: str
            - 'SAT_SATr'
            - 'SAT'
            - 'SATr
        - savepath: str
    ### returns:
        - gdf: geopandas.GeoDataFrame
    '''
    print('Start cutting gridded gdf into region from CT Mask')
    if region_name=='SAT_SATr':
        igdf1 = gdf.within(CT_Mask[(CT_Mask.transcom == 'SAT')].geometry.iloc[0])
        igdf2 = gdf.within(CT_Mask[(CT_Mask.transcom == 'SATr')].geometry.iloc[0])
        gdf_SAT = gdf.loc[igdf1]
        gdf_SATr = gdf.loc[igdf2]
        gdf = pd.concat([gdf_SAT, gdf_SATr])
    elif region_name=='north_SAT':
        gdf.drop(gdf[gdf.latitude < -25].index, inplace=True) # north SAT all above -25° latitude
        gdf = gdf.reset_index(drop=True)
    elif region_name=='mid_SAT':
        gdf.drop(gdf[gdf.latitude >= -25].index, inplace=True)
        gdf.drop(gdf[gdf.latitude <=-38].index, inplace=True) # mid SAT between -25° and -38° latitude
        gdf = gdf.reset_index(drop=True)
    elif region_name=='south_SAT':
        gdf.drop(gdf[gdf.latitude > -38].index, inplace=True) # south SAT all below -38° latitude
        gdf = gdf.reset_index(drop=True)
    elif region_name=='west_SAT':
        gdf.drop(gdf[gdf.longitude > -65].index, inplace=True) # west SAT all long below -65° longitude
        gdf = gdf.reset_index(drop=True)
    elif region_name=='mid_long_SAT':
        gdf.drop(gdf[gdf.longitude <= -65].index, inplace=True)
        gdf.drop(gdf[gdf.longitude >= -50].index, inplace=True) # mid_long SAT between -65° and -50° longitude
        gdf = gdf.reset_index(drop=True)
    elif region_name=='east_SAT':
        gdf.drop(gdf[gdf.longitude < -50].index, inplace=True) # east SAT all long above -50° longitude
        gdf = gdf.reset_index(drop=True)
    else:
        igdf = gdf.within(CT_Mask[(CT_Mask.transcom == region_name)].geometry.iloc[0])
        gdf = gdf.loc[igdf]
    print('Save cutted gdf to pickle file')
    gdf.to_pickle(savepath+savename+'_'+region_name+'_CT_Mask.pkl')
    print('Done saving')
    return gdf

def calculate_subregional_total_flux(gdf: geopandas.GeoDataFrame=None, model: str='TM5', savepath: str=None, savename: str=None) -> geopandas.GeoDataFrame:
    '''Documentation
    The function takes a geodataframe with gridded fluxes per gridcell [TgC/gridcell/month] and calculates the total flux for each subregion [TgC/subregion/month].
    ### args
        gdf: geodataframe
            Geodataframe with gridded fluxes in TgC/gridcell/month
        model: str
            - 'TM5', 'MIP'
    ### return
        gdf: geodataframe
            Geodataframe with added columns for the total flux of each subregion in TgC/subregion/month
    '''
    print('Calculating total flux per subregion...')
    if model=='TM5':
        gdf_flux_per_subregion = gdf.groupby(['MonthDate'])[['CO2_NEE_flux_monthly_TgC_per_gridcell','CO2_fire_flux_monthly_TgC_per_gridcell',
                                                            'CO2_ocean_flux_monthly_TgC_per_gridcell','CO2_fossil_flux_monthly_TgC_per_gridcell',
                                                            'CO2_NEE_fire_flux_monthly_TgC_per_gridcell']].sum().reset_index()
        gdf_flux_per_subregion.rename(columns={'CO2_NEE_flux_monthly_TgC_per_gridcell':'CO2_NEE_flux_monthly_TgC_per_subregion',
                                            'CO2_fire_flux_monthly_TgC_per_gridcell':'CO2_fire_flux_monthly_TgC_per_subregion',
                                            'CO2_ocean_flux_monthly_TgC_per_gridcell':'CO2_ocean_flux_monthly_TgC_per_subregion',
                                            'CO2_fossil_flux_monthly_TgC_per_gridcell':'CO2_fossil_flux_monthly_TgC_per_subregion',
                                            'CO2_NEE_fire_flux_monthly_TgC_per_gridcell':'CO2_NEE_fire_flux_monthly_TgC_per_subregion'}, inplace=True)
    elif model=='MIP_ens':
        gdf_flux_per_subregion = gdf.groupby(['MonthDate'])[['LandStdtot', 'Landtot']].sum().reset_index()
        #gdf_flux_per_subregion.rename(columns={'LandStdtot':'LandStdtot_per_subregion', 'Landtot':'Landtot_per_subregion'}, inplace=True)
    elif model=='MIP_TM5':
        gdf_flux_per_subregion = gdf.groupby(['MonthDate'])[['Landtot']].sum().reset_index()
    gdf_flux_per_subregion['year'] = gdf_flux_per_subregion.apply(lambda x: int(x.MonthDate.year), axis=1)
    gdf_flux_per_subregion.to_pickle(savepath+savename)
    return gdf_flux_per_subregion


if __name__=='__main__':
    # read out CarbonTracker region mask
    CT_Mask = pd.read_pickle('/home/eschoema/masks/CTTranscomMask1x1Borders.pkl')
    
    max_region_list = ['global', 'SA_square', 'SAT_SATr', 'SAT', 'SATr']
    region_list_CT_Mask = ['SAT_SATr', 'SAT', 'SATr']
    region_list_subregions = ['north_SAT', 'mid_SAT', 'south_SAT', 'west_SAT', 'mid_long_SAT', 'east_SAT']
    
    list_of_satellite_xCO2_measurements = ['GOSAT_ACOS', 'GOSAT_RT', 'OCO2'] # ACOS & RT & OCO2
    
    #list_of_TM5_xCO2_cos_RT = ['TM5-4DVar_IS_ACOS_ass_cos_RT', 'TM5-4DVar_IS_RT_ass_cos_RT', 'TM5-4DVar_IS_ass_cos_RT', 'TM5-4DVar_apri_cos_RT']
    assimilation_list_TM5_xCO2_cos_RT = ['IS_ACOS', 'IS_RT', 'IS', 'apri']
    # TM5/IS_ACOS_cos_RemoteC & TM5/IS_cos_RemoteC_apri & TM5/IS_cos_RemoteC & TM5/IS_RemoteC_cos_RemoteC
    assimilation_list_TM5_xCO2_NOT_cos = ['IS', 'IS_RT', 'ACOS']
    assimilation_list_TM5_xCO2_NOT_cos_short = ['IS_ACOS']
    
    assimilation_list_TM5_flux_gridded = ['IS_ACOS', 'IS_RT', 'IS', 'prior'] # TM5 gridded 1x1 resolution
    assimilation_list_MIP_flux_gridded = ['IS', 'IS_OCO2'] # MIP
    model_list_MIP_flux_gridded = ['MIP_ens', 'MIP_TM5']
    
    
    #1 cut satellite measurements into CT regions
    '''
    for satellite in list_of_satellite_xCO2_measurements:
        if satellite!='OCO2_measurements':
            ds = load_datasets.load_sat_measurements(satellite=satellite, region=max_region_list[0])
        else:
            ds = load_datasets.load_sat_measurements(satellite=satellite, region=max_region_list[1])
        for region in region_list_CT_Mask_short:
            cut_gridded_gdf_into_region_from_CT_Mask(ds, CT_Mask, region_name=region, 
                                                     savepath='/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/Satellite_measurements/',
                                                     savename=satellite)
    '''
    #100 cut satellite measurements into subregions based on CT Mask
    '''
    for satellite in list_of_satellite_xCO2_measurements:
        for region in region_list_subregions:
            ds = load_datasets.load_sat_measurements_CT_Mask(satellite=satellite, region=max_region_list[3])
            cut_gridded_gdf_into_region_from_CT_Mask(ds, CT_Mask, region_name=region, 
                                                     savepath='/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/Satellite_measurements/',
                                                     savename=satellite)
    '''
    
    #2 cut TM5_xCO2_cos_RT into CT regions
    '''
    for assimilation in assimilation_list_TM5_xCO2_cos_RT:
        ds = load_datasets.load_gdf_TM5_xCO2_cos_RT(assimilated=assimilation, region_name=max_region_list[0])
        for region in region_list_CT_Mask_short:
            ds = cut_gridded_gdf_into_region_from_CT_Mask(ds, CT_Mask, region_name=region, 
                                                          savepath='/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_xCO2_cos_RT/',
                                                          savename='TM5-4DVar_'+assimilation+'ass_cos_RT')
    '''
    #200 cut TM5_xCO2_cos_RT into subregions based on CT Mask
    '''
    for assimilation in assimilation_list_TM5_xCO2_cos_RT:
        for region in region_list_subregions:
            ds = load_datasets.load_gdf_TM5_xCO2_cos_RT_CT_Mask(assimilated=assimilation, region_name=max_region_list[3])
            cut_gridded_gdf_into_region_from_CT_Mask(ds, CT_Mask, region_name=region, 
                                                     savepath='/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_xCO2_cos_RT/',
                                                     savename='TM5-4DVar_'+assimilation+'_ass_cos_RT')
    '''
    
    #3 cut TM5 1x1 gridded fluxes into CT regions
    '''
    for assimilation in assimilation_list_TM5_flux_gridded:
        ds = load_datasets.return_dataframe_from_name_TM5_gridded_1x1_right_units(assimilated=assimilation, region=max_region_list[1], unit='per_gridcell')
        for region in region_list_CT_Mask_short:
            ds1 = cut_gridded_gdf_into_region_from_CT_Mask(ds, CT_Mask, region_name=region,
                                                           savepath='/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_flux_gridded_1x1/',
                                                           savename='TM5-4DVar_'+assimilation+'_ass_flux_gridded_1x1_per_gridcell')
    '''
    #30 calculate TM5 1x1 flux per_subregion
    '''
    for assimilation in assimilation_list_TM5_flux_gridded:
        for region in region_list_CT_Mask:
            ds = load_datasets.return_dataframe_from_name_TM5_gridded_1x1_CT_Mask(assimilated=assimilation, region=region, unit='per_gridcell')
            ds_per_subregion = calculate_subregional_total_flux(ds, model='TM5', 
                                                                savepath='/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_flux_gridded_1x1/',
                                                                savename='TM5-4DVar_'+assimilation+'_ass_flux_gridded_1x1_per_subregion_'+region+'_CT_Mask.pkl')
    '''
    #300 cut TM5 1x1 gridded fluxes into subregions based on CT Mask
    '''
    for assimilation in assimilation_list_TM5_flux_gridded:
        for region in region_list_subregions:
            ds = load_datasets.return_dataframe_from_name_TM5_gridded_1x1_CT_Mask(assimilated=assimilation, region=max_region_list[3], unit='per_gridcell')
            cut_gridded_gdf_into_region_from_CT_Mask(ds, CT_Mask, region_name=region,
                                                     savepath='/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_flux_gridded_1x1/',
                                                     savename='TM5-4DVar_'+assimilation+'_ass_flux_gridded_1x1_per_gridcell')
    '''
    #3000 calculate TM5 1x1 flux per_subregion
    '''
    for assimilation in assimilation_list_TM5_flux_gridded:
        for region in region_list_subregions:
            ds = load_datasets.return_dataframe_from_name_TM5_gridded_1x1_CT_Mask(assimilated=assimilation, region=region, unit='per_gridcell')
            ds_per_subregion = calculate_subregional_total_flux(ds, model='TM5',
                                                                savepath='/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_flux_gridded_1x1/',
                                                                savename='TM5-4DVar_'+assimilation+'_ass_flux_gridded_1x1_per_subregion_'+region+'_CT_Mask.pkl')
    '''
    
    #4 cut TM5 3x2 gridded fluxes into CT regions
    '''
    for assimilation in assimilation_list_TM5_flux_gridded:
        ds = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_right_units(assimilated=assimilation, region=max_region_list[1], unit='per_gridcell')
        for region in region_list_CT_Mask:
            ds1 = cut_gridded_gdf_into_region_from_CT_Mask(ds, CT_Mask, region_name=region,
                                                           savepath='/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_flux_gridded_3x2_from_coarsen/',
                                                           savename='TM5-4DVar_'+assimilation+'_ass_flux_gridded_3x2_per_gridcell')
    '''
    #40 calculate TM5 3x2 flux per_subregion
    '''
    for assimilation in assimilation_list_TM5_flux_gridded:
        for region in region_list_CT_Mask:
            ds = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated=assimilation, region=region, unit='per_gridcell')
            ds_per_subregion = calculate_subregional_total_flux(ds, model='TM5', 
                                                                savepath='/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_flux_gridded_3x2_from_coarsen/',
                                                                savename='TM5-4DVar_'+assimilation+'_ass_flux_gridded_3x2_per_subregion_'+region+'_CT_Mask.pkl')
    '''
    #400 cut TM5 3x2 gridded fluxes into subregions based on CT Mask
    '''
    for assimilation in assimilation_list_TM5_flux_gridded:
        for region in region_list_subregions:
            ds = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated=assimilation, region=max_region_list[3], unit='per_gridcell')
            cut_gridded_gdf_into_region_from_CT_Mask(ds, CT_Mask, region_name=region,
                                                     savepath='/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_flux_gridded_3x2_from_coarsen/',
                                                     savename='TM5-4DVar_'+assimilation+'_ass_flux_gridded_3x2_per_gridcell')
    '''
    #4000 calculate TM5 3x2 flux per_subregion for subregions like north_SAT
    '''
    for assimilation in assimilation_list_TM5_flux_gridded:
        for region in region_list_subregions:
            ds = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated=assimilation, region=region, unit='per_gridcell')
            calculate_subregional_total_flux(ds, model='TM5', 
                                             savepath='/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_flux_gridded_3x2_from_coarsen/',
                                             savename='TM5-4DVar_'+assimilation+'_ass_flux_gridded_3x2_per_subregion_'+region+'_CT_Mask.pkl')
    '''
    
    #5 cut MIP 1x1 gridded fluxes into CT regions
    '''
    for assimilation in assimilation_list_MIP_flux_gridded:
        for model in model_list_MIP_flux_gridded:
            ds = load_datasets.return_dataframe_from_name_MIP_gridded(model=model, assimilated=assimilation, region=max_region_list[1])
            for region in region_list_CT_Mask:
                if model=='MIP_ens':
                    savepath = '/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/MIP_ens_flux/'
                    savename = 'MIP_ens_'+assimilation+'_ass_flux_gridded_1x1_per_gridcell'
                elif model=='MIP_TM5':
                    savepath = '/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_flux_gridded_1x1/TM5-4DVar_from_MIP_flux/'
                    savename = 'TM5-4DVar_from_MIP_'+assimilation+'_ass_flux_gridded_1x1_per_gridcell'
                ds1 = cut_gridded_gdf_into_region_from_CT_Mask(ds, CT_Mask, region_name=region, savepath=savepath, savename=savename)
    '''
    #50 calculate MIP flux per_subregion
    '''
    for assimilation in assimilation_list_MIP_flux_gridded:
        for model in model_list_MIP_flux_gridded:
            for region in region_list_CT_Mask:
                ds = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model=model, assimilated=assimilation, region=region, unit='per_gridcell')
                if model=='MIP_ens':
                    ds_per_subregion = calculate_subregional_total_flux(ds, model=model,
                                                    savepath='/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/MIP_ens_flux/',
                                                    savename='MIP_ens_'+assimilation+'_ass_flux_gridded_1x1_per_subregion_'+region+'_CT_Mask.pkl')
                elif model=='MIP_TM5':
                    ds_per_subregion = calculate_subregional_total_flux(ds, model=model,
                                                    savepath='/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_flux_gridded_1x1/TM5-4DVar_from_MIP_flux/',
                                                    savename='TM5-4DVar_from_MIP_'+assimilation+'_ass_flux_gridded_1x1_per_subregion_'+region+'_CT_Mask.pkl')
                else:
                    print('model type not defined!')
    '''
    #500 cut MIP 1x1 gridded fluxes into subregions based on CT Mask
    '''
    for assimilation in assimilation_list_MIP_flux_gridded:
        for model in model_list_MIP_flux_gridded:
            for region in region_list_subregions:
                ds = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model=model, assimilated=assimilation, region=max_region_list[3])
                if model=='MIP_ens':
                    savepath = '/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/MIP_ens_flux/'
                    savename = 'MIP_ens_'+assimilation+'_ass_flux_gridded_1x1_per_gridcell'
                elif model=='MIP_TM5':
                    savepath = '/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_flux_gridded_1x1/TM5-4DVar_from_MIP_flux/'
                    savename = 'TM5-4DVar_from_MIP_'+assimilation+'_ass_flux_gridded_1x1_per_gridcell'
                ds1 = cut_gridded_gdf_into_region_from_CT_Mask(ds, CT_Mask, region_name=region, savepath=savepath, savename=savename)
    '''
    #5000 calculate MIP flux per_subregion for subregions
    '''
    for assimilation in assimilation_list_MIP_flux_gridded:
        for model in model_list_MIP_flux_gridded:
            for region in region_list_subregions:
                ds = load_datasets.return_dataframe_from_name_MIP_gridded_CT_Mask(model=model, assimilated=assimilation, region=region, unit='per_gridcell')
                if model=='MIP_ens':
                    ds_per_subregion = calculate_subregional_total_flux(ds, model=model,
                                                    savepath='/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/MIP_ens_flux/',
                                                    savename='MIP_ens_'+assimilation+'_ass_flux_gridded_1x1_per_subregion_'+region+'_CT_Mask.pkl')
                elif model=='MIP_TM5':
                    ds_per_subregion = calculate_subregional_total_flux(ds, model=model,
                                                    savepath='/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_flux_gridded_1x1/TM5-4DVar_from_MIP_flux/',
                                                    savename='TM5-4DVar_from_MIP_'+assimilation+'_ass_flux_gridded_1x1_per_subregion_'+region+'_CT_Mask.pkl')
                else:
                    print('model type not defined!')
    '''
    
    #6 cut TM5_xCO2 NOT cosampled into CT regions
    '''
    for assimilation in assimilation_list_TM5_xCO2_NOT_cos_short:
        ds = load_datasets.load_gdf_TM5_xCO2_NOT_cos(assimilated=assimilation, region_name=max_region_list[1])
        for region in region_list_CT_Mask:
            ds1 = cut_gridded_gdf_into_region_from_CT_Mask(ds, CT_Mask, region_name=region,
                                                           savepath='/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_xCO2_NOT_cos/',
                                                           savename='TM5-4DVar_'+assimilation+'_ass_xCO2_NOT_cos')
    '''
    #600 cut TM5_xCO2 NOT cosampled into subregion based on CT Mask
    '''
    for assimilation in assimilation_list_TM5_xCO2_NOT_cos_short:
        for region in region_list_subregions:
            ds = load_datasets.load_gdf_TM5_xCO2_NOT_cos_CT_Mask(assimilated=assimilation, region_name=max_region_list[3])
            cut_gridded_gdf_into_region_from_CT_Mask(ds, CT_Mask, region_name=region, 
                                                     savepath='/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_xCO2_NOT_cos/',
                                                     savename='TM5-4DVar_'+assimilation+'_ass_xCO2_NOT_cos')
    '''
    
    