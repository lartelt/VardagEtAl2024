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


def cut_gridded_gdf_into_region_from_CT_Mask(gdf: geopandas.GeoDataFrame=None, CT_Mask: pd.DataFrame=None, region_name: str='SAT', savepath: str=None, savename: str=None) -> geopandas.GeoDataFrame:
    '''Documentation
    Function to cut a gridded gdf into a region from the CT Mask. Returned is a gdf with gridded cells that are inside the region.
    ### arguments:
        - gdf: geopandas.GeoDataFrame
        - region_name: str
            - 'SAT_SATr'
            - 'SAT'
            - 'SATr
            - 'arid'
            - 'humid'
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
    elif region_name=='arid' or region_name=='humid':
        igdf = gdf.within(CT_Mask[(CT_Mask.aridity == region_name)].geometry.iloc[0])
        gdf = gdf.loc[igdf]
    elif region_name=='arid_west':
        gdf.drop(gdf[gdf.longitude > -60].index, inplace=True) # south SAT all west of -60° longitude
        gdf = gdf.reset_index(drop=True)
    elif region_name=='arid_east':
        gdf.drop(gdf[gdf.longitude < -60].index, inplace=True) # south SAT all east of -60° longitude
        gdf = gdf.reset_index(drop=True)
    else:
        igdf = gdf.within(CT_Mask[(CT_Mask.transcom == region_name)].geometry.iloc[0])
        gdf = gdf.loc[igdf]
    print('Save cutted gdf to pickle file')
    gdf.to_pickle(savepath+savename+'.pkl')
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
    '''# Cut TM5 3x2 gridded fluxes into arid/humid regions based on CT Mask
    # read out CarbonTracker region mask
    strict_bound = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_0_001_minMonths_5_SAT.pkl')
    loose_bound = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_0_001_minMonths_4_SAT.pkl')

    region_list = ['arid', 'humid']
    assimilation_list_TM5_flux_gridded = ['IS_ACOS', 'IS_RT', 'IS', 'prior'] # TM5 gridded 1x1 resolution
    assimilation_list_MIP_flux_gridded = ['IS', 'IS_OCO2'] # MIP
    
    for region in region_list:
        print(region)
        for boundary_type in [strict_bound, loose_bound]:
            if boundary_type is strict_bound:
                boundary = 'strict'
            elif boundary_type is loose_bound:
                boundary = 'loose'
            print('cut gdf into '+region+' region')
            for assimilation in assimilation_list_TM5_flux_gridded:
                ds = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated=assimilation, region='SAT', unit='per_gridcell')
                gdf_subregion = cut_gridded_gdf_into_region_from_CT_Mask(gdf=ds, CT_Mask=boundary_type, region_name=region, 
                                        savepath='/mnt/data/users/lartelt/MA/TM5_MIP_fluxes_cut_with_arid_humid_mask/TM5-4DVar_flux_gridded_3x2_from_coarsen/',
                                        savename='TM5-4DVar_'+assimilation+'_ass_flux_gridded_3x2_per_gridcell_'+boundary+'limit_'+region)
                print('done cutting, now calc subregional flux...')
                calculate_subregional_total_flux(gdf_subregion, model='TM5',
                                        savepath='/mnt/data/users/lartelt/MA/TM5_MIP_fluxes_cut_with_arid_humid_mask/TM5-4DVar_flux_gridded_3x2_from_coarsen/',
                                        savename='TM5-4DVar_'+assimilation+'_ass_flux_gridded_3x2_per_subregion_'+boundary+'limit_'+region+'.pkl')
        print('done')
        '''
        
        # Cut TM5 3x2 gridded fluxes into arid_west, arid_east subregions based on CT Mask
    
    strict_bound = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5_precipitation/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_0_001_minMonths_5_SAT.pkl')
    loose_bound = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5_precipitation/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_0_001_minMonths_4_SAT.pkl')
    
    region_list = ['arid_west', 'arid_east']
    assimilation_list_TM5_flux_gridded = ['IS_ACOS', 'IS_RT', 'IS', 'prior'] # TM5 gridded 1x1 resolution
    assimilation_list_MIP_flux_gridded = ['IS', 'IS_OCO2'] # MIP
    
    for region in region_list:
        print(region)
        for boundary_type in [strict_bound, loose_bound]:
            if boundary_type is strict_bound:
                boundary = 'strict'
            elif boundary_type is loose_bound:
                boundary = 'loose'
            print('cut gdf into '+region+' region')
            for i, assimilation in enumerate(assimilation_list_TM5_flux_gridded):
                ds = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated=assimilation, arid_type=boundary+'limit_arid', region='SAT', unit='per_gridcell')
                print(ds.head())
                gdf_subregion = cut_gridded_gdf_into_region_from_CT_Mask(gdf=ds, CT_Mask=boundary_type, region_name=region, 
                                        savepath='/mnt/data/users/lartelt/MA/TM5_MIP_fluxes_cut_with_arid_humid_mask/TM5-4DVar_flux_gridded_3x2_from_coarsen/',
                                        savename='TM5-4DVar_'+assimilation+'_ass_flux_gridded_3x2_per_gridcell_'+boundary+'limit_'+region)
                print('done cutting, now calc subregional flux...')
                if i==0:
                    plot_map_func.plotGDFspatially_simple_world(gdf=gdf_subregion, savepath_w_savename='/mnt/data/users/lartelt/MA/TM5_MIP_fluxes_cut_with_arid_humid_mask/TM5-4DVar_flux_gridded_3x2_from_coarsen/map_TM5_'+assimilation+'_SAT_'+boundary+'limit_'+region+'.png')
                calculate_subregional_total_flux(gdf_subregion, model='TM5',
                                        savepath='/mnt/data/users/lartelt/MA/TM5_MIP_fluxes_cut_with_arid_humid_mask/TM5-4DVar_flux_gridded_3x2_from_coarsen/',
                                        savename='TM5-4DVar_'+assimilation+'_ass_flux_gridded_3x2_per_subregion_'+boundary+'limit_'+region+'.pkl')
        print('done')
            
            

#NOT FINAL YET from here on
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
    
    