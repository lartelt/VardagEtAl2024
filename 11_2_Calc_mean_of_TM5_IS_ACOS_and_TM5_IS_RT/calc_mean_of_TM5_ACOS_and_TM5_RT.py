# By Lukas Artelt
# import packages
import numpy as np
import xarray as xr
import pandas as pd
import geopandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
sns.set(rc={'axes.facecolor':'gainsboro'})
import functions_to_load_datasets as load_datasets


if __name__=='__main__':
    # for SAT regions and subregions
    region_list = ['SAT_SATr', 'SAT', 'SATr', 'north_SAT', 'mid_SAT', 'south_SAT', 'east_SAT', 'mid_long_SAT', 'west_SAT']
    #region_list = ['SATr']
    
    '''# Cut TM5 3x2 gridded fluxes into subregions based on CT Mask
    for region in region_list:
        print(region)
        ds_IS_ACOS = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_ACOS', region='SAT', unit='per_subregion')
        #print(ds_IS_ACOS.head())
        ds_IS_RT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_RT', region='SAT', unit='per_subregion')
        #print(ds_IS_RT.head())
        
        ds_concat = ds_IS_ACOS.merge(ds_IS_RT, on=['MonthDate', 'year', 'Month'])
        ds_mean = ds_concat[['MonthDate', 'year', 'Month']]
        
        for column in ['CO2_NEE_flux_monthly_TgC_per_subregion','CO2_fire_flux_monthly_TgC_per_subregion',
                       'CO2_ocean_flux_monthly_TgC_per_subregion','CO2_fossil_flux_monthly_TgC_per_subregion',
                       'CO2_NEE_fire_flux_monthly_TgC_per_subregion']:
            print(column)
            ds_mean[column] = ds_concat.apply(lambda x: np.mean((x[column+'_x'], x[column+'_y'])), axis=1)
            ds_mean[column+'_std'] = ds_concat.apply(lambda x: np.std((x[column+'_x'], x[column+'_y']), ddof=0), axis=1)
        
        ds_mean.to_pickle('/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_flux_gridded_3x2_from_coarsen/TM5-4DVar_mean_of_IS_ACOS_and_IS_RT_ass_flux_gridded_3x2_per_subregion_'+region+'_CT_Mask.pkl')
    '''
    
    '''# for SAT arid/humid region
    # Cut TM5 3x2 gridded fluxes into subregions based on ERA5 arid/humid mask
    for aridity_limit in ['strict', 'loose']:
        for aridity in ['arid', 'humid']:
            print(aridity_limit+'limit_'+aridity)
            ds_IS_ACOS = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_ACOS', region=aridity_limit+'limit_'+aridity, unit='per_subregion')
            #print(ds_IS_ACOS.head())
            ds_IS_RT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_RT', region=aridity_limit+'limit_'+aridity, unit='per_subregion')
            #print(ds_IS_RT.head())
            
            ds_concat = ds_IS_ACOS.merge(ds_IS_RT, on=['MonthDate', 'year', 'Month'])
            ds_mean = ds_concat[['MonthDate', 'year', 'Month']]
            
            for column in ['CO2_NEE_flux_monthly_TgC_per_subregion','CO2_fire_flux_monthly_TgC_per_subregion',
                           'CO2_ocean_flux_monthly_TgC_per_subregion','CO2_fossil_flux_monthly_TgC_per_subregion',
                           'CO2_NEE_fire_flux_monthly_TgC_per_subregion']:
                print(column)
                ds_mean[column] = ds_concat.apply(lambda x: np.mean((x[column+'_x'], x[column+'_y'])), axis=1)
                ds_mean[column+'_std'] = ds_concat.apply(lambda x: np.std((x[column+'_x'], x[column+'_y']), ddof=0), axis=1)
            
            ds_mean.to_pickle('/mnt/data/users/lartelt/MA/TM5_MIP_fluxes_cut_with_arid_humid_mask/TM5-4DVar_flux_gridded_3x2_from_coarsen/TM5-4DVar_mean_of_IS_ACOS_and_IS_RT_ass_flux_gridded_3x2_per_subregion_'+aridity_limit+'limit_'+aridity+'.pkl')
    '''
        

    # for SAT arid subregions
    # Cut TM5 3x2 gridded fluxes into subregions based on ERA5 arid mask
    for aridity_limit in ['strict', 'loose']:
        for aridity in ['arid_west', 'arid_east']:
            print(aridity_limit+'limit_'+aridity)
            ds_IS_ACOS = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_ACOS', region=aridity_limit+'limit_'+aridity, unit='per_subregion')
            print(ds_IS_ACOS.head())
            ds_IS_RT = load_datasets.return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated='IS_RT', region=aridity_limit+'limit_'+aridity, unit='per_subregion')
            print(ds_IS_RT.head())
            
            ds_concat = ds_IS_ACOS.merge(ds_IS_RT, on=['MonthDate', 'year', 'Month'])
            ds_mean = ds_concat[['MonthDate', 'year', 'Month']]
            
            for column in ['CO2_NEE_flux_monthly_TgC_per_subregion','CO2_fire_flux_monthly_TgC_per_subregion',
                           'CO2_ocean_flux_monthly_TgC_per_subregion','CO2_fossil_flux_monthly_TgC_per_subregion',
                           'CO2_NEE_fire_flux_monthly_TgC_per_subregion']:
                print(column)
                ds_mean[column] = ds_concat.apply(lambda x: np.mean((x[column+'_x'], x[column+'_y'])), axis=1)
                ds_mean[column+'_std'] = ds_concat.apply(lambda x: np.std((x[column+'_x'], x[column+'_y']), ddof=0), axis=1)
            
            ds_mean.to_pickle('/mnt/data/users/lartelt/MA/TM5_MIP_fluxes_cut_with_arid_humid_mask/TM5-4DVar_flux_gridded_3x2_from_coarsen/TM5-4DVar_mean_of_IS_ACOS_and_IS_RT_ass_flux_gridded_3x2_per_subregion_'+aridity_limit+'limit_'+aridity+'.pkl')
            
