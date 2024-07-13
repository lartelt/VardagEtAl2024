# Script by Lukas Artelt, 05.05.2023
# import packages
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
sns.set(rc={'axes.facecolor':'gainsboro'})


# Satellite measurements
def load_sat_measurements_Evas_mask(satellite: str='GOSAT_ACOS', region: str='SAT'):
    '''### Documentation
    ### Function to load the measurements of the satellites
     Input: satellite (str) - name of the satellite, e.g. GOSAT_ACOS, GOSAT_RemoteC, OCO2
             region (str) - name of the region, e.g. SAT, SATr, global or region_around_SA for OCO2
     Output: gdf (geopandas dataframe) - dataframe with the measurements
    '''
    if satellite=='SAT_SATr' or region=='SAT' or region=='SATr':
        if satellite=='GOSAT_ACOS':
            path='/mnt/data/users/lartelt/MA/ACOS/ACOS_GOSAT_measurements_with_xco2_raw/ACOS_GOSAT_all_ACOS_datasets_used_15_03_2023_with_MonthDate_'+region+'.pkl'
        elif satellite=='GOSAT_RT':
            path='/mnt/data/users/lartelt/MA/RemoteC_measured/RemoteC_GOSAT_measured_with_MonthDate_used_17_01_2023_'+region+'.pkl'
        elif satellite=='OCO2':
            if region=='SA_square':
                region='region_around_SA'
            path='/mnt/data/users/lartelt/MA/OCO2/measurements_xCO2/OCO2_measurements_used_05_04_2023_with_MonthDate_'+region+'.pkl'
        else:
            print('Satellite type not known!')
    else:
        if satellite=='GOSAT_RT':
            satellite='GOSAT_RemoteC'
        path='/mnt/data/users/lartelt/MA/Satellite_xCO2_measured_subregional/'+satellite+'_measured_xCO2_region_'+region+'.pkl'
    print('loading measurements for satellite: ', satellite, ' and region: ', region)
    gdf = pd.read_pickle(path)
    if satellite == 'GOSAT_ACOS':
        gdf = gdf.rename(columns={'time': 'date'})
    if satellite == 'GOSAT_RT' or satellite=='GOSAT_RemoteC':
        gdf = gdf.rename(columns={'CO2': 'xco2'})
        gdf = gdf.rename(columns={'CO2_uncorr': 'xco2_raw'})
    print('Done loading dataset')
    return gdf

def load_sat_measurements_CT_Mask(satellite: str='GOSAT_ACOS', region: str='SAT', start_year: int=None, end_year: int=None):
    '''### Documentation
    ### Function to load the measurements of the satellites cutted into the CT Mask
     Input: satellite (str) - name of the satellite, e.g. GOSAT_ACOS, GOSAT_RemoteC, OCO2
             region (str) - name of the region, e.g. SAT, SATr, SAT_SATr
     Output: gdf (geopandas dataframe) - dataframe with the measurements
    '''
    print('loading measurements for satellite: ', satellite, ' and region: ', region)
    if region=='SAT_SATr' or region=='SAT' or region=='SATr':
        path = '/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/Satellite_measurements/'+satellite+'_measurements_'+region+'_CT_Mask.pkl'
    else:
        path = '/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/Satellite_measurements/'+satellite+'_'+region+'_CT_Mask.pkl'
    gdf = pd.read_pickle(path)
    if satellite == 'GOSAT_RT':
        gdf = gdf.rename(columns={'Lat': 'latitude', 'Long': 'longitude'})
        gdf = gdf.rename(columns={'CO2': 'xco2'})
        gdf = gdf.rename(columns={'CO2_uncorr': 'xco2_raw'})
        gdf = gdf.rename(columns={'Year': 'year'})
    if start_year!=None:
        gdf.drop(gdf[(gdf['year'] < start_year)].index, inplace=True)
    if end_year!=None:
        gdf.drop(gdf[(gdf['year'] > end_year)].index, inplace=True)
    print('Done loading dataset')
    return gdf


# TM5 xCO2 cosampled RT
def load_gdf_TM5_xCO2_cos_RT_Evas_mask(assimilated: str='IS', region_name: str='SAT'):
    '''Documentation
    Function to return a pd.dataframe from the TM5 xCO2 cos RT in SAT region
    ### arguments:
    # - assimilated: str
    #      - apri
    #      - IS
    #      - IS_ACOS
    #      - IS_RT
    # - region_name: str
    #      - global
    #      - SAT_SATr
    #      - SAT
    #      - SATr
    '''
    if assimilated=='apri':
        path = '/mnt/data/users/lartelt/MA/ACOS/TM5_apriori_xco2/TM5-4DVar_IS_cos_RemoteC/TM5_IS_cosampled_RemoteC_01-2009_07-2019_used_22_03_2023_'+region_name+'.pkl'
    elif assimilated=='IS':
        path = '/mnt/data/users/lartelt/MA/ACOS/results_from_cosampling_TM5-4DVar_IS_assimilated/cosampling_RemoteC/TM5_IS_ass_cosampling_RemoTeC_results_01-2009_07-2019_used_23_01_2023_'+region_name+'.pkl'
    elif assimilated=='IS_ACOS':
        path = '/mnt/data/users/lartelt/MA/ACOS/results_from_cosampling_TM5-4DVar_IS_ACOS_assimilated/cosampling_on_RemoTeC/TM5_IS_ACOS_assimilated_GOSAT_cosampled_RemoteC_results_01-2009_07-2019_used_27_03_2023_'+region_name+'.pkl'
    elif assimilated=='IS_RT':
        path = '/mnt/data/users/lartelt/MA/ACOS/results_from_cosampling_TM5-4DVar_IS_RemoteC_assimilated/TM5_IS_RemoteC_assimilated_GOSAT_cosampling_results_04-2009_06-2019_used_20_01_2023_'+region_name+'.pkl'
    else:
        print('ERROR: assimilation type not defined!')
    print('reading data for ' + assimilated + ' ass, region ' + region_name + ' from path: ' + path)
    gdf = pd.read_pickle(path)
    return gdf

# TM5 xCO2 cosampled RT CT Mask
def load_gdf_TM5_xCO2_cos_RT_CT_Mask(assimilated: str='IS', region_name: str='SAT'):
    '''Documentation
    Function to return a pd.dataframe from the TM5 xCO2 cos RT regionally cut with CT Mask
    ### arguments:
        - assimilated: str
             - apri
             - IS
             - IS_ACOS
             - IS_RT
        - region_name: str
             - SAT_SATr
             - SAT
             - SATr
    '''
    if region_name=='SAT_SATr':
        path = '/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_xCO2_cos_RT/TM5-4DVar_'+assimilated+'_ass_cos_RT_'+region_name+'_CT_Mask_04_07_2023.pkl'
    else:
        path = '/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_xCO2_cos_RT/TM5-4DVar_'+assimilated+'_ass_cos_RT_'+region_name+'_CT_Mask.pkl'
    print('reading data for ' + assimilated + ' ass, region ' + region_name + ' from path: ' + path)
    gdf = pd.read_pickle(path)
    return gdf


# TM5 xCO2 cos ACOS & NOT cos
def load_gdf_TM5_xCO2_cos_ACOS_Evas_mask(assimilated: str='IS', region_name: str='SAT'):
    '''Documentation
    Function to return a pd.dataframe from the TM5 xCO2 cos ACOS in SAT region
    ATTENTION: TM5/IS+RT cos ACOS is NOT available
    ### arguments:
    # - assimilated: str
    #      - apri
    #      - IS
    #      - IS_ACOS
    #      - IS_RemoteC
    # - region_name: str
    #      - 'global' ('square_around_SA')
    #      - SAT_SATr
    #      - SAT
    #      - SATr
    '''
    if assimilated=='apri':
        path = '/mnt/data/users/lartelt/MA/ACOS/TM5_apriori_xco2/TM5-4DVar_IS_cos_ACOS/TM5_IS_cosampled_ACOS_01-2009_07-2019_used_22_03_2023_'+region_name+'.pkl'
    elif assimilated=='IS':
        path = '/mnt/data/users/lartelt/MA/ACOS/results_from_cosampling_TM5-4DVar_IS_assimilated/cosampling_ACOS/TM5_ACOS_GOSAT_cosampling_results_01-2009_07-2019_used_07_12_2022_'+region_name+'.pkl'
    elif assimilated=='IS_RT':
        print('ATTENTION! For TM5/IS+RT there is no cosampling with ACOS available!')
    elif assimilated=='IS_ACOS':
        path = '/mnt/data/users/lartelt/MA/ACOS/results_from_cosampling_TM5-4DVar_IS_ACOS_assimilated/TM5_IS_ACOS_assimilated_GOSAT_cosampling_results_01-2009_07-2019_used_07_12_2022_'+region_name+'.pkl'
    print('reading data for ' + assimilated + ' ass cos ACOS, region ' + region_name + ' from path: ' + path)
    gdf = pd.read_pickle(path)
    return gdf

def load_gdf_TM5_xCO2_NOT_cos_Evas_mask(assimilated: str='IS', region_name: str='SAT'):
    '''Documentation
    Function to return a pd.dataframe from the TM5 xCO2 NOT cosampled in SAT region
    ATTENTION: TM5/IS+ACOS and TM5_apri NOT available for not cosampled
    ### arguments:
    - assimilated: str
            - apri
            - IS
            - ACOS
            - IS_ACOS
            - IS_RemoteC
    - region_name: str
            - 'SA_square' for all now. Name is changed inside this function to 'square_around_SA' for TM5/IS and TM5/IS+RT
            - SAT_SATr
            - SAT
            - SATr
    '''
    if assimilated=='apri':
        print('ATTENTION! For TM5_apri there is ONLY cosampled results available! Either cosampled on ACOS or RT.')
    elif assimilated=='IS':
        if region_name=='SA_square':
            region_name='square_around_SA'
        path = '/mnt/data/users/lartelt/MA/ACOS/results_from_NOT_cosampled_TM5-4DVar_IS_assimilated/TM5_ACOS_GOSAT_NOT_cosampled_results_01-2009_06-2019_used_27_12_2022_'+region_name+'.pkl'
    elif assimilated=='ACOS':
        path = '/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_xCO2_NOT_cos/TM5-4DVar_ACOS_ass_NOT_cos_'+region_name+'.pkl'
    elif assimilated=='IS_RT':
        if region_name=='SA_square':
            region_name='square_around_SA'
        path = '/mnt/data/users/lartelt/MA/ACOS/results_from_NOT_cosampled_TM5-4DVar_IS_RemoteC_assimilated/TM5_IS_RemoteC_assimilated_GOSAT_NOT_cosampled_results_01-2009_06-2019_used_13_01_2023_'+region_name+'.pkl'
    elif assimilated=='IS_ACOS':
        path = '/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_xCO2_NOT_cos/TM5-4DVar_IS_ACOS_ass_NOT_cos_'+region_name+'.pkl'
    print('reading data for ' + assimilated + ' ass NOT cosampled, region ' + region_name + ' from path: ' + path)
    gdf = pd.read_pickle(path)
    gdf['modeled_XCO2'] = gdf.XCO2
    gdf['month'] = gdf['MonthDate'].apply(lambda x: x.month)
    gdf['year'] = gdf['MonthDate'].apply(lambda x: x.year)
    gdf.drop(columns='time', inplace=True)
    gdf['time'] = gdf['date']
    
    return gdf

def load_gdf_TM5_xCO2_NOT_cos_CT_Mask(assimilated: str='IS', region_name: str='SAT'):
    '''Documentation
    Function to return a pd.dataframe from the TM5 xCO2 NOT cosampled regional based on the CarbonTracker Mask
    ATTENTION: TM5/IS+ACOS and TM5_apri NOT available for NOT cosampled
    ### arguments:
    - assimilated: str
            - IS
            - ACOS
            - IS_ACOS
            - IS_RemoteC
    - region_name: str
            - SAT_SATr
            - SAT
            - SATr
            - north_SAT, 'mid_SAT', ...
    '''
    if assimilated=='apri':
        print('ATTENTION! For TM5_apri there is ONLY cosampled results available! Either cosampled on ACOS or RT.')
    elif assimilated=='IS':
        path = '/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_xCO2_NOT_cos/TM5-4DVar_IS_ass_xCO2_NOT_cos_'+region_name+'_CT_Mask.pkl'
    elif assimilated=='ACOS':
        path = '/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_xCO2_NOT_cos/TM5-4DVar_ACOS_ass_xCO2_NOT_cos_'+region_name+'_CT_Mask.pkl'
    elif assimilated=='IS_RT' or assimilated=='IS_RemoteC':
        path = '/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_xCO2_NOT_cos/TM5-4DVar_IS_RT_ass_xCO2_NOT_cos_'+region_name+'_CT_Mask.pkl'
    elif assimilated=='IS_ACOS':
        path = '/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_xCO2_NOT_cos/TM5-4DVar_IS_ACOS_ass_xCO2_NOT_cos_'+region_name+'_CT_Mask.pkl'
    print('reading data for ' + assimilated + ' ass NOT cosampled, region ' + region_name + ' from path: ' + path)
    gdf = pd.read_pickle(path)
    print('Done loading dataset!')
    return gdf
    

# Fluxes
# MIP 1x1 gridded flux
def return_dataframe_from_name_MIP_gridded(model: str, assimilated: str, region: str=None):
    '''Documentation
    Function to return a pd.dataframe from the name of the file/dataframe; fluxes given in TgC/gridcell/month
    ### arguments:
        - model: str
            - 'MIP_ens'
            - 'MIP_TM5'
        - assimilated: str
            - IS_OCO2
            - IS
        - region: str
            - 'SA_square'
    ### returns:
        - pandas dataframe
    '''
    if model=='MIP_ens':
        if assimilated=='IS':
            path = '/mnt/data/users/lartelt/MA/MIP_fluxes/Only_IS_ObsPack_data_assimilated/gdf_MIP_IS_ass_ens_SA_square_region.pkl'
        elif assimilated=='IS_OCO2':
            path = '/mnt/data/users/lartelt/MA/MIP_fluxes/gdf_MIP_IS_OCO2_ass_ens_SA_square_region.pkl'
        else:
            print('assimilation type not defined!')
    elif model=='MIP_TM5':
        if assimilated=='IS':
            path = '/mnt/data/users/lartelt/MA/MIP_fluxes/Only_IS_ObsPack_data_assimilated/gdf_MIP_IS_ass_TM5_model_SA_square_region.pkl'
        elif assimilated=='IS_OCO2':
            path = '/mnt/data/users/lartelt/MA/MIP_fluxes/gdf_MIP_IS_OCO2_ass_TM5_model_SA_square_region.pkl'
        else:
            print('assimilation type not defined!')
    else:
        print('model type not defined!')
    
    print('loading dataset ' + model + ' ' + assimilated + '_assimilated from path ' + path)
    df = pd.read_pickle(path)
    print('Done loading dataset')
    return df
# MIP 1x1 gridded flux from CT Mask
def return_dataframe_from_name_MIP_gridded_CT_Mask(model: str, assimilated: str, end_year: int=None, region: str='SAT', unit: str='per_gridcell'):
    '''Documentation
    ### arguments
       - model: str
           - 'MIP_ens'
           - 'MIP_TM5'
       - assimilated: str
           - 'IS'
           - 'IS_OCO2'
       - region: str
           - 'SAT_SATr'
           - 'SAT'
           - 'SATr'
       - end_year: str; last year that is included, e.g. end_year=2019 if data should only cover (including) 2019
       - unit: str
           - 'per_gridcell'
           - 'per_subregion'
    ### returns
       - pd.dataframe
    '''
    if region=='SAT_SATr':
        if model=='MIP_ens':
            path = '/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/MIP_ens_flux/MIP_ens_'+assimilated+'_ass_flux_gridded_1x1_'+unit+'_'+region+'_CT_Mask_04_07_2023.pkl'
        elif model=='MIP_TM5':
            path = '/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_flux_gridded_1x1/TM5-4DVar_from_MIP_flux/TM5-4DVar_from_MIP_'+assimilated+'_ass_flux_gridded_1x1_'+unit+'_'+region+'_CT_Mask_04_07_2023.pkl'
        else:
            print('model type not defined!')
    else:
        if model=='MIP_ens':
            path = '/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/MIP_ens_flux/MIP_ens_'+assimilated+'_ass_flux_gridded_1x1_'+unit+'_'+region+'_CT_Mask.pkl'
        elif model=='MIP_TM5':
            path = '/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_flux_gridded_1x1/TM5-4DVar_from_MIP_flux/TM5-4DVar_from_MIP_'+assimilated+'_ass_flux_gridded_1x1_'+unit+'_'+region+'_CT_Mask.pkl'
        else:
            print('model type not defined!')
    
    print('loading dataset ' + model + ' ' + assimilated + '_assimilated in '+region+' region from path ' + path)
    df = pd.read_pickle(path)
    if end_year!=None:
        df.drop(df[(df['year'] > end_year)].index, inplace=True)#.reset_index(drop=True)
    gdf = df.rename(columns={'Lat': 'latitude', 'Long': 'longitude'})
    gdf['Month'] = gdf['MonthDate'].apply(lambda x: x.month)
    print('Done loading dataset')
    return gdf

# TM5 1x1 gridded flux
def return_dataframe_from_name_TM5_gridded_1x1_right_units(assimilated: str, start_year: int=2009, region: str=None, unit: str=None):
    '''Documentation
    Function to return a pd.dataframe from the name of the file/dataframe
    ### arguments:
     - assimilated: str
           - IS
           - IS_ACOS
           - IS_RT
           - prior
     - start_year: int (default: 2009): Year of the first month in the returned dataset
     - region: str
           - 'SA_square'   
           - 'SAT_SATr'
           - 'SAT'
           - 'SATr'
           - 'north_SAT'
           - 'mid_SAT'
           - 'south_SAT'
     - unit: str
           - 'per_gridcell': units in returned gdf are: TgC/month/gridcell
           - 'per_subregion': units in returned gdf are: TgC/month/region
    ### returns:
     - pandas dataframe
    '''
    path='/mnt/data/users/lartelt/MA/TM5-4DVar_Fluxes_subregional/gridded_fluxes_1x1_res/TM5_'+assimilated+'_flux_grid_region_'+region+'_flux_'+unit+'.pkl'
    print('loading dataset TM5_'+assimilated+'_flux_grid_region_'+region+'_flux_'+unit+ ' from path ' + path)
    df = pd.read_pickle(path)
    df.drop(df[(df['year'] < start_year)].index, inplace=True)#.reset_index(drop=True)
    print('Done loading dataset')
    return df
# TM5 1x1 gridded flux from CT_Mask
def return_dataframe_from_name_TM5_gridded_1x1_CT_Mask(assimilated: str, start_year: int=2009, region: str=None, unit: str=None):
    '''Documentation
    Function to return a pd.dataframe for TM5-4DVar fluxes in 1x1 resolution from CT_Mask
    ### arguments:
        - assimilated: str
           - IS
           - IS_ACOS
           - IS_RT
           - prior
        - start_year: int (default: 2009): Year of the first month in the returned dataset
        - region: str
           - 'SAT_SATr'
           - 'SAT'
           - 'SATr'
        - unit: str
           - 'per_gridcell': units in returned gdf are: TgC/month/gridcell
           - NOT YET CALCULATED: 'per_subregion': units in returned gdf are: TgC/month/subregion
    ### returns:
        - pandas dataframe: gdf
    '''
    path='/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_flux_gridded_1x1/TM5-4DVar_'+assimilated+'_ass_flux_gridded_1x1_'+unit+'_'+region+'_CT_Mask.pkl'
    print('loading dataset TM5-4DVar_'+assimilated+'_ass_flux_gridded_1x1_'+unit+'_'+region+'_CT_Mask from path ' + path)
    df = pd.read_pickle(path)
    df.drop(df[(df['year'] < start_year)].index, inplace=True)#.reset_index(drop=True)
    print('Done loading dataset')
    return df


# TM5 3x2 gridded flux
def return_dataframe_from_name_TM5_gridded_3x2_right_units(assimilated: str, start_year: int=2009, region: str=None, unit: str=None):
    '''Documentation
    Function to return a pd.dataframe for TM5-4DVar fluxes in 3x2 resolution
    ### arguments:
    - assimilated: str
           - IS
           - IS_ACOS
           - IS_RT
           - prior
     - start_year: int (default: 2009): Year of the first month in the returned dataset
     - region: str
           - 'SA_square'
     - unit: str
           - 'per_gridcell': units in returned gdf are: TgC/month/gridcell
           - NOT YET CALCULATED: 'per_subregion': units in returned gdf are: TgC/month/region
    ### returns:
     - pandas dataframe
    '''
    path='/mnt/data/users/lartelt/MA/TM5-4DVar_Fluxes_subregional/gridded_fluxes_3x2_res_from_coarsen/TM5_'+assimilated+'_flux_grid_region_'+region+'_'+unit+'.pkl'
    print('loading dataset TM5_'+assimilated+'_flux_grid_region_'+region+'_flux_'+unit+ ' from path ' + path)
    df = pd.read_pickle(path)
    df['year'] = df['MonthDate'].apply(lambda x: x.year)
    df.drop(df[(df['year'] < start_year)].index, inplace=True)#.reset_index(drop=True)
    print('Done loading dataset')
    return df

# TM5 3x2 gridded flux from CT_Mask
def return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated: str, start_year: int=2009, region: str=None, unit: str=None):
    '''Documentation
    Function to return a pd.dataframe for TM5-4DVar fluxes in 3x2 resolution from CT_Mask
    ### arguments:
        - assimilated: str
           - IS
           - IS_ACOS
           - IS_RT
           - prior
        - start_year: int (default: 2009): Year of the first month in the returned dataset
        - region: str
           - 'SAT_SATr'
           - 'SAT'
           - 'SATr'
        - unit: str
           - 'per_gridcell': units in returned gdf are: TgC/month/gridcell
           - NOT YET CALCULATED: 'per_subregion': units in returned gdf are: TgC/month/subregion
    ### returns:
        - pandas dataframe: gdf
    '''
    path='/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_flux_gridded_3x2_from_coarsen/TM5-4DVar_'+assimilated+'_ass_flux_gridded_3x2_'+unit+'_'+region+'_CT_Mask.pkl'
    print('loading dataset TM5-4DVar_'+assimilated+'_flux_gridded_3x2_'+unit+'_region_'+region+'_CT_Mask from path ' + path)
    df = pd.read_pickle(path)
    df.drop(df[(df['year'] < start_year)].index, inplace=True)#.reset_index(drop=True)
    print('Done loading dataset')
    return df




# TM5 flux per region from Sourish
def dataframe_from_name_TM5_flux_per_region_Sourish(assimilated: str, region_name: str, start_year: int=2015):
    '''Documentation
    Function to return a pd.dataframe from the TM5 flux per region Sourish
    ### arguments:
     - assimilated: str
          - prior
          - IS
          - ACOSIS
          - RemoTeCISloc
     - region_name: str
          - 'SAT_SATr'
          - 'SAT'
          - 'SATr'
     - OLD: region_number: int
          - 2: northern_SATr
          - 3: southern_SATr
          - 4: SAT
          - 203: SATr = northern_SATr + southern_SATr
          - 234: SAT_SATr = northern_SATr + southern_SATr + SAT
          [18=East_Pacific_Tropical; 19=South Pacific Temperate; 22=Atlantic Tropics; 23=South Atlandic Temperate; 24=Southern Ocean]
    '''
    if assimilated == 'prior':
        assimilated = 'IS'
    path = '/mnt/data/users/lartelt/MA/TM5-4DVar_Fluxes_subregional/with_regions_18_to_24/DF1_'+assimilated+'glb3x2.pkl'
    print('loading dataset TM5_'+assimilated+'_flux_per_region_'+region_name+' from path ' + path)
    df = pd.read_pickle(path)
    if region_name == 'SAT_SATr':
        region_number = 234
    elif region_name == 'SATr':
        region_number = 203
    elif region_name == 'SAT':
        region_number = 4
    else: 
        print('Region name not specified')
    if region_number==234:
        df_nSATr = df[(df.Region == 2)]
        df_sSATr = df[(df.Region == 3)]
        df_SAT   = df[(df.Region == 4)]
        df_SAT_SATr = pd.concat([df_nSATr, df_sSATr, df_SAT])
        df_SAT_SATr = df_SAT_SATr.groupby(['Year', 'Month', 'MonthDate'])[['Flux_ocn_apri','Flux_bio_apri','Flux_ff_apri','Flux_fire_apri','Flux_apri','Flux_ocn','Flux_bio','Flux_ff','Flux_fire','Flux']].sum().reset_index()
        df_SAT_SATr.drop(df_SAT_SATr[(df_SAT_SATr['Year'] < start_year)].index, inplace=True)
        return df_SAT_SATr
    elif region_number==203:
        df_nSATr = df[(df.Region == 2)]
        df_sSATr = df[(df.Region == 3)]
        df_SATr = pd.concat([df_nSATr, df_sSATr])
        df_SATr = df_SATr.groupby(['Year', 'Month', 'MonthDate'])[['Flux_ocn_apri','Flux_bio_apri','Flux_ff_apri','Flux_fire_apri','Flux_apri','Flux_ocn','Flux_bio','Flux_ff','Flux_fire','Flux']].sum().reset_index()
        df_SATr.drop(df_SATr[(df_SATr['Year'] < start_year)].index, inplace=True)
        return df_SATr
    else:
        df = df[df.Region == region_number]
        df.drop(df[(df['Year'] < start_year)].index, inplace=True)
        return df

#NOT FINAL YET: TM5 xCO2 per region from Eva's Mask
def dataframe_from_name_TM5_xCO2_per_region_Eva_mask(assimilated: str='IS', region_name: str='SAT', start_year: int=2009):
    '''Documentation
    Function to return a pd.dataframe from the TM5 xCO2 per region from Eva's Mask
    ### arguments:
    '''



def load_GFED_model_gdf(region:str='SAT', unit:str='per_subregion', start_year:int=None, end_year:int=None):
    '''Documentation
    # arguments:
        - region: str
            - 'SAT'
            - 'SATr'
            - 'SAT_SATr'
        - unit: str
            - 'per_gridcell'
            - 'per_subregion'
    '''
    path='/mnt/data/users/lartelt/MA/GFED4/GFED_cut_with_CT_Mask/'
    gdf = pd.read_pickle(path+'GFED_CT_Mask_'+region+'_TgC_per_month_'+unit+'.pkl')
    gdf['Year'] = gdf['MonthDate'].apply(lambda x: x.year)
    gdf['Month'] = gdf['MonthDate'].apply(lambda x: x.month)
    if start_year!=None:
        gdf.drop(gdf[(gdf['Year'] < start_year)].index, inplace=True)
    if end_year!=None:
        gdf.drop(gdf[(gdf['Year'] > end_year)].index, inplace=True)
    print('Done loading data from '+path+' GFED_CT_Mask_'+region+'_TgC_per_month_'+unit+'.pkl')
    return gdf



if __name__ == "__main__":
    print('File not made to be main file!')