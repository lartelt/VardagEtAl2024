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

# calculations
def getNumDayOfMonth(year,month):
    """returns list of number of days within given month in given year"""
    listDays = [31,28,31,30,31,30,31,31,30,31,30,31]
    listDaysl = [31,29,31,30,31,30,31,31,30,31,30,31]
    if year < 1900:
        print('year is out of implemented range, check code')
    elif year in list(range(1904,2100,4)):
        days = listDaysl[month-1]
    else:
        days = listDays[month-1]

    return days


# load datasets
# TM5 3x2 gridded flux from CT_Mask
def return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated: str, start_year: int=2009, end_year: int=2020, region: str=None, unit: str=None):
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
           - 'per_subregion': units in returned gdf are: TgC/month/subregion
    ### returns:
        - pandas dataframe: gdf
    '''
    path='/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_flux_gridded_3x2_from_coarsen/TM5-4DVar_'+assimilated+'_ass_flux_gridded_3x2_'+unit+'_'+region+'_CT_Mask.pkl'
    print('loading dataset TM5-4DVar_'+assimilated+'_flux_gridded_3x2_'+unit+'_region_'+region+'_CT_Mask from path ' + path)
    df = pd.read_pickle(path)
    df.drop(df[(df['year'] < start_year)].index, inplace=True)#.reset_index(drop=True)
    df.drop(df[(df['year'] > end_year)].index, inplace=True)
    df['Month'] = df['MonthDate'].apply(lambda x: x.month)
    
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


def load_TRENDY_model_gdf(model: str='CABLE-POP', variable: str='gpp', unit: str=None, model_category: str='bad_low_ampl_models', start_year: int=None, end_year: int=None):
    """
    Load TRENDY model data from pkl files and return a geodataframe with the model data and the corresponding coordinates.
    # arguments:
        - unit:
            - None: Column=variable: kgC/(m^2*s) ; Column=variable+'tot': kgC/(s * gridcell)
            - 'TgC_per_gridcell_per_month': Column=variable+'tot_TgC_month': TgC/month/gridcell
            - 'TgC_per_region_per_month': Column=variable+'tot_TgC_month': TgC/month/region
            - 'mean_of_variable': get df with the monthly mean values of the desired variable from ALL models that have this variable covered 
                                  AND the mean, std, count of all models combined
                                  Columns: mean, std, count in UNIT TgC/month/region
            - mean_of_variable_model_category: get df with the monthly mean values of the desired variable from ONLY models inside the category defined in "model_category"
    """
    if unit=='mean_of_variable':
        path = '/mnt/data/users/lartelt/MA/TRENDY/mean_of_all_models/'
        gdf = pd.read_pickle(path+'Ensemble_mean_for_'+variable+'_SAT_721.pkl')
        gdf['Year'] = gdf['MonthDate'].apply(lambda x: x.year)
    elif unit=='mean_of_variable_model_category':
        variable = variable.lower()
        path = '/mnt/data/users/lartelt/MA/TRENDY/mean_of_all_models/mean_of_variable_'+model_category+'/'
        gdf = pd.read_pickle(path+'Mean_'+model_category+'_for_'+variable+'_SAT_721.pkl')
        gdf['Year'] = gdf['MonthDate'].apply(lambda x: x.year)
    else:
        path = '/mnt/data/users/lartelt/MA/TRENDY/'+model+'/'
        if unit==None:
            gdf = pd.read_pickle(path+'gdf1x1_'+model+'_'+variable+'_SAT_721.pkl')
            if not 'Days_in_month' in gdf.columns:
                print('Column Days_in_month not in gdf, adding column...')
                gdf['days_in_month'] = gdf.apply(lambda x: getNumDayOfMonth(int(x['Year']),int(x['Month'])), axis=1)
        elif unit=='TgC_per_gridcell_per_month':
            gdf = pd.read_pickle(path+'gdf1x1_'+model+'_'+variable+'_SAT_721_TgC_per_gridcell_per_month.pkl')
        elif unit=='TgC_per_region_per_month':
            gdf = pd.read_pickle(path+'gdf1x1_'+model+'_'+variable+'_SAT_721_TgC_per_region_per_month.pkl')
            #gdf['Year'] = gdf['MonthDate'].apply(lambda x: x.year)
            #gdf['Month'] = gdf['MonthDate'].apply(lambda x: x.month)
    #gdf.drop(gdf[(gdf['Year'] < start_year)].index, inplace=True)
    if start_year!=None:
        gdf.drop(gdf[(gdf['Year'] < start_year)].index, inplace=True)
    if end_year!=None:
        gdf.drop(gdf[(gdf['Year'] > end_year)].index, inplace=True)
    print('Done loading data for gdf1x1_'+model+'_'+variable+'_SAT_721.pkl')
    return gdf


def load_FLUXCOM_model_gdf(variable: str='NEE', region: str='SAT', unit: str='per_gridcell', start_year:int=None, end_year: int=None):
    '''Documentation
    # arguments:
        - variable: str: 'NEE', 'GPP', 'TER'
        - region: 'SAT'
    '''
    path='/mnt/data/users/lartelt/MA/FLUXCOM/'
    if variable.endswith('tot'):
        variable=variable[:-3]
    gdf = pd.read_pickle(path+'gdf_'+variable+'_'+region+'_per_month_'+unit+'.pkl')
    gdf['Year'] = gdf['MonthDate'].apply(lambda x: x.year)
    gdf['Month'] = gdf['MonthDate'].apply(lambda x: x.month)
    if start_year!=None:
        gdf.drop(gdf[(gdf['Year'] < start_year)].index, inplace=True)
    if end_year!=None:
        gdf.drop(gdf[(gdf['Year'] > end_year)].index, inplace=True)
    print('Done loading data from '+path+ 'gdf_'+variable+'_SAT_per_month_'+unit+'.pkl')
    return gdf


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
    #gdf['Year'] = gdf['MonthDate'].apply(lambda x: x.year)
    #gdf['Month'] = gdf['MonthDate'].apply(lambda x: x.month)
    if start_year!=None:
        gdf.drop(gdf[(gdf['Year'] < start_year)].index, inplace=True)
    if end_year!=None:
        gdf.drop(gdf[(gdf['Year'] > end_year)].index, inplace=True)
    print('Done loading data from '+path+' GFED_CT_Mask_'+region+'_TgC_per_month_'+unit+'.pkl')
    return gdf


def load_ERA5_precip_gdf(region:str='SAT', unit:str='per_subregion', threshold:float=0.0, resolution: float=1., start_year:int=None, end_year:int=None):
    '''Documentation
    # arguments:
        - region: str
           - 'SAT', 'SATr', 'SAT_SATr'
        - unit: str
           - 'per_gridcell', 'per_subregion', 'YearlyLowPrecipitation', 'YearlyMEANLowPrecipitation'
        - threshold: float: needed only if unit=='YearlyLowPrecipitation' or unit=='YearlyMEANLowPrecipitation'
        - resolution: float: defines with which resolution of the grid the ERA5 dataset should be loaded
    '''
    print('start loading ERA5 data...')
    if resolution==1.:
        path = '/mnt/data/users/lartelt/MA/ERA5_precipitation/ERA5_1x1_grid/'
    elif resolution==0.25:
        path = '/mnt/data/users/lartelt/MA/ERA5_precipitation/ERA5_0,25x0,25_grid/'
    else:
        print('resolution not defined!')
    if unit=='YearlyLowPrecipitation':
        gdf = pd.read_pickle(path+'ERA5_YearlyLowPrecipitation_of_'+str(threshold).replace('.','_')+'_per_gridcell_'+region+'.pkl')
    elif unit=='YearlyMEANLowPrecipitation':
        gdf = pd.read_pickle(path+'ERA5_YearlyMEAN_LowPrecipitation_of_'+str(threshold).replace('.','_')+'_per_gridcell_'+region+'.pkl')
        return gdf
    else:
        gdf = pd.read_pickle(path+'ERA5_MonthlyPrecipitation_'+unit+'_'+region+'.pkl')
    if start_year!=None:
        gdf.drop(gdf[(gdf['Year'] < start_year)].index, inplace=True)
    if end_year!=None:
        gdf.drop(gdf[(gdf['Year'] > end_year)].index, inplace=True)
    print('Done loading data from '+path+'ERA5_MonthlyPrecipitation_'+unit+'_'+region+'.pkl')
    return gdf


def load_ERA5_soilmoisture_gdf(layer:str='VSWL1', region:str='SAT', unit:str='per_subregion', start_year:int=None, end_year:int=None):
    '''Documentation
    # arguments:
        - layer: int: layer of the soil moisture that should be loaded [1,2,3,4]
        - region: str
           - 'SAT', 'SATr', 'SAT_SATr'
        - unit: str
           - 'per_gridcell', 'per_subregion'
        - resolution: float: defines with which resolution of the grid the ERA5 dataset should be loaded
    '''
    print('start loading ERA5 data...')
    path = '/mnt/data/users/lartelt/MA/ERA5_soilmoisture/ERA5_1x1_grid/'+layer+'/'
    gdf = pd.read_pickle(path+'ERA5_MonthlySoilmoisture_'+unit+'_'+region+'.pkl')
    if start_year!=None:
        gdf.drop(gdf[(gdf['Year'] < start_year)].index, inplace=True)
    if end_year!=None:
        gdf.drop(gdf[(gdf['Year'] > end_year)].index, inplace=True)
    print('Done loading data from '+path+'ERA5_MonthlySoilmoisture_'+unit+'_'+region+'.pkl')
    return gdf


if __name__=='__main__':
    print('This file is not made to be the main file! Only use as main file if you want to convert units or calculate ensemble mean!')
    


