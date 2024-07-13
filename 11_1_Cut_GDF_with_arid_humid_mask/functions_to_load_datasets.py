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
import functions_to_load_datasets as load_func

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

def convert_units_TRENDY(model: str='CABLE-POP', variable: str='gpp', to_unit: str='TgC_per_gridcell_per_month'):
    """
    Convert units of TRENDY model data from kgC/gridcell/year to TgC/gridcell/month.
    Unit of input gdf: Column=variable: kgC/(m^2*s) ; Column=variable+'tot': kgC/(s * gridcell)
    Unit of returned gdf: 
        - to_unit=='TgC_per_gridcell_per_month' : Monthly mean value for each gridcell with Column=variable+'tot_TgC_month': TgC/month/gridcell
        - to_unit=='TgC_per_region_per_month'   : Monthly mean value for total flux per region with Column=variable+'tot_TgC_month': TgC/month/month
    """
    # load data
    gdf = load_TRENDY_model_gdf(model=model, variable=variable)
    # convert units
    print('Converting units of gdf...')
    if to_unit=='TgC_per_gridcell_per_month':
        if variable=='nbp':
            gdf[variable+'tot_TgC_month'] = gdf[variable+'tot']*60*60*24*gdf['days_in_month']*1e-9*(-1)
        else:
            gdf[variable+'tot_TgC_month'] = gdf[variable+'tot']*60*60*24*gdf['days_in_month']*1e-9
    if to_unit=='TgC_per_region_per_month':
        if variable=='nbp':
            gdf[variable+'tot_TgC_month'] = gdf[variable+'tot']*60*60*24*gdf['days_in_month']*1e-9*(-1)
        else:
            gdf[variable+'tot_TgC_month'] = gdf[variable+'tot']*60*60*24*gdf['days_in_month']*1e-9
        gdf = gdf.groupby(['MonthDate'])[[variable+'tot_TgC_month']].sum().reset_index()
    # save data
    gdf.to_pickle('/mnt/data/users/lartelt/MA/TRENDY/'+model+'/gdf1x1_'+model+'_'+variable+'_SAT_721_'+to_unit+'.pkl')
    print('Done saving new gdf!')
    del gdf

def convert_units_TRENDY_loop(TrendyModels: list=[], Variables: list=[]):
    for model in TrendyModels:
            print('analyze model: '+model)
            for variable in Variables:
                print('analyze variable: '+variable)
                # convert units
                try:
                    convert_units_TRENDY(model=model, variable=variable, to_unit='TgC_per_gridcell_per_month')
                    convert_units_TRENDY(model=model, variable=variable, to_unit='TgC_per_region_per_month')
                except:
                    print('Model '+model+' has no variable '+variable)
                    continue

def calculate_TRENDY_ensemble_mean(model_list: list=[], variable_list: str='nbp'):
    '''Documatation
    # arguments:
        - model_list: list of models that are used to calculate the ensemble mean    
        - variable_list: list of variables that are used to calculate the ensemble mean
    # returns:
        - gdf: gdf with columns: variable+'tot_TgC_month_ens_mean', variable+'tot_TgC_month_ens_std', variable+'tot_TgC_month_ens_count', MonthDate
    '''
    for variable in variable_list:
        gdf_new = pd.DataFrame()
        for i, model in enumerate(model_list):
            try:
                print('Calculate ensemble mean for '+variable+' of model '+model)
                if gdf_new.empty:
                    gdf_new = load_func.load_TRENDY_model_gdf(model=model, variable=variable, unit='TgC_per_region_per_month')
                    gdf_new.rename(columns={variable+'tot_TgC_month': variable+'tot_TgC_month_'+model}, inplace=True)
                else:
                    gdf = load_func.load_TRENDY_model_gdf(model=model, variable=variable, unit='TgC_per_region_per_month')
                    gdf.rename(columns={variable+'tot_TgC_month': variable+'tot_TgC_month_'+model}, inplace=True)
                    gdf_new = gdf_new.merge(gdf, on='MonthDate')
            except:
                print('Model '+model+' has no variable '+variable)
                continue
        gdf_old = gdf_new.copy()
        gdf_new['mean'] = gdf_new.mean(axis=1)
        gdf_new['std'] = gdf_old.std(ddof=0, axis=1)
        gdf_new['count'] = gdf_old.count(axis=1) - 1
        gdf_new['Month'] = gdf_new['MonthDate'].apply(lambda x: x.month)
        gdf_new.to_pickle('/mnt/data/users/lartelt/MA/TRENDY/mean_of_all_models/Ensemble_mean_for_'+variable+'_SAT_721.pkl')
        del gdf_new, gdf_old


# load datasets
# TM5 3x2 gridded flux from CT_Mask
def return_dataframe_from_name_TM5_gridded_3x2_CT_Mask(assimilated: str, arid_type: str=None, start_year: int=2009, end_year: int=2020, region: str=None, unit: str=None):
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
        - arid_type: str
           - 'looselimit_arid', 'looselimit_humid', 'strictlimit_arid', 'strictlimit_humid'
    ### returns:
        - pandas dataframe: gdf
    '''
    if arid_type is None:
        path='/mnt/data/users/lartelt/MA/Datasets_cut_with_CT_Mask/TM5-4DVar_flux_gridded_3x2_from_coarsen/TM5-4DVar_'+assimilated+'_ass_flux_gridded_3x2_'+unit+'_'+region+'_CT_Mask.pkl'
        print('loading dataset TM5-4DVar_'+assimilated+'_flux_gridded_3x2_'+unit+'_region_'+region+'_CT_Mask from path ' + path)
    else:
        path='/mnt/data/users/lartelt/MA/TM5_MIP_fluxes_cut_with_arid_humid_mask/TM5-4DVar_flux_gridded_3x2_from_coarsen/TM5-4DVar_'+assimilated+'_ass_flux_gridded_3x2_'+unit+'_'+arid_type+'.pkl'
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


def load_TRENDY_model_gdf(model: str='CABLE-POP', variable: str='gpp', unit: str=None, model_category: str='bad_low_amplitude_models', arid_type: str=None, start_year: int=None, end_year: int=None):
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
            - 'looselimit_arid', 'looselimit_humid', 'strictlimit_arid', 'strictlimit_humid' --> get always unit PER REGION, not per gridcell
    """
    if unit=='mean_of_variable':
        path = '/mnt/data/users/lartelt/MA/TRENDY/mean_of_all_models/'
        if arid_type is None:
            gdf = pd.read_pickle(path+'Ensemble_mean_for_'+variable+'_SAT_721.pkl')
            gdf['Year'] = gdf['MonthDate'].apply(lambda x: x.year)
        else:
            gdf = pd.read_pickle(path+'Ensemble_mean_for_'+variable+'_SAT_'+arid_type+'.pkl')
    elif unit=='mean_of_variable_model_category':
        if 'fFire' not in variable:
            variable = variable.lower()
        path = '/mnt/data/users/lartelt/MA/TRENDY/mean_of_all_models/mean_of_variable_'+model_category+'/'
        if arid_type is None:
            gdf = pd.read_pickle(path+'Mean_'+model_category+'_for_'+variable+'_SAT_721.pkl')
        else:
            gdf = pd.read_pickle(path+'Mean_'+model_category+'_for_'+variable+'_SAT_'+arid_type+'.pkl')
        gdf['Year'] = gdf['MonthDate'].apply(lambda x: x.year)
    #elif unit in ['looselimit_arid', 'looselimit_humid', 'strictlimit_arid', 'strictlimit_humid']:
    #    print('unit given for arid/humid area')
    #    #path = '/mnt/data/users/lartelt/MA/TRENDY/'+model+'/'
    #    gdf = pd.read_pickle(path+'gdf1x1_'+model+'_'+variable+'_SAT_'+unit+'_TgC_per_region_per_month.pkl')
    else:
        path = '/mnt/data/users/lartelt/MA/TRENDY/'+model+'/'
        if unit==None:
            gdf = pd.read_pickle(path+'gdf1x1_'+model+'_'+variable+'_SAT_721.pkl')
            if not 'Days_in_month' in gdf.columns:
                print('Column Days_in_month not in gdf, adding column...')
                gdf['days_in_month'] = gdf.apply(lambda x: getNumDayOfMonth(int(x['Year']),int(x['Month'])), axis=1)
        elif unit=='TgC_per_gridcell_per_month':
            if arid_type is None:
                gdf = pd.read_pickle(path+'gdf1x1_'+model+'_'+variable+'_SAT_721_TgC_per_gridcell_per_month.pkl')
            else:
                gdf = pd.read_pickle(path+'gdf1x1_'+model+'_'+variable+'_SAT_'+arid_type+'_TgC_per_gridcell_per_month.pkl')
        elif unit=='TgC_per_region_per_month':
            if arid_type is None:
                gdf = pd.read_pickle(path+'gdf1x1_'+model+'_'+variable+'_SAT_721_TgC_per_region_per_month.pkl')
            else:
                gdf = pd.read_pickle(path+'gdf1x1_'+model+'_'+variable+'_SAT_'+arid_type+'_TgC_per_region_per_month.pkl')
            #gdf['Year'] = gdf['MonthDate'].apply(lambda x: x.year)
            #gdf['Month'] = gdf['MonthDate'].apply(lambda x: x.month)
    #gdf.drop(gdf[(gdf['Year'] < start_year)].index, inplace=True)
    if start_year!=None:
        gdf.drop(gdf[(gdf['Year'] < start_year)].index, inplace=True)
    if end_year!=None:
        gdf.drop(gdf[(gdf['Year'] > end_year)].index, inplace=True)
    print('Done loading data for gdf1x1_'+model+'_'+variable+'_SAT_721.pkl')
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



if __name__=='__main__':
    #Create Dataframe with all importent characteristics of the TRENDY model
    TrendyModels = ['CABLE-POP', 'CLASSIC', 'DLEM', 'IBIS', 'ISAM', 'JSBACH', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'ORCHIDEE-CNP', 'ORCHIDEEv3', 'CLM5.0',
                    'ISBA-CTRIP', 'JULES-ES-1p0', 'LPJ', 'SDGVM', 'VISIT', 'YIBs']
    #TrendyModels_short = ['CABLE-POP']
    
    #which variables were processed for each model
    ModelVars = ['gpp-npp-ra-rh', 'gpp-nbp-npp-ra-rh-fFire', 'gpp-npp-ra-rh', 'gpp-nbp-npp-ra-rh', 'gpp-nbp-npp-ra-rh', 'gpp-nbp-npp-ra-rh-fFire-fLuc', 
                 'gpp-nbp-npp-ra-rh-fFire', 'gpp-nbp-npp-ra-rh', 'gpp-nbp-npp-ra-rh-fFire', 'gpp-nbp-npp-ra-rh-fFire', 'gpp-nbp-npp-ra-rh', 'gpp-nbp-npp-ra-rh-fFire',
                 'gpp-nbp-npp-ra-rh-fFire', 'gpp-nbp-rh-npp_nlim', 'gpp-nbp-npp-ra-rh-fFire', 'gpp-nbp-npp-ra-rh-fFire', 'gpp-nbp-ra-rh-fFire', 'gpp-nbp-npp-ra-rh']
    Variables = ['gpp', 'nbp', 'npp', 'ra', 'rh', 'fFire', 'fLuc']
    Variables_short = ['nbp']
    print('This file is not made to be the main file! Only use as main file if you want to convert units or calculate ensemble mean!')
    
    #convert_units_TRENDY_loop(TrendyModels, Variables_short)
    #calculate_TRENDY_ensemble_mean(TrendyModels, Variables_short)
    


