# Author: Lukas Artelt
import pandas as pd
import xarray as xr
import numpy as np
import datetime
import geopandas
import netCDF4 as nc
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'axes.facecolor':'gainsboro'})

# Annahme: emission data given in unit: TgC/month/gridcell


def plotGDFspatially_simple_world(gdf: geopandas.GeoDataFrame, savepath_w_savename: str, savefig: bool = True):
    print('start plotting map...')
    fig, ax2 = plt.subplots(figsize = (11,8))
    # define region map to plot as contour in background
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    world.boundary.plot(ax=ax2, color = 'black')
    gdf.plot(ax=ax2, column = 'geometry', legend=True, markersize=1, color='darkgreen', alpha=0.9, label="GFED gridcells")
    print('Done plotting, now saving map...')
    ax2.set_xlabel('Longitude', fontsize=14)
    ax2.set_ylabel('Latitude', fontsize=14)
    ax2.tick_params(bottom=True, left=True, axis='both', color='slategray')
    ax2.legend(framealpha=0.25, facecolor='grey', loc='upper right', fontsize=13)
    if savefig:
        plt.savefig(savepath_w_savename, dpi=600, bbox_inches='tight')
        print('Done saving map!')


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
    print('Start cutting gridded gdf from CT Mask into region '+region_name)
    if region_name=='SAT_SATr':
        igdf1 = gdf.within(CT_Mask[(CT_Mask.transcom == 'SAT')].geometry.iloc[0])
        igdf2 = gdf.within(CT_Mask[(CT_Mask.transcom == 'SATr')].geometry.iloc[0])
        gdf_SAT = gdf.loc[igdf1]
        gdf_SATr = gdf.loc[igdf2]
        gdf = pd.concat([gdf_SAT, gdf_SATr])
    elif region_name=='north_SAT':
        gdf.drop(gdf[gdf.Lat < -25].index, inplace=True) # north SAT all above -25° latitude
        gdf = gdf.reset_index(drop=True)
    elif region_name=='mid_SAT':
        print('drop 1')
        gdf.drop(gdf[gdf.Lat >= -25].index, inplace=True)
        print('drop 2')
        gdf.drop(gdf[gdf.Lat <=-38].index, inplace=True) # mid SAT between -25° and -38° latitude
        print('reset index')
        gdf = gdf.reset_index(drop=True)
    elif region_name=='south_SAT':
        gdf.drop(gdf[gdf.Lat > -38].index, inplace=True) # south SAT all below -38° latitude
        gdf = gdf.reset_index(drop=True)
    elif region_name=='west_SAT':
        gdf.drop(gdf[gdf.Long > -65].index, inplace=True) # west SAT all long below -65° longitude
        gdf = gdf.reset_index(drop=True)
    elif region_name=='mid_long_SAT':
        gdf.drop(gdf[gdf.Long <= -65].index, inplace=True)
        gdf.drop(gdf[gdf.Long >= -50].index, inplace=True) # mid_long SAT between -65° and -50° longitude
        gdf = gdf.reset_index(drop=True)
    elif region_name=='east_SAT':
        gdf.drop(gdf[gdf.Long < -50].index, inplace=True) # east SAT all long above -50° longitude
        gdf = gdf.reset_index(drop=True)
    elif region_name=='arid' or region_name=='humid':
        igdf = gdf.within(CT_Mask[(CT_Mask.aridity == region_name)].geometry.iloc[0])
        gdf = gdf.loc[igdf]
    elif region_name=='arid_west':
        gdf.drop(gdf[gdf.Long > -60].index, inplace=True) # south SAT all west of -60° longitude
        gdf = gdf.reset_index(drop=True)
    elif region_name=='arid_east':
        gdf.drop(gdf[gdf.Long < -60].index, inplace=True) # south SAT all east of -60° longitude
        gdf = gdf.reset_index(drop=True)
    else:
        igdf = gdf.within(CT_Mask[(CT_Mask.transcom == region_name)].geometry.iloc[0])
        gdf = gdf.loc[igdf]
    print('Save cutted gdf to pickle file')
    #gdf['MonthDate'] = gdf.Date
    #gdf.drop(columns=['index_x', 'index_y', 'Date', 'Year'], inplace=True)
    #gdf['Month'] = gdf.apply(lambda x: int(x.MonthDate.month), axis=1)
    #gdf['Year'] = gdf.apply(lambda x: int(x.MonthDate.year), axis=1)
    if savename=='GFED':
        gdf.to_pickle(savepath+savename+'_CT_Mask_'+region_name+'_TgC_per_month_per_gridcell.pkl')
    else:
        gdf.to_pickle(savepath+savename+'.pkl')
    print('Done saving')
    return gdf


def calculate_subregional_total_flux(gdf: geopandas.GeoDataFrame=None, model: str='GFED', region_name:str='SAT', savepath: str=None, savename: str=None) -> geopandas.GeoDataFrame:
    '''Documentation
    The function takes a geodataframe with gridded fluxes per gridcell [TgC/gridcell/month] and calculates the total flux for each subregion [TgC/subregion/month].
    ### args
        gdf: geodataframe
            Geodataframe with gridded fluxes in TgC/gridcell/month
        model: str
            - 'GFED'
    ### return
        gdf: geodataframe
            Geodataframe with added columns for the total flux of each subregion in TgC/subregion/month
    '''
    print('Calculating total flux per subregion...')
    if model=='GFED':
        gdf_flux_per_subregion = gdf.groupby(['MonthDate'])[['total_emission']].sum().reset_index()
    #gdf_flux_per_subregion['Year'] = gdf_flux_per_subregion.apply(lambda x: int(x.MonthDate.year), axis=1)
    #gdf_flux_per_subregion['Month'] = gdf_flux_per_subregion.apply(lambda x: int(x.MonthDate.month), axis=1)
    print('Done calculating, now saving...')
    gdf_flux_per_subregion.to_pickle(savepath+savename+'_CT_Mask_'+region_name+'_TgC_per_month_per_subregion.pkl')
    print('Done saving!')
    return gdf_flux_per_subregion


if __name__=='__main__':
    # read out CarbonTracker region mask
    CT_Mask = pd.read_pickle('/home/eschoema/masks/CTTranscomMask1x1Borders.pkl')
    ##region = ['SAT_SATr']
    region_list_subregions = ['SAT_SATr', 'SAT', 'SATr', 'north_SAT', 'mid_SAT', 'south_SAT', 'west_SAT', 'mid_long_SAT', 'east_SAT']
    region_list_regions = ['SAT', 'SATr']
    region_list_subregions = ['north_SAT', 'mid_SAT', 'south_SAT', 'west_SAT', 'mid_long_SAT', 'east_SAT']
    region_list_subregions_short = ['south_SAT', 'west_SAT', 'mid_long_SAT', 'east_SAT']
    
    '''for i, region in enumerate(region_list_subregions_short):
        print(region)
        print('loading GFED data')
        ds = pd.read_pickle('/mnt/data/users/lartelt/MA/GFED4/GFED_cut_with_CT_Mask/GFED_CT_Mask_SAT_TgC_per_month_per_gridcell.pkl')
        print('Done loading GFED data!')
        gdf_regional = cut_gridded_gdf_into_region_from_CT_Mask(ds, CT_Mask, region_name=region, 
                                                                savepath='/mnt/data/users/lartelt/MA/GFED4/GFED_cut_with_CT_Mask/',
                                                                savename='GFED')
        print(gdf_regional.head())
        plotGDFspatially_simple_world(gdf_regional, '/mnt/data/users/lartelt/MA/GFED4/GFED_cut_with_CT_Mask/map_GFED_datapoints_'+region+'_region.png', savefig=True)
        calculate_subregional_total_flux(gdf_regional, model='GFED', region_name=region, savepath='/mnt/data/users/lartelt/MA/GFED4/GFED_cut_with_CT_Mask/', savename='GFED')
        del gdf_regional
    '''
    
    
    '''# calculate flux in TgC/month/gridcell or /subregion from the above resulting TgCO2/month/gridcell or /subregion
    for region in ['SAT_SATr', 'SAT', 'SATr', 'north_SAT']:
        print(region)
        df_per_gridcell = pd.read_pickle('/mnt/data/users/lartelt/MA/GFED4/GFED_cut_with_CT_Mask/GFED_CT_Mask_'+region+'_TgCO2_per_month_per_gridcell.pkl')
        df_per_subregion = pd.read_pickle('/mnt/data/users/lartelt/MA/GFED4/GFED_cut_with_CT_Mask/GFED_CT_Mask_'+region+'_TgCO2_per_month_per_subregion.pkl')
        print('done loading datasets')
        df_per_gridcell['total_emission'] = df_per_gridcell.apply(lambda x: x['total_emission']*(12/44), axis=1)
        df_per_gridcell['emission'] = df_per_gridcell.apply(lambda x: x['emission']*(12/44), axis=1)
        df_per_subregion['total_emission'] = df_per_subregion.apply(lambda x: x['total_emission']*(12/44), axis=1)
        print('done calculating')
        df_per_gridcell.to_pickle('/mnt/data/users/lartelt/MA/GFED4/GFED_cut_with_CT_Mask/GFED_CT_Mask_'+region+'_TgC_per_month_per_gridcell.pkl')
        df_per_subregion.to_pickle('/mnt/data/users/lartelt/MA/GFED4/GFED_cut_with_CT_Mask/GFED_CT_Mask_'+region+'_TgC_per_month_per_subregion.pkl')
        print('done saving')
    '''
        
    '''# Cut in arid/humid regions
    strict_bound = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5_precipitation/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_0_001_minMonths_5_SAT.pkl')
    loose_bound = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5_precipitation/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_0_001_minMonths_4_SAT.pkl')

    region_list = ['arid', 'humid']
    
    for region in region_list:
        print(region)
        for boundary_type in [strict_bound, loose_bound]:
            if boundary_type is strict_bound:
                boundary = 'strict'
            elif boundary_type is loose_bound:
                boundary = 'loose'
            print('cut gdf into '+region+' region')

            ds = pd.read_pickle('/mnt/data/users/lartelt/MA/GFED4/GFED_cut_with_CT_Mask/GFED_CT_Mask_SAT_TgC_per_month_per_gridcell.pkl')
            gdf_subregion = cut_gridded_gdf_into_region_from_CT_Mask(gdf=ds, CT_Mask=boundary_type, region_name=region, 
                                    savepath='/mnt/data/users/lartelt/MA/GFED4/GFED_cut_with_CT_Mask/',
                                    savename='GFED_CT_Mask_SAT_'+boundary+'limit_'+region+'_TgC_per_month_per_gridcell')
            print('done cutting, now calc subregional flux...')
            calculate_subregional_total_flux(gdf_subregion, model='GFED', region_name='SAT_'+boundary+'limit_'+region, 
                                             savepath='/mnt/data/users/lartelt/MA/GFED4/GFED_cut_with_CT_Mask/', 
                                             savename='GFED')
            print('done')
    '''
    
    
    # Cut in arid subregions
    strict_bound = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5_precipitation/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_0_001_minMonths_5_SAT.pkl')
    loose_bound = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5_precipitation/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_0_001_minMonths_4_SAT.pkl')

    region_list = ['arid_west', 'arid_east']
    
    for region in region_list:
        print(region)
        for boundary_type in [strict_bound, loose_bound]:
            if boundary_type is strict_bound:
                boundary = 'strict'
            elif boundary_type is loose_bound:
                boundary = 'loose'
            print('cut gdf into '+region+' region')

            ds = pd.read_pickle('/mnt/data/users/lartelt/MA/GFED4/GFED_cut_with_CT_Mask/GFED_CT_Mask_SAT_'+boundary+'limit_arid'+'_TgC_per_month_per_gridcell.pkl')
            print(ds.head())
            gdf_subregion = cut_gridded_gdf_into_region_from_CT_Mask(gdf=ds, CT_Mask=boundary_type, region_name=region, 
                                    savepath='/mnt/data/users/lartelt/MA/GFED4/GFED_cut_with_CT_Mask/',
                                    savename='GFED_CT_Mask_SAT_'+boundary+'limit_'+region+'_TgC_per_month_per_gridcell')
            print('done cutting, now calc subregional flux...')
            calculate_subregional_total_flux(gdf_subregion, model='GFED', region_name='SAT_'+boundary+'limit_'+region, 
                                             savepath='/mnt/data/users/lartelt/MA/GFED4/GFED_cut_with_CT_Mask/', 
                                             savename='GFED')
            print('done')
        
        
        