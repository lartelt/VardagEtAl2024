import numpy as np
import xarray as xr
import pandas as pd
import geopandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import datetime
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
    gdf.to_pickle(savepath+savename+'_'+region_name+'.pkl')
    print('Done saving')
    return gdf


def calculate_subregional_total_flux(gdf: geopandas.GeoDataFrame=None, model: str='ERA5', to_unit:str='per_subregion', subregion:str='SAT', savepath: str=None, savename: str=None) -> geopandas.GeoDataFrame:
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
    elif model=='ERA5':
        if to_unit=='per_gridcell':
            gdf_flux_per_subregion = gdf
        elif to_unit=='per_subregion':
            gdf_flux_per_subregion_mean = gdf.groupby(['MonthDate'])[['tp']].mean().reset_index()
            gdf_flux_per_subregion_std = gdf.groupby(['MonthDate'])[['tp']].std(ddof=0).reset_index()
            gdf_flux_per_subregion = gdf_flux_per_subregion_mean.merge(gdf_flux_per_subregion_std, on='MonthDate', how='outer', suffixes=('_mean', '_std'))
    gdf_flux_per_subregion['Year'] = gdf_flux_per_subregion.apply(lambda x: int(x.MonthDate.year), axis=1)
    gdf_flux_per_subregion['Month'] = gdf_flux_per_subregion.apply(lambda x: int(x.MonthDate.month), axis=1)
    #gdf_flux_per_subregion.to_pickle(savepath+savename+'_'+to_unit+'_'+subregion+'.pkl')
    gdf_flux_per_subregion.to_pickle(savepath+savename+'_'+subregion+'.pkl')
    return gdf_flux_per_subregion


def get_regionsWithLowPrecipForNmonths(gdf: geopandas.GeoDataFrame, threshold:float=0.0, calc_mean:bool=False, savepath:str=None, region:str='SAT') -> geopandas.GeoDataFrame:
    '''Documentation
    Function to get gdf with entries: MonthDate, N_months_with_low_precip, Month_of_first_low_precip, Month_of_first_high_precip, latitude, longitude, geometry
    # arguments:
        - gdf: geopandas.GeoDataFrame
        - threshold: int, threshold for low precipitation in m/month/gridcell
    # returns:
        - gdf_new: geopandas.GeoDataFrame
            ATTENTION: Values in 'Month_of_first_low_precip' and 'Month_of_last_low_precip' seem to be wrong!
    '''
    print('Start calculating regions with low precipitation for N months')
    if calc_mean==True:
        gdf_mean = gdf.groupby(['Month', 'latitude', 'longitude'])['tp'].mean().reset_index()
        gdf_new = gdf.groupby(['latitude', 'longitude'])['tp'].mean().reset_index().drop(columns=['tp'])
        print(gdf_new.head())
        gdf_new['N_months_with_low_precip'] = gdf_mean.groupby(['latitude', 'longitude'])[['tp']].apply(lambda x: len(x[x.tp<=threshold])).reset_index()[0] # [0] defines the column we want to write into the new gdf
        gdf_new['Month_of_first_low_precip'] = gdf_mean.groupby(['latitude', 'longitude'])[['tp', 'Month']].apply(lambda x: x[x.tp<=threshold]).reset_index()['Month'].min()#[0]
        gdf_new['Month_of_last_low_precip'] = gdf_mean.groupby(['latitude', 'longitude'])[['tp', 'Month']].apply(lambda x: x[x.tp<=threshold]).reset_index()['Month'].max()#[0]
        print('done calculating, now saving')
        gdf_new.to_pickle(savepath+'ERA5_YearlyMEAN_LowPrecipitation_of_'+str(threshold).replace('.','_')+'_per_gridcell_'+region+'.pkl')
    else:
        gdf_new = gdf.groupby(['Year', 'latitude', 'longitude'])[['tp']].sum().reset_index()
        gdf_new.drop(columns=['tp'], inplace=True)
        
        gdf_new['N_months_with_low_precip'] = gdf.groupby(['Year', 'latitude', 'longitude'])[['tp']].apply(lambda x: len(x[x.tp<=threshold])).reset_index()[0] # [0] defines the column we want to write into the new gdf
        gdf_new['Month_of_first_low_precip'] = gdf.groupby(['Year', 'latitude', 'longitude'])[['tp', 'Month']].apply(lambda x: x[x.tp<=threshold]).reset_index()['Month'].min()#[0]
        gdf_new['Month_of_last_low_precip'] = gdf.groupby(['Year', 'latitude', 'longitude'])[['tp', 'Month']].apply(lambda x: x[x.tp<=threshold]).reset_index()['Month'].max()#[0]
        print('done calculating, now saving')
        gdf_new.to_pickle(savepath+'ERA5_YearlyLowPrecipitation_of_'+str(threshold).replace('.','_')+'_per_gridcell_'+region+'.pkl')
    print('done saving')
    return gdf_new


def get_bordersOfAreaWithLowAndHighPrecipitation(df: pd.DataFrame, threshold: float=0.0015, min_dry_months: int=6, savepath: str=None, region: str='SAT') -> geopandas.GeoDataFrame:
    '''Documentation
    Function to generate a mask of the area with low precipitation and the area with high precipitation.
    # arguments:
        - df: pandas.DataFrame: MUST be a DataFrame! Will be converted to a GeoDataFrame in the function. Region can be SAT_SATr or SAT
                                Must contain columns: 'latitude', 'longitude', 'N_months_with_low_precip'
        - threshold: float, threshold for low precipitation in m/day
        - min_dry_months: int, defines at which minimum number of months with low precipitation a gridcell is considered as arid
        - region: str='SAT': defines the region for which the mask should be generated.
    # returns:
        - polygon: geopandas.GeoDataFrame: contains the polygons of the area with low precipitation and the area with high precipitation
    '''
    print('Start generating mask of area with low and high precipitation')
    CT_Mask = pd.read_pickle('/home/eschoema/masks/CTTranscomMask1x1Borders.pkl')
    grid = pd.read_pickle('/home/eschoema/masks/CTTranscomMask1x1Grid.pkl')
    if region=='SAT':
        grid_SAT = grid[grid.transcom_regions==4]
    elif region=='SAT_SATr':
        grid_SAT_X = grid[grid.transcom_regions==4]
        grid_SATr_X = grid[grid.transcom_regions==5]
        gdf_SAT = pd.concat([grid_SAT_X, grid_SATr_X])
    else:
        print('Region not defined!')

    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    igdf = gdf.within(CT_Mask[(CT_Mask.transcom == region)].geometry.iloc[0])
    gdf_SAT = gdf.loc[igdf]

    gdf_SAT['aridity'] = np.zeros(len(gdf_SAT), str)
    gdf_SAT['aridity'] = np.where(gdf_SAT['N_months_with_low_precip'].values>=min_dry_months, 'arid', 'humid')
    df_new = gdf_SAT.merge(grid_SAT, on=['longitude', 'latitude'], how='outer')
    gdf_new = geopandas.GeoDataFrame(df_new, geometry=df_new['geometry_y'], crs="EPSG:4326")
    gdf_new.drop(columns=['geometry_x', 'geometry_y'], inplace=True)

    polygon = gdf_new.dissolve(by='aridity').reset_index()
    print('done generating mask, now saving mask')
    polygon.to_pickle(savepath+'ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_'+str(threshold).replace('.', '_')+'_minMonths_'+str(min_dry_months)+'_'+region+'.pkl')
    print('done saving mask')
    return polygon



# main
'''# new: first coarsen ERA5 data to 1x1 degree grid
ds_for_grid = xr.open_dataset('/mnt/data/users/eschoema/ERA5/SouthAmerica/NewGrid_OnlyForRegriddingUse.nc')
ds = xr.open_mfdataset('/mnt/data/users/eschoema/ERA5/SouthAmerica/MonthlyPrecipitation.nc')
ds_regridded = ds.interp_like(ds_for_grid)

first_longitude = ds_regridded.longitude.values[0]
print(first_longitude)
mask = ds_regridded.longitude != first_longitude
ds_regridded = ds_regridded.sel(longitude=mask)

ds_1x1 = ds_regridded[['tp']].coarsen(latitude=4, longitude=4, boundary='trim', coord_func='mean').mean()

df = ds_1x1.to_dataframe().reset_index()
df.to_pickle('/mnt/data/users/lartelt/MA/ERA5/ERA5_1x1_grid/df_ERA5_MonthlyPrecipitation_per_gridcell_SA.pkl')
gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')
gdf.to_pickle('/mnt/data/users/lartelt/MA/ERA5/ERA5_1x1_grid/ERA5_MonthlyPrecipitation_per_gridcell_SA.pkl')
'''


'''# get subregional ERA5 data
CT_Mask = pd.read_pickle('/home/eschoema/masks/CTTranscomMask1x1Borders.pkl')

#max_region_list = ['SA']
region_list_CT_Mask = ['SAT', 'SATr']
region_list_subregions = ['SAT', 'SATr', 'north_SAT', 'mid_SAT', 'south_SAT', 'west_SAT', 'mid_long_SAT', 'east_SAT']
region = 'SAT_SATr'

for region in region_list_subregions:
    print(region)
    print('read gdf_SAT')
    gdf_SA = load_datasets.load_ERA5_precip_gdf(region=region, unit='per_gridcell', resolution=1)
    #print(gdf_SA.head())
    #gdf_SA['MonthDate'] = gdf_SA['time'].apply(lambda x: datetime.datetime(x.year, x.month, 15))
    #gdf_SA['Month'] = gdf_SA['MonthDate'].apply(lambda x: x.month)
    #gdf_SA['Year'] = gdf_SA['MonthDate'].apply(lambda x: x.year)
    #print(gdf_SA.head())
    
    #print('cut gdf_SA into region from CT Mask')
    #gdf_subregion = cut_gridded_gdf_into_region_from_CT_Mask(gdf=gdf_SA, CT_Mask=CT_Mask, region_name=region, savepath='/mnt/data/users/lartelt/MA/ERA5/ERA5_1x1_grid/', savename='ERA5_MonthlyPrecipitation_per_gridcell')
    #plot_map_func.plotGDFspatially_simple_world(gdf=gdf_subregion, savepath_w_savename='/mnt/data/users/lartelt/MA/ERA5/ERA5_1x1_grid/map_ERA5_precipitation_'+region+'.png')
    
    print('calculate total flux per subregion')
    gdf_subregion_total_precip = calculate_subregional_total_flux(gdf=gdf_SA, model='ERA5', to_unit='per_subregion', subregion=region, savepath='/mnt/data/users/lartelt/MA/ERA5/ERA5_1x1_grid/', savename='ERA5_MonthlyPrecipitation')
    print('done')
'''


'''# get gdf with entries: MonthDate, N_months_with_low_precip, Month_of_first_high_precip, Month_of_last_high_precip
region='SAT_SATr'
gdf_SAT_SATr = load_datasets.load_ERA5_precip_gdf(region=region, unit='per_gridcell', resolution=1.)

# to calculate the mean over all years
for threshold in [0.0005, 0.001, 0.0015]:
    print(threshold)
    gdf_precipMonths = get_regionsWithLowPrecipForNmonths(gdf=gdf_SAT_SATr, region=region, calc_mean=True, threshold=threshold, savepath='/mnt/data/users/lartelt/MA/ERA5/ERA5_1x1_grid/')
    print(gdf_precipMonths)
'''


'''# correction of calculating the precipitation of a whole subregion. Now: use .mean() instead of .sum() in function
region_list = ['SAT_SATr', 'SAT', 'SATr', 'north_SAT', 'mid_SAT', 'south_SAT', 'west_SAT', 'mid_long_SAT', 'east_SAT']
for region in region_list:
    gdf_subregion = load_datasets.load_ERA5_precip_gdf(region=region, unit='per_gridcell', resolution=1)
    print('calculate total flux per subregion')
    gdf_subregion_total_precip = calculate_subregional_total_flux(gdf=gdf_subregion, model='ERA5', to_unit='per_subregion', subregion=region, savepath='/mnt/data/users/lartelt/MA/ERA5/ERA5_1x1_grid/', savename='ERA5_MonthlyPrecipitation')
    print('done')
'''


'''# get mask of area with low and high precipitation
region='SAT'
thresholds = [0.0005, 0.001, 0.0015]
min_dry_months = 3
for threshold in thresholds:
    df = load_datasets.load_ERA5_precip_gdf(region='SAT_SATr', unit='YearlyMEANLowPrecipitation', threshold=threshold, resolution=1.)
    borders = get_bordersOfAreaWithLowAndHighPrecipitation(df, threshold=threshold, min_dry_months=min_dry_months, 
                                                           savepath='/mnt/data/users/lartelt/MA/ERA5/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/', region=region)
    border = borders[borders['aridity']=='humid']
    plot_map_func.plotGDFspatially_simple_SA_for_ERA5_humid_regions(border, 
                                                                    '/mnt/data/users/lartelt/MA/ERA5/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/map_ERA5_Polygon_AreaWithHighPrecipitation_threshold_'+str(threshold).replace('.', '_')+'_minMonths_'+str(min_dry_months)+'_'+region+'.png',
                                                                    threshold=threshold, minMonths=min_dry_months)
'''


'''# cut ERA5 data into arid & humid regions
region_list = ['arid', 'humid']
gdf_ERA5 = load_datasets.load_ERA5_precip_gdf(region='SAT', unit='per_gridcell', resolution=1.)
strict_bound = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_0_001_minMonths_5_SAT.pkl')
loose_bound = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_0_001_minMonths_4_SAT.pkl')

for region in region_list:
    print(region)
    for boundary_type in [strict_bound, loose_bound]:
        if boundary_type is strict_bound:
            boundary = 'strict'
        elif boundary_type is loose_bound:
            boundary = 'loose'
        print(boundary)
        print('cut gdf_SA into humid/arid region')
        gdf_subregion = cut_gridded_gdf_into_region_from_CT_Mask(gdf=gdf_ERA5, CT_Mask=boundary_type, region_name=region, savepath='/mnt/data/users/lartelt/MA/ERA5/ERA5_1x1_grid/', savename='ERA5_MonthlyPrecipitation_per_gridcell_SAT_'+boundary+'_limit')
        #plot_map_func.plotGDFspatially_simple_world(gdf=gdf_subregion, savepath_w_savename='/mnt/data/users/lartelt/MA/ERA5/ERA5_1x1_grid/map_ERA5_precipitation_'+boundary+'_limit_'+region+'.png')
        print('calculate total flux per subregion')
        gdf_subregion_total_precip = calculate_subregional_total_flux(gdf=gdf_subregion, model='ERA5', to_unit='per_subregion', subregion=region, 
                                                                      savepath='/mnt/data/users/lartelt/MA/ERA5/ERA5_1x1_grid/',
                                                                      savename='ERA5_MonthlyPrecipitation_per_subregion_SAT_'+boundary+'_limit')
        print('done')
'''
        


# cut ERA5 data into arid subregions
region_list = ['arid_west', 'arid_east']
strict_bound = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5_precipitation/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_0_001_minMonths_5_SAT.pkl')
loose_bound = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5_precipitation/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_0_001_minMonths_4_SAT.pkl')

for region in region_list:
    print(region)
    for boundary_type in [strict_bound, loose_bound]:
        if boundary_type is strict_bound:
            boundary = 'strict'
        elif boundary_type is loose_bound:
            boundary = 'loose'
        print(boundary)
        gdf_ERA5 = load_datasets.load_ERA5_precip_gdf(region='SAT_'+boundary+'_limit_arid', unit='per_gridcell')
        print('cut gdf_SA into arid subregion')
        gdf_subregion = cut_gridded_gdf_into_region_from_CT_Mask(gdf=gdf_ERA5, CT_Mask=boundary_type, region_name=region, savepath='/mnt/data/users/lartelt/MA/ERA5_precipitation/ERA5_1x1_grid/', savename='ERA5_MonthlyPrecipitation_per_gridcell_SAT_'+boundary+'_limit')
        plot_map_func.plotGDFspatially_simple_world(gdf=gdf_subregion, savepath_w_savename='/mnt/data/users/lartelt/MA/ERA5_precipitation/ERA5_1x1_grid/map_ERA5_precipitation_'+boundary+'_limit_'+region+'.png')
        print('calculate total flux per subregion')
        gdf_subregion_total_precip = calculate_subregional_total_flux(gdf=gdf_subregion, model='ERA5', to_unit='per_subregion', subregion=region, 
                                                                      savepath='/mnt/data/users/lartelt/MA/ERA5_precipitation/ERA5_1x1_grid/',
                                                                      savename='ERA5_MonthlyPrecipitation_per_subregion_SAT_'+boundary+'_limit')
        print('done')


