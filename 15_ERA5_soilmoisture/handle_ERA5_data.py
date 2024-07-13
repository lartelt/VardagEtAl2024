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
    gdf.to_pickle(savepath+savename+'.pkl')
    print('Done saving')
    return gdf


def calculate_subregional_total_flux(gdf: geopandas.GeoDataFrame=None, model: str='ERA5', layer:str='swvl1', to_unit:str='per_subregion', subregion:str='SAT', savepath: str=None, savename: str=None) -> geopandas.GeoDataFrame:
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
            #gdf_flux_per_subregion_mean = gdf.groupby(['MonthDate'])[['swvl1']].mean().reset_index() # 'Volumetric soil water layer 1' in m3/m3
            gdf_flux_per_subregion_mean = gdf.groupby(['MonthDate'])[[layer]].mean().reset_index() # 'Volumetric soil water layer 1' in m3/m3
            #gdf_flux_per_subregion_std = gdf.groupby(['MonthDate'])[['swvl1']].std(ddof=0).reset_index()
            gdf_flux_per_subregion_std = gdf.groupby(['MonthDate'])[[layer]].std(ddof=0).reset_index()
            gdf_flux_per_subregion = gdf_flux_per_subregion_mean.merge(gdf_flux_per_subregion_std, on='MonthDate', how='outer', suffixes=('_mean', '_std'))
    gdf_flux_per_subregion['Year'] = gdf_flux_per_subregion.apply(lambda x: int(x.MonthDate.year), axis=1)
    gdf_flux_per_subregion['Month'] = gdf_flux_per_subregion.apply(lambda x: int(x.MonthDate.month), axis=1)
    #gdf_flux_per_subregion.to_pickle(savepath+savename+'_'+to_unit+'_'+subregion+'.pkl')
    gdf_flux_per_subregion.to_pickle(savepath+savename+'_'+subregion+'.pkl')
    return gdf_flux_per_subregion



# main for VSWL1 - the first layer of the soil:

'''# new: first coarsen ERA5 data to 1x1 degree grid
ds = xr.open_mfdataset('/mnt/data/users/eschoema/ERA5/SouthAmerica/MonthlyVSWL1.nc')
ds_1x1 = ds[['swvl1']].coarsen(latitude=10, longitude=10, boundary='pad', coord_func='mean').mean()

df = ds_1x1.to_dataframe().reset_index()
df['latitude'] = df['latitude'].astype(int).apply(lambda x: x+0.5 if x>0 else x-0.5)
df['longitude'] = df['longitude'].astype(int).apply(lambda x: x+0.5 if x>0 else x-0.5)

df.to_pickle('/mnt/data/users/lartelt/MA/ERA5_soilmoisture/ERA5_1x1_grid/df_ERA5_MonthlySoilmoisture_per_gridcell_SA.pkl')
print(df.head())
gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')
gdf.to_pickle('/mnt/data/users/lartelt/MA/ERA5_soilmoisture/ERA5_1x1_grid/ERA5_MonthlySoilmoisture_per_gridcell_SA.pkl')
'''

'''# get subregional ERA5 data
CT_Mask = pd.read_pickle('/home/eschoema/masks/CTTranscomMask1x1Borders.pkl')
region_list_CT_Mask = ['SAT', 'SATr']
region_list_subregions = ['north_SAT', 'mid_SAT', 'south_SAT', 'west_SAT', 'mid_long_SAT', 'east_SAT']

#for region in region_list_CT_Mask:
for region in region_list_subregions:
    print(region)
    print('read gdf_SAT')
    gdf_SA = load_datasets.load_ERA5_soilmoisture_gdf(region='SAT', unit='per_gridcell')
    print(gdf_SA.head())
    gdf_SA['MonthDate'] = gdf_SA['time'].apply(lambda x: datetime.datetime(x.year, x.month, 15))
    gdf_SA['Month'] = gdf_SA['MonthDate'].apply(lambda x: x.month)
    gdf_SA['Year'] = gdf_SA['MonthDate'].apply(lambda x: x.year)
    print(gdf_SA.head())
    
    print('cut gdf_SA into region from CT Mask')
    gdf_subregion = cut_gridded_gdf_into_region_from_CT_Mask(gdf=gdf_SA, CT_Mask=CT_Mask, region_name=region, savepath='/mnt/data/users/lartelt/MA/ERA5_soilmoisture/ERA5_1x1_grid/', savename='ERA5_MonthlySoilmoisture_per_gridcell_'+region)
    plot_map_func.plotGDFspatially_simple_world(gdf=gdf_subregion, savepath_w_savename='/mnt/data/users/lartelt/MA/ERA5_soilmoisture/ERA5_1x1_grid/map_ERA5_soilmoisture_'+region+'.png')
    
    print('calculate total flux per subregion')
    gdf_subregion_total_precip = calculate_subregional_total_flux(gdf=gdf_subregion, model='ERA5', to_unit='per_subregion', subregion=region, savepath='/mnt/data/users/lartelt/MA/ERA5_soilmoisture/ERA5_1x1_grid/', savename='ERA5_MonthlySoilmoisture')
    print('done')
'''


'''# cut ERA5 data into arid & humid regions
region_list = ['arid', 'humid']
gdf_ERA5 = load_datasets.load_ERA5_soilmoisture_gdf(region='SAT', unit='per_gridcell')
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
        print('cut gdf_SA into humid/arid region')
        gdf_subregion = cut_gridded_gdf_into_region_from_CT_Mask(gdf=gdf_ERA5, CT_Mask=boundary_type, region_name=region, savepath='/mnt/data/users/lartelt/MA/ERA5_soilmoisture/ERA5_1x1_grid/', savename='ERA5_MonthlySoilmoisture_per_gridcell_SAT_'+boundary+'limit_'+region)
        plot_map_func.plotGDFspatially_simple_world(gdf=gdf_subregion, savepath_w_savename='/mnt/data/users/lartelt/MA/ERA5_soilmoisture/ERA5_1x1_grid/map_ERA5_soilmoisture_'+boundary+'limit_'+region+'.png')
        print('calculate total flux per subregion')
        gdf_subregion_total_precip = calculate_subregional_total_flux(gdf=gdf_subregion, model='ERA5', to_unit='per_subregion', subregion=region, 
                                                                      savepath='/mnt/data/users/lartelt/MA/ERA5_soilmoisture/ERA5_1x1_grid/',
                                                                      savename='ERA5_MonthlySoilmoisture_per_subregion_SAT_'+boundary+'limit_'+region)
        print('done')
'''
        

'''# cut ERA5 data into arid subregions
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
        gdf_ERA5 = load_datasets.load_ERA5_soilmoisture_gdf(region='SAT_'+boundary+'limit_arid', unit='per_gridcell')
        print('cut gdf_SA into humid/arid region')
        gdf_subregion = cut_gridded_gdf_into_region_from_CT_Mask(gdf=gdf_ERA5, CT_Mask=boundary_type, region_name=region, savepath='/mnt/data/users/lartelt/MA/ERA5_soilmoisture/ERA5_1x1_grid/', savename='ERA5_MonthlySoilmoisture_per_gridcell_SAT_'+boundary+'limit_'+region)
        plot_map_func.plotGDFspatially_simple_world(gdf=gdf_subregion, savepath_w_savename='/mnt/data/users/lartelt/MA/ERA5_soilmoisture/ERA5_1x1_grid/map_ERA5_soilmoisture_'+boundary+'limit_'+region+'.png')
        print('calculate total flux per subregion')
        gdf_subregion_total_precip = calculate_subregional_total_flux(gdf=gdf_subregion, model='ERA5', to_unit='per_subregion', subregion=region, 
                                                                      savepath='/mnt/data/users/lartelt/MA/ERA5_soilmoisture/ERA5_1x1_grid/',
                                                                      savename='ERA5_MonthlySoilmoisture_per_subregion_SAT_'+boundary+'limit_'+region)
        print('done')
'''



# main for VSWL2-4 (all below):

CT_Mask = pd.read_pickle('/home/eschoema/masks/CTTranscomMask1x1Borders.pkl')
region_list_CT_Mask = ['SAT', 'SATr']
region_list_subregions = ['north_SAT', 'mid_SAT', 'south_SAT', 'west_SAT', 'mid_long_SAT', 'east_SAT']
region_list = ['SAT', 'SATr', 'north_SAT', 'mid_SAT', 'south_SAT', 'west_SAT', 'mid_long_SAT', 'east_SAT']

# first coarsen ERA5 data to 1x1 degree grid
layers1 = ['VSWL2', 'VSWL3', 'VSWL4']
layers2 = ['swvl2', 'swvl3', 'swvl4']

for l, layer in enumerate(layers1):
    print('START with '+layer)
    ds = xr.open_mfdataset('/mnt/data/users/eschoema/ERA5/SouthAmerica/Monthly'+layer+'.nc')
    print('coarsen...')
    ds_1x1 = ds[[layers2[l].lower()]].coarsen(latitude=10, longitude=10, boundary='pad', coord_func='mean').mean()
    df = ds_1x1.to_dataframe().reset_index()
    df['latitude'] = df['latitude'].astype(int).apply(lambda x: x+0.5 if x>0 else x-0.5)
    df['longitude'] = df['longitude'].astype(int).apply(lambda x: x+0.5 if x>0 else x-0.5)
    df.to_pickle('/mnt/data/users/lartelt/MA/ERA5_soilmoisture/ERA5_1x1_grid/'+layer+'/df_ERA5_MonthlySoilmoisture_per_gridcell_SA.pkl')
    print('df to gdf...')
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')
    gdf.to_pickle('/mnt/data/users/lartelt/MA/ERA5_soilmoisture/ERA5_1x1_grid/'+layer+'/ERA5_MonthlySoilmoisture_per_gridcell_SA.pkl')
    print('DONE calc and save gdf for '+layer)
    
    # get subregional ERA5 data
    for i, region in enumerate(region_list): #for region in region_list_CT_Mask:
        print(region)
        if i==0 or i==1:
            if region not in ['SAT', 'SATr']:
                raise ValueError('region is not SAT or SATr')
            print('read gdf for '+region)
            gdf_SA = load_datasets.load_ERA5_soilmoisture_gdf(layer=layer, region='SA', unit='per_gridcell')
        else:
            print('read gdf for '+region)
            gdf_SA = load_datasets.load_ERA5_soilmoisture_gdf(layer=layer, region='SAT', unit='per_gridcell')
        gdf_SA['MonthDate'] = gdf_SA['time'].apply(lambda x: datetime.datetime(x.year, x.month, 15))
        gdf_SA['Month'] = gdf_SA['MonthDate'].apply(lambda x: x.month)
        gdf_SA['Year'] = gdf_SA['MonthDate'].apply(lambda x: x.year)
        gdf_subregion = cut_gridded_gdf_into_region_from_CT_Mask(gdf=gdf_SA, CT_Mask=CT_Mask, region_name=region, savepath='/mnt/data/users/lartelt/MA/ERA5_soilmoisture/ERA5_1x1_grid/'+layer+'/', savename='ERA5_MonthlySoilmoisture_per_gridcell_'+region)
        print(gdf_subregion.columns)
        plot_map_func.plotGDFspatially_simple_world(gdf=gdf_subregion, savepath_w_savename='/mnt/data/users/lartelt/MA/ERA5_soilmoisture/ERA5_1x1_grid/'+layer+'/map_ERA5_soilmoisture_'+region+'.png')
        gdf_subregion_total_precip = calculate_subregional_total_flux(gdf=gdf_subregion, model='ERA5', layer=layers2[l], to_unit='per_subregion', subregion=region, savepath='/mnt/data/users/lartelt/MA/ERA5_soilmoisture/ERA5_1x1_grid/'+layer+'/', savename='ERA5_MonthlySoilmoisture_per_subregion')
        print('DONE with region '+region)


    print('DONE cutting subregions, START cutting arid/humid regions...')
    # cut ERA5 data into arid & humid regions
    region_list_arid_humid = ['arid', 'humid', 'arid_west', 'arid_east']
    strict_bound = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5_precipitation/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_0_001_minMonths_5_SAT.pkl')
    loose_bound = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5_precipitation/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_0_001_minMonths_4_SAT.pkl')

    for j, region in enumerate(region_list_arid_humid):
        print(region)
        for boundary_type in [strict_bound, loose_bound]:
            if boundary_type is strict_bound:
                boundary = 'strict'
            elif boundary_type is loose_bound:
                boundary = 'loose'
            print(boundary)
            if j>1:
                if region not in ['arid_west', 'arid_east']:
                    raise ValueError('region is not arid_west or arid_east')
                gdf_ERA5 = load_datasets.load_ERA5_soilmoisture_gdf(layer=layer, region='SAT_'+boundary+'limit_arid', unit='per_gridcell')
            else:
                gdf_ERA5 = load_datasets.load_ERA5_soilmoisture_gdf(layer=layer, region='SAT', unit='per_gridcell')
            print('cut gdf_SA into humid/arid region')
            gdf_subregion = cut_gridded_gdf_into_region_from_CT_Mask(gdf=gdf_ERA5, CT_Mask=boundary_type, region_name=region, savepath='/mnt/data/users/lartelt/MA/ERA5_soilmoisture/ERA5_1x1_grid/'+layer+'/', savename='ERA5_MonthlySoilmoisture_per_gridcell_SAT_'+boundary+'limit_'+region)
            plot_map_func.plotGDFspatially_simple_world(gdf=gdf_subregion, savepath_w_savename='/mnt/data/users/lartelt/MA/ERA5_soilmoisture/ERA5_1x1_grid/'+layer+'/map_ERA5_soilmoisture_'+boundary+'limit_'+region+'.png')
            print('calculate total flux per subregion')
            gdf_subregion_total_precip = calculate_subregional_total_flux(gdf=gdf_subregion, model='ERA5', layer=layers2[l], to_unit='per_subregion', subregion=region, 
                                                                        savepath='/mnt/data/users/lartelt/MA/ERA5_soilmoisture/ERA5_1x1_grid/'+layer+'/',
                                                                        savename='ERA5_MonthlySoilmoisture_per_subregion_SAT_'+boundary+'limit')
    print('DONE with '+layer)

