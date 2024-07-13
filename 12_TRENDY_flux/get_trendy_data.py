#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 17:38:52 2021

@author: eschoema, updated by Lukas Artelt
"""

import datetime
#from datetime import timedelta
import numpy as np
import pandas as pd
#from functions import getAreaOfGrid, getGdfOfGrid 
import glob, os
import geopandas
import xarray as xr
import math
from shapely.geometry import Polygon
from skimage.measure import block_reduce
from RegionParam import getRegion
from functools import reduce
import cftime
from netCDF4 import Dataset
import warnings
warnings.filterwarnings("ignore")
import functions_to_load_datasets as load_datasets
import plotting_map_functions as plot_map_func

#https://gis.stackexchange.com/questions/326408/how-aggregate-data-in-a-geodataframe-by-the-geometry-in-a-geoseries

#-----------------------------------------------------------------------------------
# Cut TRENDY data into subregions (arid/humid region)
#-----------------------------------------------------------------------------------

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
        gdf.drop(gdf[gdf.Long > -60].index, inplace=True) # south SAT all west of -60° longitude
        gdf = gdf.reset_index(drop=True)
    elif region_name=='arid_east':
        gdf.drop(gdf[gdf.Long < -60].index, inplace=True) # south SAT all east of -60° longitude
        gdf = gdf.reset_index(drop=True)
    else:
        igdf = gdf.within(CT_Mask[(CT_Mask.transcom == region_name)].geometry.iloc[0])
        gdf = gdf.loc[igdf]
    print('Save cutted gdf to pickle file')
    gdf.to_pickle(savepath+savename+'.pkl')
    print('Done saving')
    return gdf

def calculate_TRENDY_ensemble_mean(model_list: list=[], arid_type: str='looselimit_arid', variable_list: list=['nbp', 'gpp']):
    '''Documatation
    # arguments:
        - model_list: list of models that are used to calculate the ensemble mean    
        - variable_list: list of variables that are used to calculate the ensemble mean
        - arid_type: str = 'looselimit_arid', 'looselimit_humid', 'strictlimit_arid', 'strictlimit_humid'
    # returns:
        - gdf: gdf with columns: variable+'tot_TgC_month_ens_mean', variable+'tot_TgC_month_ens_std', variable+'tot_TgC_month_ens_count', MonthDate
    '''
    for variable in variable_list:
        gdf_new = pd.DataFrame()
        for i, model in enumerate(model_list):
            try:
                print('Calculate ensemble mean for '+variable+' of model '+model)
                if gdf_new.empty:
                    gdf_new = load_datasets.load_TRENDY_model_gdf(model=model, variable=variable, unit='TgC_per_region_per_month', arid_type=arid_type)
                    gdf_new.rename(columns={variable+'tot_TgC_month': variable+'tot_TgC_month_'+model}, inplace=True)
                else:
                    gdf = load_datasets.load_TRENDY_model_gdf(model=model, variable=variable, unit='TgC_per_region_per_month', arid_type=arid_type)
                    gdf.rename(columns={variable+'tot_TgC_month': variable+'tot_TgC_month_'+model}, inplace=True)
                    gdf_new = gdf_new.merge(gdf, on='MonthDate')
            except:
                print('Model '+model+' has no variable '+variable)
                continue
        gdf_for_std = gdf_new.copy().drop(columns=['MonthDate'])
        gdf_new['mean'] = gdf_for_std.mean(axis=1)
        gdf_new['std'] = gdf_for_std.std(ddof=0, axis=1)
        gdf_new['count'] = gdf_for_std.count(axis=1) - 1
        gdf_new['Month'] = gdf_new['MonthDate'].apply(lambda x: x.month)
        gdf_new['Year'] = gdf_new['MonthDate'].apply(lambda x: x.year)
        gdf_new.to_pickle('/mnt/data/users/lartelt/MA/TRENDY/mean_of_all_models/Ensemble_mean_for_'+variable+'_SAT_'+arid_type+'.pkl')
        #print(gdf_new.head())
        del gdf_new, gdf_for_std

#-----------------------------------------------------------------------------------
# functions to retrieve TRENDY data
#-----------------------------------------------------------------------------------

def CreateDF(DS,Datatype,ModelInfo,ncTime):
    timeVar = ModelInfo.tVar.values[0]
    
    # get the index of the first timevalue after 2008/1/1 and before 2008/1/2
    # cftime can not deal with month calendars
    if ModelInfo.tformat.values[0] == 'm':
        itstart = int(math.ceil((2008-ModelInfo.starttime.values[0].year)*12
                      +1-ModelInfo.starttime.values[0].month
                      -DS.variables[timeVar].values[0]))
    else:
        try:
            itstart = cftime.date2index(datetime.datetime(2008,1,1), ncTime, calendar = ncTime.calendar, select = 'exact')
            print('exact')
        except:
            itstart = cftime.date2index(datetime.datetime(2008,1,1), ncTime, calendar = ncTime.calendar, select = 'after')
        #check whether itstart before 2008/2/1
        try:
            if itstart == cftime.date2index(datetime.datetime(2008,2,1), ncTime, calendar = ncTime.calendar, select = 'exact'):  
                raise Exception('Error: startindex at 2008/2/1')
        except:
            if itstart == cftime.date2index(datetime.datetime(2008,2,1), ncTime, calendar = ncTime.calendar, select = 'after'):
                raise Exception('Error: startindex after 2008/2/1')
    print(itstart)
    
    print('start reading the data')
    for t in range(itstart,len(DS.variables[Datatype])):
        if len(DS.variables['lat'+ModelInfo.LatName.values[0]].values) == 180: 
            #the data is on a 1°x1° grid    
            dfFlux = pd.DataFrame(
                            data=DS.variables[Datatype][t].values,#data=DS.variables[Datatype][t][:][0].values,
                            index = DS.variables['lat'+ModelInfo.LatName.values[0]].values, 
                            columns = DS.variables['lon'+ModelInfo.LonName.values[0]].values, 
                            dtype='float')
        elif len(DS.variables['lat'+ModelInfo.LatName.values[0]].values) == 360: 
            #the data is on a 0.5°x0.5° grid
            dfFlux = pd.DataFrame(
                            data=block_reduce(DS.variables[Datatype][t].values,#block_reduce(DS.variables[Datatype][t][:][0].values,
                                              block_size=(2,2),
                                              func=np.nanmean),
                            index = block_reduce(DS.variables['lat'+ModelInfo.LatName.values[0]].values, block_size=((2,)), func=np.mean),
                            columns = block_reduce(DS.variables['lon'+ModelInfo.LonName.values[0]].values,
                                                   block_size=((2,)), func=np.mean), dtype='float')
        else:
            #the regridding is done when creating the geodataframe   
            dfFlux = pd.DataFrame(data=DS.variables[Datatype][t].values,#data=DS.variables[Datatype][t][0].values,
                                  index = DS.variables['lat'+ModelInfo.LatName.values[0]].values, 
                                  columns = DS.variables['lon'+ModelInfo.LonName.values[0]].values, 
                                  dtype='float')
        #Unpivot dataframe (df)     
        dfFlux = pd.melt(dfFlux.reset_index(), id_vars='index', value_name=Datatype,var_name = 'Long')
        dfFlux["Lat"] = dfFlux['index']
        
        #check that Longitude from -180 to 180, if from 0 to 360, shift by 180, case for ISAM, CLM5.0, Jules
        if dfFlux.Long.max() > 180:
            dfFlux['Long'] = (((dfFlux.Long + 180) % 360) - 180)

        #### Latitude problem in CABLE-POP
        # The CABLE-POP caontains latitudes from -85.3 to 93.5 
        # Maps of the data show that there is no shift in the latitude. 
        # The coordinates are right and unphysical coordinates >90 get deleted
        dfFlux = dfFlux.drop(dfFlux[(dfFlux.Lat >90)].index,axis=0).reset_index()

        #get date
        if ModelInfo.tformat.values[0] == 'm':
            date = datetime.date(int(ModelInfo.starttime.values[0].year + math.floor(DS.variables[timeVar].values[t]/12)),
                                 int(ModelInfo.starttime.values[0].month +DS.variables[timeVar].values[t]%12),
                                 15)
        else:
            cfdate = cftime.num2date(ncTime[t].data, ncTime.units, ncTime.calendar)
            #the 'only_use_cftime_datetimes=False' option to convert directely 
            #into datetime.date does not wirk for noleap calendar
            date = datetime.date(cfdate.year,cfdate.month,cfdate.day)
        year = date.year
        month = date.month
        day = date.day
        dfFlux.insert(loc=1,column='Year', value= np.ones(len(dfFlux["Lat"]))*int(year))        
        dfFlux.insert(loc=1,column='Month', value= np.ones(len(dfFlux["Lat"]))*int(month))
        dfFlux.insert(loc=1,column='Day', value= np.ones(len(dfFlux["Lat"]))*int(day))
        dfFlux.insert(loc=1,column='Date', value= [date]*len(dfFlux["Lat"]))
        dfFlux.insert(loc=1,column='MonthDate', value= [datetime.date(year,month,15)]*len(dfFlux["Lat"]))
        if t == itstart:
            print('the first time step i on '+str(year)+str(month).zfill(2)+str(day).zfill(2))
            df = dfFlux.copy()
        else:
            df = df.append(dfFlux)
                     
    return df

def CreateDataFrameTRENDYflux(Lat_min,Lat_max,Long_min,Long_max,RegionName,Num,VegModel,ModelInfo):        
    datapath = "/mnt/data/users/eschoema/TRENDY/"+VegModel+"/"  
    for num, Datatype in enumerate(ModelInfo.Variables.values[0].split('-')):    
        # one file per model and each variable
        filepath = datapath + VegModel + "_S3_" + Datatype + ".nc"
        print('reading mode ' + VegModel + ' from path: ' + filepath)
        #decode time false as 365day/noleap calender can not be decoded
        DS = xr.open_mfdataset(filepath, combine='by_coords', decode_times=False)
        
        #read time variable as netCDF4._netCDF4.Variable to use cftime module
        ncTime = Dataset(filepath).variables[ModelInfo.tVar.values[0]]
        #check that ncTime and DS.time are in the same order
        if not (ncTime== DS[ModelInfo.tVar.values[0]].values).all():
            print('ERROR in time varaible comparison of netcdf and xarray')
        
        #create Dataframe
        df3= CreateDF(DS,Datatype,ModelInfo,ncTime)
        print(df3)
        df = df3[(df3.Long >= Long_min) & (df3.Long <= Long_max) & (df3.Lat >= Lat_min) & (df3.Lat <= Lat_max)]
       
        print("finished reading data")
        df.to_pickle("/mnt/data/users/lartelt/MA/TRENDY/"+ModelInfo.Model.values[0]+"/DF1_"+
                     ModelInfo.Model.values[0]+"_"+Datatype.split('_')[0]+"_"+RegionName+".pkl")
        
        #create GeoDataFrame
        gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Long, df.Lat),crs=4326)

        #interpolate grid on 1x1° grid, if not already done in CreateDF (-> orig. grid not on 0.5° or 1° res)     
        if len(DS.variables['lat'+ModelInfo.LatName.values[0]].values) not in [180,360]:
            # get 1x1° grid of region
            try:
                gdfGrid = pd.read_pickle("/home/lartelt/MA/region_masks/Grid1_1"+str(Num)+".pkl")
            except:
                gdfGrid = getGdfOfGrid(Num)
            gdf.rename(columns = {'Lat':'Lat_orig','Long':'Long_orig'}, inplace = True)
                
            if not os.path.isfile('/mnt/data/users/lartelt/MA/TRENDY/intersection_matrices/intersectionMatrix_'+VegModel+'.pkl'):
                # intersection matrix not calculated yet
                
                #geodataframe (gdf) with borders of the Transcom regions
                if Num >= 900:
                    Transcom = pd.read_pickle("/home/eschoema/Transcom_Regions.pkl")
                elif Num >= 700:
                    Transcom = pd.read_pickle("/home/eschoema/masks/CTTranscomMask1x1Borders.pkl")

                print('Interpolate on 1°x1° grid')
                #get grid cell size
                dLong = (DS.variables['lon'+ModelInfo.LonName.values[0]].values[2] - DS.variables['lon'+ModelInfo.LonName.values[0]].values[1])
                dLat = (DS.variables['lat'+ModelInfo.LatName.values[0]].values[2] - DS.variables['lat'+ModelInfo.LatName.values[0]].values[1])
                
                #create polygon geometry for geodataframe
                geom = gdf.apply(lambda x: Polygon(zip([x.Long_orig-dLong/2,x.Long_orig-dLong/2,x.Long_orig+dLong/2,x.Long_orig+dLong/2],
                                                       [x.Lat_orig-dLat/2,x.Lat_orig+dLat/2,x.Lat_orig+dLat/2,x.Lat_orig-dLat/2])),axis = 1)
                print(geom)
                print(gdf)
                gdf.insert(loc = 1, value = geom, column = 'geomPoly')
                gdf = gdf.set_geometry(gdf.geomPoly)
                
                #cut the gdf grid with the 1x1° grid and keep all subgridcells of the intersection
                inter = geopandas.overlay(gdfGrid,gdf,how='intersection')
                #fig2,ax2=plt.subplots(figsize=(12,9)); inter[(inter.Year == 2012)&(inter.Month == 1)].boundary.plot(color='blue',ax = ax2); inter[(np.isnan(inter.gpp))&(inter.Year == 2012)&(inter.Month == 1)].boundary.plot(color='red',ax = ax2)
                #fig2,ax2=plt.subplots(figsize=(12,9)); gdf[(gdf.Year == 2012)&(gdf.Month == 1)].boundary.plot(color='red',ax = ax2);inter[(inter.Year == 2012)&(inter.Month == 1)].boundary.plot(color='blue',ax = ax2)
                
                # to m² insteadt of degree to get area of the subgridcells
                inter = inter.to_crs({'proj':'cea'}) 
                inter.insert(loc = 1, value = inter.area,column = 'subarea')           
                # back to degrees
                inter = inter.to_crs(crs=4326)
                
                # save intersection matrix to convert old grid to new grid (Long_orig&Lat_orig -> GridID)
                inter_matrix = inter[['Lat_orig','Long_orig','GridID','subarea']].copy(deep = True)
                # 'Lat_orig','Long_orig','GridID','subarea' exists for each month but is the same, only keep once
                inter_matrix.drop_duplicates(inplace = True, ignore_index = True)
                inter_matrix.to_pickle('/mnt/data/users/lartelt/MA/TRENDY/intersection_matrices/intersectionMatrix_'+VegModel+'_'+RegionName+'_'+str(Num)+'.pkl')
            else:
                inter_matrix = pd.read_pickle('/mnt/data/users/lartelt/MA/TRENDY/intersection_matrices/intersectionMatrix_'+VegModel+'.pkl')    
                inter = pd.merge(inter_matrix,gdf,on=['Long_orig','Lat_orig'])
            
            #get indices of all not nan values 
            nonanindex = ~np.isnan(inter[Datatype])
            #calculate weighted mean on the 1x1° gris (using GridID)    
            newdf = inter.loc[inter.index[nonanindex]].groupby(['GridID','Month','Year','Day'])[Datatype].agg(
                                                               lambda x: np.average(x, weights=inter.loc[x.index, "subarea"])).reset_index()
            # change geometry column in reference grid to prepare for merging
            gdfGrid = gdfGrid.drop(columns='geometry')
            gdfGrid = gdfGrid.rename(columns={'geomPoint':'geometry'})
            gdfGrid = gdfGrid.set_geometry(gdfGrid.geometry)
            del gdf
            #get  Lat, Lon and geometry in the new gdf 'newdf'->'gdf'
            gdf = pd.merge(gdfGrid[['GridID','geometry','Lat','Long']],newdf,on=['GridID'])
            gdf = gdf.drop(columns = 'GridID')

            #insert date columns
            date = gdf.apply(lambda x: datetime.date(int(x.Year), int(x.Month), int(x.Day)),axis=1)
            gdf.insert(loc=1,column='Date',value=date)
            Monthdate = gdf.apply(lambda x: datetime.date(int(x.Year), int(x.Month), 15),axis=1)
            gdf.insert(loc=1,column='MonthDate',value=Monthdate)
        else:
            if Num >= 700:
                #geodataframe (gdf) with borders of the Transcom regions
                if Num >= 900:
                    Transcom = pd.read_pickle("/home/eschoema/Transcom_Regions.pkl")
                elif Num >= 700:
                    Transcom = pd.read_pickle("/home/eschoema/masks/CTTranscomMask1x1Borders.pkl")
                #not needed for the interpolated gdf as gdfGrid already cut by Transcom border
                igdf = gdf.within(Transcom[(Transcom.transcom == RegionName)].geometry.iloc[0])
                gdf = gdf.loc[igdf]
        
        #possibility to apply additional masks
        if Num == 950: 
            SemiArid = pd.read_pickle("/mnt/data/users/eschoema/ERA5/Australia/DataFrames/MaskDryPoly0.02_4.pkl")
            iSemiAridgdf = gdf.within(SemiArid.geometry.iloc[0])
            gdf = gdf.loc[iSemiAridgdf]
        
        print("calculate total fluxes") 
        #gdf with latitude dependent area of a 1x1° cell
        Area = getAreaOfGrid()

        gdf = pd.merge(gdf,Area,on=['Lat'],how='left')
        gdf.insert(loc=1,column=Datatype.split('_')[0] + 'tot', value= gdf[Datatype]*gdf['Area']) 
        gdf = gdf.rename(columns={Datatype:Datatype.split('_')[0]})
        
        gdf.to_pickle("/mnt/data/users/lartelt/MA/TRENDY/"+ModelInfo.Model.values[0]+"/gdf1x1_"+ModelInfo.Model.values[0]+"_"+Datatype.split('_')[0]+"_"+RegionName+"_"+str(Num)+".pkl")
        
        del gdf,df,DS

#------------------------------------------------------------------------------------------------------
# helper functions
#------------------------------------------------------------------------------------------------------

def getAreaOfGrid():
    """Get Dataframe with Area dependend on Latitude for a 1°x1° grid"""
    AreaLat = []
    Lat = range(895,-905,-10)
    for i in Lat:
        geom = [Polygon(zip([100,100,101,101],[i/10-0.5,i/10+0.5,i/10+0.5,i/10-0.5]))]
        GData = geopandas.GeoDataFrame({'data':[1]}, geometry=geom)
        GData.crs = 'epsg:4326'
        GData = GData.to_crs({'proj':'cea'})
        AreaLat.append(GData.geometry.area[0])
    dfArea = pd.DataFrame({'Area':AreaLat,'Lat':np.array(Lat)/10})
    
    return(dfArea)
    
def getGdfOfGrid(Num):
    """Get Geodatframe of 1°x1° Grid for a certain transcom region"""
    RegionName, Long_min, Long_max, Lat_min, Lat_max = getRegion(Num)
    if Num >= 900:
        Transcom = pd.read_pickle("/home/eschoema/Transcom_Regions.pkl")
        first = True
        AreaLat = []
        Lat = range(Lat_max*10+5, Lat_min*10-5,-10)
        if Long_max >= 179.5:
            Long_max = Long_max -1
        Long = range(Long_max*10+5, Long_min*10-5,-10)
        for i in Lat:
            for j in Long:
                geom = [Polygon(zip([j/10-0.5,j/10-0.5,j/10+0.5,j/10+0.5],[i/10-0.5,i/10+0.5,i/10+0.5,i/10-0.5]))]
                GData = geopandas.GeoDataFrame({'data':[1]}, geometry=geom)
                GData.crs = 'epsg:4326'
                GData.insert(loc=1,column = 'Lat',value = [i/10])
                GData.insert(loc=1,column = 'Long',value = [j/10])
                if first:
                    gdf = GData.copy()
                    first = False
                else:
                    gdf = gdf.append(GData, ignore_index=True)
                GData = GData.to_crs({'proj':'cea'})
                AreaLat.append(GData.geometry.area[0])
        
        gdf.insert(loc=1, value = AreaLat,column = 'AreaGrid') 
        gdf = gdf.drop(columns = 'data')
        gdf.insert(loc=1, value = gdf.geometry,column = 'geomPoly') 
        gdf.insert(loc=1, column = 'geomPoint', value = geopandas.points_from_xy(gdf.Long, gdf.Lat))
        gdf = gdf.set_geometry(gdf.geomPoint)
        igdf = gdf.within(Transcom[(Transcom.transcom == RegionName)].geometry.iloc[0])
        gdf = gdf.loc[igdf]
        
        gdf = gdf.set_geometry(gdf.geomPoly)
        
    elif Num >= 700:
        Transcom = pd.read_pickle("/home/eschoema/masks/CTTranscomMask1x1Borders.pkl")
        gdf = pd.read_pickle("/home/eschoema/masks/CTTranscomMask1x1Grid.pkl")

        gdf = gdf.set_geometry(gdf.geomPoint)
        igdf = gdf.within(Transcom[(Transcom.transcom == RegionName)].geometry.iloc[0])
        gdf = gdf.loc[igdf]
        gdf = gdf.set_geometry(gdf.geomPoly)

    gdf = gdf.reset_index()
    gdf.insert(loc=1,value=gdf.index,column='GridID')
    gdf.to_pickle("/home/lartelt/MA/region_masks/Grid1_1"+RegionName+".pkl")
    gdf = gdf[(gdf.Long >= Long_min) & (gdf.Long <= Long_max) & (gdf.Lat >= Lat_min) & (gdf.Lat <= Lat_max)]
    gdf.to_pickle("/home/lartelt/MA/region_masks/Grid1_1"+str(Num)+".pkl")
    
    return gdf

#--------------------------------------------------------------------------
# MAIN
#--------------------------------------------------------------------------
#Create Dataframe with all importent characteristics of the TRENDY model
TrendyModels = ['CABLE-POP', 
                'CLASSIC', 
                'DLEM', 
                'IBIS', 
                'ISAM', 
                'JSBACH', 
                'LPX-Bern', 
                'OCN', 
                'ORCHIDEE', 
                'ORCHIDEE-CNP', 
                'ORCHIDEEv3', 
                'LPJ-GUESS',
                'CLM5.0',
                'ISBA-CTRIP',
                'JULES-ES-1p0',
                'LPJ',
                'SDGVM',
                'VISIT',
                'YIBs']

TrendyModels_short = ['CABLE-POP', 
                      'CLASSIC', 
                      'DLEM', 
                      'IBIS', 
                      'ISAM', 
                      'JSBACH', 
                      'LPX-Bern', 
                      'OCN', 
                      'ORCHIDEE', 
                      'ORCHIDEE-CNP', 
                      'ORCHIDEEv3', 
                      'CLM5.0',
                      'ISBA-CTRIP',
                      'JULES-ES-1p0',
                      'LPJ',
                      'SDGVM',
                      'VISIT',
                      'YIBs']
#Leave out Model 'LPJ-GUESS' because it only has yearly resolution which leads to the Error: Start index after 2008/2/1. istime=3696

tCalendar = ['standard','365_day','noleap','','noleap','proleptic_gregorian',
             'noleap','365_day','noleap','noleap','noleap',
             '','noleap','gregorian','365_day','365_day','360_day',
             '','']

#name of latitude variable 'lat'+ LatName
LatName = ['itude','itude','','itude','','itude',
             'itude','itude','itude','','',
             'itude','','_FULL','itude','','itude',
             '','itude']
#name of longitude variable 'lon'+ LatName
LonName = ['gitude','gitude','','gitude','','gitude',
             'gitude','gitude','gitude','','',
             'gitude','','_FULL','gitude','','gitude',
             '','gitude']

#which variables shall be processed, needs to be addapted
ModelVars = ['gpp-npp-ra-rh',
             'gpp-nbp-npp-ra-rh-fFire',
             'gpp-npp-ra-rh',
             'gpp-nbp-npp-ra-rh',
             'gpp-nbp-npp-ra-rh',
             'gpp-nbp-npp-ra-rh-fFire-fLuc',#'rh', #
             'gpp-nbp-npp-ra-rh-fFire',
             'gpp-nbp-npp-ra-rh',
             'gpp-nbp-npp-ra-rh-fFire',
             'gpp-nbp-npp-ra-rh-fFire',
             'gpp-nbp-npp-ra-rh',
             'gpp-nbp-npp-ra-fFire',
             'gpp-nbp-npp-ra-rh-fFire',
             'gpp-nbp-npp-ra-rh-fFire',
             'gpp-nbp-rh-npp_nlim',
             'gpp-nbp-npp-ra-rh-fFire',
             'gpp-nbp-npp-ra-rh-fFire',
             'gpp-nbp-ra-rh-fFire',
             'gpp-nbp-npp-ra-rh']

'''# in case soil moisture needs to be processed
#msl contains multiple depth. When treating this varialbe in the func CreateDF muss ein [0] hinter [t]
#ModelVars = ['msl','msl','msl','msl',
#            'msl','msl','msl','msl',
#            'msl','msl','msl','msl',
#            'msl','msl','msl','msl',
#            'msl','msl','msl']
'''

#the time format is only relevant for models with monthly values as for those the cftime function can not be applied
tformat = ['s','d','m','m','d','h',
           'd','d','d','d','d',
           'd','d','m','s','d','h',
           'm','m']
tVar = ['time','time','time','time','time','time',
        'time','time','time','time','time_counter',
        'time','time','time_counter','time','time','time',
        'time','time']
tstart = [datetime.date(1700,1,1),
          datetime.date(1700,12,31),
          datetime.date(1700,1,1),
          datetime.date(1700,1,1),
          datetime.date(1700,1,1),
          datetime.date(1700,1,16),
          datetime.date(1700,1,1),
          datetime.date(1700,1,1),
          datetime.date(1700,1,1),
          datetime.date(1701,1,1),
          datetime.date(1700,1,1),
          datetime.date(1700,1,1),
          datetime.date(1700,1,1),
          datetime.date(1700,1,16),
          datetime.date(2010,1,1),
          datetime.date(1700,1,1),
          datetime.date(1900,1,1),
          datetime.date(1860,1,1),
          datetime.date(1700,1,1)]

dfModelInfo = pd.DataFrame({'Model':TrendyModels,
                            'tcalendar':tCalendar,
                            'Variables': ModelVars,
                            'tformat':tformat,
                            'tVar': tVar,
                            'starttime':tstart,
                            'LatName':LatName,
                            'LonName':LonName} )



#-----------------------------------------------------------------------------------
# MAIN
#-----------------------------------------------------------------------------------

'''# retrieve TRENDY data
Numm = 721 #davor: 56
print('get region...')
RegionName, Long_min, Long_max, Lat_min, Lat_max = getRegion(Numm)
print('region name = '+RegionName)
for model in TrendyModels_short:
    ModelInfo = dfModelInfo[(dfModelInfo.Model == model)]
    CreateDataFrameTRENDYflux(Lat_min,Lat_max,Long_min,Long_max,RegionName,Numm,model,ModelInfo)
    '''
    
'''# cut TRENDY data into subregion (arid/humid region)
vg_ok_TRENDY_models = ['CLASSIC', 'OCN', 'ISBA-CTRIP', 'ISAM', 'YIBs', 'ORCHIDEE']
bad_TRENDY_models = ['ORCHIDEE-CNP', 'LPJ', 'JULES-ES-1p0', 'JSBACH', 'IBIS', 'SDGVM', 'VISIT', 'LPX-Bern', 'CLM5.0', 'ORCHIDEEv3']

#for model in vg_ok_TRENDY_models:
for model in bad_TRENDY_models:
    print(model)
    variable_list = ['nbp', 'gpp', 'npp', 'ra', 'rh', 'fFire']
    #if model in ['CLASSIC', 'ISBA-CTRIP']:
    #    variable_list = ['nbp', 'gpp', 'npp', 'ra', 'rh', 'fFire']
    #else:
    #    variable_list = ['nbp', 'gpp', 'npp', 'ra', 'rh']
    for var in variable_list:
        print(var)
        try:
            gdf = load_datasets.load_TRENDY_model_gdf(model=model, variable=var, unit='TgC_per_gridcell_per_month')
        except:
            continue
        strict_bound = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_0_001_minMonths_5_SAT.pkl')
        loose_bound = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_0_001_minMonths_4_SAT.pkl')
        region_list = ['arid', 'humid']
        for region in region_list:
            print(region)
            for boundary_type in [strict_bound, loose_bound]:
                if boundary_type is strict_bound:
                    boundary = 'strict'
                elif boundary_type is loose_bound:
                    boundary = 'loose'
                print(boundary)
                print('cut gdf into humid/arid region')
                gdf_subregion = cut_gridded_gdf_into_region_from_CT_Mask(gdf=gdf, CT_Mask=boundary_type, region_name=region, savepath='/mnt/data/users/lartelt/MA/TRENDY/'+model+'/', savename='gdf1x1_'+model+'_'+var+'_SAT_'+boundary+'limit_'+region+'_TgC_per_gridcell_per_month')
                #plot_map_func.plotGDFspatially_simple_world(gdf=gdf_subregion, savepath_w_savename='/mnt/data/users/lartelt/MA/ERA5/ERA5_1x1_grid/map_ERA5_precipitation_'+boundary+'_limit_'+region+'.png')
                print('calculate total flux per subregion')
                gdf_regional = gdf_subregion.groupby(['MonthDate'])[[var+'tot_TgC_month']].sum().reset_index()
                gdf_regional.to_pickle('/mnt/data/users/lartelt/MA/TRENDY/'+model+'/gdf1x1_'+model+'_'+var+'_SAT_'+boundary+'limit_'+region+'_TgC_per_region_per_month.pkl')
                
                print('done')
'''
    

'''# Calculate TRENDY ensemble mean for arid/humid region
vg_ok_TRENDY_models = ['CLASSIC', 'OCN', 'ISBA-CTRIP', 'ISAM', 'YIBs', 'ORCHIDEE']
bad_TRENDY_models = ['ORCHIDEE-CNP', 'LPJ', 'JULES-ES-1p0', 'JSBACH', 'IBIS', 'SDGVM', 'VISIT', 'LPX-Bern', 'CLM5.0', 'ORCHIDEEv3']
all_TRENDY_models = ['CLASSIC', 'OCN', 'ISBA-CTRIP', 'ISAM', 'YIBs', 'ORCHIDEE', 'ORCHIDEE-CNP', 'LPJ', 'JULES-ES-1p0', 'JSBACH', 'IBIS', 'SDGVM', 
                     'VISIT', 'LPX-Bern', 'CLM5.0', 'ORCHIDEEv3']
variable_list = ['nbp', 'gpp', 'npp', 'ra', 'rh', 'fFire']
arid_list = ['looselimit_arid', 'looselimit_humid', 'strictlimit_arid', 'strictlimit_humid']

for aridity in arid_list:
    gdf_mean = calculate_TRENDY_ensemble_mean(model_list=all_TRENDY_models, arid_type=aridity, variable_list=variable_list)
'''


'''# cut TRENDY data into subregions of arid region (arid_east, arid_west)
good_TRENDY_models = ['CLASSIC', 'OCN', 'ISBA-CTRIP', 'ISAM', 'YIBs', 'ORCHIDEE']
bad_TRENDY_models = ['ORCHIDEE-CNP', 'LPJ', 'JULES-ES-1p0', 'JSBACH', 'IBIS', 'SDGVM', 'VISIT', 'LPX-Bern', 'CLM5.0', 'ORCHIDEEv3']

for model in bad_TRENDY_models:
#for model in ['LPJ']:
    print(model)
    variable_list = ['nbp', 'gpp', 'npp', 'ra', 'rh', 'fFire']
    for i, var in enumerate(variable_list):
        print(var)
        strict_bound = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5_precipitation/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_0_001_minMonths_5_SAT.pkl')
        loose_bound = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5_precipitation/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_0_001_minMonths_4_SAT.pkl')
        region_list = ['arid_east', 'arid_west']
        for region in region_list:
            print(region)
            for boundary_type in [strict_bound, loose_bound]:
                if boundary_type is strict_bound:
                    boundary = 'strict'
                elif boundary_type is loose_bound:
                    boundary = 'loose'
                print(boundary)
                try:
                    gdf = load_datasets.load_TRENDY_model_gdf(model=model, variable=var, arid_type=boundary+'limit_arid', unit='TgC_per_gridcell_per_month')
                    print(gdf.head())
                except:
                    print('no data for '+model+' '+var+' '+boundary+'limit_arid')
                    continue
                gdf_subregion = cut_gridded_gdf_into_region_from_CT_Mask(gdf=gdf, CT_Mask=boundary_type, region_name=region, savepath='/mnt/data/users/lartelt/MA/TRENDY/'+model+'/', 
                                                                         savename='gdf1x1_'+model+'_'+var+'_SAT_'+boundary+'limit_'+region+'_TgC_per_gridcell_per_month')
                if i==0:
                    plot_map_func.plotGDFspatially_simple_world(gdf=gdf_subregion, savepath_w_savename='/mnt/data/users/lartelt/MA/TRENDY/'+model+'/map_gdf1x1_'+model+'_'+var+'_SAT_'+boundary+'limit_'+region+'.png')
                print('calculate total flux per subregion')
                gdf_regional = gdf_subregion.groupby(['MonthDate'])[[var+'tot_TgC_month']].sum().reset_index()
                gdf_regional.to_pickle('/mnt/data/users/lartelt/MA/TRENDY/'+model+'/gdf1x1_'+model+'_'+var+'_SAT_'+boundary+'limit_'+region+'_TgC_per_region_per_month.pkl')
                
                print('done')
'''


# Calculate TRENDY ensemble mean for subregions of arid region (arid_east, arid_south, arid_west)
vg_ok_TRENDY_models = ['CLASSIC', 'OCN', 'ISBA-CTRIP', 'ISAM', 'YIBs', 'ORCHIDEE']
bad_TRENDY_models = ['ORCHIDEE-CNP', 'LPJ', 'JULES-ES-1p0', 'JSBACH', 'IBIS', 'SDGVM', 'VISIT', 'LPX-Bern', 'CLM5.0', 'ORCHIDEEv3']
all_TRENDY_models = ['CLASSIC', 'OCN', 'ISBA-CTRIP', 'ISAM', 'YIBs', 'ORCHIDEE', 'ORCHIDEE-CNP', 'LPJ', 'JULES-ES-1p0', 'JSBACH', 'IBIS', 'SDGVM', 
                     'VISIT', 'LPX-Bern', 'CLM5.0', 'ORCHIDEEv3']
variable_list = ['nbp', 'gpp', 'npp', 'ra', 'rh', 'fFire']
arid_list = ['looselimit_arid', 'strictlimit_arid']

for aridity in arid_list:
    for subregion in ['west', 'east']:
        print('NOW calc: '+aridity+'_'+subregion)
        gdf_mean = calculate_TRENDY_ensemble_mean(model_list=all_TRENDY_models, arid_type=aridity+'_'+subregion, variable_list=variable_list)




# DO NOT USE, not working - use ipynb "calc_var1+var2+uvm.ipynb" instead! There is a section specially for arid/humid regions in the end
'''# Calculate var1+var2 for TRENDY models in arid/humid regions: FOR SAT region this was done in "calc_var1+var2+uvm.ipynb"
def calc_sum_of_variables_TRENDY(variables_to_sum_list: list=['ra', 'rh'], operator_list: list=['+'], arid_type: str='looselimit_arid', savepath: str=''):
    for i, var in enumerate(variables_to_sum_list):
        df_TRENDY_var1 = load_datasets.load_TRENDY_model_gdf(variable=variables_to_sum_list[0], arid_type=arid_type, unit='mean_of_variable').drop(columns=['mean', 'std', 'count', 'Month', 'Year'])
        columns_var1 = df_TRENDY_var1.columns.drop(['MonthDate', 'Month', 'Year'])
    #print(df_TRENDY_var1.head())
    df_TRENDY_var2 = load_datasets.load_TRENDY_model_gdf(model='CABLE-POP', variable=var2, unit='mean_of_variable').drop(columns=['mean', 'std', 'count'])
    columns_var2 = df_TRENDY_var2.columns.drop(['MonthDate', 'Month', 'Year'])
    #print(df_TRENDY_var2.head())
    #df_TRENDY = pd.merge(df_TRENDY_var1, df_TRENDY_var2, on=['MonthDate', 'Month', 'Year'], how='outer')
    df_TRENDY_sum = df_TRENDY_var1[['MonthDate', 'Month', 'Year']]

    model_list = []
    for column in columns_var1:
        col = 'rh'+column[2:]
        if col in columns_var2:
            model_list.append(column[16:])
    print(model_list)

    for model in model_list:
        df_TRENDY_sum[var1+'tot+'+var2+'tot_TgC_month_'+model] = df_TRENDY_var1[var1+'tot_TgC_month_'+model] + df_TRENDY_var2[var2+'tot_TgC_month_'+model]

    df_TRENDY_sum_old = df_TRENDY_sum.copy()
    df_TRENDY_sum['mean'] = df_TRENDY_sum.mean(axis=1)
    df_TRENDY_sum['std'] = df_TRENDY_sum_old.std(axis=1)
    df_TRENDY_sum['count'] = df_TRENDY_sum_old.count(axis=1)
    #print(df_TRENDY_sum.head())
    #save df_TRENDY_sum to pickle

    df_TRENDY_sum.to_pickle('/mnt/data/users/lartelt/MA/TRENDY/mean_of_all_models/Ensemble_mean_for_ra+rh_SAT_721.pkl')
'''



