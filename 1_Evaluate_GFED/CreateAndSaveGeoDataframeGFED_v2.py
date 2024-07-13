#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06.10.2022
Based on improved createDataFrameGFED(filepath,datayear,datatype) in functions.py

@author: eschoema
"""
from calendar import monthcalendar
import h5py 
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cf
import numpy as np
import datetime
import geopandas
import geoplot as gplt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker  # added, not in code form Ines
import seaborn as sns
import math
import warnings
warnings.filterwarnings("ignore")

def createAnnualDataFrameGFED(filepath,datayear,datatype):
    '''
    # function to create a pandas dataframe out of one GFED4 file
    # input:
    #           filepath: file of GFED4 data
    #           datayear: year of the datafile
    #           datatype: 0=carbon emissions, 1=CO2 emissions, 2=CO emissions
    #
    # output:
    #           df: pandas DataFrame containing the variables
    #           -- 'emission',
    #           -- 'Month' integer
    #           -- 'Date' datetime object with day = 15
    #           -- 'Year' integer
    # resulting unit for 'total_emission' for type 'CO2': TgCO2/month/gridcell
    # Documentation GFED see https://www.geo.vu.nl/~gwerf/GFED/GFED4/Readme.pdf
    '''
    DS = h5py.File(filepath,'r')
    typeL = ['SAVA','BORF','TEMF','DEFO','PEAT','AGRI']
    CO2L = [1686,1489,1647,1643,1703,1585] #emission factors for CO2 from https://www.geo.vu.nl/~gwerf/GFED/GFED4/ancill/GFED4_Emission_Factors.txt
    COL = [63,127,88,93,210,102] # emission factors for CO, see above
    
    #get grid cell areas
    dfy_gridarea = pd.DataFrame(data=DS['ancill/grid_cell_area'][:],index = DS['lat'][:,0],
                                columns = DS['lon'][0,:], dtype='float' )  #creates 2d dataframe, uses latitudes as index/rows & longitudes as columns
    dfy_gridarea = pd.melt(dfy_gridarea.reset_index(), id_vars='index',
                                value_name ='Grid_area',var_name = 'Long') # reshapes df to shape 1036800 rows and 3 columns.
                                                                           # columns = identifyer variable = "index"(=lat), name=long, values="Grid_area"
    dfy_gridarea["Lat"] = dfy_gridarea['index'] # just create an additional column with latitudes (has same value as index row)

    #store the values for each month
    for m in range(1,13):
        mo = str(m)
        print(m)
        if datatype == 0: # == carbon emission, datatype given as argument    # mo.zfill(2) = 05 for mo=5 and =11 for mo=11
            dfy = pd.DataFrame(data=DS['emissions/'+mo.zfill(2)+'/C'][:],index = DS['lat'][:,0],  # get datafor Carbon ("/C") from DS "emission" part
                                  columns = DS['lon'][0,:], dtype='float' ) # makes df (matrix) with rows=lat & columns=long. Data is emission of carbon 
            if datayear < 2017:
                dfy_area = pd.DataFrame(data=DS['burned_area/'+mo.zfill(2)+'/burned_fraction'][:],index = DS['lat'][:,0],
                                  columns = DS['lon'][0,:], dtype='float' )
            else: #cburned fraction only given in this format until 2016, add nans for later years
                arr = np.empty([720,1440])  # because there are 720 latitudes and 1440 longitudes resolved in the measurement/dataframe
                arr[:] = np.nan
                dfy_area = pd.DataFrame(data=arr,index = DS['lat'][:,0],
                                  columns = DS['lon'][0,:], dtype='float' )
            dfy_area = pd.melt(dfy_area.reset_index(), id_vars='index',
                                  value_name='burned_fraction',var_name='Long')
            dfy_area["Lat"] = dfy_area['index']

        elif datatype == 1:  # == CO2 emission, datatype given as argument
            datae = DS['emissions/'+mo.zfill(2)+'/DM'][:]  # DM = dry matter (burned)
            CO2emission = np.zeros((720, 1440))
            for t in range(6): #for each of the eco systems, add up emissions
                contribution = DS['emissions/'+mo.zfill(2)+'/partitioning/DM_'+typeL[t]][:]
                CO2emission += datae * contribution * CO2L[t]  # CO2L = emission factors given in beginning
            #Create Pandas DataFrame
            dfy = pd.DataFrame(data = CO2emission, index = DS['lat'][:,0],
                                  columns = DS['lon'][0,:], dtype='float')
       
        elif datatype == 2:  # == CO emission, datatype given as argument
            datae = DS['emissions/'+mo.zfill(2)+'/DM'][:]
            CO2emission = np.zeros((720, 1440))
            for t in range(6): #for each of the eco systems, add up emissions
                contribution = DS['emissions/'+mo.zfill(2)+'/partitioning/DM_'+typeL[t]][:]
                CO2emission += datae * contribution * COL[t]
            #Create Pandas DataFrame
            dfy = pd.DataFrame(data=CO2emission,index = DS['lat'][:,0],
                                  columns = DS['lon'][0,:], dtype='float' )
        #unpivot dataframe (2D array -> columns: Lat, Long, emission)
        dfy = pd.melt(dfy.reset_index(), id_vars='index',
                                  value_name ='emission',var_name = 'Long')
        dfy["Lat"] = dfy['index']

        #merge burned area if applicable
        if datatype == 0:
            dfy = pd.merge(dfy, dfy_area, on=['Lat','Long'], how = 'outer')
            # this merges dfy with dfy_area: for the same lat,long there are values from both df's given (with all possible combinations). 
            # If one lat,long value is only given in one of the two df's, NaN is inserted in the field of the df with the missing lat,long

        #merge grid area into df and calculate total emissions
        dfy = pd.merge(dfy, dfy_gridarea, on=['Lat','Long'], how = 'outer') # merges two df's, see right above
        dfy.insert(loc=1, column='total_emission', value = dfy.Grid_area*dfy['emission']/(1000000000000)) # inserts one column after the index (lat) column 
        # Divide by 10^12 to convert gCO2 to Terra-gCO2 (TgCO2)

        dfy.insert(loc=1,column='Month',value= int(m)*np.ones(len(dfy['emission'])))
        dfy.insert(loc=1,column='Year',value= int(datayear)*np.ones(len(dfy['emission'])))
        MonthDate = dfy.apply(lambda x: datetime.date(int(x.Year),int(x.Month),15),axis=1) # applies the function "lambda" on the dataframe
        # function lambda is defined after the ":", "x" is the argument of the function. datetime.date just creates a date out of the previously given year, month
        dfy.insert(loc=1,column='Date',value=MonthDate)

        if m==1: # first time the function runs we return a exact copy of the before created dataframe dfy
            df = dfy.copy()
        else: # else, we append the created dfy dataframe to the already existing dataframe df 
            df = df.append(dfy)#merge dataframes
    
    return df


def createDataFrameGFED(datapath,savepath,datatype,Long_min,Long_max,Lat_min,Lat_max,RegionName,Transcom):
    # create df for several years starting in 2009
    # resulting unit for 'total_emission' for type 'CO2': TgCO2/month/gridcell
    # datapath = path to .hdf5 file
    typelist = ['C','CO2','CO']
    if not os.path.isfile(savepath+"DFglobal_"+typelist[datatype]+".pkl"): # modify path so that it suits you
        #tests whether (modified) path is an existing file. Here: if it is NOT existing
        print('start creating Dataframe')
        print('process 2009')
        df = createAnnualDataFrameGFED(datapath+'GFED4.1s_2009.hdf5',2009,datatype)
        for j in range(2010,2022): # 2022 not included
            print('process year'+str(j))
            if j < 2017:
                df2 = createAnnualDataFrameGFED(datapath+'GFED4.1s_'+str(j)+'.hdf5',j,datatype)
            else:
                df2 = createAnnualDataFrameGFED(datapath+'GFED4.1s_'+str(j)+'_beta.hdf5',j,datatype) # starting for year 2017 the names are modified with "_beta"
            df = df.append(df2)
        df.to_pickle(savepath+"DFglobal_"+typelist[datatype]+".pkl") # modify path so that it suits you
        # saves the df to pickle file that can be used for analyzing & plotting
    else: # if the file "DFglobal_"+typelist[datatype]+".pkl" already exists
        print('reading dataframe, this may take some minutes')
        df = pd.read_pickle(savepath+"DFglobal_"+typelist[datatype]+".pkl")
    df = df[(df.Long >= Long_min) & (df.Long <= Long_max) & (df.Lat >= Lat_min) & (df.Lat <= Lat_max)] # select specified latitudes, longitudes
    print('create GeoDataFrame')
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Long, df.Lat),crs = 'EPSG:4326') 
    # geometry: plots points at x-y position of the long & lat 
    # crs: coordinate reference system: selects the used projection of the map that is printed

    if UseTranscom:
        print('cut Transcom Region, this too needs a cuple of minutes')
        if not os.path.isfile(savepath+"DFglobal_"+typelist[datatype]+"_"+RegionName+".pkl"):# modify path so that it suits you
            Transcom = pd.read_pickle("/home/eschoema/Transcom_Regions.pkl")
            igdf = gdf.within(Transcom[(Transcom.transcom == RegionName)].geometry.iloc[0]) # returns bools if gdf point lies within the Transcom area
            gdf = gdf.loc[igdf]
    gdf.to_pickle(savepath+"DFglobal_"+typelist[datatype]+"_"+RegionName+".pkl")# modify path so that it suits you

    return gdf


def getMonthSum(gdf,date_min,date_max,ValueName):
    # function to create monthly sum of indicated column inside given borders
    # input:
    #       gdf: Geodataframe containing the data
    #       year_max, month_max, day_max: enddate
    #       year_min,month_min, day_min: startdate
    #       Long_min, Long_max: Longitude range
    #       Lat_min, Lat_max: Latitude range
    #       ValueName: Name of column to sum up
    #
    # output:
    #       df with monthly summed values
    print('calculate monthly means')
    gdf2 = gdf[(gdf.Date >= date_min) & (gdf.Date <= date_max)]

    output = gdf2.groupby(['Year','Month'])[ValueName].sum().reset_index()  # returns a df of the sum for each month for all lat,long
    MonthDate = output.apply(lambda x: datetime.date(int(x.Year),int(x.Month),15),axis=1)  # creates a date from the numbers in "year" and "month"
    output.insert(loc=1,column='MonthDate',value=MonthDate)  # make another column with the created date as value
    print(output.info())
    print(output)
    return output


def plotMonthlyMeansTimeSeries(savepath, MonthDate, total_emission, savefig=True):
    plt.plot(MonthDate,total_emission, color = 'r', label = 'GFED')
    plt.grid(axis='x')
    plt.ylabel('Fire CO$_2$ emissions [TgCO$_2$/month]')
    plt.xlabel('Date')
    plt.title('GFED Fire CO$_2$ Emissions in South America') 

    if savefig == True: 
      plt.savefig(savepath+"GDF_CO2(time)_2009_2021.png")  # modify path so that it suits you


def plotMonthlyMeanSpatially(savepath, syear, smonth, gdf, savefig=True):
    # input: syear = start year, same for smonth
    sdf = gdf[(gdf['Year']==syear)&(gdf['Month']==smonth)]
    '''
    sdf = sdf[(145<sdf["Lat"])&(sdf["Lat"]<155)&(-20>sdf["Long"])&(sdf["Long"]>-40)]
    sdf = sdf[(sdf["Lat"]<155)]
    sdf = sdf.set_geometry("centroid")
    sdf = gdf.set_index(['Lat', 'Long'])
    xsdf = xr.Dataset.from_dataframe(sdf)
    foo = xr.DataArray(xsdf['xco2'][:])
    '''
    sdf = sdf.set_index(['Lat', 'Long'])  # set two index: First index = Lat ; Second index = Long
    xsdf = xr.Dataset.from_dataframe(sdf)  # convert pandas df to an xarray dataset
    foo = xr.DataArray(xsdf['total_emission'][:])  # creates a DataArray with the xsdf data: ndarrays with values & labeled dimensions & coordinates
    
    plt.figure(figsize=(14, 10))
    plt.rcParams.update({'font.size': 13})
    
    ax = plt.axes(projection=ccrs.PlateCarree()) # cartopy ccrs.PlateCarree() == normal 2D world map
    #ax.set_xlim((146,156))
    #ax.set_ylim((-41,-22))
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--') # creates gridlines of continents
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False
    
    ''' for Australia:
    gl.xlocator = mticker.FixedLocator([120,130,140,150,160,170,180])
    gl.ylocator = mticker.FixedLocator([-10,-20,-30,-40, -50])
    '''
    
    # for South-America:
    gl.xlocator = mticker.FixedLocator([-30,-40,-50,-60,-70,-80,-90])
    gl.ylocator = mticker.FixedLocator([20,10,0,-10,-20,-30,-40,-50,-60])

    #cmap_sns = sns.color_palette("YlOrBr", as_cmap=True)
    N = 17 # number of contour lines; default = 7
    foo.plot.contourf(x = 'Long', y = 'Lat', levels=N, extend = 'max', ax=ax, vmin=0, vmax=2.4, transform = ccrs.PlateCarree(), cmap = 'Reds', #prev. 'OrRd', better: 'Reds'
                      cbar_kwargs=dict(orientation='vertical', shrink=0.65, label='CO$_2$ Emissions [TgCO$_2$/month]', ticks = np.arange(0,2.5,0.3)))
    '''
    plots contours of world and fills it with the data (colorcode "cmap")
        extend: determines the coloring of values that are outside the "levels" range - default: do not color them
                with 'max' --> color values that are above the "levels" range
        vmax = defines the max data range that the color map covers
    '''
    
    ax.coastlines()

    if savefig == True: 
        #plt.savefig(savepath+'cartopy_example.png')
        plt.savefig(savepath+"GDF_CO2map_{}_{}_xr.png".format(smonth, syear), bbox_inches='tight')

    #plt.show()
    

#main
if __name__ == "__main__": #if this script is executed; this part is not executed if function of this script is imported in another script
    datapath = "/mnt/data/users/eschoema/GFED4/"
    savepath = "/mnt/data/users/lartelt/MA/GFED4/"
    datatype = 1 # CO2
    Long_min = -90
    Long_max = -30
    Lat_min = -60
    Lat_max = 20
    #RegionName = 'SAT' # South America multipolygen, see "/home/eschoema/Transcom_Regions.pkl" or as test in "test_GFED_file.ipynb"
    RegionName = 'square_region_lukas'
    UseTranscom = False # True: Fire data is cut along Eva's chosen TRANSCOM region

    typelist = ['C','CO2','CO']
    
    if not os.path.isfile(savepath+"DFglobal_"+typelist[datatype]+"_"+RegionName+".pkl"):
        print('creating new DataFrame GFED...')
        gdf = createDataFrameGFED(datapath,savepath,datatype,Long_min,Long_max,Lat_min,Lat_max,RegionName, UseTranscom)
    else:
        print("loading data frame")
        gdf = pd.read_pickle(savepath+"DFglobal_"+typelist[datatype]+"_"+RegionName+".pkl")
    #print(gdf['Month'])
    print(gdf)
    print("calculating monthly means")
    MonthMeans = getMonthSum(gdf,datetime.date(2009,1,1),datetime.date(2021,12,31),'total_emission')
    
    #print(gdf.columns)
    #print(gdf[(gdf["Lat"]>-35)&(gdf["Lat"]<-34)&(gdf["Long"]==140.125)&(gdf["Year"]==2019)])
    #print(gdf["geometry"])
    #print(gdf["Grid_area"])
    '''
    print('plotting monthly means time series')
    plotMonthlyMeansTimeSeries(savepath, MonthMeans.MonthDate,MonthMeans.total_emission)
    '''
    # why previously "MonthMeans.total_emission*12/44" ?
    # --> mol-masse: CO2=44g/mol ; C=12g/mol
    
    
    '''
    print('plotting spatially year 2009')
    plotMonthlyMeanSpatially(savepath, 2009, 1, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2009, 2, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2009, 3, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2009, 4, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2009, 5, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2009, 6, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2009, 7, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2009, 8, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2009, 9, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2009, 10, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2009, 11, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2009, 12, gdf, savefig=True)
    print('plotting spatially year 2010')
    plotMonthlyMeanSpatially(savepath, 2010, 1, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2010, 2, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2010, 3, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2010, 4, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2010, 5, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2010, 6, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2010, 7, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2010, 8, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2010, 9, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2010, 10, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2010, 11, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2010, 12, gdf, savefig=True)
    print('plotting spatially year 2011')
    plotMonthlyMeanSpatially(savepath, 2011, 1, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2011, 2, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2011, 3, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2011, 4, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2011, 5, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2011, 6, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2011, 7, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2011, 8, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2011, 9, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2011, 10, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2011, 11, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2011, 12, gdf, savefig=True)
    print('plotting spatially year 2012')
    plotMonthlyMeanSpatially(savepath, 2012, 1, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2012, 2, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2012, 3, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2012, 4, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2012, 5, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2012, 6, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2012, 7, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2012, 8, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2012, 9, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2012, 10, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2012, 11, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2012, 12, gdf, savefig=True)
    print('plotting spatially year 2013')
    plotMonthlyMeanSpatially(savepath, 2013, 1, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2013, 2, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2013, 3, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2013, 4, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2013, 5, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2013, 6, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2013, 7, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2013, 8, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2013, 9, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2013, 10, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2013, 11, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2013, 12, gdf, savefig=True)
    print('plotting spatially year 2014')
    plotMonthlyMeanSpatially(savepath, 2014, 1, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2014, 2, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2014, 3, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2014, 4, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2014, 5, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2014, 6, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2014, 7, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2014, 8, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2014, 9, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2014, 10, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2014, 11, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2014, 12, gdf, savefig=True)
    print('plotting spatially year 2015')
    plotMonthlyMeanSpatially(savepath, 2015, 1, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2015, 2, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2015, 3, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2015, 4, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2015, 5, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2015, 6, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2015, 7, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2015, 8, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2015, 9, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2015, 10, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2015, 11, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2015, 12, gdf, savefig=True)
    print('plotting spatially year 2016')
    plotMonthlyMeanSpatially(savepath, 2016, 1, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2016, 2, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2016, 3, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2016, 4, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2016, 5, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2016, 6, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2016, 7, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2016, 8, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2016, 9, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2016, 10, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2016, 11, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2016, 12, gdf, savefig=True)
    print('plotting spatially year 2017')
    plotMonthlyMeanSpatially(savepath, 2017, 1, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2017, 2, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2017, 3, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2017, 4, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2017, 5, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2017, 6, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2017, 7, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2017, 8, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2017, 9, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2017, 10, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2017, 11, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2017, 12, gdf, savefig=True)
    print('plotting spatially year 2018')
    plotMonthlyMeanSpatially(savepath, 2018, 1, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2018, 2, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2018, 3, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2018, 4, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2018, 5, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2018, 6, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2018, 7, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2018, 8, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2018, 9, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2018, 10, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2018, 11, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2018, 12, gdf, savefig=True)
    print('plotting spatially year 2019')
    plotMonthlyMeanSpatially(savepath, 2019, 1, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2019, 2, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2019, 3, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2019, 4, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2019, 5, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2019, 6, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2019, 7, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2019, 8, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2019, 9, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2019, 10, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2019, 11, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2019, 12, gdf, savefig=True)
    print('plotting spatially year 2020')
    plotMonthlyMeanSpatially(savepath, 2020, 1, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2020, 2, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2020, 3, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2020, 4, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2020, 5, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2020, 6, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2020, 7, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2020, 8, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2020, 9, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2020, 10, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2020, 11, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2020, 12, gdf, savefig=True)
    print('plotting spatially year 2021')
    plotMonthlyMeanSpatially(savepath, 2021, 1, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2021, 2, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2021, 3, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2021, 4, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2021, 5, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2021, 6, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2021, 7, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2021, 8, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2021, 9, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2021, 10, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2021, 11, gdf, savefig=True)
    plotMonthlyMeanSpatially(savepath, 2021, 12, gdf, savefig=True)
    '''