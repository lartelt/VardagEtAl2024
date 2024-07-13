#Author: Eva-Marie Schoemann, modified by Lukas Artelt
#Date 08.08.2023

import datetime
#from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import glob, os
import geopandas
import xarray as xr
import time
from shapely.geometry import Polygon
import math
from skimage.measure import block_reduce
from RegionParam import getRegion
import warnings
#warnings.filterwarnings('ignore', '.*Mean of empty slice.*', )
warnings.filterwarnings('ignore')
import functions_to_load_datasets as load_func
'''
def createPandasDateFrame(dic):
    
    d = {'CO2': dic['REMOTEC-OUTPUT_X_COLUMN_CORR'].T[0], 
         'CH4': dic['REMOTEC-OUTPUT_X_COLUMN_CORR'].T[1],
         'CO2_uncorr': dic['REMOTEC-OUTPUT_X_COLUMN'].T[0],
         'CH4_uncorr': dic['REMOTEC-OUTPUT_X_COLUMN'].T[1],
         'Sec': dic['GOSAT_TIME_SEC'],
         'Min': dic['GOSAT_TIME_MIN'], 
         'Hour': dic['GOSAT_TIME_HOUR'],
         'Day': dic['GOSAT_TIME_DAY'], 
         'Month': dic['GOSAT_TIME_MONTH'],
         'Year': dic['GOSAT_TIME_YEAR'],
         'Lat':dic['GOSAT_LATITUDE'],
         'Long':dic['GOSAT_LONGITUDE'],
         'CT_CO2': dic['REMOTEC-OUTPUT_X_APR_COLUMN'].T[0],
         'CT_CH4': dic['REMOTEC-OUTPUT_X_APR_COLUMN'].T[1],
         'CT_error': dic['REMOTEC-OUTPUT_X_APR_COLUMN_ERR'].T[0],
         'CO2_error': dic['REMOTEC-OUTPUT_X_COLUMN_ERR'].T[0],
         'meas_geom' : dic['(0=NADIR,1=GLINT)'],
         'quality' : dic['REMOTEC-OUTPUT_FLAG_QUALITY'], #NICHT VERWENDEN, DA NICHT ZUVERLÄSSIG
         'gain': dic["GOSAT_GAIN"]}
    df = pd.DataFrame(data=d)
    
    df["Lat_round"] = np.floor(np.array(df["Lat"])) 
    df["Long_round"] = np.floor(np.array(df["Long"]))

    return df
'''


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


def getReferenceDate(year_min,year_max,month_min,month_max):
    yea = []
    mon = []
    for k in range(year_min,year_max +1):
        if k == year_min:
            for p in range(month_min,12+1):
                yea.append(k)
                mon.append(p)
        elif k == year_max:
            for p in range(1,month_max+1):
                yea.append(k)
                mon.append(p)
        else:
            for p in range(1,13):
                yea.append(k)
                mon.append(p)

    dateData = {"Year":yea,"Month":mon}
    DateRef = pd.DataFrame(data=dateData)
    MonthDate2 = []
    for j in range(len(DateRef.Year)):
        MonthDate2.append(datetime.date(DateRef.Year[j],DateRef.Month[j],15))
    DateRef['MonthDate'] = MonthDate2

    #year from july to june index column
    jjYear = []
    for i in range(len(DateRef.Year)):
        if DateRef.Month.values[i] <= 6:
            jjYear.append(DateRef.Year.values[i]-1)
        elif DateRef.Month.values[i] >= 7:
            jjYear.append(DateRef.Year.values[i])
        else:
            print('Error')
    DateRef.insert(loc = 1, column= 'JJYear',value = jjYear)

    return DateRef


def getReferencesDateDay(year_min,year_max,month_min,month_max,day_min,day_max):
    yea = []
    mon = []
    day = []
    for k in range(year_min,year_max +1):
        if k == year_min:
            for p in range(month_min,12+1):
                if p == month_min:
                    for d in range(day_min, getNumDayOfMonth(k,p)+1):
                        day.append(d)
                        yea.append(k)
                        mon.append(p)
                else:
                    for d in range(1, getNumDayOfMonth(k,p)+1):
                        day.append(d)
                        yea.append(k)
                        mon.append(p)
        elif k == year_max:
            for p in range(1,month_max+1):
                if p == month_max:
                    for d in range(1, day_max):
                        day.append(d)
                        yea.append(k)
                        mon.append(p)
                else:
                    for d in range(1, getNumDayOfMonth(k,p)+1):
                        day.append(d)
                        yea.append(k)
                        mon.append(p)
        else:
            for p in range(1,13):
                for d in range(1, getNumDayOfMonth(k,p)+1):
                    day.append(d)
                    yea.append(k)
                    mon.append(p)

    dateData = {"Year":yea,"Month":mon,"Day":day}
    DateRef = pd.DataFrame(data=dateData)
    MonthDate2 = []
    for j in range(len(DateRef.Year)):
        MonthDate2.append(datetime.date(DateRef.Year[j],DateRef.Month[j],DateRef.Day[j]))
    DateRef['MonthDate'] = MonthDate2

    return DateRef

def getNumDayOfMonth(year,month):
    listDays = [31,28,31,30,31,30,31,31,30,31,30,31]
    listDaysl = [31,29,31,30,31,30,31,31,30,31,30,31]
    if year < 1900:
        print('year is out of implemented range, check code')
    elif year in list(range(1904,2100,4)):
        days = listDaysl[month-1]
    else:
        days = listDays[month-1]

    return days


#-------------------------------------------------------------------------------------------------------------------------------------
def CreateDF(DS,Datatype):
    for d in range(len(DS[Datatype])):
        for nam in [Datatype,Datatype+'_mad']:#,'GPP_mad','GPP_n']:        
            dfyCO2 = pd.DataFrame(data=block_reduce(DS.variables[nam][d].values,block_size=(12,12),func=np.nanmean),
                                  index = list(np.array(range(895,-905,-10))/10), columns = list(np.array(range(-1795,1805,10))/10), dtype='float')
            dfyCO2 = pd.melt(dfyCO2.reset_index(), id_vars='index', value_name =nam,var_name = 'Long')
            dfyCO2["Lat"] = dfyCO2['index']
            dfyCO2.insert(loc=1,column='Year1',value= np.ones(len(dfyCO2["Lat"]))*int(str(DS.time.values[d])[0:4]))        
            dfyCO2.insert(loc=1,column='Month1',value= np.ones(len(dfyCO2["Lat"]))*int(str(DS.time.values[d])[5:7]))
            dfyCO2.insert(loc=1,column='Day1',value= np.ones(len(dfyCO2["Lat"]))*int(str(DS.time.values[d])[8:10]))
            dfyCO2.insert(loc=1,column='Year2',value= np.ones(len(dfyCO2["Lat"]))*((datetime.date(int(str(DS.time.values[d])[0:4]),int(str(DS.time.values[d])[5:7]),int(str(DS.time.values[d])[8:10]))+datetime.timedelta(7,0)).year))
            dfyCO2.insert(loc=1,column='Month2',value= np.ones(len(dfyCO2["Lat"]))*((datetime.date(int(str(DS.time.values[d])[0:4]),int(str(DS.time.values[d])[5:7]),int(str(DS.time.values[d])[8:10]))+datetime.timedelta(7,0)).month))
            dfyCO2.insert(loc=1,column='Day2',value= np.ones(len(dfyCO2["Lat"]))*((datetime.date(int(str(DS.time.values[d])[0:4]),int(str(DS.time.values[d])[5:7]),int(str(DS.time.values[d])[8:10]))+datetime.timedelta(7,0)).day))
            
            if nam in Datatype:
                dfy = dfyCO2.copy()
            else:
                dfy = pd.merge(dfyCO2, dfy, on=['Lat','Long','Day2', 'Month2', 'Year2', 'Day1', 'Month1','Year1'])

        if d == 0:
            df = dfy.copy()
        else:
            df = df.append(dfy)
                     
    return df


def CreateDataFrameSIFFC(Lat_min,Lat_max,Long_min,Long_max,RegionName,Num,Datatype=''):
    a_corr2 = []
    for i in range(895,-905,-10):
        geom = [Polygon(zip([100,100,101,101],[i/10-0.5,i/10+0.5,i/10+0.5,i/10-0.5]))]
        GData = geopandas.GeoDataFrame({'data':[1]}, geometry=geom)
        GData.crs = 'epsg:4326'
        GData = GData.to_crs({'proj':'cea'})
        a_corr2.append(GData.geometry.area[0])
    
    path1="/mnt/data/users/lartelt/MA/FLUXCOM/df2_"+Datatype+RegionName+str(Num)+".pkl"
    if not os.path.isfile(path1):
        path2="/mnt/data/users/lartelt/MA/FLUXCOM/dfd02_"+Datatype+RegionName+str(Num)+".pkl"
        #print(path2)
        if not os.path.isfile(path2):
            #print("/mnt/data/users/lartelt/MA/FLUXCOM/dfd02_"+Datatype+RegionName+str(Num)+".pkl")
            print("Start reading data:")
            if Datatype == 'NEE':
                print('get files for NEE')
                dp = glob.glob("/mnt/data/users/eschoema/FLUXCOM/NEE.RS_V006.FP-NONE.MLM-ALL.METEO-NONE.4320_2160.8daily.201*.nc")
                #dp = glob.glob("/mnt/data/users/lschrein/FLUXCOM/GPP.RS_V006.FP-ALL.MLM-ALL.METEO-NONE.4320_2160.8daily.2010.nc")
                dp.extend(glob.glob("/mnt/data/users/eschoema/FLUXCOM/NEE.RS_V*2009.nc"))
            elif Datatype == 'TER':
                print('get files for TER')
                dp = glob.glob("/mnt/data/users/eschoema/FLUXCOM/"+Datatype+".RS_V006.FP-ALL.MLM-ALL.METEO-NONE.4320_2160.8daily.201*.nc")
                #dp = glob.glob("/mnt/data/users/lschrein/FLUXCOM/GPP.RS_V006.FP-ALL.MLM-ALL.METEO-NONE.4320_2160.8daily.2010.nc")
                dp.extend(glob.glob("/mnt/data/users/eschoema/FLUXCOM/"+Datatype+".RS_V*2009.nc"))
            elif Datatype == 'GPP':
                dp = glob.glob("/mnt/data/users/lschrein/FLUXCOM/"+Datatype+".RS_V006.FP-ALL.MLM-ALL.METEO-NONE.4320_2160.8daily.201*.nc")
                dp.extend(glob.glob("/mnt/data/users/lschrein/FLUXCOM/"+Datatype+".RS_V*2009.nc"))
            else:
                print('Datatype not known')
                raise('ERROR!')
            for num, filepath in enumerate(dp):
                print(filepath)
                DS = xr.open_mfdataset(filepath, combine='by_coords',drop_variables = 'time_bnds')#decode_times=False)
       
                #create Dataframe
                df3= CreateDF(DS,Datatype)
                
                df2 = df3[(df3.Long >= Long_min) & (df3.Long <= Long_max) & (df3.Lat >= Lat_min) & (df3.Lat <= Lat_max)]

                if num == 0:
                    df = df2.copy()
                else:
                    df = df.append(df2, ignore_index=True)
        
            print("finished reading data") 
            df.to_pickle(path2)
        else:
            print('reading dataset from pickle: /mnt/data/users/lartelt/MA/FLUXCOM/dfd02_'+Datatype+RegionName+str(Num)+'.pkl')
            df = pd.read_pickle("/mnt/data/users/lartelt/MA/FLUXCOM/dfd02_"+Datatype+RegionName+str(Num)+".pkl")
        #create date variable
        print("create timestamp")
        
        dfd = getReferencesDateDay(2009,2019,1,12,1,31)
       
        gpp = []
        gpp_mad = []
        #gpp_n = []
        gppt = []
        gpp_madt = []
        #gpp_nt = []
        lat = []
        lon = []
        date = []
        ml = []
        dl = []
        yl = []
        # the reference date set has an entry for ever day, each day gets a NEE value (as FLUXCOM has 8 day means, 8 days will get the same value)
        print('...in for loop')
        for i,row in dfd.iterrows():
            #for ind in df[(df.Year == row.Year)&(row.Month <= df.Month2)&(row.Month >= df.Month1)&(row.Day <= df.Day2)&(row.Day >= df.Day1)].index.values:!WRONG!!!!!
            for ind in df[(row.Year <= df.Year2)&(row.Year >= df.Year1)&(row.Month <= df.Month2)&(row.Month >= df.Month1)&(row.Day <= df.Day2+31*(df.Month2-row.Month))&(row.Day >= df.Day1-31*(row.Month-df.Month1))].index.values:
                #print(df[(df.Year == row.Year)&(row.Month <= df.Month2)&(row.Month >= df.Month1)&(row.Day <= df.Day2)&(row.Day >= df.Day1)].Lat)
                gpp.append(df.iloc[ind][Datatype])
                gpp_mad.append(df.iloc[ind][Datatype+'_mad'])
                #gpp_n.append(df[(df.Year == row.Year)&(row.Month <= df.Month2)&(row.Month >= df.Month1)&(row.Day <= df.Day2)&(row.Day >= df.Day1)].GPP_n)
                gppt.append((df.iloc[ind][Datatype]*a_corr2[int(90-(df.iloc[ind].Lat+0.5))]))
                gpp_madt.append((df.iloc[ind][Datatype+'_mad']*a_corr2[int(90-(df.iloc[ind].Lat+0.5))]))
                #gpp_nt.append((df[(df.Year == row.Year)&(row.Month <= df.Month2)&(row.Month >= df.Month1)&(row.Day <= df.Day2)&(row.Day >= df.Day1)].GPP_n*111319*111000*math.cos(math.radians(df[(df.Year == row.Year)&(row.Month <= df.Month2)&(row.Month >= df.Month1)&(row.Day <= df.Day2)&(row.Day >= df.Day1)].Lat))).mean())
                lat.append(df.iloc[ind].Lat)
                lon.append(df.iloc[ind].Long)            
                ml.append(int(row.Month))
                yl.append(int(row.Year))
                dl.append(int(row.Day))
                date.append(datetime.date(int(row.Year),int(row.Month),int(row.Day)))    
        del dfd
        print('done for loop')
        datad = {'Date':date,Datatype:gpp,Datatype+'tot':gppt,'Lat':lat,'Long':lon,'Year':yl,'Month':ml,'Day':dl}
        dfd = pd.DataFrame(data=datad)
        dfd.insert(loc=1,column=Datatype+'_mad',value=gpp_mad)
        #dfd.insert(loc=1,column='GPP_n',value=gpp_n)
        dfd.insert(loc=1,column=Datatype+'_madtot',value=gpp_madt)
        #dfd.insert(loc=1,column='GPP_ntot',value=gpp_nt)

        dfd.to_pickle("/mnt/data/users/lartelt/MA/FLUXCOM/df2_"+Datatype+RegionName+str(Num)+".pkl")
    #create GeoDataFrame
    else:
        print('reading dataset from pickle: /mnt/data/users/lartelt/MA/FLUXCOM/df2_'+Datatype+RegionName+str(Num)+".pkl")
        dfd = pd.read_pickle("/mnt/data/users/lartelt/MA/FLUXCOM/df2_"+Datatype+RegionName+str(Num)+".pkl") 

    print("create GeoDataFrame")
    gdf = geopandas.GeoDataFrame(dfd, geometry=geopandas.points_from_xy(dfd.Long, dfd.Lat))
    gdf.crs = {'init' :'epsg:4326'}
    
    #if Num >= 900:
    #    Transcom = pd.read_pickle("/home/eschoema/Transcom_Regions.pkl")
    #    igdf = gdf.within(Transcom[(Transcom.transcom == RegionName)].geometry.iloc[0])
    #    gdf = gdf.loc[igdf]
    if Num < 900 and Num > 700:
        print('cut in SAT region by CT-Mask')
        CT_Mask = pd.read_pickle('/home/eschoema/masks/CTTranscomMask1x1Borders.pkl')
        igdf1 = gdf.within(CT_Mask[(CT_Mask.transcom == 'SAT')].geometry.iloc[0])
        #igdf2 = gdf.within(CT_Mask[(CT_Mask.transcom == 'SATr')].geometry.iloc[0])
        gdf_SAT = gdf.loc[igdf1]
        #gdf_SATr = gdf.loc[igdf2]
    print('saving gdf to pickle...')
    gdf.to_pickle("/mnt/data/users/lartelt/MA/FLUXCOM/gdf2_"+Datatype+RegionName+str(Num)+".pkl")
    print('done saving!')


#-------------------------------------------------------------------------------------------------------------------------------------
def calculate_flux_per_month_per_gridcell_or_subregion(gdf: geopandas.GeoDataFrame, variable: str='NEE', to_unit: str='per_gridcell', region:str='SAT', savename:str=None):
    print('start calculating flux per month '+to_unit)
    # calc flux per month per gridcell
    if to_unit == 'per_gridcell':
        df_monthly_gridcell = (gdf.groupby(['MonthDate', 'Lat', 'Long'])[[variable+'tot', variable+'_madtot']].sum()*10**-12).reset_index()
        gdf_monthly_gridcell = geopandas.GeoDataFrame(df_monthly_gridcell, geometry=geopandas.points_from_xy(df_monthly_gridcell.Long, df_monthly_gridcell.Lat))
        gdf_monthly_gridcell.to_pickle('/mnt/data/users/lartelt/MA/FLUXCOM/gdf_'+variable+'_'+region+'_per_month_per_gridcell.pkl')
    elif to_unit == 'per_subregion':
        gdf_monthly_subregion = (gdf.groupby(['MonthDate'])[[variable+'tot', variable+'_madtot']].sum()*10**-12).reset_index()
        gdf_monthly_subregion.to_pickle('/mnt/data/users/lartelt/MA/FLUXCOM/gdf_'+variable+'_'+region+'_per_month_per_subregion.pkl')
    elif to_unit == 'per_subregion_arid_and_humid':
        gdf_monthly_subregion = (gdf.groupby(['MonthDate'])[[variable+'tot', variable+'_madtot']].sum()).reset_index()
        gdf_monthly_subregion.to_pickle('/mnt/data/users/lartelt/MA/FLUXCOM/'+savename+'.pkl')



# main
'''
Numm  = 721
RegionName, Long_min, Long_max, Lat_min, Lat_max = getRegion(Numm)

#for variable in ['NEE','TER']:
variable = 'GPP'
print(variable)
CreateDataFrameSIFFC(Lat_min,Lat_max,Long_min,Long_max,RegionName,Numm,variable)
print('DONE!')
'''

'''# create gdf for SAT region out of the output in "main" (see above)
CT_Mask = pd.read_pickle('/home/eschoema/masks/CTTranscomMask1x1Borders.pkl')
#for variable in ['NEE','TER', 'GPP']:
for variable in ['GPP']:
    print(variable)
    print('loading gdf...')
    gdf = pd.read_pickle('/mnt/data/users/lartelt/MA/FLUXCOM/output_of_skript_get_FLUXCOM_data_py_NOT_cut_for_SAT_region/gdf2_'+variable+'SAT721.pkl')
    print('done loading gdf, now get rid of NaNs')
    gdf_new = gdf.dropna(axis='index')
    print('done getting rid of NaNs, now cutting into SAT region')
    igdf1 = gdf_new.within(CT_Mask[(CT_Mask.transcom == 'SAT')].geometry.iloc[0])
    gdf_SAT = gdf_new.loc[igdf1]
    print('done cutting, now adding new column MonthDate...')
    gdf_SAT['MonthDate'] = gdf_SAT.apply(lambda x: datetime.date(x.Year, x.Month, 15), axis=1)
    print('done adding MonthDate, now saving gdf to pickle...')
    gdf_SAT.to_pickle("/mnt/data/users/lartelt/MA/FLUXCOM/gdf_"+variable+"_SAT_per_day_per_gridcell.pkl")
    print('done saving!')
'''

'''# create gdf with correct units
for variable in ['NEE', 'TER', 'GPP']:
    print(variable)
    gdf = pd.read_pickle('/mnt/data/users/lartelt/MA/FLUXCOM/gdf_'+variable+'_SAT_per_day_per_gridcell.pkl')
    calculate_flux_per_month_per_gridcell_or_subregion(gdf, variable, to_unit='per_gridcell')
    calculate_flux_per_month_per_gridcell_or_subregion(gdf, variable, to_unit='per_subregion')
    print('DONE ' + variable)
'''

'''# create FLUXCOM nbp data
FLUXCOM_NEE = load_func.load_FLUXCOM_model_gdf('NEE', unit='per_subregion')
GFED_fire = load_func.load_GFED_model_gdf(region='SAT', unit='per_subregion', end_year=2019)

FLUXCOM_nbp = FLUXCOM_NEE.copy().drop(columns=['NEEtot'])
FLUXCOM_nbp['NBPtot'] = FLUXCOM_NEE['NEEtot'] + GFED_fire['total_emission']
FLUXCOM_nbp.rename(columns={'NEE_madtot': 'NBP_madtot'}, inplace=True)
FLUXCOM_nbp.to_pickle('/mnt/data/users/lartelt/MA/FLUXCOM/gdf_NBP_SAT_per_month_per_subregion.pkl')
'''


'''# Cut FLUXCOM data into subregion arid/humid
strict_bound = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5_precipitation/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_0_001_minMonths_5_SAT.pkl')
loose_bound = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5_precipitation/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_0_001_minMonths_4_SAT.pkl')

region_list = ['arid', 'humid']
variable_list = ['NEE', 'TER', 'GPP']
for region in region_list:
    print(region)
    for boundary_type in [strict_bound, loose_bound]:
        if boundary_type is strict_bound:
            boundary = 'strict'
        elif boundary_type is loose_bound:
            boundary = 'loose'
        print('cut gdf into '+region+' region')
        for var in variable_list:
            ds = load_func.load_FLUXCOM_model_gdf(variable=var, region='SAT', unit='per_gridcell')
            gdf_subregion = cut_gridded_gdf_into_region_from_CT_Mask(gdf=ds, CT_Mask=boundary_type, region_name=region, 
                                    savepath='/mnt/data/users/lartelt/MA/FLUXCOM/',
                                    savename='gdf_'+var+'_'+boundary+'limit_'+region+'_per_month_per_gridcell')
            print('done cutting, now calc subregional flux...')
            calculate_flux_per_month_per_gridcell_or_subregion(gdf_subregion, variable=var, to_unit='per_subregion_arid_and_humid',
                                    savename='gdf_'+var+'_'+boundary+'limit_'+region+'_per_month_per_subregion')
            print('done')
        
        # calc NBP
        FLUXCOM_NEE = load_func.load_FLUXCOM_model_gdf(variable='NEE', region=boundary+'limit_'+region, unit='per_subregion')
        GFED_fire = load_func.load_GFED_model_gdf(region='SAT_'+boundary+'limit_'+region, unit='per_subregion', end_year=2019)

        FLUXCOM_nbp = FLUXCOM_NEE.copy().drop(columns=['NEEtot'])
        FLUXCOM_nbp['NBPtot'] = FLUXCOM_NEE['NEEtot'] + GFED_fire['total_emission']
        FLUXCOM_nbp.rename(columns={'NEE_madtot': 'NBP_madtot'}, inplace=True)
        FLUXCOM_nbp.to_pickle('/mnt/data/users/lartelt/MA/FLUXCOM/gdf_NBP_'+boundary+'limit_'+region+'_per_month_per_subregion.pkl')
        
        print('done')
'''

# Cut FLUXCOM data into arid subregions
strict_bound = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5_precipitation/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_0_001_minMonths_5_SAT.pkl')
loose_bound = pd.read_pickle('/mnt/data/users/lartelt/MA/ERA5_precipitation/ERA5_1x1_grid/Mask_for_HighLowPrecip_Areas/ERA5_Polygon_AreaWithLowAndHighPrecipitation_threshold_0_001_minMonths_4_SAT.pkl')

region_list = ['arid_west', 'arid_east']
variable_list = ['NEE', 'TER', 'GPP']
for region in region_list:
    print(region)
    for boundary_type in [strict_bound, loose_bound]:
        if boundary_type is strict_bound:
            boundary = 'strict'
        elif boundary_type is loose_bound:
            boundary = 'loose'
        print('cut gdf into '+region+' region')
        '''
        for var in variable_list:
            ds = load_func.load_FLUXCOM_model_gdf(variable=var, region=boundary+'limit_arid', unit='per_gridcell')
            print(ds.head())
            gdf_subregion = cut_gridded_gdf_into_region_from_CT_Mask(gdf=ds, CT_Mask=boundary_type, region_name=region, 
                                    savepath='/mnt/data/users/lartelt/MA/FLUXCOM/',
                                    savename='gdf_'+var+'_'+boundary+'limit_'+region+'_per_month_per_gridcell')
            print('done cutting, now calc subregional flux...')
            calculate_flux_per_month_per_gridcell_or_subregion(gdf_subregion, variable=var, to_unit='per_subregion_arid_and_humid',
                                    savename='gdf_'+var+'_'+boundary+'limit_'+region+'_per_month_per_subregion')
            print('done')
        '''
        # calc NBP
        FLUXCOM_NEE = load_func.load_FLUXCOM_model_gdf(variable='NEE', region=boundary+'limit_'+region, unit='per_subregion')
        GFED_fire = load_func.load_GFED_model_gdf(region='SAT_'+boundary+'limit_'+region, unit='per_subregion', end_year=2019)

        FLUXCOM_nbp = FLUXCOM_NEE.copy().drop(columns=['NEEtot'])
        FLUXCOM_nbp['NBPtot'] = FLUXCOM_NEE['NEEtot'] + GFED_fire['total_emission']
        FLUXCOM_nbp.rename(columns={'NEE_madtot': 'NBP_madtot'}, inplace=True)
        FLUXCOM_nbp.to_pickle('/mnt/data/users/lartelt/MA/FLUXCOM/gdf_NBP_'+boundary+'limit_'+region+'_per_month_per_subregion.pkl')
        print('done')
        


