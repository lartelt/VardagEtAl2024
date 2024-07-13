# Author: Lukas Artelt
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cf
import numpy as np
import datetime
import geopandas
import geoplot as gplt
import netCDF4 as nc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.colors as mcolors
mpl.use('Agg')
from matplotlib import ticker as mticker
from matplotlib import cm
import matplotlib.pyplot as plt
import warnings
import imageio
warnings.filterwarnings("ignore")
import seaborn as sns
#sns.set_theme(style="darkgrid")
sns.set(rc={'axes.facecolor':'gainsboro'})

# plotting functions
def plotGDFspatially(gdf: geopandas.GeoDataFrame, RegionName: str, savepath: str, satellite_type:str='OCO2', mean_vertical_and_horizontal: str = None, 
                    plot_title: str = None, savefig: bool = False):
    '''# Documentation
    Function to plot the monthly mean xco2 values spatially - Need RegionName for fig title
    # input:
        - gdf: geodataframe including only measurements inside the 'RegionName' area 
        - RegionName (str): SA (all of South America), SAT (temperate SA = southern part of SA), SATr (tropical SA = northern part of SA)
        - savepath (str): Path to save the figure
        - satellite_type (str): 'TM5_gridded_flux'
        - mean_value_vertical: if != None, a vertical line will be plotted with the mean value for the specified column, e.g. the longitude if mean_value_vertical='longitude'
        - mean_value_horizontal: if != None, a horizontal line will be plotted with the mean value for the specified column, e.g. the latitude if mean_value_vertical='latitude'
        - season: changes the figure title, e.g. 'dry_season' or 'rainy_season' or 'after_2014' or 'before_2014'
        - plot_title: chance to personally define the figure title. If None, a standard figure title will be plotted
    # output:
            plot: map of SA with points representing the measurements, color of points indicates the monthly mean xco2 value
    '''
    print('start plotting...')
    Transcom_data_path = '/home/eschoema/Transcom_Regions.pkl'
    # define region map to plot as contour in background
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    countries = world[['geometry','name','continent']]
    SA = countries[(countries.continent == 'South America')]
    fig, ax2 = plt.subplots(figsize = (11,8))
    SA.boundary.plot(ax=ax2, color = 'black')
    Transcom = pd.read_pickle(Transcom_data_path)
    SAT=Transcom[(Transcom.transcom == 'SAT_SATr')].to_crs('EPSG:4326')
    SAT.boundary.plot(ax=ax2, color='black')
    GDF_clip = geopandas.overlay(gdf, SA, how='union')
    # select GOSAT measurements located inside the SA borders
    #if satellite_type==('TM5_gridded_flux' or 'TM5_gridded_3x2_coarsen_flux'):
        #GDF_clip.plot(ax=ax2, column = 'xco2', legend=True, markersize=3, cmap='Reds', vmin=370, vmax=420, legend_kwds={'label': "OCO2 measured xCO2 [ppm]"})
    GDF_clip.plot(ax=ax2, column = 'geometry', legend=True, markersize=5, color='darkgreen', alpha=0.9, label="TM5_gridded_flux")
    
    #Transcom = pd.read_pickle(Transcom_data_path) # South America: SAT, SATr, SATr_SE, SAT_SATr
    #SA=Transcom[(Transcom.transcom == RegionName)]
    print('done plotting, now saving...')
    ax2.set_xlabel('Longitude', fontsize=14)
    ax2.set_ylabel('Latitude', fontsize=14)
    ax2.tick_params(bottom=True, left=True, axis='both', color='slategray')
    if mean_vertical_and_horizontal != None:
        ax2.axvline(np.mean(gdf['longitude'].values), ymin=0, ymax=1, color='slategray', 
                    label=('mean longitude = ' + str(round(np.mean(gdf['longitude'].values), 2))))
        ax2.axhline(np.mean(gdf['latitude'].values), xmin=0, xmax=1, color='gray', 
                    label=('mean latitude = ' + str(round(np.mean(gdf['latitude'].values), 2))))
    
    #ax2.legend(framealpha=0.25, facecolor='grey', loc='upper right', fontsize=13)
    
    if plot_title!=None:
        plt.title(plot_title)
    else:
        plt.title(satellite_type, fontsize=18, pad=-16)
    
    if savefig:
        if mean_vertical_and_horizontal!=None:
            plt.savefig(savepath+satellite_type+"_all_datapoints_for_" + RegionName + "_with_mean_longitude_and_latitude.png", dpi=300, bbox_inches='tight')
        else:
            plt.savefig(savepath+satellite_type+"_all_datapoints_for_" + RegionName + ".png", dpi=300, bbox_inches='tight')


def plotGDFspatially_simple_world(gdf: geopandas.GeoDataFrame, savepath_w_savename: str, savefig: bool = True):
    Transcom_data_path = '/home/eschoema/Transcom_Regions.pkl'
    fig, ax2 = plt.subplots(figsize = (11,8))
    # define region map to plot as contour in background
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    world.boundary.plot(ax=ax2, color = 'black')
    gdf.plot(ax=ax2, column = 'geometry', legend=True, markersize=1, color='darkgreen', alpha=0.9, label="TM5_gridded_flux")
    print('done plotting, now saving...')
    ax2.set_xlabel('Longitude', fontsize=14)
    ax2.set_ylabel('Latitude', fontsize=14)
    ax2.tick_params(bottom=True, left=True, axis='both', color='slategray')
    ax2.legend(framealpha=0.25, facecolor='grey', loc='upper right', fontsize=13)
    if savefig:
        plt.savefig(savepath_w_savename, dpi=600, bbox_inches='tight')


def plotGDFspatially_simple_SA(gdf: geopandas.GeoDataFrame, savepath_w_savename: str):
    CT_Mask = pd.read_pickle('/home/eschoema/masks/CTTranscomMask1x1Borders.pkl')
    fig, ax2 = plt.subplots(figsize = (11,8))
    # define region map to plot as contour in background
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    countries = world[['geometry','name','continent']]
    SA = countries[(countries.continent == 'South America')]
    SA = countries[(countries.continent == 'South America')]
    SA.boundary.plot(ax=ax2, color = 'black')
    
    #SAT=CT_Mask[(CT_Mask.transcom == 'SAT')].to_crs('EPSG:4326')
    #SATr=CT_Mask[(CT_Mask.transcom == 'SATr')].to_crs('EPSG:4326')
    #SAT.boundary.plot(ax=ax2, color='black')
    #SATr.boundary.plot(ax=ax2, color='black')
    #GDF_clip = geopandas.overlay(gdf, SA, how='intersection')
    
    gdf.plot(ax=ax2, column = 'geometry', legend=True, markersize=20, color='darkgreen', alpha=0.9, label='ERA5 grid')
    ax2.set_xlabel('Longitude', fontsize=14)
    ax2.set_ylabel('Latitude', fontsize=14)
    ax2.tick_params(bottom=True, left=True, axis='both', color='slategray')
    ax2.legend(framealpha=0.25, facecolor='grey', loc='upper right', fontsize=13)
    ax2.set_title('ERA5 humid regions', fontsize=18)
    print('done plotting, now saving...')
    plt.savefig(savepath_w_savename, dpi=300, bbox_inches='tight')
    
    
def plotGDFspatially_simple_SA_for_ERA5_humid_regions(gdf: geopandas.GeoDataFrame, savepath_w_savename: str, threshold: float=1, minMonths: int=6):
    fig, ax2 = plt.subplots(figsize = (11,8))
    # define region map to plot as contour in background
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    countries = world[['geometry','name','continent']]
    SA = countries[(countries.continent == 'South America')]
    SA = countries[(countries.continent == 'South America')]
    SA.boundary.plot(ax=ax2, color = 'black')
    
    gdf.plot(ax=ax2, column = 'geometry', legend=True, markersize=20, color='darkgreen', alpha=0.9, label='humid regions')
    ax2.set_xlabel('Longitude', fontsize=14)
    ax2.set_ylabel('Latitude', fontsize=14)
    ax2.tick_params(bottom=True, left=True, axis='both', color='slategray')
    #ax2.legend(framealpha=0.25, facecolor='grey', loc='upper right', fontsize=13)
    ax2.set_title('ERA5 humid regions threshold='+str(threshold)+' minMonths='+str(minMonths), fontsize=12)
    print('done plotting, now saving...')
    plt.savefig(savepath_w_savename, dpi=300, bbox_inches='tight')    



# plot maps as xarray
def plotMapForERA5_tp(gdf_ERA5: geopandas.GeoDataFrame, Year_or_Month_plotted: int=1, savepath: str='/home/lartelt/MA/software/'): 
    '''Documentation
    Function to plot the ERA5 precipitation data for a specific month or year in SA
    # arguments:
        - gdf_ERA5: geopandas gdf with the ERA5 precip per_gridcell
        - Year_or_Month_plotted: int, year or month that should be plotted. Month=[1,12], Year=[2009,2021]
    '''   
    
    if Year_or_Month_plotted>12:
        what_is_plotted = 'Year'
        print('grouping ERA5 by Year...')
        df_ERA5_new = gdf_ERA5.groupby(['Year', 'latitude', 'longitude'])['tp'].sum().reset_index()
    else:
        what_is_plotted = 'Month'
        print('grouping ERA5 by Month...')
        df_ERA5_new = gdf_ERA5.groupby(['Month', 'latitude', 'longitude'])['tp'].sum().reset_index()
    savename='ERA5_precipitation_for_'+what_is_plotted+'_'+str(Year_or_Month_plotted)
    
    sdf = df_ERA5_new[(df_ERA5_new[what_is_plotted]==Year_or_Month_plotted)]
    sdf = sdf.set_index(['latitude', 'longitude'])  # set two index: First index = Lat ; Second index = Long
    xsdf = xr.Dataset.from_dataframe(sdf)  # convert pandas df to an xarray dataset
    foo = xr.DataArray(xsdf['tp'][:])  # creates a DataArray with the xsdf data: ndarrays with values & labeled dimensions & coordinates
    
    plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree()) # cartopy ccrs.PlateCarree() == normal 2D world map
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4, color='white', alpha=0.5, linestyle='-') # creates gridlines of continents
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False
    # for South-America:
    gl.xlocator = mticker.FixedLocator([-30,-40,-50,-60,-70,-80,-90])
    gl.ylocator = mticker.FixedLocator([20,10,0,-10,-20,-30,-40,-50,-60])
    
    cmap = cm.Blues
    norm = mpl.colors.Normalize(vmin=0, vmax=foo.max())
    foo.plot.contourf(x = 'longitude', y = 'latitude', extend = 'max', ax=ax, transform = ccrs.PlateCarree(), levels=256, cmap=cmap, vmin=0, vmax=0.1,
                      cbar_kwargs=dict(orientation='vertical', norm=norm, shrink=1, pad=0.03, 
                                       ticks=np.arange(0,0.125,0.025),
                                       label='precipitation [m/'+what_is_plotted+'/gridcell]'))
    ax.coastlines()

    # define region map to plot as contour in background
    CT_Mask = pd.read_pickle('/home/eschoema/masks/CTTranscomMask1x1Borders.pkl')
    SAT=CT_Mask[(CT_Mask.transcom == 'SAT')].to_crs('EPSG:4326')
    SAT.boundary.plot(ax=ax, color='black', linewidth=0.8)
    
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.tick_params(bottom=True, left=True, axis='both', color='slategray')
    ax.set_title('ERA5 precipitation for '+what_is_plotted+' '+str(Year_or_Month_plotted), fontsize=18)
    print('done plotting, now saving...')
    plt.savefig(savepath+savename+'.png', dpi=300, bbox_inches='tight')
    


# plot maps as xarray
def plotMapForERA_low_precip(gdf_ERA5: geopandas.GeoDataFrame, Year_plotted: int=1, column_to_plot: str='N_months', threshold: float=0.0, plot_mean:bool=False, savepath: str='/home/lartelt/MA/software/'): 
    '''Documentation
    Function to plot the Month where the ERA5 data shows low precipitation in a year in SA
    # arguments:
        - gdf_ERA5: geopandas gdf with the ERA5 precip per_gridcell
        - Year_plotted: int, year that is plotted. Year=[2009,2021]
        - column_to_plot: str, column that is plotted. column_to_plot=['N_months_with_low_precip', 'Month_of_first_low_precip', 'Month_of_last_low_precip']
    '''   
    
    if plot_mean==False:
        what_is_plotted = 'Year'
        sdf = gdf_ERA5[(gdf_ERA5[what_is_plotted]==Year_plotted)]
    else:
        what_is_plotted = 'mean'
        sdf = gdf_ERA5
    sdf = sdf.set_index(['latitude', 'longitude'])  # set two index: First index = Lat ; Second index = Long
    xsdf = xr.Dataset.from_dataframe(sdf)  # convert pandas df to an xarray dataset
    foo = xr.DataArray(xsdf[column_to_plot][:])  # creates a DataArray with the xsdf data: ndarrays with values & labeled dimensions & coordinates
    
    plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree()) # cartopy ccrs.PlateCarree() == normal 2D world map
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4, color='white', alpha=0.5, linestyle='-') # creates gridlines of continents
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False
    # for South-America:
    gl.xlocator = mticker.FixedLocator([-30,-40,-50,-60,-70,-80,-90])
    gl.ylocator = mticker.FixedLocator([20,10,0,-10,-20,-30,-40,-50,-60])
    
    if column_to_plot=='N_months_with_low_precip':
        #cmap = cm.Set1
        cmap_colors = ['#00429d', '#2166ac', '#4386c2', '#6699cc', '#8bafcf', '#aebbd6', 'lightgrey']
        cmap = mcolors.ListedColormap(cmap_colors)
        vmin=0
        vmax=6
        levels=7
        label='#month with low precip'
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    elif column_to_plot=='Month_of_first_low_precip' or column_to_plot=='Month_of_last_low_precip':
        #cmap = cm.tab20
        distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b','#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#b17b2e', '#66c2a5']
        #distinct_colors_13 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b','#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#b17b2e', '#66c2a5', '#9e9ac8']
        cmap = mcolors.ListedColormap(distinct_colors)
        vmin=1
        vmax=12
        norm = mcolors.BoundaryNorm(np.arange(0, 13), cmap.N)  # 13 boundaries for 12 colors
        #norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        label='first month with low precip'
        levels=12
    foo.plot.contourf(x = 'longitude', y = 'latitude', extend = 'max', ax=ax, transform = ccrs.PlateCarree(), levels=levels, cmap=cmap, vmin=vmin, vmax=vmax,
                      cbar_kwargs=dict(orientation='vertical', norm=norm, shrink=1, pad=0.03,
                                       ticks=np.arange(vmin, vmax+.5), boundaries=np.arange(vmin, vmax+.5),
                                       label=label))
    
    ax.coastlines()

    # define region map to plot as contour in background
    CT_Mask = pd.read_pickle('/home/eschoema/masks/CTTranscomMask1x1Borders.pkl')
    SAT=CT_Mask[(CT_Mask.transcom == 'SAT')].to_crs('EPSG:4326')
    SAT.boundary.plot(ax=ax, color='black', linewidth=0.8)
    
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.tick_params(bottom=True, left=True, axis='both', color='slategray')
    ax.set_title('Months w/ low precip ('+str(threshold)+'m/day) for '+what_is_plotted+' '+str(Year_plotted), fontsize=18)
    print('done plotting, now saving...')
    savename='ERA5_low_precipitation_'+column_to_plot+'_'+str(threshold).replace('.','_')+'_for_'+what_is_plotted
    plt.savefig(savepath+savename+'.png', dpi=300, bbox_inches='tight')



# plot GIF of maps
def create_GIF_from_images(image_list: list, what_is_plotted:str='tp', savepath_images: str='/home/lartelt/MA/', monthly_or_yearly: str='monthly') -> None:
    '''Creates a GIF from a list of images
    Parameters
    ----------
    image_list : list
        List of images
    savepath_images : str
        Path to images
    monthly_or_yearly : str, optional
        Whether the GIF is a 'monthly' or 'yearly' GIF
    '''
    images = []
    for name in image_list:
        images.append(imageio.imread(savepath_images + name))
    if what_is_plotted=='tp':
        imageio.mimsave(savepath_images + 'GIF_ERA5_precipitation_'+monthly_or_yearly+'.gif', images, duration=1200, loop=0)
    else:
        imageio.mimsave(savepath_images + 'GIF_ERA5_'+what_is_plotted+'_'+monthly_or_yearly+'.gif', images, duration=1200, loop=0)



if __name__=='__main__':
    print('Do not use this file as main file, only to import in different files')
    
    