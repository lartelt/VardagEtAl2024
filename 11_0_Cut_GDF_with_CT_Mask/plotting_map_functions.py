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
mpl.use('Agg')
import matplotlib.pyplot as plt
import warnings
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
    fig, ax2 = plt.subplots(figsize = (11,8))
    # define region map to plot as contour in background
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    world.boundary.plot(ax=ax2, color = 'black')
    gdf.plot(ax=ax2, column = 'geometry', legend=True, markersize=3, color='darkgreen', alpha=0.9, label="TM5_gridded_flux")
    print('done plotting, now saving...')
    ax2.set_xlabel('Longitude', fontsize=14)
    ax2.set_ylabel('Latitude', fontsize=14)
    ax2.tick_params(bottom=True, left=True, axis='both', color='slategray')
    ax2.legend(framealpha=0.25, facecolor='grey', loc='upper right', fontsize=13)
    if savefig:
        plt.savefig(savepath_w_savename, dpi=300, bbox_inches='tight')



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
    
    gdf.plot(ax=ax2, column = 'geometry', legend=True, markersize=20, color='darkgreen', alpha=0.9, label="model grid")
    ax2.set_xlabel('Longitude', fontsize=14)
    ax2.set_ylabel('Latitude', fontsize=14)
    ax2.tick_params(bottom=True, left=True, axis='both', color='slategray')
    ax2.legend(framealpha=0.25, facecolor='grey', loc='upper right', fontsize=13)
    ax2.set_title('TM5/IS+RT cosampled RT', fontsize=18)
    print('done plotting, now saving...')
    plt.savefig(savepath_w_savename, dpi=300, bbox_inches='tight')



if __name__=='__main__':
    print('Do not use this file as main file, only to import in different files')
    
    