"""Library of functions to create datasets of Sentinel-1 SAR ratio images, Sentinel-1 local incidence angle, landcover classification, forest cover fraction, snow classification, and snow cover.

Author: Eric Gagliano (egagli@uw.edu)
Updated: 01/2024
"""

import numpy as np
import geopandas as gpd
import rasterio as rio
import xarray as xr
import rioxarray as rxr
import sys
sys.path.append('../generate_sentinel1_local_incidence_angle_maps')
import generate_lia
import pandas as pd
import odc.stac,odc
import pystac_client
import planetary_computer
import glob
import os
import shutil
import earthaccess





def get_s1_rtc(bbox_gdf,start_time='2019-01-01',end_time='2019-12-31',resolution=10, epsg=None, resampling=None):
    
    if epsg == None:
        epsg = bbox_gdf.estimate_utm_crs().to_epsg()
        
        
    catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,)
    bbox = bbox_gdf.total_bounds
    search = catalog.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=f"{start_time}/{end_time}",limit=1000)
    
    items = search.item_collection()

    ds = odc.stac.load(
    search.get_items(), 
    chunks={'x':512,'y':512}, 
    bands={"vv","vh"},
    groupby = 'sat:absolute_orbit',
    crs=epsg,
    resampling=resampling,
    fail_on_error = False,
    resolution=odc.geo.Resolution(resolution, -resolution)).where(lambda x: x > 0, other=np.nan)
    
    bounding_box_utm_gf = bbox_gdf.to_crs(ds.rio.crs)
    xmin, ymax, xmax, ymin = bounding_box_utm_gf.bounds.values[0]
    
    scenes = ds.sel(x=slice(xmin,xmax),y=slice(ymin,ymax)).sortby('time')
    
    df = gpd.GeoDataFrame.from_features(items.to_dict())
    df = df.groupby(['sat:absolute_orbit']).first().sort_values('datetime')

    scenes = scenes.assign_coords({'sat:orbit_state':('time',df['sat:orbit_state'])})
    scenes = scenes.assign_coords({'sat:relative_orbit':('time',df['sat:relative_orbit'].astype('int16'))}).to_array(dim='band').transpose('time','band','y','x')
    
    scenes = scenes.assign_attrs({'resolution':resolution})
    
    scenes.data = scenes.data.rechunk((100,1,512,512))
    
    return scenes 



def aggregate_reference_scenes(ts_ds, time_reference_scenes, reference_scenes_aggregation_technique = 'median'):
    
    if reference_scenes_aggregation_technique == 'mean':
        composite_reference_scenes = ts_ds.sel(time=time_reference_scenes).groupby('sat:relative_orbit').mean()
    elif reference_scenes_aggregation_technique == 'median':
        composite_reference_scenes = ts_ds.sel(time=time_reference_scenes).groupby('sat:relative_orbit').median()
    elif reference_scenes_aggregation_technique == 'max':
        composite_reference_scenes = ts_ds.sel(time=time_reference_scenes).groupby('sat:relative_orbit').max()
        
    return composite_reference_scenes
    
    
def calculate_ratio_images(ts_ds,composite_reference_scenes):
    
    ratio_images = 10*np.log10(ts_ds.groupby('sat:relative_orbit')/composite_reference_scenes)
    
    return ratio_images
    
    
def select_closest(group,time_of_interest):
    
    closest = group.sel(time=time_of_interest,method='nearest')
    
    return closest


def select_closest_ratio_images(ratio_images, time_target_scene):
    
    ratio_images_closest = ratio_images.groupby('sat:relative_orbit').map(lambda group: select_closest(group, time_target_scene))
    
    return ratio_images_closest


def get_ratio_images(bbox_gdf, year, time_reference_scenes, time_target_scene, reference_scenes_aggregation_technique, resolution):
    
    
    ts_ds = get_s1_rtc(bbox_gdf,
                       start_time=f'{year}-01-01',
                       end_time=f'{year}-12-31',
                       resolution=resolution,
                       resampling='bilinear')
    
    ts_ds = ts_ds.where(ts_ds>0.006) # remove border noise
    
    composite_reference_scene = aggregate_reference_scenes(ts_ds, time_reference_scenes, reference_scenes_aggregation_technique = 'median')
    ratio_images = calculate_ratio_images(ts_ds, composite_reference_scene)
    ratio_images_closest = select_closest_ratio_images(ratio_images, time_target_scene)
    
    return ratio_images_closest

def get_lia(bbox_gdf):
    
    dem_fp = '/tmp/dems'
    output_fp = '/tmp/lia_maps'
    text_fp = '/tmp/lia_data.txt'
    
    geojson = bbox_gdf.attrs['filename']
    
    
    if os.path.isfile(text_fp):
        with open(text_fp,'r') as f:
            lia_data = f.read()
    else: 
        lia_data = ''
    
    if geojson == lia_data:
        print('Using already present LIA data.')
        
    else:
        print('Creating new LIA data.')
        if os.path.isdir(dem_fp):
            shutil.rmtree(dem_fp)
        if os.path.isdir(output_fp):
            shutil.rmtree(output_fp)
        
        generate_lia.geojson_to_lia_rasters_and_lia_stack(geojson,dem_folder_path = dem_fp, output_folder_path = output_fp,res=20)
        
        with open(text_fp, 'w') as f:
            f.write(geojson)

    
    dem = rxr.open_rasterio(f'{dem_fp}/dem_UTM.tif')
    lia_stack_fn = glob.glob(f'{output_fp}/lia_stack_orbits_*.nc')[0]
    lia_stack = xr.open_dataarray(lia_stack_fn,decode_coords='all').rio.write_crs(dem.rio.crs)
    mean_local_incidence_angles = np.rad2deg(lia_stack)
    mean_local_incidence_angles = mean_local_incidence_angles.where(mean_local_incidence_angles<100)
    lia = mean_local_incidence_angles.rio.clip(bbox_gdf.geometry.values,bbox_gdf.crs, drop=True)
    
    
    
    return lia


def get_worldcover(bbox_gdf, return_classmap=False):
        
    catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,)

    search = catalog.search(
    collections=["esa-worldcover"],
    bbox=bbox_gdf.total_bounds)
    
    items = list(search.get_items())
    class_list = items[0].assets["map"].extra_fields["classification:classes"]
    classmap = {
        c["value"]: {"description": c["description"], "hex": c["color-hint"]}
        for c in class_list
    }

    stack_lc = odc.stac.load(search.get_items(),
                             resolution=0.0001,
                             bbox=bbox_gdf.total_bounds,
                             epsg = bbox_gdf.estimate_utm_crs().to_epsg(),
                             bands=["map"]).isel(time=-1).where(lambda x: x > 0, other=np.nan)
    
    stack_lc = stack_lc['map'].rio.write_nodata(0,encoded =True)
    
    if return_classmap == False:
        return stack_lc
    else:
        return stack_lc, classmap
    
    
def get_fcf(bbox_gdf):
    
    fcf = rxr.open_rasterio('https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif',mask_and_scale=True)
    xmin, ymin, xmax, ymax = bbox_gdf.total_bounds
    fcf = fcf.rio.clip_box(xmin, ymin, xmax, ymax).squeeze()
    
    return fcf


def get_snow_class(bbox_gdf):
    
    snow_class = rxr.open_rasterio('https://snowmelt.blob.core.windows.net/snowmelt/eric/snow_classification/SnowClass_GL_300m_10.0arcsec_2021_v01.0.tif', chunks=True, mask_and_scale=True)
    xmin, ymin, xmax, ymax = bbox_gdf.total_bounds
    snow_class = snow_class.rio.clip_box(xmin, ymin, xmax, ymax).squeeze().rio.write_nodata(9,encoded=True)
    
    return snow_class


# def get_snow_extent(dataset):
    
#     bbox = dataset.rio.transform_bounds(rio.crs.CRS.from_epsg(4326))
#     year = int(dataset.time.dt.year[0])
    
#     catalog = pystac_client.Client.open(
#         "https://planetarycomputer.microsoft.com/api/stac/v1",
#         modifier=planetary_computer.sign_inplace,)
    
#     search = catalog.search(collections=["modis-10A2-061"], bbox=bbox, datetime=f"{year}-04-01/{year}-06-30",limit=1000)
#     items = search.item_collection()
        
#     modis_snow = odc.stac.load(
#         search.get_items(), 
#         bands="Maximum_Snow_Extent",
#         like = dataset, 
#         resampling = "nearest")['Maximum_Snow_Extent']
    
#     modis_binary_snow = xr.where(modis_snow==200,1,0)
    
#     modis_binary_snow_interp = modis_binary_snow.interp_like(dataset['ratio_images'].swap_dims({'sat:relative_orbit':'time'}),method='nearest').swap_dims({'time':'sat:relative_orbit'})
    
#     return modis_binary_snow_interp



def get_modis_ndsi(dataset):
    
    modis_ndsi_dict = {}
    
    for rel_orbit in np.unique(dataset['sat:relative_orbit'].values):
        
        time = pd.to_datetime(dataset.where(dataset['sat:relative_orbit']==rel_orbit, drop=True).time.values[0]).strftime('%Y-%m-%d')

        results = earthaccess.search_data(
            short_name='MOD10A1F',
            cloud_hosted=True,
            bounding_box=dataset.rio.transform_bounds(rio.crs.CRS.from_epsg(4326)),
            temporal=(f"{time}",f"{time}"),
            count=10
        )

        temp_download_fp = '/tmp/local_folder'

        files = earthaccess.download(results, temp_download_fp)

        modis = rxr.open_rasterio(*files,mask_and_scale=True).squeeze()


        modis_ndsi = modis['CGF_NDSI_Snow_Cover'].where(modis['CGF_NDSI_Snow_Cover']<=100,np.nan)
        
        modis_ndsi_dict[rel_orbit] = modis_ndsi.rio.reproject_match(dataset)

        shutil.rmtree(temp_download_fp)
    
    return modis_ndsi_dict
    
    
    
    
        
def create_dataset(bbox_gdf, year, time_reference_scenes, time_target_scene, reference_scenes_aggregation_technique, resolution):
    
    ratio_images = get_ratio_images(bbox_gdf, year, time_reference_scenes, time_target_scene, reference_scenes_aggregation_technique, resolution) # 10m but we pull at 100m
    dataset = ratio_images.to_dataset(name='ratio_images')
    dataset = dataset.rio.write_crs(bbox_gdf.estimate_utm_crs().to_epsg())
    

    lia = get_lia(bbox_gdf) # 20m
    dataset['local_incidence_angle'] = lia.rio.reproject_match(dataset, Resampling = rio.enums.Resampling.bilinear)
    
    worldcover = get_worldcover(bbox_gdf) # 10m
    dataset['esa_worldcover'] = worldcover.rio.reproject_match(dataset, Resampling = rio.enums.Resampling.mode)
    
    fcf = get_fcf(bbox_gdf) # 100m
    dataset['forest_cover_fraction'] = fcf.rio.reproject_match(dataset, Resampling = rio.enums.Resampling.bilinear)
    
    snow_class = get_snow_class(bbox_gdf) # 300m
    dataset['snow_class'] = snow_class.rio.reproject_match(dataset, Resampling = rio.enums.Resampling.nearest)
    
#    modis_8day_snow_extent = get_snow_extent(dataset) # 500m
#    dataset['modis_8day_max_snow_extent'] = modis_8day_snow_extent
    
    modis_ndsi_dict = get_modis_ndsi(dataset) # 500m
    modis_ndsi_da = xr.concat(modis_ndsi_dict.values(), dim=pd.Index(modis_ndsi_dict.keys(), name='sat:relative_orbit'))
    dataset = dataset.assign(modis_ndsi=modis_ndsi_da) 
    
    return dataset



def dataset_to_dataframe(dataset):
    
    dataframe = dataset.to_dataframe()
    dataframe_obs = dataframe.reset_index()
    dataframe_obs = dataframe_obs.dropna(how='any')
    
    return dataframe_obs



def dataframe_ndsi_to_binary(dataframe, ndsi_thresh):
    if isinstance(ndsi_thresh, list) and len(ndsi_thresh) == 2:
        dataframe['modis_binary'] = dataframe['modis_ndsi'].apply(lambda x: 0 if x < ndsi_thresh[0] else (1 if x > ndsi_thresh[1] else np.nan))
        dataframe = dataframe.dropna(how='any')
    else:
        dataframe['modis_binary'] = dataframe['modis_ndsi'].apply(lambda x: 0 if x <= ndsi_thresh else 1)
    
    return dataframe

def dataframe_numbers_to_classes(dataframe):
    
    description_dict = {10:'Tree cover', 20:'Shrubland', 30:'Grassland', 40:'Cropland', 50:'Built-up', 60:'Bare / sparse vegetation', 70:'Snow and ice', 80:'Permanent water bodies', 90:'Herbaceous wetland', 95:'Mangroves', 100:'Moss and lichen'}
    dataframe['esa_worldcover'] = dataframe['esa_worldcover'].replace(description_dict)

    snow_classes_dict = {1:'Tundra',2:'Boreal Forest',3:'Maritime',4:'Ephemeral (includes no snow)',5:'Prairie',6:'Montane Forest',7:'Ice (glaciers and ice sheets)',8:'Ocean',9:'Fill'}
    dataframe['snow_class'] = dataframe['snow_class'].replace(snow_classes_dict)
    
    if 'modis_binary' in dataframe.columns:
        snow_nosnow_dict = {0:'no snow',1:'snow'}
        dataframe['modis_binary'] = dataframe['modis_binary'].replace(snow_nosnow_dict)
    
    return dataframe


    
    

