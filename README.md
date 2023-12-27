# wet_snow_threshold_localization

This repository is for SAR binary wet snow threshold localization
google slides with some ideas and background: https://docs.google.com/presentation/d/1y1C19CQyOyy0HFLKyf9x4ZlzIvoxL7Bz3Yzis7T1P6w/edit?usp=sharing

## the problem with with binary wet snow detection
for SAR binary wet snow detection, there are a ton of methods for thresholding (e.g. for SAR: -1dB to -2dB to -3dB, VV, VH, VV/VH combinations weighted on LIA, vegetation, different reference images, etc), but very little spatially distributed wetness data to validate against :( 

## an idea for an empirical approach

* for a ratio image with wet snow (choose a scene where most/all of snow is wet), compare histograms of dB change values
  * compare dB drop within snow / no snow classes using optical imagery (figure on the right from Nagler et al. 2016)
  * hopefully find bimodal distribution in dB drop (snow / no snow) which would allow us to optimize a wet snow threshold
* iterate at locations with different landcover/veg, different snow classification types, snow depths, variable topography (to get at incidence angle dependence) 
  * build up a huge datacube containing [VVdB drop,VHdB drop, snow/nosnow, LIA, LCC/veg, snowclass, where in the melt season are we, fSCA, othervars?] for each pixel at all locations
* find trends, optimize, or use ML/DL approach
  * will help us characterize how binary wet snow threshold changes with these variables allowing localization of this threshold
  * end goal would be some sort of heuristic: given a pixel with x vegetation/LC type, y snow class, z incidence angle, we expect at least a backscatter change of -XdB in VV and -YdB in VH from the reference image to the wet snow image
  * if it works, could be the foundation of a “smart” binary wet snow algorithm, where the binary wet snow threshold is variable across a scene as a function of [landcover class, snow class, local incidence angle, etc]

## To do list:
* notebook to build dataset
  * data sources
    * for Sentinel-1 RTC data: https://github.com/egagli/sar_snowmelt_timing/blob/main/dev/collaborations/sierra_nevada/sierra_nevada_binary_wet_snow_map_timeseries.ipynb 
    * LIA: https://github.com/egagli/generate_sentinel1_local_incidence_angle_maps
    * LC: https://planetarycomputer.microsoft.com/dataset/esa-worldcover
    * FCF: ?
    * Snow classification: https://nsidc.org/data/nsidc-0768/versions/1
    * Snow cover (snow / no snow): https://planetarycomputer.microsoft.com/dataset/modis-10A1-061 (or 8 day)
  * loop over a bunch of areas, multiple years, save a bunch of these as zarr files?
      * scene selection criteria
         * global? what are the extents of our individual datasets
  * questions to consider
    * reference scene
      * time span?
      * mean or median?
      * multi-year?
    * how to find when most snow is melting
      * morning / afternoon overpass?
      * temperature data?
      * remote sensing thermal?
      * in-situ weather station?
      * reanalysis?
* notebook to ingest saved zarr files, stitch all together. possibly ouput as singular large dataframe with each row being a single pixel
* notebook to analyze full datas
  * aggregate statistics
  * histograms analyzing each variable
    * for example, LIA on x axis, dB change on right axis, then two box plots for each LIA, one for snow and one for no snow, maybe each of those broken into VV and VH
  * test VV and VH for higher wet snow seperability
  * given snow / no snow distributions, which threshold provides the most seperability between distributions
    * create confusion matrix based on optimal threholds
  * analysis of variance
  * multiple linear regression
  * unsupervised machine learning
    * kmeans clustering
  * deep learning approach 




