import os

from osgeo import gdal, ogr

def get_corners(src):
    ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)
    return ulx, uly, lrx, lry

def load_shapefile(path: str):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(path, 0) # 0 means read-only. 1 means writeable.
    if dataSource is None:
        print('Could not open %s' % (path))
        return
    else:
        print('Opened %s' % (path))
        layer = dataSource.GetLayer()
        return layer


ROOT_DIRS = "/Users/yc/projects/dali/data/s2_india_fields/imgs/"
patch_dirs = os.listdir(ROOT_DIRS)
for patch_dir in patch_dirs:
    time_dir = os.listdir(os.path.join(ROOT_DIRS, patch_dir))[0] # only one season for mask creation
    filepath = os.path.join(ROOT_DIRS, patch_dir, time_dir, "all_bands.tif")
    print("Processing: ", filepath)

    # open raster tiles to get corners
    src = gdal.Open(filepath)
    xmin, ymax, xmax, ymin = get_corners(src)

    # load shapefile to get information
    shppath = os.path.join("/Users/yc/projects/dali/data", "india_fields", "india_10k_fields.shp")
    layer = load_shapefile(shppath)

    # crop shapefile
    mask_shp_path = filepath.replace(os.path.basename(filepath), "mask.shp")
    ogr_command = "ogr2ogr -clipsrc {} {} {} {} ".format(xmin,ymin,xmax,ymax,"s")
    ogr_command += "{} {}".format(mask_shp_path, shppath)
    os.system(ogr_command)

    # rasterize
    mask_raster_path = mask_shp_path.replace("shp", "tif")
    gdal_command = "gdal_rasterize -burn 1 -ts 264 264 -te {} {} {} {} {} {}".format(xmin, ymin, xmax, ymax, mask_shp_path, mask_raster_path)
    os.system(gdal_command)
