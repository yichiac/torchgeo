import os
import numpy as np
from osgeo import gdal, ogr
import csv
import time
import shapefile
# import matplotlib.pyplot as plt


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
patch_dirs = patch_dirs[0]
time_dirs = os.listdir(os.path.join(ROOT_DIRS, patch_dirs))
time_dirs = time_dirs[0]
filepath = os.path.join(ROOT_DIRS, patch_dirs, time_dirs, "all_bands.tif")
print("filepath", filepath)

# open raster tiles to get corners
src = gdal.Open(filepath)
xmin, ymax, xmax, ymin = get_corners(src)

# I think it is okay to load all gdal corners first then loop through shapefiles to reduce the impacts of O2

# load shapefile to get information
shppath = os.path.join("/Users/yc/projects/dali/data", "india_fields", "india_10k_fields.shp")
layer = load_shapefile(shppath)

# crop and rasterize shapefile within the bounding box
p1 = (xmin, ymin)
p2 = (xmax, ymin)
p3 = (xmax, ymax)
p4 = (xmin, ymax)

# crop
i = 0
mask_path = filepath.replace(os.path.basename(filepath), "mask.shp")
w = shapefile.Writer(mask_path)
w.field("sample", i)  # pyshp needs at least one field
w.poly([[p4, p3, p2, p1]])  # generate bbox polygon
w.record('bbox')
w.close()

# generate .PRJ file
crs_wkt = src.GetSpatialRef()
prj = open(mask_path.replace("shp","prj"), "w")
prj.write(crs_wkt.ExportToWkt())
prj.close()

# root = 'D:/data/extent/'
# path = 'D:/data/extent_merge/'
# shapefiles = glob.glob(root + '/' + '/*.shp')
# gdf = pandas.concat([
#     geopandas.read_file(shp)
#     for shp in shapefiles
# ]).pipe(geopandas.GeoDataFrame)
# gdf.crs = {'init': 'epsg:32610'}
# gdf.to_file(path + '/' + 'compiled.shp')



# rasterize


# use the coordinate to select polygon(s) and assign no data for blank area
start = time.time()
for poly in layer:
    geom = poly.GetGeometryRef()
    x = geom.Centroid().GetX()
    y = geom.Centroid().GetY()
    if ulx < x and x < lrx:
        if lry < y and y < uly:
            print('Got the polygon: ', (x, y))

print('Shapefile Time: ', time.time()-start)

# crop and produce masks


# plot