import os

import numpy as np
from osgeo import gdal, ogr
# import matplotlib.pyplot as plt
import csv

# load shapefile to get information
shapefile = os.path.join("india_fields", "india_10k_fields.shp")
driver = ogr.GetDriverByName('ESRI Shapefile')
dataSource = driver.Open(shapefile, 0) # 0 means read-only. 1 means writeable.

if dataSource is None:
    print('Could not open %s' % (shapefile))
else:
    print('Opened %s' % (shapefile))
    layer = dataSource.GetLayer()
    extents = layer.GetExtent()
    featureCount = layer.GetFeatureCount()
    print("Number of features in %s: %d" % (os.path.basename(shapefile),featureCount))

# retrieve geolocation in the shapefile
centroid_x = []
centroid_y = []
centroid_xy = []
for poly in layer:
    geom = poly.GetGeometryRef()
    x = geom.Centroid().GetX()
    y = geom.Centroid().GetY()
    centroid_x.append(x)
    centroid_y.append(y)
    centroid_xy.append((x,y))

# plt.scatter(centroid_x, centroid_y) # plot all scatter points on the map

with open('centroids.csv', 'w', newline='') as f: # write out the centroids
    writer = csv.writer(f, delimiter=',')
    key = list(range(len(centroid_x)))
    for i in zip(key, centroid_x, centroid_y):
        writer.writerow(i)

# access Sentinel-2 with the retrieved geolocation
# root = os.path.join("sen2")
# datamodule = Sentinel2(root=root, crs=None, res=10, bands=None, transforms=None, cache=True)
    # root=root, batch_size=64, num_workers=4, download=True)


# visualization


## downstream tasks
# root = os.path.join("eurosat")
# datamodule = CV4AKenyaCropType(root=root, batch_size=64, num_workers=4, download=True)


print('done')