import os

ROOT_DIRS = "/Users/yc/projects/dali/data/s2_india_fields/imgs/"
patch_dirs = os.listdir(ROOT_DIRS)
not_processed = []
processed = []

for patch_dir in patch_dirs:
    # create mask directory
    mask_dir = os.path.join(ROOT_DIRS, patch_dir, "mask")
    if not os.path.exists(mask_dir):
        time_dir = os.listdir(os.path.join(ROOT_DIRS, patch_dir))[0] # only one season for mask creation

        os.mkdir(mask_dir)

        filepath = os.path.join(ROOT_DIRS, patch_dir, time_dir, "all_bands.tif")
        mask_shp_path = filepath.replace(os.path.basename(filepath), "mask.shp")
        mask_raster_path = mask_shp_path.replace("shp", "tif")
        new_mask_raster_path = os.path.join(ROOT_DIRS, patch_dir, "mask", "mask.tif")
        new_mask_shp_path = os.path.join(ROOT_DIRS, patch_dir, "mask", "mask.shp")

        new_mask_dbf_path = os.path.join(ROOT_DIRS, patch_dir, "mask", "mask.dbf")
        new_mask_prj_path = os.path.join(ROOT_DIRS, patch_dir, "mask", "mask.prj")
        new_mask_shx_path = os.path.join(ROOT_DIRS, patch_dir, "mask", "mask.shx")

        print('Moving ', mask_raster_path, ' to ', new_mask_raster_path)
        os.rename(mask_raster_path, new_mask_raster_path)
        os.rename(mask_shp_path, new_mask_shp_path)
        os.rename(mask_shp_path.replace("shp", "dbf"), new_mask_dbf_path)
        os.rename(mask_shp_path.replace("shp", "prj"), new_mask_prj_path)
        os.rename(mask_shp_path.replace("shp", "shx"), new_mask_shx_path)

        processed.append(patch_dir)

    else:
        print("Mask directory already exists for ", patch_dir)
        not_processed.append(patch_dir)

print("Not processed: ", not_processed)
print("Total not processed: ", len(not_processed))
print("Total processed: ", len(processed))