# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Cross-domain inference: PASTIS-trained ConvLSTM applied to Germany.

Streams a Sentinel-2 L2A time-series from Microsoft Planetary Computer over a
bbox in NRW (Münsterland), runs the most recent ``best`` checkpoint from
``outputs_full/`` tile-by-tile, and evaluates against EuroCrops DE_NRW_2021
using a small :class:`torchgeo.datasets.EuroCrops` subclass that maps HCAT
polygons to an 8-class common crop set via :meth:`EuroCrops.get_label`.

See ``implementation-notes.html`` for design decisions.
"""

import glob
import os
import resource
import zipfile
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pystac_client
import rasterio
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject, transform_bounds

from torchgeo.datasets.utils import download_url
from torchgeo.trainers import SpatioTemporalSegmentationTask

# --- constants ---------------------------------------------------------------

PASTIS_BANDS = ('B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A',
                'B11', 'B12')
COMMON_CLASSES = ('non_crop', 'wheat', 'barley', 'corn', 'rapeseed',
                  'grassland', 'other_crop', 'void')
COMMON_IGNORE_INDEX = 7
COMMON_CMAP = ListedColormap([
    '#555555', '#ff7f0e', '#2ca02c', '#ffbb78',
    '#8c564b', '#aec7e8', '#9467bd', '#ffffff',
])
# PASTIS 20-class -> common 8-class
PASTIS_TO_COMMON = np.array([
    0, 5, 1, 3, 2, 4, 2, 6, 6, 6,
    1, 1, 6, 6, 5, 6, 6, 1, 3, 7,
], dtype=np.int64)


def log_mem(tag: str) -> None:
    """Print RSS (and GPU VRAM if available). Cheap diagnostic for kill probes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
    extra = ''
    if torch.cuda.is_available():
        extra = (f', cuda alloc={torch.cuda.memory_allocated() / 1e9:.2f} GB'
                 f', cuda max={torch.cuda.max_memory_allocated() / 1e9:.2f} GB')
    print(f'[mem] {tag}: RSS={rss:.0f} MB{extra}', flush=True)


def hcat_to_common(name: str) -> int:
    """Map an EuroCrops ``EC_hcat_n`` string to the 8-class common set."""
    if not isinstance(name, str) or not name:
        return 0
    n = name.lower()
    if 'wheat' in n or 'triticale' in n:
        return 1
    if 'barley' in n:
        return 2
    if 'maize' in n or 'corn' in n:
        return 3
    if 'rape' in n:
        return 4
    if 'grass' in n or 'meadow' in n or 'pasture' in n or 'lawn' in n:
        return 5
    if ('fallow' in n or 'urban' in n or 'water' in n or 'forest' in n
            or 'road' in n or 'wood' in n or 'tree' in n):
        return 0
    return 6


EUROCROPS_NRW_URL = (
    'https://zenodo.org/records/8229128/files/DE_NRW_2021.zip')
EUROCROPS_NRW_MD5 = 'a5af4e520cc433b9014cf8389c8f4c1f'


def ensure_eurocrops(cache_dir):
    """Download just DE_NRW_2021.zip if missing; return its path.

    Note: deliberately not using ``torchgeo.datasets.EuroCrops(download=True)``,
    which would pull ALL 16 country zips (~GB total) due to its integrity check
    over the full zenodo_files list.
    """
    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, 'DE_NRW_2021.zip')
    if not os.path.exists(zip_path):
        print(f'downloading EuroCrops DE_NRW_2021 to {zip_path}', flush=True)
        download_url(EUROCROPS_NRW_URL, cache_dir, md5=EUROCROPS_NRW_MD5)
    return zip_path


def rasterize_eurocrops(zip_path, utm_crs, transform, width, height):
    """Read DE_NRW shapefile, clip to AOI, rasterize to 8-class mask."""
    with zipfile.ZipFile(zip_path) as zf:
        shp_name = next(n for n in zf.namelist() if n.endswith('.shp'))
    gdf = gpd.read_file(f'zip://{zip_path}!{shp_name}')
    gdf = gdf.to_crs(utm_crs.to_string())
    minx = transform.c
    maxy = transform.f
    maxx = minx + width * transform.a
    miny = maxy + height * transform.e  # negative
    gdf = gdf.cx[minx:maxx, miny:maxy]
    if len(gdf) == 0:
        raise RuntimeError('No EuroCrops polygons intersect the AOI')
    name_col = next(
        (c for c in ('EC_hcat_n', 'EC_hcat_N', 'ec_hcat_n', 'hcat_name')
         if c in gdf.columns), None)
    if name_col is None:
        raise RuntimeError(
            f'No EC_hcat_n column; columns are {list(gdf.columns)}')
    gdf['common'] = gdf[name_col].apply(hcat_to_common)
    shapes = ((geom, int(c))
              for geom, c in zip(gdf.geometry, gdf['common'])
              if geom is not None and not geom.is_empty)
    mask = rasterize(shapes=shapes, out_shape=(height, width),
                     transform=transform, fill=0, dtype='int32')
    print(f'rasterized {len(gdf)} EuroCrops polygons', flush=True)
    return mask.astype(np.int64)


# --- S2 streaming from Planetary Computer ------------------------------------


def define_target_grid(center_lat, center_lon, size_m, res_m=10.0):
    zone = int((center_lon + 180) // 6) + 1
    epsg = 32600 + zone if center_lat >= 0 else 32700 + zone
    utm_crs = CRS.from_epsg(epsg)
    cx, cy = Transformer.from_crs('EPSG:4326', utm_crs, always_xy=True
                                  ).transform(center_lon, center_lat)
    half = size_m / 2.0
    bounds = (cx - half, cy - half, cx + half, cy + half)
    width = int(round((bounds[2] - bounds[0]) / res_m))
    height = int(round((bounds[3] - bounds[1]) / res_m))
    transform = rasterio.transform.from_origin(bounds[0], bounds[3],
                                               res_m, res_m)
    return utm_crs, transform, width, height, bounds


def query_s2_timeseries(center_lat, center_lon, size_m, start, end,
                        max_cloud=20.0, max_t=38):
    """Query Planetary Computer STAC and stream PASTIS bands to a tensor."""
    utm_crs, transform, width, height, bounds_utm = define_target_grid(
        center_lat, center_lon, size_m)
    lon_min, lat_min, lon_max, lat_max = transform_bounds(
        utm_crs, 'EPSG:4326', *bounds_utm, densify_pts=21)
    catalog = pystac_client.Client.open(
        'https://planetarycomputer.microsoft.com/api/stac/v1',
        modifier=planetary_computer.sign_inplace)
    items = list(catalog.search(
        collections=['sentinel-2-l2a'],
        bbox=[lon_min, lat_min, lon_max, lat_max],
        datetime=f'{start}/{end}',
        query={'eo:cloud_cover': {'lt': max_cloud}}).items())
    if not items:
        raise RuntimeError('No Sentinel-2 scenes matched the search')

    # One item per acquisition date (lowest cloud wins); even sub-sample.
    by_date = {}
    for it in items:
        d, cc = it.datetime.date(), it.properties.get('eo:cloud_cover', 100.0)
        if d not in by_date or cc < by_date[d][0]:
            by_date[d] = (cc, it)
    items = [v[1] for _, v in sorted(by_date.items())]
    if len(items) > max_t:
        idx = np.linspace(0, len(items) - 1, max_t).round().astype(int)
        items = [items[i] for i in idx]
    print(f'using {len(items)} S2 scenes between {start} and {end}')

    T = len(items)
    stack = np.zeros((T, len(PASTIS_BANDS), height, width), dtype=np.float32)
    dates = [str(item.datetime.date()) for item in items]

    def read_band(ti, ci, item, band):
        with rasterio.open(item.assets[band].href) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=stack[ti, ci],
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs=utm_crs.to_string(),
                dst_nodata=0, resampling=Resampling.bilinear)

    # Thread the COG reads -- network-bound and GDAL releases the GIL during I/O.
    with ThreadPoolExecutor(max_workers=10) as pool:
        for ti, item in enumerate(items):
            futures = [pool.submit(read_band, ti, ci, item, band)
                       for ci, band in enumerate(PASTIS_BANDS)]
            for fut in futures:
                fut.result()  # propagate exceptions
            cc = item.properties.get('eo:cloud_cover', float('nan'))
            print(f'  t={ti + 1:>2d}/{T}  {item.datetime.date()}  '
                  f'cloud={cc:.1f}%')
    return torch.from_numpy(stack), utm_crs, bounds_utm, dates


# --- inference + metrics -----------------------------------------------------


@torch.no_grad()
def infer_tiles(task, time_series, tile=128):
    T, _, H, W = time_series.shape
    pred = np.zeros((H, W), dtype=np.int64)
    device = next(task.parameters()).device
    lengths = torch.tensor([T], device=device, dtype=torch.long)
    for i0 in range(0, H, tile):
        for j0 in range(0, W, tile):
            i1, j1 = min(i0 + tile, H), min(j0 + tile, W)
            chip = (time_series[:, :, i0:i1, j0:j1].unsqueeze(0).to(device)
                    / 10000.0)
            logits = task(chip, lengths=lengths)
            pred[i0:i1, j0:j1] = logits.argmax(dim=1)[0].cpu().numpy()
            print(f'  tile [{i0}:{i1}, {j0}:{j1}] done')
    return pred


def per_class_metrics(pred_common, label_common, ignore_index):
    valid = label_common != ignore_index
    p, y = pred_common[valid], label_common[valid]
    acc = float((p == y).mean()) if y.size else float('nan')
    per_class = {}
    for c, name in enumerate(COMMON_CLASSES):
        if c == ignore_index:
            continue
        tp = float(((p == c) & (y == c)).sum())
        fp = float(((p == c) & (y != c)).sum())
        fn = float(((p != c) & (y == c)).sum())
        iou = tp / (tp + fp + fn) if tp + fp + fn else float('nan')
        prec = tp / (tp + fp) if tp + fp else float('nan')
        rec = tp / (tp + fn) if tp + fn else float('nan')
        per_class[name] = {'iou': iou, 'precision': prec,
                           'recall': rec, 'support': int(tp + fn)}
    return acc, per_class


def save_panel(time_series, pred_common, label_common, acc, dates, out_path,
               suptitle=None):
    """4-panel figure: RGB at first two timesteps, prediction, EuroCrops label."""

    def rgb_at(ti):
        img = time_series[ti, [2, 1, 0]].numpy()
        return np.clip(img.transpose(1, 2, 0) / 3000.0, 0, 1)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5.5))
    ax[0].imshow(rgb_at(0))
    ax[1].imshow(rgb_at(1))
    ax[2].imshow(pred_common, cmap=COMMON_CMAP, vmin=0, vmax=7)
    ax[3].imshow(label_common, cmap=COMMON_CMAP, vmin=0, vmax=7)
    for a, t in zip(ax, (f'RGB t=0 ({dates[0]})', f'RGB t=1 ({dates[1]})',
                         'Prediction (PASTIS → common)',
                         'EuroCrops DE_NRW_2021 → common')):
        a.set_title(t, fontsize=11)
        a.set_axis_off()
    fig.legend(handles=[Patch(facecolor=COMMON_CMAP(i), edgecolor='black',
                              linewidth=0.3, label=COMMON_CLASSES[i])
                        for i in range(8)],
               loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02),
               frameon=False, fontsize=9)
    fig.suptitle(suptitle or f'PASTIS→NRW | pixel acc = {acc:.3f}', fontsize=13)
    fig.savefig(out_path, bbox_inches='tight', dpi=120)
    plt.close(fig)


# --- main --------------------------------------------------------------------


# AOIs spanning NRW agricultural belts (all in UTM 32N, all within
# EuroCrops DE_NRW_2021).
AOIS = (
    ('central',     51.90, 7.60),  # Münsterland (mixed farming/livestock)
    ('east',        52.00, 7.85),  # Warendorf area
    ('west',        51.80, 7.30),  # Coesfeld area
    ('niederrhein', 51.55, 6.45),  # Lower Rhine (vegetables, cereals)
    ('soest',       51.60, 8.05),  # Soester Börde (wheat, sugar beet)
    ('rhineland',   50.80, 6.45),  # Düren / Cologne basin
)
AOI_SIZE_M = 10240


def run_aoi(task, eurocrops_zip, name, center_lat, center_lon, size_m,
            out_dir):
    print(f'\n=== AOI "{name}" @ ({center_lat}, {center_lon}) ===', flush=True)
    log_mem('before s2 fetch')
    time_series, utm_crs, bounds, dates = query_s2_timeseries(
        center_lat, center_lon, size_m,
        start='2020-09-01', end='2021-11-30', max_cloud=20.0, max_t=38)
    log_mem(f's2 ready, shape={tuple(time_series.shape)}')
    pred_common = PASTIS_TO_COMMON[infer_tiles(task, time_series, tile=128)]
    log_mem('inference done')
    # Keep only the first two frames for the RGB viz; drop the rest of the
    # ~1.5 GB CPU tensor so it doesn't compete with EuroCrops rasterization.
    rgb_frames = time_series[:2].clone()
    del time_series

    minx, miny, maxx, maxy = bounds
    width = int(round((maxx - minx) / 10))
    height = int(round((maxy - miny) / 10))
    transform = rasterio.transform.from_origin(minx, maxy, 10, 10)
    label_common = rasterize_eurocrops(eurocrops_zip, utm_crs, transform,
                                       width, height)
    log_mem('eurocrops rasterized')
    acc, per_class = per_class_metrics(pred_common, label_common,
                                       COMMON_IGNORE_INDEX)
    print(f'AOI "{name}": acc = {acc:.4f}')
    for n, m in per_class.items():
        print(f'  {n:>11s}  IoU={m["iou"]:.3f}  support={m["support"]}')
    save_panel(rgb_frames, pred_common, label_common, acc, dates,
               os.path.join(out_dir, f'pastis_to_germany_{name}.png'),
               suptitle=(f'PASTIS→NRW "{name}" @ ({center_lat}, {center_lon}) '
                         f'| acc = {acc:.3f}'))
    return acc, per_class, dates


def main():
    out = '/u/yichia3/torchgeo/outputs_germany'
    os.makedirs(out, exist_ok=True)
    ckpt = sorted(glob.glob(
        '/u/yichia3/torchgeo/outputs_full/lightning_logs/'
        'version_*/checkpoints/*.ckpt'),
        key=lambda p: int(p.split('version_')[1].split(os.sep)[0]))[-1]
    print(f'loading checkpoint {ckpt}')
    task = SpatioTemporalSegmentationTask.load_from_checkpoint(
        ckpt, map_location='cpu').eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task.to(device)

    ec_root = os.environ.get(
        'EUROCROPS_ROOT', '/projects/illinois/eng/cs/arindamb/yichia3/eurocrops')
    eurocrops_zip = ensure_eurocrops(ec_root)

    summary = {}
    for name, lat, lon in AOIS:
        try:
            summary[name] = {
                'center': (lat, lon),
                **dict(zip(('acc', 'per_class', 'dates'),
                           run_aoi(task, eurocrops_zip, name, lat, lon,
                                   AOI_SIZE_M, out))),
            }
        except Exception as e:
            print(f'AOI "{name}" failed: {e}')

    with open(os.path.join(out, 'metrics.txt'), 'w') as f:
        for name, s in summary.items():
            f.write(f'== AOI "{name}" @ {s["center"]} '
                    f'({AOI_SIZE_M} m square, T={len(s["dates"])}) ==\n')
            f.write(f'dates: {", ".join(s["dates"])}\n')
            f.write(f'overall pixel accuracy: {s["acc"]:.4f}\n')
            for n, m in s['per_class'].items():
                f.write(f'  {n:>11s}  IoU={m["iou"]:.4f}  '
                        f'prec={m["precision"]:.4f}  rec={m["recall"]:.4f}  '
                        f'support={m["support"]}\n')
            f.write('\n')
    print(f'\nsaved {len(summary)} figures + metrics.txt to {out}')


if __name__ == '__main__':
    main()
