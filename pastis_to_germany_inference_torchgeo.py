# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Cross-domain inference: PASTIS ConvLSTM applied to NRW Germany.

Uses ``Sentinel2(time_series=True) & EuroCropsCommon`` + a single bbox index
to pull one ``(image, mask)`` sample at the full AOI scale, then runs the
model once. The Planetary-Computer download helper writes COGs in
Sentinel-2 naming convention to a local cache so torchgeo's Sentinel2 can
read them. See ``implementation-notes.html``.
"""

import glob
import os

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
from rasterio.warp import Resampling, reproject

from torchgeo.datasets import EuroCrops, Sentinel2
from torchgeo.trainers import SpatioTemporalSegmentationTask

BANDS = ('B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12')
COMMON = ('non_crop', 'wheat', 'barley', 'corn', 'rapeseed', 'grassland',
          'other_crop', 'void')
CMAP = ListedColormap(['#555', '#ff7f0e', '#2ca02c', '#ffbb78',
                       '#8c564b', '#aec7e8', '#9467bd', '#fff'])
P2C = np.array([0, 5, 1, 3, 2, 4, 2, 6, 6, 6, 1, 1, 6, 6, 5, 6, 6, 1, 3, 7])


def hcat2common(name: str) -> int:
    if not isinstance(name, str):
        return 0
    n = name.lower()
    rules = [(('wheat', 'triticale'), 1), (('barley',), 2),
             (('maize', 'corn'), 3), (('rape',), 4),
             (('grass', 'meadow', 'pasture'), 5),
             (('fallow', 'urban', 'water', 'forest', 'wood', 'tree', 'road'), 0)]
    for kws, c in rules:
        if any(k in n for k in kws):
            return c
    return 6


class EuroCropsCommon(EuroCrops):
    def _load_class_map(self, classes):
        self.class_map = {n: i for i, n in enumerate(COMMON)}

    def get_label(self, feature):
        for col in ('EC_hcat_n', 'EC_hcat_N', 'ec_hcat_n', 'hcat_name'):
            if col in feature.index:
                return hcat2common(feature[col])
        return 0


def cache_s2(cache_dir, bounds, crs, start='2020-09-01', end='2021-11-30',
             max_cloud=20.0, max_t=38, res=10.0):
    os.makedirs(cache_dir, exist_ok=True)
    minx, miny, maxx, maxy = bounds
    W, H = int(round((maxx - minx) / res)), int(round((maxy - miny) / res))
    T = rasterio.transform.from_origin(minx, maxy, res, res)
    tll = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    bb_ll = (*tll.transform(minx, miny), *tll.transform(maxx, maxy))
    cat = pystac_client.Client.open(
        'https://planetarycomputer.microsoft.com/api/stac/v1',
        modifier=planetary_computer.sign_inplace)
    items = list(cat.search(
        collections=['sentinel-2-l2a'], bbox=bb_ll, datetime=f'{start}/{end}',
        query={'eo:cloud_cover': {'lt': max_cloud}}).items())
    # one item per date (lowest cloud wins) + even sub-sample to max_t
    by_date = {}
    for it in items:
        d, cc = it.datetime.date(), it.properties.get('eo:cloud_cover', 100)
        if d not in by_date or cc < by_date[d][0]:
            by_date[d] = (cc, it)
    items = [v[1] for _, v in sorted(by_date.items())]
    if len(items) > max_t:
        idx = np.linspace(0, len(items) - 1, max_t).round().astype(int)
        items = [items[i] for i in idx]
    print(f'caching {len(items)} S2 scenes')
    for i, it in enumerate(items):
        tile = it.properties.get('s2:mgrs_tile', '00ZZZ')
        ts = it.datetime.strftime('%Y%m%dT%H%M%S')
        for b in BANDS:
            p = os.path.join(cache_dir, f'T{tile}_{ts}_{b}.tif')
            if os.path.exists(p):
                continue
            with rasterio.open(it.assets[b].href) as src:
                dst = np.zeros((H, W), dtype=np.uint16)
                reproject(rasterio.band(src, 1), dst,
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=T, dst_crs=crs, dst_nodata=0,
                          resampling=Resampling.bilinear)
                with rasterio.open(p, 'w', driver='GTiff', height=H, width=W,
                                   count=1, dtype='uint16', crs=crs,
                                   transform=T, compress='deflate') as out:
                    out.write(dst, 1)
        print(f'  {i + 1:>2d}/{len(items)} {ts[:8]}')


def main():
    out = '/u/yichia3/torchgeo/outputs_germany_tg'
    os.makedirs(out, exist_ok=True)
    ckpt = sorted(glob.glob(
        '/u/yichia3/torchgeo/outputs_full/lightning_logs/'
        'version_*/checkpoints/*.ckpt'),
        key=lambda p: int(p.split('version_')[1].split(os.sep)[0]))[-1]
    print(f'ckpt: {ckpt}')
    task = SpatioTemporalSegmentationTask.load_from_checkpoint(
        ckpt, map_location='cpu').eval()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task.to(dev)

    # AOI: SW NRW near Düren, 2.56 km square @ 10 m
    crs = CRS.from_epsg(32632)
    cx, cy = Transformer.from_crs('EPSG:4326', crs, always_xy=True
                                  ).transform(6.45, 50.80)
    bounds = (cx - 1280, cy - 1280, cx + 1280, cy + 1280)
    cache_s2(os.path.join(out, 's2_cache'), bounds, crs)

    ec_root = os.environ.get('EUROCROPS_ROOT',
                             '/projects/illinois/eng/cs/arindamb/yichia3/eurocrops')
    s2 = Sentinel2(paths=os.path.join(out, 's2_cache'), crs=crs, res=10,
                   bands=BANDS, time_series=True)
    ec = EuroCropsCommon(paths=ec_root, crs=crs, res=10,
                         download=not os.path.isdir(
                             os.path.join(ec_root, 'DE_NRW_2021')))
    sample = (s2 & ec)[bounds[0]:bounds[2], bounds[1]:bounds[3]]

    image = sample['image'].unsqueeze(0).to(dev, torch.float32) / 10000.0
    T = image.shape[1]
    with torch.no_grad():
        logits = task(image, lengths=torch.tensor([T], device=dev))
    pred_c = P2C[logits.argmax(1)[0].cpu().numpy()]
    label = sample['mask'].squeeze().cpu().numpy().astype(np.int64)

    valid = label != 7
    acc = float((pred_c[valid] == label[valid]).mean())
    print(f'\npixel acc = {acc:.3f}  (T={T})')
    for c, n in enumerate(COMMON[:-1]):
        tp = float(((pred_c == c) & (label == c)).sum())
        fp = float(((pred_c == c) & (label != c) & valid).sum())
        fn = float(((pred_c != c) & (label == c)).sum())
        iou = tp / (tp + fp + fn) if tp + fp + fn else float('nan')
        print(f'  {n:>10s}  IoU={iou:.3f}  support={int(tp + fn)}')

    rgb = torch.median(image[0, :, [2, 1, 0]], 0).values.cpu().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(np.clip(rgb.transpose(1, 2, 0) / 0.3, 0, 1))
    ax[1].imshow(pred_c, cmap=CMAP, vmin=0, vmax=7)
    ax[2].imshow(label, cmap=CMAP, vmin=0, vmax=7)
    for a, t in zip(ax, ('Median RGB', 'Prediction', 'EuroCrops')):
        a.set_title(t)
        a.set_axis_off()
    fig.legend(handles=[Patch(facecolor=CMAP(i), edgecolor='k', label=COMMON[i])
                        for i in range(8)],
               loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02),
               frameon=False)
    fig.suptitle(f'PASTIS→NRW (torchgeo) | acc = {acc:.3f}')
    fig.savefig(os.path.join(out, 'pastis_to_germany.png'),
                bbox_inches='tight', dpi=120)
    print(f'saved {out}/pastis_to_germany.png')


if __name__ == '__main__':
    main()
