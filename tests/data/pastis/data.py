#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import json
import os
import shutil
from datetime import datetime, timedelta

import numpy as np

SIZE = 32
NUM_SAMPLES = 5
MAX_NUM_TIME_STEPS = 10
START_DATE = datetime(2018, 9, 24)
# Spacing between time steps, in days. Much longer than the true revisit period so
# that these short time-series still span every season, like the real dataset.
SPACING: dict[str, int] = {'S2': 45, 'S1A': 50, 'S1D': 50}
np.random.seed(0)

FILENAME_HIERARCHY = dict[str, 'FILENAME_HIERARCHY'] | list[str]

filenames: FILENAME_HIERARCHY = {
    'DATA_S2': ['S2'],
    'DATA_S1A': ['S1A'],
    'DATA_S1D': ['S1D'],
    'ANNOTATIONS': ['TARGET'],
    'INSTANCE_ANNOTATIONS': ['INSTANCES'],
}

# Number of time steps of each time-series, keyed by sensor and sample
num_time_steps: dict[str, dict[int, int]] = {sensor: {} for sensor in SPACING}


def create_file(path: str) -> None:
    for i in range(NUM_SAMPLES):
        new_path = f'{path}_{i}.npy'
        fn = os.path.basename(new_path)
        t = np.random.randint(1, MAX_NUM_TIME_STEPS)
        if fn.startswith('S2'):
            num_time_steps['S2'][i] = t
            data = np.random.randint(0, 256, size=(t, 10, SIZE, SIZE)).astype(np.int16)
        elif fn.startswith(('S1A', 'S1D')):
            num_time_steps[fn.split('_')[0]][i] = t
            data = np.random.randint(0, 256, size=(t, 3, SIZE, SIZE)).astype(np.float16)
        elif fn.startswith('TARGET'):
            data = np.random.randint(0, 20, size=(3, SIZE, SIZE)).astype(np.uint8)
        elif fn.startswith('INSTANCES'):
            data = np.random.randint(0, 100, size=(SIZE, SIZE)).astype(np.int64)
        np.save(new_path, data)


def create_directory(directory: str, hierarchy: FILENAME_HIERARCHY) -> None:
    if isinstance(hierarchy, dict):
        # Recursive case
        for key, value in hierarchy.items():
            path = os.path.join(directory, key)
            os.makedirs(path, exist_ok=True)
            create_directory(path, value)
    else:
        # Base case
        for value in hierarchy:
            path = os.path.join(directory, value)
            create_file(path)


def create_dates(sensor: str, i: int) -> dict[str, int]:
    dates = {}
    for t in range(num_time_steps[sensor][i]):
        date = START_DATE + timedelta(days=t * SPACING[sensor])
        dates[str(t)] = int(date.strftime('%Y%m%d'))
    return dates


if __name__ == '__main__':
    create_directory('PASTIS-R', filenames)

    features = []
    for i in range(NUM_SAMPLES):
        features.append(
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
                },
                'id': str(i),
                'properties': {
                    'Fold': (i % 5) + 1,
                    'ID_PATCH': i,
                    'dates-S2': create_dates('S2', i),
                    'dates-S1A': create_dates('S1A', i),
                    'dates-S1D': create_dates('S1D', i),
                },
            }
        )

    with open(os.path.join('PASTIS-R', 'metadata.geojson'), 'w') as f:
        json.dump({'type': 'FeatureCollection', 'features': features}, f)

    filename = 'PASTIS-R.zip'
    shutil.make_archive(filename.replace('.zip', ''), 'zip', '.', 'PASTIS-R')
