#!/usr/bin/env bash

set -euo pipefail

# User-specific parameters
ROOT_DIR=/Users/yc/projects/dali/data
SAVE_PATH="$ROOT_DIR/s2_india_fields"
MATCH_FILE="$ROOT_DIR/s2_india_fields/centroids.csv"
NUM_WORKERS=10
START_INDEX=0
END_INDEX=10000

# Satellite-specific parameters
COLLECTION=COPERNICUS/S2 # S2_SR
META_CLOUD_NAME=CLOUDY_PIXEL_PERCENTAGE
YEAR=2020
BANDS=(B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12)
# RES=30
# QA_BAND=QA_PIXEL
# ORIGINAL_RESOLUTIONS=(60 10 10 10 20 20 20 10 20 60 60 20 20)
# NEW_RESOLUTIONS=${ORIGINAL_RESOLUTIONS[@]}

# Generic parameters
SCRIPT_DIR=/Users/yc/Documents/GitHub/torchgeo/experiments/ssl4eo
CLOUD_PCT=20
SIZE=264
DTYPE=uint16
LOG_FREQ=100

time python3 "$SCRIPT_DIR/download_ssl4eo.py" \
    --save-path "$SAVE_PATH" \
    --collection $COLLECTION \
    --meta-cloud-name $META_CLOUD_NAME \
    --cloud-pct $CLOUD_PCT \
    --dates $YEAR-03-20 $YEAR-06-21 $YEAR-09-23 $YEAR-12-21 \
    --bands ${BANDS[@]} \
    --radius 1320 \
    --dtype $DTYPE \
    --num-workers $NUM_WORKERS \
    --log-freq $LOG_FREQ \
    --match-file "$MATCH_FILE" \
    --indices-range $START_INDEX $END_INDEX \
    --debug
~

# --qa-band $QA_BAND \
# --qa-cloud-bit $QA_CLOUD_BIT \
# --original-resolutions ${ORIGINAL_RESOLUTIONS[@]} \
# --new-resolutions $NEW_RESOLUTIONS \

