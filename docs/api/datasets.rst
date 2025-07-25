torchgeo.datasets
=================

.. module:: torchgeo.datasets

In :mod:`torchgeo`, we define two types of datasets: :ref:`Geospatial Datasets` and :ref:`Non-geospatial Datasets`. These abstract base classes are documented in more detail in :ref:`Base Classes`.

.. _Geospatial Datasets:

Geospatial Datasets
-------------------

:class:`GeoDataset` is designed for datasets that contain geospatial information, like latitude, longitude, coordinate system, and projection. Datasets containing this kind of information can be combined using :class:`IntersectionDataset` and :class:`UnionDataset`.

.. csv-table::
   :widths: 30 15 20 36 20 15
   :header-rows: 1
   :align: center
   :file: datasets/geo_datasets.csv

Aboveground Woody Biomass
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: AbovegroundLiveWoodyBiomassDensity

AgriFieldNet
^^^^^^^^^^^^

.. autoclass:: AgriFieldNet

Airphen
^^^^^^^

.. autoclass:: Airphen

Aster Global DEM
^^^^^^^^^^^^^^^^

.. autoclass:: AsterGDEM

Canadian Building Footprints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: CanadianBuildingFootprints

Chesapeake Land Cover
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Chesapeake
.. autoclass:: ChesapeakeDC
.. autoclass:: ChesapeakeDE
.. autoclass:: ChesapeakeMD
.. autoclass:: ChesapeakeNY
.. autoclass:: ChesapeakePA
.. autoclass:: ChesapeakeVA
.. autoclass:: ChesapeakeWV
.. autoclass:: ChesapeakeCVPR

GlobalBuildingMap
^^^^^^^^^^^^^^^^^

.. autoclass:: GlobalBuildingMap

Global Mangrove Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: CMSGlobalMangroveCanopy

Cropland Data Layer
^^^^^^^^^^^^^^^^^^^

.. autoclass:: CDL

EDDMapS
^^^^^^^

.. autoclass:: EDDMapS

EnMAP
^^^^^

.. autoclass:: EnMAP

EnviroAtlas
^^^^^^^^^^^

.. autoclass:: EnviroAtlas

Esri2020
^^^^^^^^

.. autoclass:: Esri2020

EU-DEM
^^^^^^

.. autoclass:: EUDEM

EuroCrops
^^^^^^^^^

.. autoclass:: EuroCrops

GBIF
^^^^

.. autoclass:: GBIF

GlobBiomass
^^^^^^^^^^^

.. autoclass:: GlobBiomass

iNaturalist
^^^^^^^^^^^

.. autoclass:: INaturalist

I/O Bench
^^^^^^^^^

.. autoclass:: IOBench

L7 Irish
^^^^^^^^

.. autoclass:: L7Irish

L8 Biome
^^^^^^^^

.. autoclass:: L8Biome

LandCover.ai Geo
^^^^^^^^^^^^^^^^

.. autoclass:: LandCoverAIBase
.. autoclass:: LandCoverAIGeo

Landsat
^^^^^^^

.. autoclass:: Landsat
.. autoclass:: Landsat9
.. autoclass:: Landsat8
.. autoclass:: Landsat7
.. autoclass:: Landsat5TM
.. autoclass:: Landsat5MSS
.. autoclass:: Landsat4TM
.. autoclass:: Landsat4MSS
.. autoclass:: Landsat3
.. autoclass:: Landsat2
.. autoclass:: Landsat1

MMFlood
^^^^^^^
.. autoclass:: MMFlood

NAIP
^^^^

.. autoclass:: NAIP

NCCM
^^^^

.. autoclass:: NCCM

NLCD
^^^^

.. autoclass:: NLCD

Open Buildings
^^^^^^^^^^^^^^

.. autoclass:: OpenBuildings

PRISMA
^^^^^^

.. autoclass:: PRISMA

Sentinel
^^^^^^^^

.. autoclass:: Sentinel
.. autoclass:: Sentinel1
.. autoclass:: Sentinel2

South Africa Crop Type
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: SouthAfricaCropType

South America Soybean
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: SouthAmericaSoybean

.. _Non-geospatial Datasets:

Non-geospatial Datasets
-----------------------

:class:`NonGeoDataset` is designed for datasets that lack geospatial information. These datasets can still be combined using :class:`ConcatDataset <torch.utils.data.ConcatDataset>`.

.. csv-table:: C = classification,  R = regression, S = semantic segmentation, I = instance segmentation, T = time series, CD = change detection, OD = object detection, IC = image captioning
   :widths: 15 7 15 20 12 11 12 15 13
   :header-rows: 1
   :align: center
   :file: datasets/non_geo_datasets.csv

ADVANCE
^^^^^^^

.. autoclass:: ADVANCE

Benin Cashew Plantations
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: BeninSmallHolderCashews

BigEarthNet
^^^^^^^^^^^

.. autoclass:: BigEarthNet
.. autoclass:: BigEarthNetV2

BioMassters
^^^^^^^^^^^

.. autoclass:: BioMassters

BRIGHT
^^^^^^

.. autoclass:: BRIGHTDFC2025

CaBuAr
^^^^^^

.. autoclass:: CaBuAr

CaFFe
^^^^^

.. autoclass:: CaFFe

ChaBuD
^^^^^^

.. autoclass:: ChaBuD

Cloud Cover Detection
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: CloudCoverDetection

Copernicus-Pretrain
^^^^^^^^^^^^^^^^^^^

.. autoclass:: CopernicusPretrain

COWC
^^^^

.. autoclass:: COWC
.. autoclass:: COWCCounting
.. autoclass:: COWCDetection

CropHarvest
^^^^^^^^^^^

.. autoclass:: CropHarvest

Kenya Crop Type
^^^^^^^^^^^^^^^

.. autoclass:: CV4AKenyaCropType

DeepGlobe Land Cover
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: DeepGlobeLandCover

DFC2022
^^^^^^^

.. autoclass:: DFC2022

DIOR
^^^^

.. autoclass:: DIOR


Digital Typhoon
^^^^^^^^^^^^^^^

.. autoclass:: DigitalTyphoon

DL4GAM
^^^^^^
.. autoclass:: DL4GAMAlps

DOTA
^^^^
.. autoclass:: DOTA

ETCI2021 Flood Detection
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ETCI2021

EuroSAT
^^^^^^^

.. autoclass:: EuroSAT
.. autoclass:: EuroSATSpatial
.. autoclass:: EuroSAT100

EverWatch
^^^^^^^^^

.. autoclass:: EverWatch

FAIR1M
^^^^^^

.. autoclass:: FAIR1M

Fields Of The World
^^^^^^^^^^^^^^^^^^^

.. autoclass:: FieldsOfTheWorld

FireRisk
^^^^^^^^

.. autoclass:: FireRisk

Forest Damage
^^^^^^^^^^^^^

.. autoclass:: ForestDamage

GeoNRW
^^^^^^^

.. autoclass:: GeoNRW

GID-15
^^^^^^

.. autoclass:: GID15

HySpecNet-11k
^^^^^^^^^^^^^

.. autoclass:: HySpecNet11k

IDTReeS
^^^^^^^

.. autoclass:: IDTReeS

Inria Aerial Image Labeling
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: InriaAerialImageLabeling

LandCover.ai
^^^^^^^^^^^^

.. autoclass:: LandCoverAI
.. autoclass:: LandCoverAI100

LEVIR-CD
^^^^^^^^

.. autoclass:: LEVIRCDBase
.. autoclass:: LEVIRCD

LEVIR-CD+
^^^^^^^^^

.. autoclass:: LEVIRCDPlus

LoveDA
^^^^^^

.. autoclass:: LoveDA

MapInWild
^^^^^^^^^

.. autoclass:: MapInWild

MDAS
^^^^

.. autoclass:: MDAS

Million-AID
^^^^^^^^^^^

.. autoclass:: MillionAID

MMEarth
^^^^^^^^

.. autoclass:: MMEarth

NASA Marine Debris
^^^^^^^^^^^^^^^^^^

.. autoclass:: NASAMarineDebris

OSCD
^^^^

.. autoclass:: OSCD

PASTIS
^^^^^^

.. autoclass:: PASTIS

PatternNet
^^^^^^^^^^

.. autoclass:: PatternNet

Potsdam
^^^^^^^

.. autoclass:: Potsdam2D

QuakeSet
^^^^^^^^

.. autoclass:: QuakeSet

ReforesTree
^^^^^^^^^^^

.. autoclass:: ReforesTree

RESISC45
^^^^^^^^

.. autoclass:: RESISC45

Rwanda Field Boundary
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: RwandaFieldBoundary

SatlasPretrain
^^^^^^^^^^^^^^

.. autoclass:: SatlasPretrain

Seasonal Contrast
^^^^^^^^^^^^^^^^^

.. autoclass:: SeasonalContrastS2

SeasoNet
^^^^^^^^

.. autoclass:: SeasoNet

SEN12MS
^^^^^^^

.. autoclass:: SEN12MS

SKIPP'D
^^^^^^^

.. autoclass:: SKIPPD

SkyScript
^^^^^^^^^

.. autoclass:: SkyScript

So2Sat
^^^^^^

.. autoclass:: So2Sat

Solar Plants Brazil
^^^^^^^^^^^^^^^^^^^

.. autoclass:: SolarPlantsBrazil

SODA
^^^^

.. autoclass:: SODAA

SSL4EO
^^^^^^

.. autoclass:: SSL4EO
.. autoclass:: SSL4EOL
.. autoclass:: SSL4EOS12

SSL4EO-L Benchmark
^^^^^^^^^^^^^^^^^^

.. autoclass:: SSL4EOLBenchmark

Substation
^^^^^^^^^^

.. autoclass:: Substation

SustainBench Crop Yield
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: SustainBenchCropYield

TreeSatAI
^^^^^^^^^

.. autoclass:: TreeSatAI

Tropical Cyclone
^^^^^^^^^^^^^^^^

.. autoclass:: TropicalCyclone

UC Merced
^^^^^^^^^

.. autoclass:: UCMerced

USAVars
^^^^^^^

.. autoclass:: USAVars

Vaihingen
^^^^^^^^^

.. autoclass:: Vaihingen2D

VHR-10
^^^^^^

.. autoclass:: VHR10

Western USA Live Fuel Moisture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: WesternUSALiveFuelMoisture

xView2
^^^^^^

.. autoclass:: XView2

ZueriCrop
^^^^^^^^^

.. autoclass:: ZueriCrop

.. _Base Classes:

Copernicus-Bench
----------------

Copernicus-Bench is a comprehensive evaluation benchmark with 15 downstream tasks hierarchically organized across preprocessing (e.g., cloud removal), base applications (e.g., land cover classification), and specialized applications (e.g., air quality estimation). This benchmark enables systematic assessment of foundation model performances across various Sentinel missions on different levels of practical applications.

.. csv-table:: C = classification,  R = regression, S = semantic segmentation, T = time series, CD = change detection
   :widths: 5 15 7 15 20 12 11 12 15 13
   :header-rows: 1
   :align: center
   :file: datasets/copernicus_bench.csv

.. autoclass:: CopernicusBench
.. autoclass:: CopernicusBenchBase
.. autoclass:: CopernicusBenchCloudS2
.. autoclass:: CopernicusBenchCloudS3
.. autoclass:: CopernicusBenchEuroSATS1
.. autoclass:: CopernicusBenchEuroSATS2
.. autoclass:: CopernicusBenchBigEarthNetS1
.. autoclass:: CopernicusBenchBigEarthNetS2
.. autoclass:: CopernicusBenchLC100ClsS3
.. autoclass:: CopernicusBenchLC100SegS3
.. autoclass:: CopernicusBenchDFC2020S1
.. autoclass:: CopernicusBenchDFC2020S2
.. autoclass:: CopernicusBenchFloodS1
.. autoclass:: CopernicusBenchLCZS2
.. autoclass:: CopernicusBenchBiomassS3
.. autoclass:: CopernicusBenchAQNO2S5P
.. autoclass:: CopernicusBenchAQO3S5P

SpaceNet
--------

The `SpaceNet Dataset <https://spacenet.ai/datasets/>`_ is hosted as an Amazon Web Services (AWS) `Public Dataset <https://registry.opendata.aws/spacenet/>`_. It contains ~67,000 square km of very high-resolution imagery, >11M building footprints, and ~20,000 km of road labels to ensure that there is adequate open source data available for geospatial machine learning research. SpaceNet Challenge Dataset's have a combination of very high resolution satellite imagery and high quality corresponding labels for foundational mapping features such as building footprints or road networks.

.. csv-table:: I = instance segmentation
   :widths: 15 7 15 20 12 11 12 15 13
   :header-rows: 1
   :align: center
   :file: datasets/spacenet.csv

.. autoclass:: SpaceNet
.. autoclass:: SpaceNet1
.. autoclass:: SpaceNet2
.. autoclass:: SpaceNet3
.. autoclass:: SpaceNet4
.. autoclass:: SpaceNet5
.. autoclass:: SpaceNet6
.. autoclass:: SpaceNet7
.. autoclass:: SpaceNet8

Base Classes
------------

If you want to write your own custom dataset, you can extend one of these abstract base classes.

GeoDataset
^^^^^^^^^^

.. autoclass:: GeoDataset

RasterDataset
^^^^^^^^^^^^^

.. autoclass:: RasterDataset

VectorDataset
^^^^^^^^^^^^^

.. autoclass:: VectorDataset

NonGeoDataset
^^^^^^^^^^^^^

.. autoclass:: NonGeoDataset

NonGeoClassificationDataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: NonGeoClassificationDataset

IntersectionDataset
^^^^^^^^^^^^^^^^^^^

.. autoclass:: IntersectionDataset

UnionDataset
^^^^^^^^^^^^

.. autoclass:: UnionDataset

Utilities
---------

Collation Functions
^^^^^^^^^^^^^^^^^^^

.. autofunction:: stack_samples
.. autofunction:: concat_samples
.. autofunction:: merge_samples
.. autofunction:: unbind_samples

Splitting Functions
^^^^^^^^^^^^^^^^^^^

.. autofunction:: random_bbox_assignment
.. autofunction:: random_bbox_splitting
.. autofunction:: random_grid_cell_assignment
.. autofunction:: roi_split
.. autofunction:: time_series_split

Errors
------

.. autoclass:: DatasetNotFoundError
.. autoclass:: DependencyNotFoundError
.. autoclass:: RGBBandsMissingError
