from torchgeo.datasets import Sentinel2
from torchgeo.datasets import CDL

Sentinel2(paths="/projects/dali/data/sentinel2/")
CDL(paths="/projects/dali/data/cdl/", download=True, years=[2021,2022,2023])


from torchgeo.datamodules import CDLSentinel2DataModule

datamodule = CDLSentinel2DataModule(batch_size=64, patch_size=16)
task = SemanticSegmentationTask(
    loss="ce",
    model="unet",
    backbone="resnet50",
    weights=True,
    in_channels=13,
    num_classes=134,
    ignore_index=0,
    lr=0.1,
    patience=6,
)

trainer = Trainer(default_root_dir="...")
trainer.fit(model=task, datamodule=datamodule)
