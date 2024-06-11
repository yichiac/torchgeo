import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch

# from torchgeo.datamodules import Sentinel2CDLDataModule
# from torchgeo.datamodules import Sentinel2NCCMDataModule
# from torchgeo.datamodules import Sentinel2RasterizedEuroCropsDataModule
from torchgeo.datamodules import AgriFieldNetMaskDataModule
# from torchgeo.datamodules import Sentinel2SouthAmericaSoybeanDataModule
from torchgeo.datasets import unbind_samples
import time

device = torch.device("cpu")

# path = "/Users/yc/Datasets/ID/cdl/checkpoints/epoch=99-step=700.ckpt"
# path = "/Users/yc/Datasets/ID/nccm/checkpoints/epoch=99-step=700.ckpt"
# path = "/Users/yc/Datasets/ID/eurocrops/checkpoints/epoch=99-step=700.ckpt"
path = "/Users/yc/Datasets/ID/agrifieldnet/checkpoints/epoch=99-step=700.ckpt"
# path = "/Users/yc/Datasets/ID/sas/checkpoints/epoch=99-step=700.ckpt"
# path = "/Users/yc/Datasets/ID/sact/checkpoints/epoch=99-step=700.ckpt"

state_dict = torch.load(path, map_location=device)["state_dict"]
state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}


model = smp.Unet(encoder_name="resnet50", in_channels=13, classes=6)
model.to(device)
model.load_state_dict(state_dict, strict=True)

# datamodule = Sentinel2CDLDataModule(
#     crs="epsg:3857",
#     batch_size=128,
#     patch_size=256,
#     cdl_paths="/Users/yc/Datasets/cdl_harmonized_block",
#     sentinel2_paths="/Users/yc/Datasets/sentinel2_subsample_1000/sentinel2_cdl_2023_subsampled",
# )

# datamodule = Sentinel2NCCMDataModule(
#     crs="epsg:3857",
#     batch_size=128,
#     patch_size=256,
#     nccm_paths="/Users/yc/Datasets/nccm_harmonized_block",
#     sentinel2_paths="/Users/yc/Datasets/sentinel2_subsample_1000/sentinel2_nccm_2019_subsampled",
# )

# datamodule = Sentinel2RasterizedEuroCropsDataModule(
    # crs="epsg:3857",
    # batch_size=128,
    # patch_size=256,
    # eurocrops_paths="/Users/yc/Datasets/eurocrops_cropped_subsampled",
    # sentinel2_paths="/Users/yc/Datasets/sentinel2_subsample_1000/sentinel2_eurocrops_subsampled",
# )

datamodule = AgriFieldNetMaskDataModule(
    crs="epsg:3857",
    batch_size=128,
    patch_size=256,
    agrifieldnet_paths="/Users/yc/Datasets/agrifieldnet_harmonized",
    sentinel2_paths="/Users/yc/Datasets/sentinel2_subsample_1000/sentinel2_agrifieldnet_2021_subsampled",
)
# datamodule = Sentinel2SouthAmericaSoybeanDataModule(
#     crs="epsg:3857",
#     batch_size=128,
#     patch_size=256,
#     south_america_soybean_paths="/Users/yc/Datasets/sas_harmonized",
#     sentinel2_paths="/Users/yc/Datasets/sentinel2_subsample_1000/sentinel2_sas_2021_subsampled",
# )

datamodule.setup("test")

for batch in datamodule.test_dataloader():
    image = batch["image"]
    mask = batch["mask"]
    image.to(device)

    # Make a prediction
    start_time = time.time()
    print("start prediction")

    prediction = model(image)
    prediction = prediction.argmax(dim=1)
    prediction.detach().to("cpu")

    batch["prediction"] = prediction

    print("Finish prediction in {} seconds".format(time.time() - start_time))

    count = 0
    for sample in unbind_samples(batch):
        # Skip nodata pixels
        if 0 in sample["mask"]:
            continue

        # Skip boring images
        # if len(sample["mask"].unique()) < 3:
        #     continue

        # Plot
        datamodule.plot(sample)
        plt.savefig(f"results/sas_{count}_img.png")
        # plt.show()
        plt.close()
        count += 1
