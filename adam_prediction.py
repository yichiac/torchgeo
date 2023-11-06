mport matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch

from torchgeo.datamodules import L7IrishDataModule
from torchgeo.datasets import unbind_samples

device = torch.device("cpu")

# Load weights
path = "/scratch.local/yichia3/output/l7irish_output/l7irish_unet_resnet18_0.0003_ce_moco/checkpoint-epoch=26-val_loss=0.68.ckpt"
state_dict = torch.load(path, map_location=device)["state_dict"]
state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}

# Initialize model
model = smp.Unet(encoder_name="resnet18", in_channels=9, classes=5)
model.to(device)
model.load_state_dict(state_dict)

# Initialize data loaders
datamodule = L7IrishDataModule(
    root="data/l7irish", crs="epsg:3857", download=True, batch_size=64, patch_size=224
)
datamodule.setup("test")

for batch in datamodule.test_dataloader():
    image = batch["image"]
    mask = batch["mask"]
    image.to(device)

    # Make a prediction
    prediction = model(image)
    prediction = prediction.argmax(dim=1)
    prediction.detach().to("cpu")

    batch["prediction"] = prediction

    for sample in unbind_samples(batch):
        # Skip nodata pixels
        if 0 in sample["mask"]:
            continue

        # Skip boring images
        if len(sample["mask"].unique()) < 3:
            continue

        # Plot
        datamodule.test_dataset.plot(sample)
        plt.show()
