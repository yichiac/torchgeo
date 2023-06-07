import torch
import glob
import os


paths = glob.glob(os.path.join("/projects/dali/data/pretrained_weights/*/*/", "checkpoint-epoch=199.ckpt"), recursive=True)
for path in paths:
    ckpt = torch.load(path)

    state_dict = {key.replace("backbone.", "model.backbone.model."): val for key, val in ckpt["state_dict"].items() if key.startswith("backbone.")}
    state_to_save = {
        "state_dict": state_dict,
        "hyper_parameters": {
            "backbone": os.path.dirname(path).split(os.sep)[0].split("-")[-1]
        }
    }
    save_path = os.path.join(os.getcwd(), os.path.dirname(path), "backbone.ckpt")
    torch.save(state_to_save, os.path.join(os.path.dirname(path), path.split(os.sep)[-3] + "_backbone.ckpt"))
