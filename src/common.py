from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms as T
from src import config as C

IMG_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def eval_transform():
    """Deterministic preprocessing. Same code is used at inference time (predict.py)."""
    return T.Compose([T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE), T.ToTensor(), T.Normalize(MEAN, STD)])


def build_model(arch="resnet18", pretrained=True, num_classes=len(C.CLASSES)):
    """Return (model, head_modules, last_block_modules) for the 2-stage fine-tuning."""
    tv = torchvision.models
    if arch == "resnet18":
        m = tv.resnet18(weights=tv.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m, [m.fc], [m.layer4]
    if arch == "mobilenet_v3_small":
        m = tv.mobilenet_v3_small(weights=tv.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        return m, [m.classifier], [m.features[-3:]]
    if arch == "efficientnet_b0":
        m = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m, [m.classifier], [m.features[-2:]]
    raise ValueError(arch)


def load_model(path, device="cpu"):
    """Load a checkpoint saved by 04_train.py. Returns (eval-mode model, checkpoint dict)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model, _, _ = build_model(ck["arch"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


@torch.no_grad()
def predict_df(model, df, device="cpu", images_root=None, bs=64):
    """Softmax probabilities (N x 6) for the rows of a split dataframe (columns: path, cls)."""
    tf = eval_transform()
    model.to(device).eval()
    paths, clss, out = list(df["path"]), list(df["cls"]), []
    for i in range(0, len(paths), bs):
        xs = []
        for p, c in zip(paths[i:i + bs], clss[i:i + bs]):
            p = Path(p)
            if not p.exists() and images_root:
                p = Path(images_root) / c / p.name
            with Image.open(p) as im:
                xs.append(tf(im.convert("RGB")))
        out.append(torch.softmax(model(torch.stack(xs).to(device)).float(), 1).cpu().numpy())
    return np.concatenate(out)
