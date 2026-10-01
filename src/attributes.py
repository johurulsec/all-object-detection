"""Market-1501 pedestrian attributes from a ResNet50 with 30 binary heads (ResNet50 backbone + per-attribute heads)."""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "market_attr_resnet50.pth"

# Output order of the 30 binary heads in the checkpoint.
LABELS = ["young", "teenager", "adult", "old", "backpack", "bag", "handbag", "clothes", "down", "up",
          "hair", "hat", "gender", "upblack", "upwhite", "upred", "uppurple", "upyellow", "upgray",
          "upblue", "upgreen", "downblack", "downwhite", "downpink", "downpurple", "downyellow",
          "downgray", "downblue", "downgreen", "downbrown"]
AGES = ["young", "teenager", "adult", "old"]
UP_COLORS = [l[2:] for l in LABELS if l.startswith("up") and l not in ("up",)]
DOWN_COLORS = [l[4:] for l in LABELS if l.startswith("down") and l not in ("down",)]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_cache = {}


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_block = nn.Sequential(nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.1), nn.Dropout(0.5))
        self.classifier = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, x):
        return self.classifier(self.add_block(x))


class _Net(nn.Module):
    def __init__(self, n=len(LABELS)):
        super().__init__()
        backbone = models.resnet50()
        backbone.fc = nn.Sequential()
        self.features = backbone
        for i in range(n):
            setattr(self, f"class_{i}", _Head())
        self.n = n

    def forward(self, x):
        f = self.features(x).flatten(1)
        return torch.cat([getattr(self, f"class_{i}")(f) for i in range(self.n)], dim=1)


def _model():
    if "m" not in _cache:
        m = _Net()
        m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        _cache["m"] = m.eval()
    return _cache["m"]


def _prep(crop):
    rgb = cv2.cvtColor(cv2.resize(crop, (144, 288), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))


def probabilities(crops):
    """Mean sigmoid output per head over a track's crops, as {label: prob}."""
    with torch.inference_mode():
        p = _model()(torch.stack([_prep(c) for c in crops])).mean(dim=0).numpy()
    return dict(zip(LABELS, p.tolist()))


def _pick(p, names, prefix=""):
    scores = [p[prefix + n] for n in names]
    i = int(np.argmax(scores))
    return names[i], scores[i]


def decode(p):
    """Turn the 30 head probabilities into Market-1501's 12 attributes with confidences."""
    age, age_c = _pick(p, AGES)
    up, up_c = _pick(p, UP_COLORS, "up")
    down, down_c = _pick(p, DOWN_COLORS, "down")

    def binary(key, no, yes):
        q = p[key]
        return (yes, q) if q >= 0.5 else (no, 1 - q)

    pairs = {
        "gender": binary("gender", "male", "female"),
        "age": (age, age_c),
        "hair_length": binary("hair", "short", "long"),
        "sleeve_length": binary("up", "long", "short"),
        "lower_clothing_length": binary("down", "long", "short"),
        "lower_clothing_type": binary("clothes", "dress", "pants"),
        "hat": binary("hat", "no", "yes"),
        "backpack": binary("backpack", "no", "yes"),
        "bag": binary("bag", "no", "yes"),
        "handbag": binary("handbag", "no", "yes"),
        "upper_color": (up, up_c),
        "lower_color": (down, down_c),
    }
    return {k: {"label": v[0], "conf": round(float(v[1]), 3)} for k, v in pairs.items()}


def predict(crops):
    return decode(probabilities(crops))
