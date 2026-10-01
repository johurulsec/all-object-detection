import warnings
from pathlib import Path

import numpy as np
import torch

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "osnet_x1_0_msmt17.pth"
_cache = {}


def _extractor():
    if "fe" not in _cache:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from torchreid.reid.utils import FeatureExtractor
            _cache["fe"] = FeatureExtractor(
                model_name="osnet_x1_0", model_path=str(MODEL_PATH), device="cpu", verbose=False)
    return _cache["fe"]


def embed(crops):
    """L2-normalised OSNet embeddings for BGR crops, shape [N, 512]."""
    rgb = [c[:, :, ::-1] for c in crops]
    with torch.inference_mode():
        f = _extractor()(rgb)
    f = f.cpu().numpy()
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def track_embedding(feats):
    """Mean of a track's crop embeddings, re-normalised."""
    e = feats.mean(axis=0)
    return e / np.linalg.norm(e)
