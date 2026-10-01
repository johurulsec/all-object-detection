"""Download the pretrained weights into models/ (skips files that already exist).

- OSNet x1.0 ReID weights (Hugging Face: kaiyangzhou/osnet)
- Market-1501 attribute ResNet50 (Google Drive, from the Person-Attribute-Recognition-MarketDuke project)
"""
import shutil
import tempfile
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"
OSNET_URL = ("https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
             "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")
ATTR_FOLDER = "https://drive.google.com/drive/folders/1JTdjuEbxSLypnfUzVuuxLj1uSKAacfd0"


def main():
    MODELS.mkdir(exist_ok=True)
    osnet = MODELS / "osnet_x1_0_msmt17.pth"
    if not osnet.exists():
        print("Downloading OSNet weights ...")
        urllib.request.urlretrieve(OSNET_URL, osnet)

    attr = MODELS / "market_attr_resnet50.pth"
    if not attr.exists():
        import gdown
        print("Downloading Market-1501 attribute weights ...")
        with tempfile.TemporaryDirectory() as tmp:
            gdown.download_folder(ATTR_FOLDER, output=tmp, quiet=True)
            shutil.copy(Path(tmp) / "checkpoints" / "market" / "resnet50_nfc" / "net_last.pth", attr)
    print("Models ready in", MODELS)


if __name__ == "__main__":
    main()
