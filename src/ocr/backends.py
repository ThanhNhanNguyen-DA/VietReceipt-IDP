"""Lazy model adapters: Paddle and Torch are never imported in the same process."""
import os
import shutil
from collections import defaultdict
from contextlib import suppress
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def package_versions() -> dict[str, str]:
    out = {}
    for name in ("paddleocr", "paddlex", "paddlepaddle-gpu", "paddlepaddle", "torch", "torchvision", "vietocr", "numpy", "opencv-python-headless"):
        with suppress(PackageNotFoundError):
            out[name] = version(name)
    return out


def _paddle_setup():
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(ROOT / "models/paddlex"))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "bos")


def paddle_device_options(device: str) -> dict:
    import paddle

    if device.startswith("gpu") and paddle.device.cuda.device_count() == 0:
        raise ValueError("Requested Paddle GPU is unavailable; refusing a silent CPU fallback. Check GPU access or use --device cpu.")
    # Paddle 3.3.1's oneDNN path currently fails for the OCR detector on this host.
    return {"device": device, **({"enable_mkldnn": False} if device == "cpu" else {})}


def _download_asset(url: str, path: Path) -> Path:
    """Publish only complete downloads; interrupted files must not become cached weights."""
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    with urlopen(url, timeout=30) as response, temp.open("wb") as f:
        shutil.copyfileobj(response, f)
    temp.replace(path)
    return path


def vietocr_image(image, dataset: dict) -> np.ndarray:
    """VietOCR 0.3.12 resize/normalization using Pillow's current LANCZOS API."""
    height = dataset["image_height"]
    width = int(height * image.shape[1] / image.shape[0])
    width = ((width + 9) // 10) * 10
    width = min(max(width, dataset["image_min_width"]), dataset["image_max_width"])
    rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    resized = rgb.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0


class PaddleDetector:
    def __init__(self, config: dict, device: str):
        _paddle_setup()
        from paddleocr import TextDetection

        self.model = TextDetection(**{k: v for k, v in config.items() if k != "min_box_side"}, **paddle_device_options(device))

    def detect(self, image):
        results = self.model.predict(image)
        if len(results) != 1:
            raise ValueError("Detector must return exactly one page")
        r = results[0]
        return [(poly.tolist(), float(score)) for poly, score in zip(r["dt_polys"], r["dt_scores"], strict=True)]


class PaddleRecognizer:
    def __init__(self, config: dict, device: str):
        _paddle_setup()
        from paddleocr import TextRecognition

        self.model = TextRecognition(**{k: v for k, v in config.items() if k != "batch_size"}, **paddle_device_options(device))

    def recognize(self, images):
        results = self.model.predict(images, batch_size=len(images))
        return [(r["rec_text"], float(r["rec_score"])) for r in results]


class VietRecognizer:
    def __init__(self, config: dict, device: str):
        from vietocr.tool.predictor import Predictor

        if config.get("config_file"):
            cfg = yaml.safe_load(Path(config["config_file"]).read_text(encoding="utf-8"))
        else:
            if config["config_name"] != "vgg_transformer":
                raise ValueError("Supply config_file for a VietOCR model other than vgg_transformer")
            cache = ROOT / "models/vietocr"
            cfg = {}
            for name in ("base.yml", "vgg-transformer.yml"):
                path = _download_asset(f"https://raw.githubusercontent.com/pbcquoc/vietocr/master/config/{name}", cache / name)
                cfg.update(yaml.safe_load(path.read_text(encoding="utf-8")))
        cfg["device"] = device
        cfg["cnn"]["pretrained"] = False  # Full OCR checkpoint already includes the backbone.
        cfg["predictor"]["beamsearch"] = False
        if config.get("weights"):
            cfg["weights"] = config["weights"]
        if cfg["weights"].startswith("http"):
            url = cfg["weights"]
            filename = ("vgg_transformer.pth" if url == "https://vocr.vn/data/vietocr/vgg_transformer.pth"
                        else f"weights-{sha256(url.encode()).hexdigest()[:16]}.pth")
            cfg["weights"] = str(_download_asset(url, ROOT / "models/vietocr" / filename))
        self.model = Predictor(cfg)

    def recognize(self, images):
        import torch
        from vietocr.tool.translate import translate

        # Bypass upstream process_input (removed Image.ANTIALIAS). Group by resized
        # width as predict_batch does, then restore the original crop order.
        buckets = defaultdict(list)
        for i, image in enumerate(images):
            array = vietocr_image(image, self.model.config["dataset"])
            buckets[array.shape[-1]].append((i, array))
        results = [None] * len(images)
        for items in buckets.values():
            batch = torch.from_numpy(np.stack([array for _, array in items])).to(self.model.device)
            sequences, scores = translate(batch, self.model.model)
            texts = self.model.vocab.batch_decode(sequences.tolist())
            for (index, _), text, score in zip(items, texts, scores, strict=True):
                results[index] = (text, float(score) if np.isfinite(score) else None)
        return results
