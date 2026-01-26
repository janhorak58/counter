from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .providers import RawTrack, TrackProvider

from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRSmall

import argparse
import inspect
import torch


@dataclass
class RFDETRDetection:
    bbox_xyxy: Tuple[int, int, int, int]
    conf: float
    class_id: int


def _get_torch_module(obj) -> Optional[torch.nn.Module]:
    if isinstance(obj, torch.nn.Module):
        return obj
    for attr in ("model", "_model", "net", "module"):
        if hasattr(obj, attr):
            cand = getattr(obj, attr)
            if isinstance(cand, torch.nn.Module):
                return cand
    for attr in ("model", "_model", "net", "module"):
        if hasattr(obj, attr):
            mod = _get_torch_module(getattr(obj, attr))
            if mod is not None:
                return mod
    return None


def _torch_load_any(weights_path: str):
    """
    Security note:
      weights_only=False can execute code embedded in the file.
      Only do this for checkpoints you produced / trust.
    """
    p = Path(weights_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    try:
        torch.serialization.add_safe_globals([argparse.Namespace])
    except Exception:
        pass

    sig = inspect.signature(torch.load)
    if "weights_only" in sig.parameters:
        try:
            return torch.load(str(p), map_location="cpu", weights_only=True)
        except Exception:
            return torch.load(str(p), map_location="cpu", weights_only=False)

    return torch.load(str(p), map_location="cpu")


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("model", "state_dict", "model_state_dict", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise ValueError("Unsupported checkpoint format (expected dict-like).")


def _filter_compatible_keys(model_sd: dict, ckpt_sd: dict) -> dict:
    filtered = {}
    for k, v in ckpt_sd.items():
        if k not in model_sd:
            continue
        if not isinstance(v, torch.Tensor):
            continue
        if model_sd[k].shape != v.shape:
            continue
        if "position_embeddings" in k:
            continue
        filtered[k] = v
    return filtered


def _load_checkpoint_partial(wrapper_model, weights_path: str) -> Tuple[int, int]:
    torch_model = _get_torch_module(wrapper_model)
    if torch_model is None:
        raise RuntimeError("Cannot find underlying torch.nn.Module on RF-DETR wrapper (no .model/.net).")

    ckpt = _torch_load_any(weights_path)
    sd = _extract_state_dict(ckpt)

    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}

    model_sd = torch_model.state_dict()
    filtered = _filter_compatible_keys(model_sd, sd)

    torch_model.load_state_dict(filtered, strict=False)
    return len(filtered), len(sd)


class RoboflowRfDetrTrackProvider(TrackProvider):
    def __init__(self, model_size: str, weights: Optional[str], conf: float = 0.5):
        model_size = (model_size or "medium").lower()
        ModelCls = {"small": RFDETRSmall, "medium": RFDETRMedium, "large": RFDETRLarge}.get(model_size, RFDETRMedium)

        self.conf = float(conf)
        self._next_id = 1
        self._latest_preview_jpg: Optional[str] = None
        self._backend = "rfdetr"
        self._variant = "tuned" if bool(weights) else "pretrained"

        # Instantiate WITHOUT passing checkpoint to constructor
        self.model: RFDETRBase = ModelCls(pretrain_weights=None)

        if weights:
            loaded, total = _load_checkpoint_partial(self.model, weights)
            print(f"[RFDETR] loaded checkpoint keys: {loaded}/{total} from {weights}")

        # put underlying torch module to eval
        if hasattr(self.model, "model"):
            self.model.model.eval()
        else:
            self.model.eval()

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def variant(self) -> str:
        return self._variant

    @property
    def latest_preview_jpg(self) -> Optional[str]:
        return self._latest_preview_jpg

    def set_preview(self, jpg_b64: str) -> None:
        self._latest_preview_jpg = jpg_b64

    def reset(self) -> None:
        self._next_id = 1
        self._latest_preview_jpg = None

    def update(self, frame_bgr: np.ndarray) -> List[RawTrack]:
        frame_rgb = frame_bgr[:, :, ::-1]
        dets = self.model.infer(frame_rgb, confidence=self.conf)

        tracks: List[RawTrack] = []
        for x1, y1, x2, y2, conf, cls in dets.xyxy:
            tracks.append(
                RawTrack(
                    track_id=self._next_id,
                    class_id=int(cls),
                    conf=float(conf),
                    bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                )
            )
            self._next_id += 1

        return tracks
