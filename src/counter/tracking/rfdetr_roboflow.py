from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from counter.domain.raw_track import RawTrack

# RF-DETR + supervision
from supervision import ByteTrack
from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRSmall

import torch
import argparse


def _torch_load_trusted(path: str) -> Any:
    """
    PyTorch 2.6: default weights_only=True může shodit legacy checkpointy.
    Tohle záměrně používá weights_only=False (pickle) -> dělej jen pro checkpointy,
    kterým věříš (u tebe: vlastní trénink).
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # starší torch nemá weights_only
        return torch.load(path, map_location="cpu")


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    # nejčastější formáty
    if isinstance(ckpt, dict):
        for k in ("model", "state_dict", "model_state_dict", "net", "module"):
            v = ckpt.get(k)
            if isinstance(v, dict):
                return v
        # občas je state_dict přímo dict v rootu
        if all(isinstance(x, torch.Tensor) for x in ckpt.values()):
            return ckpt  # type: ignore
    if isinstance(ckpt, dict):
        return ckpt  # fallback
    raise RuntimeError("Checkpoint does not look like a state_dict container.")


def _unwrap_torch_module(obj: Any, max_depth: int = 5) -> Optional[torch.nn.Module]:
    """
    rfdetr wrapper někdy drží torch model v různých atributech.
    """
    if isinstance(obj, torch.nn.Module):
        return obj
    if max_depth <= 0 or obj is None:
        return None

    # zkus typické atributy
    for attr in ("model", "net", "module", "_model", "torch_model", "detr"):
        if hasattr(obj, attr):
            m = getattr(obj, attr)
            out = _unwrap_torch_module(m, max_depth - 1)
            if out is not None:
                return out

    return None


def _load_state_dict_partial(module: torch.nn.Module, sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """
    Nahraje jen kompatibilní klíče (stejný tvar tensoru).
    Vrací (loaded_keys, skipped_keys).
    """
    current = module.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped = 0

    for k, v in sd.items():
        if k in current and hasattr(v, "shape") and hasattr(current[k], "shape") and v.shape == current[k].shape:
            filtered[k] = v
        else:
            skipped += 1

    module.load_state_dict(filtered, strict=False)
    return len(filtered), skipped


@dataclass
class RoboflowRfDetrTrackProvider:
    """
    Wrapper pro RF-DETR + ByteTrack.
    - pretrained: použije RFDETR{Small|Medium|Large}(pretrain_weights=...)
    - tuned: vytvoří model (pretrained backbone) a pak nahraje checkpoint (partial load)
    """

    size: str
    conf: float
    iou: float
    bytetrack_track_thresh: float = 0.25
    bytetrack_buffer: int = 30
    bytetrack_match_thresh: float = 0.8
    bytetrack_min_hits: int = 1
    weights: Optional[str] = None

    def __post_init__(self) -> None:
        size = (self.size or "").lower().strip()
        if size not in ("small", "medium", "large"):
            raise ValueError(f"RF-DETR size must be one of: small|medium|large, got: {self.size}")

        ModelCls = {"small": RFDETRSmall, "medium": RFDETRMedium, "large": RFDETRLarge}[size]

        # 1) vytvořit model
        # - když je tuned weights, pořád je rozumné startovat z pretrained základu (stejná architektura!)
        # - pretrain_weights necháme None -> některé verze si stáhnou defaulty podle model_name,
        #   jiné nic. Pokud chceš jistotu, dej do models.yaml i pretrained weights pro backbone.
        self.model = ModelCls(model_name=f"rfdetr-{size}", pretrain_weights=None)

        # 2) tuned checkpoint load (pokud je uveden)
        if self.weights:
            ckpt = _torch_load_trusted(self.weights)
            sd = _extract_state_dict(ckpt)

            torch_module = _unwrap_torch_module(self.model)
            if torch_module is None:
                raise RuntimeError("Could not locate underlying torch.nn.Module inside RF-DETR wrapper.")

            loaded, skipped = _load_state_dict_partial(torch_module, sd)
            if loaded == 0:
                raise RuntimeError(
                    "Loaded 0 keys from tuned checkpoint -> nejspíš nekompatibilní verze/architektura."
                )
            print(f"[RFDETR] tuned checkpoint loaded: loaded_keys={loaded}, skipped_keys={skipped}")

        self.tracker = ByteTrack(
            track_activation_threshold=float(self.bytetrack_track_thresh),
            lost_track_buffer=int(self.bytetrack_buffer),
            minimum_matching_threshold=float(self.bytetrack_match_thresh),
            minimum_consecutive_frames=int(self.bytetrack_min_hits),
        )

        # names (optional)
        self._names: Optional[dict[int, str]] = None
        try:
            names = getattr(self.model, "class_names", None) or getattr(self.model, "class_names_", None)
            if isinstance(names, (list, tuple)):
                self._names = {i: str(n) for i, n in enumerate(names)}
            elif isinstance(names, dict):
                self._names = {int(k): str(v) for k, v in names.items()}
        except Exception:
            self._names = None

    def update(self, frame_bgr: np.ndarray) -> List[RawTrack]:
        frame_rgb = frame_bgr[:, :, ::-1].copy()
        detections = self.model.predict(frame_rgb, threshold=self.conf)

        if not hasattr(detections, "xyxy"):
            return []

        tracked = self.tracker.update_with_detections(detections)
        if getattr(tracked, "tracker_id", None) is None:
            return []

        out: List[RawTrack] = []

        xyxy = tracked.xyxy
        class_id = getattr(tracked, "class_id", None)
        conf = getattr(tracked, "confidence", None)
        tid = tracked.tracker_id

        if class_id is None:
            class_id = np.zeros((len(xyxy),), dtype=int)
        if conf is None:
            conf = np.ones((len(xyxy),), dtype=float)

        for box, cid, sc, track_id in zip(xyxy, class_id, conf, tid):
            cid_i = int(cid) if cid is not None else -1
            name = str(cid_i)
            if self._names and cid_i in self._names:
                name = self._names[cid_i]

            out.append(
                RawTrack(
                    track_id=int(track_id),
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    score=float(sc),
                    raw_class_id=cid_i,
                    raw_class_name=name,
                )
            )
        return out
