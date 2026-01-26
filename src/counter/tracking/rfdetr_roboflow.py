from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from counter.tracking.providers import TrackProvider, RawTrack

try:  # pragma: no cover
    import supervision as sv  # type: ignore
except Exception:  # pragma: no cover
    sv = None

try:  # pragma: no cover
    from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRSmall, RFDETRNano  # type: ignore
except Exception:  # pragma: no cover
    RFDETRBase = RFDETRLarge = RFDETRMedium = RFDETRSmall = RFDETRNano = None

try:  # pragma: no cover
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None


def _pick_model_cls(size: str | None):
    size = (size or "base").lower()
    if size == "nano":
        return RFDETRNano
    if size == "small":
        return RFDETRSmall
    if size == "medium":
        return RFDETRMedium
    if size == "large":
        return RFDETRLarge
    return RFDETRBase


def _extract_state_dict(ckpt: Any) -> Dict[str, Any]:
    """
    Accepts common checkpoint shapes:
      - {"model": {...}}
      - {"state_dict": {...}}
      - {"model_state_dict": {...}}
      - already a raw state_dict
    """
    if isinstance(ckpt, dict):
        for k in ("model", "state_dict", "model_state_dict"):
            v = ckpt.get(k)
            if isinstance(v, dict) and v:
                return v
        # fallback: sometimes the dict itself IS the state_dict
        # (heuristic: contains tensor-like values)
        if any(hasattr(v, "shape") for v in ckpt.values()):
            return ckpt
    return {}


def _filter_by_shape(model_sd: Dict[str, Any], ckpt_sd: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """
    Keep only keys that exist in model AND have identical shape.
    Returns: (filtered, dropped_missing, dropped_shape)
    """
    filtered: Dict[str, Any] = {}
    dropped_missing: List[str] = []
    dropped_shape: List[str] = []

    for k, v in ckpt_sd.items():
        if k not in model_sd:
            dropped_missing.append(k)
            continue
        try:
            if hasattr(v, "shape") and hasattr(model_sd[k], "shape") and tuple(v.shape) != tuple(model_sd[k].shape):
                dropped_shape.append(k)
                continue
        except Exception:
            dropped_shape.append(k)
            continue
        filtered[k] = v

    return filtered, dropped_missing, dropped_shape


def _load_checkpoint_partial(model_obj: Any, weights_path: str) -> None:
    """
    Loads checkpoint into RFDETR wrapper instance partially (shape-matched only).
    Works with Roboflow `rfdetr` wrapper where torch model is typically at `.model`.
    """
    if torch is None:
        raise ImportError("torch is required to load RF-DETR checkpoints.")

    # get underlying torch.nn.Module
    torch_model = getattr(model_obj, "model", None)
    if torch_model is None:
        raise RuntimeError("RFDETR wrapper has no `.model` attribute; can't partial-load checkpoint.")

    ckpt = torch.load(weights_path, map_location="cpu")
    ckpt_sd = _extract_state_dict(ckpt)
    if not ckpt_sd:
        raise RuntimeError(f"Could not extract state_dict from checkpoint: {weights_path}")

    model_sd = torch_model.state_dict()
    filtered, dropped_missing, dropped_shape = _filter_by_shape(model_sd, ckpt_sd)

    msg = (
        f"[RFDETR] Partial load: keep={len(filtered)} / ckpt={len(ckpt_sd)} "
        f"(dropped_missing={len(dropped_missing)}, dropped_shape={len(dropped_shape)})"
    )
    print(msg)

    # finally load
    torch_model.load_state_dict(filtered, strict=False)


class RoboflowRfDetrTrackProvider(TrackProvider):
    """RF-DETR via Roboflow's `rfdetr` library + Supervision ByteTrack.

    Important:
    - Some tuned checkpoints may be incompatible with the installed `rfdetr` model config
      (e.g. patch size 16 vs 14, different pos embeddings). In that case we partial-load
      only shape-matching keys to avoid hard crashes.
    """

    def __init__(
        self,
        weights: str,
        size: str | None = None,
        conf: float = 0.35,
        bytetrack_min_hits: int = 3,
        bytetrack_track_thresh: float = 0.25,
        bytetrack_match_thresh: float = 0.8,
        bytetrack_buffer: int = 30,
    ):
        if sv is None or RFDETRBase is None:
            raise ImportError(
                "Missing dependencies for RF-DETR tracking. Install extras: "
                "uv pip install -e '.[predict]' (needs rfdetr + supervision + torch)."
            )
        self.conf = float(conf)

        ModelCls = _pick_model_cls(size)
        if ModelCls is None:
            raise ImportError("rfdetr model classes not found. Check your rfdetr install/version.")

        w = (weights or "").strip()
        use_custom = bool(w) and w.lower() not in {"pretrained", "default"}

        # 1) Create model WITHOUT loading custom weights in constructor (avoids crash on mismatch)
        self.model = ModelCls()

        # 2) If custom checkpoint provided, try partial load (shape-matched only)
        if use_custom:
            try:
                _load_checkpoint_partial(self.model, w)
            except Exception as e:
                raise RuntimeError(
                    "Failed to load RF-DETR tuned checkpoint. "
                    "This usually means checkpoint is incompatible with current `rfdetr` version/config.\n"
                    f"weights={w}\nerror={e}"
                )

        # Supervision ByteTrack tracker
        self.tracker = sv.ByteTrack(
            track_activation_threshold=float(bytetrack_track_thresh),
            lost_track_buffer=int(bytetrack_buffer),
            minimum_matching_threshold=float(bytetrack_match_thresh),
            minimum_consecutive_frames=int(bytetrack_min_hits),
        )

        # RF-DETR exposes class names
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
