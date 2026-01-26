from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from counter.tracking.providers import RawTrack, TrackProvider


def _torch_load_trusted(weights_path: str | Path) -> Any:
    """
    PyTorch 2.6+ změnil default `weights_only=True`, což rozbije starší checkpointy,
    které obsahují např. argparse.Namespace apod.

    Tohle používá weights_only=False -> POTENCIÁLNĚ NEBEZPEČNÉ pro cizí checkpoint.
    Používej jen na checkpointy, kterým věříš (tvoje trénování).
    """
    return torch.load(str(weights_path), map_location="cpu", weights_only=False)


def _is_torch_module(x: Any) -> bool:
    return hasattr(x, "state_dict") and hasattr(x, "load_state_dict")


def _find_torch_module(obj: Any, max_depth: int = 5) -> Optional[torch.nn.Module]:
    """
    Robustní unwrap pro různé verze `rfdetr` wrapperu.
    Chceme najít něco, co je torch.nn.Module (má state_dict/load_state_dict).
    """
    seen: set[int] = set()

    def walk(x: Any, depth: int) -> Optional[torch.nn.Module]:
        if x is None:
            return None
        xid = id(x)
        if xid in seen:
            return None
        seen.add(xid)

        if _is_torch_module(x):
            return x  # type: ignore[return-value]

        if depth <= 0:
            return None

        for attr in ("model", "_model", "net", "detector", "detr", "module", "torch_model"):
            try:
                y = getattr(x, attr)
            except Exception:
                y = None
            m = walk(y, depth - 1)
            if m is not None:
                return m

        d = getattr(x, "__dict__", None)
        if isinstance(d, dict):
            for v in d.values():
                if _is_torch_module(v):
                    return v  # type: ignore[return-value]

        return None

    return walk(obj, max_depth)


def _load_checkpoint_partial(model_obj: Any, weights_path: str | Path) -> Dict[str, Any]:
    """
    Tuned checkpoint load:
    - načti checkpoint dict
    - vem `model`/`state_dict`
    - do aktuálního modelu nahraj jen klíče, které sedí názvem i tvarem tensoru

    => přežije to drobné změny, ale když je checkpoint úplně jiný backbone/config,
    tak se nahraje málo a kvalita půjde do háje (to je správně vidět v logu).
    """
    ckpt = _torch_load_trusted(weights_path)

    state = None
    if isinstance(ckpt, dict):
        for k in ("model", "state_dict", "model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
        if state is None and all(hasattr(v, "shape") for v in ckpt.values()):
            state = ckpt

    if state is None:
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")

    torch_model = _find_torch_module(model_obj)
    if torch_model is None:
        raise RuntimeError(
            "Could not find underlying torch.nn.Module inside RF-DETR wrapper. "
            "Your `rfdetr` version/API likely changed."
        )

    model_sd = torch_model.state_dict()

    loadable: Dict[str, torch.Tensor] = {}
    skipped_missing = 0
    skipped_shape = 0

    for k, v in state.items():
        if k not in model_sd:
            skipped_missing += 1
            continue
        try:
            if tuple(model_sd[k].shape) != tuple(v.shape):
                skipped_shape += 1
                continue
        except Exception:
            skipped_shape += 1
            continue
        loadable[k] = v

    torch_model.load_state_dict(loadable, strict=False)

    return {
        "loaded": int(len(loadable)),
        "skipped_missing": int(skipped_missing),
        "skipped_shape": int(skipped_shape),
    }


def _build_rfdetr(RFDetrCls: Any, model_size: str) -> Any:
    """
    Některé verze `rfdetr` používají jiné názvy parametrů.
    Zkoušíme víc variant.
    """
    for kwargs in (
        {"model_size": model_size},
        {"model": model_size},
        {"size": model_size},
        {"variant": model_size},
    ):
        try:
            return RFDetrCls(**kwargs)
        except TypeError:
            continue
    # poslední pokus bez args
    return RFDetrCls()


class RoboflowRfDetrTrackProvider(TrackProvider):
    def __init__(
        self,
        *,
        weights: Optional[str],
        model_size: str,
        device: str,
        conf: float,
        iou: float,
        class_map: Optional[Dict[int, int]] = None,
    ) -> None:
        RFDetr = _import_rfdetr()

        self.conf = float(conf)
        self.iou = float(iou)

        self.model = _build_rfdetr(RFDetr, model_size=model_size)

        # move to device if supported
        try:
            if hasattr(self.model, "to"):
                self.model.to(device)
        except Exception:
            pass

        # tuned weights (optional)
        self._load_stats: Dict[str, Any] = {"pretrained": True}
        if weights and str(weights).strip():
            try:
                self._load_stats = _load_checkpoint_partial(self.model, weights)
            except Exception as e:
                raise RuntimeError(
                    "Failed to load RF-DETR tuned checkpoint. "
                    "This usually means checkpoint is incompatible with current `rfdetr` version/config.\n"
                    f"weights={weights}\n"
                    f"error={e}"
                ) from e

        # warmup (optional)
        try:
            _ = self.model.predict(np.zeros((64, 64, 3), dtype=np.uint8))
        except Exception:
            pass

    def update(self, frame_bgr: np.ndarray) -> List[RawTrack]:
        # většina wrapperů čeká RGB
        # Ensure positive strides (some libraries dislike negative strides).
        img = frame_bgr[:, :, ::-1].copy()

        dets = self.model.predict(img)

        xyxy = getattr(dets, "xyxy", None)
        class_id = getattr(dets, "class_id", None)
        confidence = getattr(dets, "confidence", None)

        if xyxy is None or class_id is None or confidence is None:
            try:
                xyxy = dets["xyxy"]
                class_id = dets["class_id"]
                confidence = dets["confidence"]
            except Exception:
                return []

        tracks: List[RawTrack] = []
        for i in range(len(xyxy)):
            conf = float(confidence[i])
            if conf < self.conf:
                continue

            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
            raw_cls = int(class_id[i])
            tracks.append(
                RawTrack(
                    track_id=int(i),  # RF-DETR nedává stabilní ID
                    bbox=(x1, y1, x2, y2),
                    score=conf,
                    raw_class_id=raw_cls,
                    raw_class_name=str(raw_cls),
                )
            )

        return tracks


def _import_rfdetr():
    """Best-effort import for RF-DETR class across versions."""
    candidates = [
        ("rfdetr", "RFDetr"),
        ("rfdetr", "RFDETR"),
        ("rfdetr", "RfDetr"),
        ("rfdetr.model", "RFDetr"),
        ("rfdetr.model", "RFDETR"),
        ("rfdetr.core", "RFDetr"),
    ]
    last_err: Exception | None = None
    for mod_name, cls_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                return cls
        except Exception as e:
            last_err = e
            continue

    # Fallback: scan the package for any class name containing "detr"
    try:
        import rfdetr as _rfdetr  # type: ignore
        candidates_found = []
        for name in dir(_rfdetr):
            if "detr" in name.lower():
                candidates_found.append(name)
        for name in candidates_found:
            cls = getattr(_rfdetr, name, None)
            if cls is not None:
                return cls
        raise ImportError(
            "Could not find RF-DETR class in installed `rfdetr` package. "
            f"Checked {candidates}. Available symbols: {', '.join(sorted(candidates_found)) or '(none)'}"
        )
    except Exception as e:
        raise ImportError(
            "Could not import RF-DETR class from `rfdetr`. "
            "Please check your installed rfdetr version and API."
        ) from (last_err or e)
