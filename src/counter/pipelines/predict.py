from __future__ import annotations
import cv2

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import platform
import subprocess

from counter.config.schema import PredictConfig
from counter.counting.net_state import NetStateCounter
from counter.domain.model_spec import ModelSpec, load_models
from counter.domain.results import CountsResult
from counter.domain.types import CanonicalClass, Detection, Track
from counter.io.export import ensure_dir, dump_json, save_counts_json
from counter.io.video import get_video_info, iter_frames
from counter.mapping.factory import MappingFactory
from counter.tracking.factory import TrackerFactory

def _encode_preview_jpg(frame_bgr, line_start, line_end, tracks, max_width: int) -> bytes:
    vis = frame_bgr.copy()

    # line
    x1, y1 = int(line_start[0]), int(line_start[1])
    x2, y2 = int(line_end[0]), int(line_end[1])
    cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # boxes
    for t in tracks:
        x1b, y1b, x2b, y2b = map(int, t.bbox)
        cv2.rectangle(vis, (x1b, y1b), (x2b, y2b), (255, 0, 0), 2)
        cv2.putText(
            vis,
            f"id={t.track_id} c={t.mapped_class_id}",
            (x1b, max(0, y1b - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    h, w = vis.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return b""
    return buf.tobytes()

def _now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_git_sha() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        return out[:12]
    except Exception:
        return None


class PredictionCancelled(RuntimeError):
    """Raised when a prediction run is cancelled from the UI."""


@dataclass(frozen=True)
class PredictHooks:
    should_stop: Optional[Callable[[], bool]] = None
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None


class PredictPipeline:
    def __init__(self, models_yaml: str = "configs/models.yaml") -> None:
        self._models: Dict[str, ModelSpec] = load_models(models_yaml)

    def run(self, cfg: PredictConfig, hooks: PredictHooks | None = None) -> Path:
        if cfg.model_id not in self._models:
            raise ValueError(f"Unknown model_id='{cfg.model_id}'. Check configs/models.yaml")

        spec = self._models[cfg.model_id]

        run_id = _now_run_id()
        run_dir = ensure_dir(Path(cfg.out_dir) / run_id)
        pred_dir = ensure_dir(run_dir / "predict")

        base_meta: Dict[str, Any] = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_sha": _safe_git_sha(),
            "host": platform.node(),
            "platform": platform.platform(),
            "backend": spec.backend,
            "variant": spec.variant,
            "model_id": spec.model_id,
            "weights": spec.weights,
            "device": cfg.device,
            "thresholds": cfg.thresholds.model_dump(),
            "tracking": cfg.tracking.model_dump(),
            "export": cfg.export.model_dump(),
            "conf": cfg.conf,
            "iou": cfg.iou,
            "line": cfg.line.model_dump(),
            "greyzone_px": cfg.greyzone_px,
            "preview": getattr(cfg, "preview", None).model_dump() if getattr(cfg, "preview", None) else None,

        }

        provider = TrackerFactory.create(
            spec=spec,
            conf=cfg.conf,
            iou=cfg.iou,
            device=cfg.device,
            tracking=cfg.tracking,
            work_dir=run_dir,
        )
        mapper = MappingFactory.create(spec)

        status = "completed"
        processed_videos: List[str] = []

        # canonical aggregates (0..3)
        aggregate_in: Dict[int, int] = {int(c): 0 for c in CanonicalClass}
        aggregate_out: Dict[int, int] = {int(c): 0 for c in CanonicalClass}
        outputs: List[str] = []

        try:
            for vid_i, vid in enumerate(cfg.videos):
                video_path = str(Path(cfg.videos_dir) / vid)
                vinfo = get_video_info(video_path)

                counter = NetStateCounter(line=cfg.line, greyzone_px=cfg.greyzone_px)

                if hooks and hooks.on_progress:
                    hooks.on_progress(
                        {
                            "type": "video_start",
                            "video": vid,
                            "video_index": vid_i,
                            "video_total": len(cfg.videos),
                            "frame_count": vinfo.frame_count,
                            "fps": vinfo.fps,
                        }
                    )

                for frame_idx, frame_bgr in iter_frames(video_path):
                    if hooks and hooks.should_stop and hooks.should_stop():
                        status = "cancelled"
                        raise PredictionCancelled("Cancelled by user")

                    raw_tracks = provider.update(frame_bgr)

                    tracks: List[Track] = []
                    for rt in raw_tracks:
                        det = Detection(
                            bbox=rt.bbox,
                            score=rt.score,
                            raw_class_id=rt.raw_class_id,
                            raw_class_name=rt.raw_class_name,
                        )
                        mapped = mapper.map_detection(det)
                        if mapped is None:
                            continue
                        tracks.append(
                            Track(
                                track_id=int(rt.track_id),
                                bbox=rt.bbox,
                                score=float(rt.score),
                                mapped_class_id=int(mapped),
                            )
                        )

                    counter.observe(tracks)

                    def _emit_counts_event(event_type: str, include_jpg: bool = False) -> None:
                        if not hooks or not hooks.on_progress:
                            return

                        raw_in, raw_out = counter.snapshot_raw_counts()
                        fin_in, fin_out = mapper.finalize_counts(raw_in, raw_out)

                        # SSOT: vždy 0..3
                        for cid in CanonicalClass:
                            fin_in.setdefault(int(cid), 0)
                            fin_out.setdefault(int(cid), 0)

                        payload: Dict[str, Any] = {
                            "type": event_type,
                            "video": vid,
                            "frame": frame_idx,
                            "frame_count": vinfo.frame_count,
                            "video_index": vid_i,
                            "video_total": len(cfg.videos),
                            "n_raw_tracks": len(raw_tracks),
                            "n_tracks": len(tracks),
                            "counts_in": fin_in,
                            "counts_out": fin_out,
                        }

                        if include_jpg:
                            # pozor: tohle může zpomalit UI -> throttle přes every_n_frames
                            jpg = _encode_preview_jpg(
                                frame_bgr,
                                cfg.line.start,
                                cfg.line.end,
                                tracks,
                                max_width=int(getattr(cfg.preview, "max_width", 960)),
                            )
                            if jpg:
                                payload["jpg"] = jpg

                        hooks.on_progress(payload)

                    # 1) Textový progress (levné) každých 200 snímků
                    if hooks and hooks.on_progress and (frame_idx % 200 == 0):
                        _emit_counts_event("progress", include_jpg=False)

                    # 2) Volitelný obrazový preview (drahé) – řízené configem
                    if (
                        hooks
                        and hooks.on_progress
                        and getattr(cfg, "preview", None) is not None
                        and cfg.preview.enabled
                        and (frame_idx % max(1, int(cfg.preview.every_n_frames)) == 0)
                    ):
                        _emit_counts_event("preview", include_jpg=True)


                raw_in, raw_out = counter.finalize_raw_counts()
                fin_in, fin_out = mapper.finalize_counts(raw_in, raw_out)

                # SSOT: vždy 0..3 i kdyby 0
                for cid in CanonicalClass:
                    fin_in.setdefault(int(cid), 0)
                    fin_out.setdefault(int(cid), 0)

                for k, v in fin_in.items():
                    aggregate_in[k] = aggregate_in.get(k, 0) + int(v)
                for k, v in fin_out.items():
                    aggregate_out[k] = aggregate_out.get(k, 0) + int(v)

                meta = dict(base_meta)
                meta["video"] = asdict(vinfo)

                res = CountsResult(
                    video=vid,
                    line_name=cfg.line.name,
                    in_count=fin_in,
                    out_count=fin_out,
                    meta=meta,
                )

                out_file = pred_dir / f"{Path(vid).stem}.counts.json"
                save_counts_json(out_file, res)
                outputs.append(str(out_file))
                processed_videos.append(vid)

        except PredictionCancelled:
            # záměrně nic, status už je nastaven
            pass

        # aggregate výstup (i když cancelled -> je to jen z toho, co stihlo doběhnout)
        agg = CountsResult(
            video="__aggregate__",
            line_name=cfg.line.name,
            in_count=aggregate_in,
            out_count=aggregate_out,
            meta=dict(base_meta) | {"status": status, "processed_videos": processed_videos},
        )
        agg_file = pred_dir / "aggregate.counts.json"
        save_counts_json(agg_file, agg)
        outputs.append(str(agg_file))

        run_json = {
            "type": "predict",
            "status": status,
            "run_id": run_id,
            "model_id": spec.model_id,
            "backend": spec.backend,
            "variant": spec.variant,
            "created_at": base_meta["created_at"],
            "outputs": outputs,
            "processed_videos": processed_videos,
        }
        dump_json(run_dir / "run.json", run_json)

        return run_dir
