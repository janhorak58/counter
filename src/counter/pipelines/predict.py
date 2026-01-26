from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2

from counter.config.schema import PredictConfig
from counter.config.loader import load_models
from counter.domain.types import CanonicalClass
from counter.io.export import ensure_dir, dump_json
from counter.io.video import get_video_info, iter_frames
from counter.mapping.factory import MappingFactory
from counter.tracking.factory import TrackerFactory
from counter.counting.counter import TrackCounter


@dataclass(frozen=True)
class RunMeta:
    type: str
    status: str
    run_id: str
    model_id: str
    backend: str
    variant: str
    created_at: str
    outputs: List[str]
    processed_videos: List[str]
    extra: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _draw_overlay(
    frame_bgr,
    tracks: List[Dict[str, Any]],
    class_names: Dict[int, str],
    frame_i: int,
    frame_n: int,
    flash_period: int = 20,
):
    # blikání: 0..19 on, 20..39 off, ...
    show = ((frame_i // flash_period) % 2) == 0
    if show:
        txt = f"Frame {frame_i}/{frame_n}"
        cv2.putText(frame_bgr, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    # boxy
    for t in tracks:
        xyxy = t.get("xyxy")
        cid = t.get("class_id")
        tid = t.get("track_id", "")
        conf = t.get("conf", None)

        if not xyxy or len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)

        label = class_names.get(int(cid), str(cid))
        if conf is not None:
            label = f"{label} {float(conf):.2f}"
        if tid != "":
            label = f"{label} id={tid}"

        cv2.putText(frame_bgr, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA)


class PredictPipeline:
    def __init__(self, models_yaml: str = "configs/models.yaml"):
        self.models_yaml = models_yaml

    def run(self, cfg: PredictConfig) -> str:
        reg = load_models(self.models_yaml)
        if cfg.model_id not in reg.models:
            raise KeyError(f"Unknown model_id={cfg.model_id}. Run: python -m counter.app.cli --models {self.models_yaml} models")

        spec = reg.models[cfg.model_id]

        # runs/predict/<run_id>/
        run_id = f"{spec.backend}_{spec.variant}__{cfg.model_id}"
        run_root = ensure_dir(Path(cfg.out_dir) / "predict" / run_id)
        predict_dir = ensure_dir(run_root / "predict")

        # jsonl log
        log_path = predict_dir / "run.log.jsonl"
        def log(event: str, payload: Dict[str, Any]):
            rec = {"t": _now_iso(), "event": event, **payload}
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        log("start", {"run_id": run_id, "model_id": cfg.model_id, "backend": spec.backend, "variant": spec.variant})

        mapper = MappingFactory.create(spec)  # !!! bez policy=

        provider = TrackerFactory.create(
            spec=spec,
            device=cfg.device,
            conf=cfg.conf,
            iou=cfg.iou,
            tracking=cfg.tracking,
        )

        counter = TrackCounter(
            line_start=tuple(cfg.line.start),
            line_end=tuple(cfg.line.end),
            greyzone_px=float(cfg.greyzone_px),
            class_ids=[int(c) for c in CanonicalClass],
        )

        outputs: List[str] = []
        processed: List[str] = []

        try:
            for rel in (cfg.videos or []):
                video_path = Path(cfg.videos_dir) / rel
                if not video_path.exists():
                    log("skip_missing_video", {"video": str(video_path)})
                    continue

                vinfo = get_video_info(str(video_path))
                fps = float(vinfo.fps) if vinfo.fps > 0 else 25.0
                frame_n = int(vinfo.frame_count) if vinfo.frame_count > 0 else 0

                # video writer optional
                writer = None
                out_video_path = None
                if cfg.export.save_video:
                    out_video_path = predict_dir / f"{Path(rel).stem}.pred.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (int(vinfo.width), int(vinfo.height)))
                    log("video_writer_open", {"video": rel, "out_video": str(out_video_path), "fps": fps})

                log("video_start", {"video": rel, "fps": fps, "frames": frame_n})

                provider.reset()
                counter.reset()

                t0 = time.time()
                frame_i = 0

                for frame_bgr in iter_frames(str(video_path)):
                    frame_i += 1

                    raw_tracks = provider.update(frame_bgr)
                    mapped = mapper.map_tracks(raw_tracks)
                    counter.update(mapped)

                    if writer is not None:
                        # připrav viz data z mapped tracků
                        vis_tracks: List[Dict[str, Any]] = []
                        for tr in mapped:
                            vis_tracks.append(
                                {
                                    "xyxy": [tr.x1, tr.y1, tr.x2, tr.y2],
                                    "class_id": tr.class_id,
                                    "track_id": tr.track_id,
                                    "conf": tr.conf,
                                }
                            )

                        # class names
                        class_names = {int(c): c.name for c in CanonicalClass}
                        _draw_overlay(frame_bgr, vis_tracks, class_names, frame_i, frame_n or frame_i)
                        writer.write(frame_bgr)

                elapsed = time.time() - t0

                counts = counter.finalize()

                # standard output counts json for eval
                out_counts = {
                    "video": rel,
                    "line_name": cfg.line.name,
                    "in_count": counts["in_count"],
                    "out_count": counts["out_count"],
                    "meta": {
                        "run_id": run_id,
                        "model_id": cfg.model_id,
                        "backend": spec.backend,
                        "variant": spec.variant,
                        "weights": spec.weights,
                        "mapping_policy": spec.mapping_policy,
                        "thresholds": {"conf": cfg.conf, "iou": cfg.iou},
                        "tracker": {"type": cfg.tracking.type, "tracker_yaml": cfg.tracking.tracker_yaml, "params": cfg.tracking.params},
                        "line": {"name": cfg.line.name, "start": cfg.line.start, "end": cfg.line.end},
                        "greyzone_px": cfg.greyzone_px,
                        "video": {"path": rel, "fps": fps, "frame_count": frame_n, "width": vinfo.width, "height": vinfo.height},
                    },
                }

                out_path = predict_dir / f"{Path(rel).stem}.counts.json"
                dump_json(out_path, out_counts)
                outputs.append(str(out_path))
                processed.append(Path(rel).stem)

                if writer is not None:
                    writer.release()
                    outputs.append(str(out_video_path))

                log("video_done", {"video": rel, "elapsed_s": elapsed, "counts_path": str(out_path)})

            # aggregate
            agg = {"outputs": outputs, "processed_videos": processed}
            dump_json(predict_dir / "aggregate.counts.json", agg)
            outputs.append(str(predict_dir / "aggregate.counts.json"))

            meta = RunMeta(
                type="predict",
                status="completed",
                run_id=run_id,
                model_id=cfg.model_id,
                backend=spec.backend,
                variant=spec.variant,
                created_at=_now_iso(),
                outputs=outputs,
                processed_videos=processed,
                extra={"source": "cli"},
            )
            dump_json(run_root / "run.json", meta.to_dict())
            log("done", {"status": "completed", "run_json": str(run_root / "run.json")})
            return run_id

        except Exception as e:
            # fail meta
            meta = RunMeta(
                type="predict",
                status="failed",
                run_id=run_id,
                model_id=cfg.model_id,
                backend=spec.backend,
                variant=spec.variant,
                created_at=_now_iso(),
                outputs=outputs,
                processed_videos=processed,
                extra={"error": repr(e)},
            )
            dump_json(run_root / "run.json", meta.to_dict())
            log("done", {"status": "failed", "error": repr(e)})
            raise
