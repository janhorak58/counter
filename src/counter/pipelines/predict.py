from __future__ import annotations

import base64
import logging
import time
from dataclasses import asdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from PIL import Image

from counter.config.loader import load_predict_config
from counter.config.schema import PredictConfig
from counter.counting.net_state import NetStateCounter
from counter.domain.model_spec import ModelSpec, load_models
from counter.domain.runmeta import CountsResult, RunMeta
from counter.io.export import dump_json, ensure_dir
from counter.io.video import VideoInfo, get_video_info
from counter.mapping.factory import create_mapper
from counter.tracking.factory import TrackerFactory
from counter.tracking.providers import TrackProvider


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"predict.{log_path.parent.parent.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def _draw_overlay(
    frame_bgr,
    tracks,
    line_xyxy: Tuple[int, int, int, int],
    frame_i_1based: int,
    frame_total: int,
    *,
    blink_every_n_frames: int = 10,
) -> None:
    """Draws tracks + counting line + blinking frame counter in-place."""
    x1, y1, x2, y2 = line_xyxy
    cv2.line(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)

    for t in tracks:
        bx1, by1, bx2, by2 = [int(v) for v in t.bbox_xyxy]
        cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

        label = f"id={t.track_id} c={t.class_id}"
        cv2.putText(
            frame_bgr,
            label,
            (bx1, max(0, by1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            label,
            (bx1, max(0, by1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    visible = True if blink_every_n_frames <= 0 else ((frame_i_1based // blink_every_n_frames) % 2) == 0
    if visible:
        txt = f"Frame {frame_i_1based}/{frame_total}"
        cv2.putText(frame_bgr, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame_bgr, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)


def _encode_preview_jpg(
    frame_bgr,
    tracks,
    line_xyxy: Tuple[int, int, int, int],
    frame_i_1based: int,
    frame_total: int,
    max_width: int = 960,
) -> str:
    """Returns base64-encoded JPEG for Streamlit preview."""
    h, w = frame_bgr.shape[:2]
    if max_width > 0 and w > max_width:
        scale = float(max_width) / float(w)
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
        x1, y1, x2, y2 = line_xyxy
        line_xyxy = (int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale))

    _draw_overlay(frame_bgr, tracks, line_xyxy, frame_i_1based, frame_total)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    bio = BytesIO()
    img.save(bio, format="JPEG", quality=85)
    return base64.b64encode(bio.getvalue()).decode("utf-8")


def dump_json_line(obj: Dict[str, Any]) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


def _emit_counts_event(
    cfg: PredictConfig,
    spec: ModelSpec,
    tracker: TrackProvider,
    counts: CountsResult,
    frame_bgr,
    tracks,
    frame_i_1based: int,
    frame_total: int,
) -> None:
    """Push a lightweight event to be polled by Streamlit UI (predict_events.jsonl)."""
    events_path = Path(cfg.export.out_dir) / spec.run_id / "predict" / "predict_events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)

    line_xyxy = (*cfg.line.start, *cfg.line.end)

    jpg_b64: Optional[str] = None
    if cfg.preview.enabled:
        try:
            jpg_b64 = _encode_preview_jpg(
                frame_bgr.copy(),
                tracks,
                line_xyxy,
                frame_i_1based,
                frame_total,
                max_width=cfg.preview.max_width,
            )
        except Exception:
            jpg_b64 = None

    event = {
        "ts": datetime.now().isoformat(),
        "video": counts.video,
        "frame": frame_i_1based,
        "frame_total": frame_total,
        "counts": {"in": counts.in_count, "out": counts.out_count},
        "jpg_b64": jpg_b64,
    }

    with open(events_path, "a", encoding="utf-8") as f:
        f.write(dump_json_line(event) + "\n")


class PredictPipeline:
    def __init__(self, models_yaml: str = "configs/models.yaml"):
        self.models_yaml = models_yaml
        self.models = load_models(models_yaml)

    def run(self, cfg: PredictConfig) -> Path:
        if cfg.model_id not in self.models:
            raise KeyError(f"Unknown model_id: {cfg.model_id}. Available: {list(self.models.keys())}")

        spec = self.models[cfg.model_id]

        run_dir = ensure_dir(Path(cfg.export.out_dir) / spec.run_id)
        predict_dir = ensure_dir(run_dir / "predict")

        logger = _setup_logger(predict_dir / "predict.log")
        logger.info("Starting predict")
        logger.info("run_id=%s model_id=%s backend=%s variant=%s", spec.run_id, cfg.model_id, spec.backend, spec.variant)
        logger.info("videos_dir=%s videos=%s", cfg.videos_dir, cfg.videos)
        logger.info("thresholds=%s tracking=%s export=%s", asdict(cfg.thresholds), asdict(cfg.tracking), asdict(cfg.export))

        provider = TrackerFactory.create(
            spec=spec,
            thresholds={"conf": cfg.conf, "iou": cfg.iou},
            tracker_cfg=cfg.tracking,
            device=cfg.device,
        )
        mapper = create_mapper(spec)

        video_counts: List[CountsResult] = []
        outputs: List[str] = []
        started_s = time.time()
        status = "completed"
        error_msg = ""

        try:
            for video in cfg.videos:
                video_path = Path(cfg.videos_dir) / video
                if not video_path.exists():
                    alt = Path(video)
                    if alt.exists():
                        video_path = alt
                    else:
                        logger.warning("Skipping missing video: %s", str(video_path))
                        continue

                vinfo: VideoInfo = get_video_info(str(video_path))
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    logger.warning("Failed to open video: %s", str(video_path))
                    continue

                logger.info("Processing video=%s fps=%.3f frames=%d", str(video_path), float(vinfo.fps), int(vinfo.frame_count))

                writer = None
                if cfg.export.save_video and vinfo.width > 0 and vinfo.height > 0:
                    out_vid_path = predict_dir / f"{video_path.stem}.pred.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(out_vid_path),
                        fourcc,
                        float(vinfo.fps) or 25.0,
                        (int(vinfo.width), int(vinfo.height)),
                    )
                    outputs.append(str(out_vid_path).replace("\\", "/"))
                    logger.info("Saving annotated video to %s", str(out_vid_path))

                provider.reset()
                mapper.reset()

                counter = NetStateCounter(
                    line_name=cfg.line.name,
                    start_xy=tuple(cfg.line.start),
                    end_xy=tuple(cfg.line.end),
                    greyzone_px=cfg.greyzone_px,
                )

                frame_idx = 0
                next_preview_frame = 1 if cfg.preview.enabled else -1
                frame_total = int(vinfo.frame_count) if int(vinfo.frame_count) > 0 else 1

                while True:
                    ok, frame_bgr = cap.read()
                    if not ok:
                        break
                    frame_idx += 1

                    raw_tracks = provider.update(frame_bgr)
                    tracks = mapper.map_tracks(raw_tracks)
                    counter.update(tracks)

                    if writer is not None:
                        _draw_overlay(frame_bgr, tracks, (*cfg.line.start, *cfg.line.end), frame_idx, frame_total)
                        writer.write(frame_bgr)

                    if cfg.preview.enabled and cfg.preview.every_n_frames > 0 and frame_idx >= next_preview_frame:
                        counts_tmp = CountsResult(
                            video=video_path.name,
                            line_name=cfg.line.name,
                            in_count=counter.in_counts(),
                            out_count=counter.out_counts(),
                            meta={},
                        )
                        _emit_counts_event(cfg, spec, provider, counts_tmp, frame_bgr, tracks, frame_idx, frame_total)
                        next_preview_frame = frame_idx + int(cfg.preview.every_n_frames)

                    if frame_idx % 500 == 0:
                        logger.info("Progress %s: frame %d/%d", video_path.stem, frame_idx, frame_total)

                cap.release()
                if writer is not None:
                    writer.release()

                counts = CountsResult(
                    video=video_path.stem,
                    line_name=cfg.line.name,
                    in_count=counter.in_counts(),
                    out_count=counter.out_counts(),
                    meta={
                        "spec": asdict(spec),
                        "cfg": cfg.model_dump(),
                        "video": asdict(vinfo),
                    },
                )
                out_counts = predict_dir / f"{video_path.stem}.counts.json"
                dump_json(out_counts, counts.to_dict())
                outputs.append(str(out_counts).replace("\\", "/"))

                video_counts.append(counts)

                logger.info(
                    "Done %s: IN_total=%d OUT_total=%d",
                    video_path.stem,
                    sum(int(v) for v in counts.in_count.values()),
                    sum(int(v) for v in counts.out_count.values()),
                )

            if video_counts:
                agg_in: Dict[str, int] = {}
                agg_out: Dict[str, int] = {}
                for c in video_counts:
                    for k, v in c.in_count.items():
                        agg_in[k] = agg_in.get(k, 0) + int(v)
                    for k, v in c.out_count.items():
                        agg_out[k] = agg_out.get(k, 0) + int(v)

                aggregate = CountsResult(
                    video="aggregate",
                    line_name=cfg.line.name,
                    in_count=agg_in,
                    out_count=agg_out,
                    meta={"spec": asdict(spec), "cfg": cfg.model_dump()},
                )
                out_agg = predict_dir / "aggregate.counts.json"
                dump_json(out_agg, aggregate.to_dict())
                outputs.append(str(out_agg).replace("\\", "/"))

        except Exception as e:
            status = "failed"
            error_msg = repr(e)
            logger.exception("Prediction failed: %s", error_msg)
            raise
        finally:
            run_meta = RunMeta(
                run_id=spec.run_id,
                created_at=datetime.now().isoformat(),
                model_id=cfg.model_id,
                backend=spec.backend,
                variant=spec.variant,
                weights=spec.weights or "",
                mapping_policy=spec.mapping_policy or "",
                thresholds={"conf": cfg.conf, "iou": cfg.iou},
                tracker={"type": cfg.tracking.type, "params": cfg.tracking.params, "tracker_yaml": cfg.tracking.tracker_yaml},
                line={"name": cfg.line.name, "start": cfg.line.start, "end": cfg.line.end},
                greyzone_px=cfg.greyzone_px,
                video={"videos_dir": cfg.videos_dir, "videos": cfg.videos},
                extra={
                    "source": "predict_pipeline",
                    "status": status,
                    "duration_s": float(time.time() - started_s),
                },
            )

            meta_dict = run_meta.to_dict()
            meta_dict["type"] = "predict"
            meta_dict["status"] = status
            meta_dict["outputs"] = outputs
            meta_dict["processed_videos"] = [c.video for c in video_counts]
            if error_msg:
                meta_dict["error"] = error_msg

            dump_json(run_dir / "run.json", meta_dict)
            logger.info("Finished predict status=%s duration_s=%.2f", status, meta_dict["extra"]["duration_s"])

        return run_dir


def run_predict_cli(config_path: str, models_yaml: str, *, model_id: Optional[str] = None, video: Optional[str] = None) -> Path:
    cfg = load_predict_config(config_path)
    if model_id:
        cfg.model_id = model_id
    if video:
        p = Path(video)
        if p.exists() and p.is_absolute():
            cfg.videos_dir = str(p.parent)
            cfg.videos = [p.name]
        else:
            cfg.videos = [video]
    pipe = PredictPipeline(models_yaml=models_yaml)
    return pipe.run(cfg)
