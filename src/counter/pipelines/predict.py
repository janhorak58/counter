from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import time

import cv2
import numpy as np

from counter.config.schema import PredictConfig
from counter.domain.model_spec import load_models
from counter.domain.results import CountsResult, RunMeta
from counter.domain.types import CanonicalClass
from counter.io.export import dump_json, ensure_dir, save_counts_json
from counter.io.video import get_video_info
from counter.mapping.factory import MappingFactory
from counter.tracking.factory import TrackerFactory


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _write_log_line(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def _class_name(cid: int) -> str:
    try:
        return CanonicalClass(cid).name
    except Exception:
        return str(cid)


def _draw_boxes_and_hud(
    frame_bgr: np.ndarray,
    tracks: List[Any],
    frame_idx: int,
    frame_count: int,
    *,
    flash_every_n_frames: int = 12,
) -> np.ndarray:
    out = frame_bgr

    for t in tracks:
        try:
            x1, y1, x2, y2 = [int(v) for v in t.xyxy]
            cls = int(getattr(t, "cls", -1))
            conf = float(getattr(t, "conf", 0.0))
        except Exception:
            continue

        label = f"{_class_name(cls)} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # blikání: každých N snímků přepnout viditelnost
    if flash_every_n_frames > 0 and (frame_idx // flash_every_n_frames) % 2 == 0:
        msg = f"Frame {frame_idx}/{max(0, frame_count-1)}"
        cv2.putText(out, msg, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    return out


class PredictPipeline:
    def __init__(self, models_yaml: str | Path = "configs/models.yaml") -> None:
        self.models_yaml = str(models_yaml)

    def run(self, cfg: PredictConfig) -> str:
        models = load_models(self.models_yaml)
        if cfg.model_id not in models:
            raise KeyError(f"Unknown model_id: {cfg.model_id}. Available: {list(models.keys())[:15]}...")

        spec = models[cfg.model_id]

        run_id = f"{spec.backend}_{spec.variant}__{cfg.model_id}"
        run_dir = ensure_dir(Path(cfg.out_dir) / "predict" / run_id)
        pred_dir = ensure_dir(run_dir / "predict")

        log_path = pred_dir / "run.log"
        _write_log_line(log_path, f"[{_now_iso()}] START run_id={run_id} model_id={cfg.model_id} backend={spec.backend} variant={spec.variant}")
        _write_log_line(log_path, f"[{_now_iso()}] cfg: videos_dir={cfg.videos_dir} videos={cfg.videos} device={cfg.device} conf={cfg.conf} iou={cfg.iou} greyzone_px={cfg.greyzone_px} save_video={cfg.save_video}")

        mapper = MappingFactory.create(
            policy=spec.mapping_policy,
            class_map=spec.class_map,
            coco_ids=spec.coco_ids,
        )

        provider = TrackerFactory.create(
            spec=spec,
            device=cfg.device,
            conf=cfg.conf,
            iou=cfg.iou,
            tracking_cfg=cfg.tracking,
            mapping=mapper,
            logger=lambda s: _write_log_line(log_path, f"[{_now_iso()}] {s}"),
        )

        created_at = datetime.utcnow().isoformat()
        base_meta = RunMeta(
            run_id=run_id,
            created_at=created_at,
            model_id=cfg.model_id,
            backend=spec.backend,
            variant=spec.variant,
            weights=str(spec.weights or ""),
            mapping_policy=str(spec.mapping_policy or ""),
            thresholds={"conf": cfg.conf, "iou": cfg.iou},
            tracker={"type": cfg.tracking.type, "params": cfg.tracking.params, "tracker_yaml": cfg.tracking.tracker_yaml},
            line=asdict(cfg.line),
            greyzone_px=float(cfg.greyzone_px),
            video={},
            extra={"log": str(log_path)},
        ).to_dict()

        outputs: List[str] = []
        processed_videos: List[str] = []
        status = "completed"

        aggregate_in: Dict[int, int] = {}
        aggregate_out: Dict[int, int] = {}

        videos = cfg.videos or []
        if not videos:
            p = Path(cfg.videos_dir)
            if p.exists():
                videos = [f.name for f in sorted(p.glob("*.mp4"))]

        try:
            for vid in videos:
                vpath = Path(cfg.videos_dir) / vid
                if not vpath.exists():
                    _write_log_line(log_path, f"[{_now_iso()}] SKIP missing video={vpath}")
                    continue

                t0 = time.time()
                vinfo = get_video_info(str(vpath))

                cap = cv2.VideoCapture(str(vpath))
                if not cap.isOpened():
                    _write_log_line(log_path, f"[{_now_iso()}] ERROR cannot open video={vpath}")
                    continue

                writer = None
                out_video_path = None
                if cfg.save_video:
                    out_video_path = pred_dir / f"{Path(vid).stem}.pred.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(out_video_path),
                        fourcc,
                        float(vinfo.fps or 25.0),
                        (int(vinfo.width), int(vinfo.height)),
                    )

                from counter.counting.line_counter import LineCounter
                counter = LineCounter(
                    line_start=tuple(cfg.line.start),
                    line_end=tuple(cfg.line.end),
                    greyzone_px=float(cfg.greyzone_px),
                )

                frame_idx = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    tracks = provider.update(frame, frame_idx)
                    counter.update(tracks)

                    if writer is not None:
                        annotated = _draw_boxes_and_hud(frame, tracks, frame_idx, int(vinfo.frame_count or 0))
                        writer.write(annotated)

                    frame_idx += 1

                cap.release()
                if writer is not None:
                    writer.release()

                fin_in, fin_out = counter.get_counts()

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
                processed_videos.append(Path(vid).stem)

                if out_video_path is not None:
                    outputs.append(str(out_video_path))

                dt = max(1e-9, time.time() - t0)
                fps_eff = float(frame_idx / dt)
                _write_log_line(
                    log_path,
                    f"[{_now_iso()}] DONE video={vid} frames={frame_idx} sec={dt:.2f} fps={fps_eff:.2f} in_total={sum(fin_in.values())} out_total={sum(fin_out.values())} saved_video={bool(out_video_path)}",
                )

        except KeyboardInterrupt:
            status = "cancelled"
            _write_log_line(log_path, f"[{_now_iso()}] CANCELLED by KeyboardInterrupt")
        except Exception as e:
            status = "failed"
            _write_log_line(log_path, f"[{_now_iso()}] FAILED error={e}")
            raise
        finally:
            agg_meta = dict(base_meta)
            agg_meta["status"] = status
            agg_meta["processed_videos"] = processed_videos

            agg = CountsResult(
                video="__aggregate__",
                line_name=cfg.line.name,
                in_count=aggregate_in,
                out_count=aggregate_out,
                meta=agg_meta,
            )
            agg_file = pred_dir / "aggregate.counts.json"
            save_counts_json(agg_file, agg)
            outputs.append(str(agg_file))

            run_json = {
                "type": "predict",
                "status": status,
                "run_id": run_id,
                "model_id": cfg.model_id,
                "backend": spec.backend,
                "variant": spec.variant,
                "created_at": created_at,
                "outputs": outputs,
                "processed_videos": processed_videos,
                "extra": {"source": "predict_pipeline", "log": str(log_path)},
            }
            dump_json(run_dir / "run.json", run_json)
            _write_log_line(log_path, f"[{_now_iso()}] END status={status} outputs={len(outputs)}")

        return run_id
