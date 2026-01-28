from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from counter.predict.tracking.providers import TrackProvider
from counter.predict.mapping.track_mapper import TrackMapper
from counter.predict.counting.counter import TrackCounter
from counter.predict.visual.renderer import FrameRenderer

try:  # pragma: no cover
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

from counter.core.io import VideoInfo, get_video_info, iter_frames
from counter.core.schema import PredictConfig
from counter.predict.logic.hist import mapped_hist, raw_hist, raw_id_hist
from counter.predict.logic.output import build_counts_object, now_iso, write_counts_json
from counter.core.pipeline.base import StageContext


def _discover_videos(videos_dir: Path) -> List[str]:
    """Return video filenames in a directory by common extensions."""
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    return [p.name for p in sorted(videos_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def _open_writer(path: Path, *, fps: float, size: tuple[int, int]):
    """Open OpenCV video writer for the requested path."""
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for save_video=True")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, float(fps), size)


def _scale_line(coords, src_w: int, src_h: int, dst_w: int, dst_h: int):
    """Scale line coordinates from a base resolution to a target resolution."""
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    x1, y1, x2, y2 = coords
    return [
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    ]


@dataclass
class PredictVideos:
    """Stage that runs tracking and counting over videos."""

    name: str = "predict_videos"

    def run(self, ctx: StageContext) -> None:
        cfg: PredictConfig = ctx.cfg
        spec = ctx.state["model_spec"]
        run_id: str = ctx.state["run_id"]
        run_root: Path = ctx.state["run_root"]
        predict_dir: Path = ctx.state["predict_dir"]
        log = ctx.assets.get("log")

        provider: TrackProvider = ctx.assets["provider"]
        mapper: TrackMapper = ctx.assets["mapper"]
        counter: TrackCounter = ctx.assets["counter"]
        renderer: FrameRenderer = ctx.assets["renderer"]

        videos_dir = Path(cfg.videos_dir)
        videos = list(cfg.videos)
        if not videos:
            videos = _discover_videos(videos_dir)

        if not videos:
            raise FileNotFoundError(f"No videos found in: {videos_dir}")

        out_counts_paths: List[str] = []

        preview_enabled = bool(cfg.preview.enabled)
        preview_every = int(cfg.preview.every_n_frames or 1)
        preview_max_w = int(cfg.preview.max_width or 0)

        for vid in videos:
            video_path = videos_dir / vid
            if not video_path.exists():
                log("video_missing", {"video": str(video_path)})
                continue

            vinfo: VideoInfo = get_video_info(str(video_path))
            # Scale the counting line to match the video resolution.
            base_w, base_h = cfg.line.default_resolution
            line_coords = _scale_line(
                cfg.line.coords,
                src_w=base_w,
                src_h=base_h,
                dst_w=vinfo.width,
                dst_h=vinfo.height,
            )
            provider.reset()
            counter.reset(video_resolution=(vinfo.width, vinfo.height))

            counter.line = tuple(line_coords)
            renderer.line = tuple(line_coords)

            writer = None
            out_video_path: Optional[Path] = None
            if cfg.save_video:
                out_video_path = predict_dir / f"{Path(vid).stem}.pred.mp4"
                writer = _open_writer(out_video_path, fps=vinfo.fps or 25.0, size=(vinfo.width, vinfo.height))

            log(
                "video_start",
                {"video": vid, "fps": vinfo.fps, "frames": vinfo.frame_count, "size": [vinfo.width, vinfo.height]},
            )

            for frame_idx, frame_bgr in iter_frames(str(video_path)):
                raw_tracks = provider.update(frame_bgr)
                mapped_tracks = mapper.map_tracks(raw_tracks)
                counter.update(mapped_tracks)

                # Overlay statistics on output or preview frames.
                if cfg.save_video or preview_enabled:
                    in_c, out_c = counter.snapshot_counts()
                    renderer.render(
                        frame_bgr,
                        tracks=mapped_tracks,
                        raw_tracks=raw_tracks,
                        in_counts=in_c,
                        out_counts=out_c,
                        frame_idx=int(frame_idx),
                        fps=float(vinfo.fps) if vinfo.fps else None,
                        total_frames=int(vinfo.frame_count) if vinfo.frame_count else None,
                    )

                if writer is not None:
                    writer.write(frame_bgr)

                if preview_enabled and cv2 is not None and (frame_idx % preview_every == 0):
                    prev = renderer.preview_frame(frame_bgr, max_width=preview_max_w)
                    cv2.imshow(f"predict::{run_id}", prev)
                    key = cv2.waitKey(1)
                    if key in (ord("q"), 27):
                        preview_enabled = False
                        cv2.destroyAllWindows()

                if frame_idx % 200 == 0:
                    log(
                        "frame_sample",
                        {
                            "video": vid,
                            "frame": int(frame_idx),
                            "raw_hist": raw_hist(raw_tracks),
                            "raw_id_hist": raw_id_hist(raw_tracks),
                            "mapped_hist": mapped_hist(mapped_tracks),
                        },
                    )

            if writer is not None:
                writer.release()

            if preview_enabled and cv2 is not None:
                cv2.destroyAllWindows()

            in_final, out_final = counter.finalize()

            counts_path = predict_dir / f"{Path(vid).stem}.counts.json"
            meta = {
                "run_id": run_id,
                "created_at": now_iso(),
                "model_id": cfg.model_id,
                "backend": spec.backend,
                "variant": spec.variant,
                "weights": spec.weights,
                "mapping": spec.mapping._get_dict() if spec.mapping is not None else None,
                "thresholds": {"conf": cfg.thresholds.conf, "iou": cfg.thresholds.iou},
                "tracker": {"type": cfg.tracking.type, "params": cfg.tracking.params},
                "line": {"name": cfg.line.name, "coords": list(cfg.line.coords)},
                "greyzone_px": float(cfg.greyzone_px),
                "video": {
                    "path": str(video_path),
                    "fps": float(vinfo.fps) if vinfo.fps is not None else None,
                    "frame_count": int(vinfo.frame_count),
                    "width": int(vinfo.width),
                    "height": int(vinfo.height),
                },
                "outputs": {
                    "counts_json": str(counts_path),
                    "pred_video": str(out_video_path) if out_video_path is not None else None,
                },
            }

            obj = build_counts_object(
                video=vid,
                line_name=cfg.line.name,
                in_count=in_final,
                out_count=out_final,
                meta=meta,
            )
            write_counts_json(counts_path, obj)
            out_counts_paths.append(str(counts_path))

            log(
                "video_done",
                {"video": vid, "counts": {"in": in_final, "out": out_final}, "counts_path": str(counts_path)},
            )

        ctx.state["counts_paths"] = out_counts_paths
