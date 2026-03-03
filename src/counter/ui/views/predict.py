from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from counter.core.config import load_models_registry
from counter.ui.services import configs, jobs, line_picker
from counter.ui.state import project_root

try:  # pragma: no cover
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:  # pragma: no cover
    from streamlit_drawable_canvas import st_canvas
except Exception:  # pragma: no cover
    st_canvas = None


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
_canvas_compat_checked = False
_canvas_compat_available = False


def _ensure_canvas_compat() -> bool:
    global _canvas_compat_checked, _canvas_compat_available
    if _canvas_compat_checked:
        return _canvas_compat_available

    _canvas_compat_checked = True
    if st_canvas is None or Image is None:
        _canvas_compat_available = False
        return _canvas_compat_available

    try:
        import streamlit.elements.image as st_image  # type: ignore
    except Exception:
        _canvas_compat_available = False
        return _canvas_compat_available

    if hasattr(st_image, "image_to_url"):
        _canvas_compat_available = True
        return _canvas_compat_available

    try:
        from streamlit.elements.lib import image_utils, layout_utils  # type: ignore

        def _compat_image_to_url(image, width, clamp, channels, output_format, image_id):
            layout = layout_utils.LayoutConfig(width=width)
            return image_utils.image_to_url(image, layout, clamp, channels, output_format, image_id)

        st_image.image_to_url = _compat_image_to_url
        _canvas_compat_available = True
    except Exception:
        _canvas_compat_available = False

    return _canvas_compat_available



def _list_videos(videos_dir: Path) -> List[str]:
    if not videos_dir.exists():
        return []
    return [p.name for p in sorted(videos_dir.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTS]



def _parse_coords_text(text: str) -> Optional[List[int]]:
    try:
        parts = [int(round(float(x.strip()))) for x in str(text).split(",") if x.strip()]
    except Exception:
        return None
    if len(parts) != 4:
        return None
    return parts



def _effective_videos_dir() -> str:
    raw = str(st.session_state.get("predict_videos_dir", "")).strip()
    if raw:
        return raw
    fallback = str(st.session_state.get("ui_videos_dir", "data/videos")).strip()
    return fallback or "data/videos"



def _seed_predict_form_from_dict(cfg: Dict[str, Any]) -> None:
    st.session_state["predict_model_id"] = str(cfg.get("model_id", ""))
    st.session_state["predict_device"] = str(cfg.get("device", "cpu"))
    st.session_state["predict_conf"] = float(((cfg.get("thresholds") or {}).get("conf", 0.35)))
    st.session_state["predict_iou"] = float(((cfg.get("thresholds") or {}).get("iou", 0.50)))

    tracking = cfg.get("tracking") or {}
    st.session_state["predict_tracking_type"] = str(tracking.get("type", "bytetrack"))
    st.session_state["predict_tracker_yaml"] = str(tracking.get("tracker_yaml", "tracker.bytetrack.yaml"))

    videos_dir = str(cfg.get("videos_dir", "")).strip() or str(st.session_state.get("ui_videos_dir", "data/videos"))
    st.session_state["predict_videos_dir"] = videos_dir
    st.session_state["predict_selected_videos"] = list(cfg.get("videos", []))

    st.session_state["predict_debug"] = bool(cfg.get("debug", False))
    st.session_state["predict_save_video"] = bool(cfg.get("save_video", True))

    export = cfg.get("export") or {}
    st.session_state["predict_export_save_video"] = bool(export.get("save_video", True))
    st.session_state["predict_export_save_raw"] = bool(export.get("save_raw", True))
    st.session_state["predict_export_save_counts_json"] = bool(export.get("save_counts_json", True))

    preview = cfg.get("preview") or {}
    st.session_state["predict_preview_enabled"] = bool(preview.get("enabled", False))
    st.session_state["predict_preview_every_n_frames"] = int(preview.get("every_n_frames", 5))
    st.session_state["predict_preview_max_width"] = int(preview.get("max_width", 1200))

    line = cfg.get("line") or {}
    coords = line.get("coords") or [727, 459, 1133, 549]
    default_resolution = line.get("default_resolution") or [1920, 1080]

    st.session_state["predict_line_name"] = str(line.get("name", "main_line"))
    st.session_state["ui_predict_coords_text"] = ",".join(str(int(round(float(x)))) for x in coords[:4])
    st.session_state["predict_line_default_w"] = int(default_resolution[0])
    st.session_state["predict_line_default_h"] = int(default_resolution[1])

    st.session_state["predict_greyzone_px"] = float(cfg.get("greyzone_px", 10.0))
    st.session_state["predict_probe_frames"] = int(cfg.get("probe_frames", 0))
    st.session_state["predict_oscillation_window_frames"] = int(cfg.get("oscillation_window_frames", 0))
    st.session_state["predict_trajectory_len"] = int(cfg.get("trajectory_len", 40))



def _seed_predict_form_from_yaml(path: Path) -> None:
    _seed_predict_form_from_dict(configs.load_yaml_dict(path))



def _build_predict_dict() -> Dict[str, Any]:
    coords = _parse_coords_text(st.session_state.get("ui_predict_coords_text", "")) or [727, 459, 1133, 549]

    out: Dict[str, Any] = {
        "run_id": str(st.session_state.get("predict_model_id", "ui_predict")).replace("/", "_").replace(" ", "_"),
        "model_id": str(st.session_state.get("predict_model_id", "")),
        "output_dir": "runs/predict",
        "device": str(st.session_state.get("predict_device", "cpu")),
        "thresholds": {
            "conf": float(st.session_state.get("predict_conf", 0.35)),
            "iou": float(st.session_state.get("predict_iou", 0.50)),
        },
        "tracking": {
            "type": str(st.session_state.get("predict_tracking_type", "bytetrack")),
            "tracker_yaml": str(st.session_state.get("predict_tracker_yaml", "tracker.bytetrack.yaml")),
            "params": {},
        },
        "videos_dir": _effective_videos_dir(),
        "videos": list(st.session_state.get("predict_selected_videos", [])),
        "probe_frames": int(st.session_state.get("predict_probe_frames", 0)),
        "export": {
            "save_video": bool(st.session_state.get("predict_export_save_video", True)),
            "save_raw": bool(st.session_state.get("predict_export_save_raw", True)),
            "save_counts_json": bool(st.session_state.get("predict_export_save_counts_json", True)),
        },
        "debug": bool(st.session_state.get("predict_debug", False)),
        "preview": {
            "enabled": bool(st.session_state.get("predict_preview_enabled", False)),
            "every_n_frames": int(st.session_state.get("predict_preview_every_n_frames", 5)),
            "max_width": int(st.session_state.get("predict_preview_max_width", 1200)),
        },
        "line": {
            "name": str(st.session_state.get("predict_line_name", "main_line")),
            "coords": [int(x) for x in coords],
            "default_resolution": [
                int(st.session_state.get("predict_line_default_w", 1920)),
                int(st.session_state.get("predict_line_default_h", 1080)),
            ],
        },
        "greyzone_px": float(st.session_state.get("predict_greyzone_px", 10.0)),
        "save_video": bool(st.session_state.get("predict_save_video", True)),
        "oscillation_window_frames": int(st.session_state.get("predict_oscillation_window_frames", 0)),
        "trajectory_len": int(st.session_state.get("predict_trajectory_len", 40)),
    }
    return out



def _apply_pending_updates() -> None:
    if "predict_line_pending_resolution" in st.session_state:
        w, h = st.session_state.pop("predict_line_pending_resolution")
        st.session_state["predict_line_default_w"] = int(w)
        st.session_state["predict_line_default_h"] = int(h)

    if "predict_coords_pending" in st.session_state:
        st.session_state["ui_predict_coords_text"] = str(st.session_state.pop("predict_coords_pending"))

    if "predict_yaml_pending" in st.session_state:
        text = str(st.session_state.pop("predict_yaml_pending"))
        st.session_state["ui_predict_yaml_text"] = text
        st.session_state["ui_predict_yaml_applied"] = text

    if "predict_form_pending_from_yaml" in st.session_state:
        cfg = st.session_state.pop("predict_form_pending_from_yaml")
        if isinstance(cfg, dict):
            _seed_predict_form_from_dict(cfg)



def _ensure_loaded_from_config(predict_cfg_path: Path) -> None:
    current = str(predict_cfg_path)
    loaded = str(st.session_state.get("ui_predict_loaded_from", ""))
    initialized = bool(st.session_state.get("predict_form_initialized", False))

    if not initialized or loaded != current:
        _seed_predict_form_from_yaml(predict_cfg_path)
        st.session_state["predict_form_initialized"] = True
        st.session_state["ui_predict_loaded_from"] = current
        st.session_state["ui_predict_yaml_text"] = ""
        st.session_state["ui_predict_yaml_applied"] = ""



def _sync_yaml_to_form_if_needed() -> None:
    if not bool(st.session_state.get("ui_predict_yaml_to_form_sync", True)):
        return

    # Prevent sync tug-of-war: when form->YAML auto-sync is enabled,
    # form edits are the source of truth.
    if bool(st.session_state.get("ui_predict_yaml_sync", True)):
        return

    text = str(st.session_state.get("ui_predict_yaml_text", "")).strip()
    if not text:
        return

    if text == str(st.session_state.get("ui_predict_yaml_applied", "")):
        return

    ok, _ = configs.validate_predict_yaml_text(text)
    if not ok:
        return

    cfg = configs.parse_yaml_text(text)
    _seed_predict_form_from_dict(cfg)
    st.session_state["ui_predict_yaml_applied"] = text



def _render_line_picker() -> None:
    st.subheader("Line Picker")
    st.caption("Load the first frame and set two points for line.coords.")

    videos_dir = Path(_effective_videos_dir())
    video_options = _list_videos(videos_dir)

    if not video_options:
        st.info(f"No videos found in: {videos_dir}")
        return

    selected = st.selectbox("Video for line picking", options=video_options, key="predict_line_picker_video")

    if st.button("Load first frame", key="predict_load_first_frame"):
        try:
            frame_rgb, w, h = line_picker.load_first_frame_rgb(videos_dir / selected)
            st.session_state["ui_line_frame_rgb"] = frame_rgb
            st.session_state["ui_line_frame_video"] = selected
            st.session_state["ui_line_frame_size"] = (w, h)
            st.session_state["predict_line_pending_resolution"] = (int(w), int(h))
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    frame_rgb = st.session_state.get("ui_line_frame_rgb")
    frame_video = st.session_state.get("ui_line_frame_video")

    if frame_rgb is None:
        return

    if frame_video != selected:
        st.caption(f"Loaded frame is from: {frame_video}. Click 'Load first frame' for current selection.")

    canvas_points: List[tuple[int, int]] = []
    if _ensure_canvas_compat():
        try:
            canvas = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=3,
                stroke_color="#ff4b4b",
                background_image=Image.fromarray(frame_rgb),
                update_streamlit=True,
                height=int(frame_rgb.shape[0]),
                width=int(frame_rgb.shape[1]),
                drawing_mode="point",
                point_display_radius=6,
                key="predict_line_canvas",
            )
            canvas_points = line_picker.extract_two_points(canvas.json_data)
        except Exception as exc:
            st.warning(f"Canvas picker unavailable ({type(exc).__name__}). Use manual point inputs below.")
    else:
        st.info("Canvas picker is unavailable in this environment. Use manual point inputs below.")

    if len(canvas_points) >= 2:
        (x1, y1), (x2, y2) = canvas_points[0], canvas_points[1]
        st.success(f"Detected points: ({x1}, {y1}) and ({x2}, {y2})")
        if st.button("Use canvas points as line.coords", key="predict_use_line_points"):
            st.session_state["predict_coords_pending"] = f"{x1},{y1},{x2},{y2}"
            st.rerun()

    st.markdown("**Manual points**")
    coords_now = _parse_coords_text(st.session_state.get("ui_predict_coords_text", "")) or [0, 0, 0, 0]
    h, w = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])
    max_x = max(0, w - 1)
    max_y = max(0, h - 1)

    c1, c2 = st.columns(2)
    with c1:
        x1 = int(
            st.number_input(
                "x1",
                min_value=0,
                max_value=max_x,
                value=min(max(coords_now[0], 0), max_x),
                step=1,
            )
        )
        y1 = int(
            st.number_input(
                "y1",
                min_value=0,
                max_value=max_y,
                value=min(max(coords_now[1], 0), max_y),
                step=1,
            )
        )
    with c2:
        x2 = int(
            st.number_input(
                "x2",
                min_value=0,
                max_value=max_x,
                value=min(max(coords_now[2], 0), max_x),
                step=1,
            )
        )
        y2 = int(
            st.number_input(
                "y2",
                min_value=0,
                max_value=max_y,
                value=min(max(coords_now[3], 0), max_y),
                step=1,
            )
        )

    if st.button("Use manual points as line.coords", key="predict_use_manual_line_points"):
        st.session_state["predict_coords_pending"] = f"{x1},{y1},{x2},{y2}"
        st.session_state["predict_line_pending_resolution"] = (w, h)
        st.rerun()



def render() -> None:
    st.header("Predict")

    predict_cfg_path = Path(st.session_state["ui_predict_config_path"])
    models_cfg_path = Path(st.session_state["ui_models_config_path"])

    _apply_pending_updates()
    _ensure_loaded_from_config(predict_cfg_path)
    _sync_yaml_to_form_if_needed()

    if st.button("Reload from predict.yaml", key="predict_reload_yaml"):
        _seed_predict_form_from_yaml(predict_cfg_path)
        st.session_state["ui_predict_loaded_from"] = str(predict_cfg_path)
        st.session_state["ui_predict_yaml_text"] = ""
        st.session_state["ui_predict_yaml_applied"] = ""
        st.success("Predict form reloaded from YAML.")
        st.rerun()

    model_options: List[str] = []
    try:
        registry = load_models_registry(models_cfg_path)
        model_options = sorted(registry.models.keys())
    except Exception as exc:
        st.warning(f"Failed to load models registry: {exc}")

    if model_options and st.session_state.get("predict_model_id") not in model_options:
        st.session_state["predict_model_id"] = model_options[0]

    col1, col2 = st.columns(2)

    with col1:
        st.selectbox("model_id", options=model_options or [""], key="predict_model_id")
        st.text_input("device", key="predict_device")
        st.number_input("thresholds.conf", min_value=0.0, max_value=1.0, step=0.01, key="predict_conf")
        st.number_input("thresholds.iou", min_value=0.0, max_value=1.0, step=0.01, key="predict_iou")
        st.selectbox("tracking.type", options=["none", "bytetrack"], key="predict_tracking_type")
        st.text_input("tracking.tracker_yaml", key="predict_tracker_yaml")
        st.text_input("videos_dir", key="predict_videos_dir")

        vids = _list_videos(Path(_effective_videos_dir()))
        st.multiselect("videos", options=vids, key="predict_selected_videos")

    with col2:
        st.checkbox("debug", key="predict_debug")
        st.checkbox("save_video (top-level)", key="predict_save_video")
        st.checkbox("export.save_video", key="predict_export_save_video")
        st.checkbox("export.save_raw", key="predict_export_save_raw")
        st.checkbox("export.save_counts_json", key="predict_export_save_counts_json")
        st.checkbox("preview.enabled", key="predict_preview_enabled")
        st.number_input("preview.every_n_frames", min_value=1, step=1, key="predict_preview_every_n_frames")
        st.number_input("preview.max_width", min_value=320, step=10, key="predict_preview_max_width")
        st.text_input("line.name", key="predict_line_name")
        st.text_input("line.coords (x1,y1,x2,y2)", key="ui_predict_coords_text")
        st.number_input("line.default_resolution.width", min_value=1, step=1, key="predict_line_default_w")
        st.number_input("line.default_resolution.height", min_value=1, step=1, key="predict_line_default_h")
        st.number_input("greyzone_px", min_value=0.0, step=1.0, key="predict_greyzone_px")
        st.number_input("probe_frames", min_value=0, step=1, key="predict_probe_frames")
        st.number_input("oscillation_window_frames", min_value=0, step=1, key="predict_oscillation_window_frames")
        st.number_input("trajectory_len", min_value=1, step=1, key="predict_trajectory_len")

    _render_line_picker()

    generated = configs.dump_yaml_text(_build_predict_dict())
    if st.session_state.get("ui_predict_yaml_sync", True) or not st.session_state.get("ui_predict_yaml_text", ""):
        st.session_state["ui_predict_yaml_text"] = generated
        st.session_state["ui_predict_yaml_applied"] = generated

    st.subheader("Predict YAML Editor")
    s1, s2 = st.columns(2)
    with s1:
        st.checkbox("Sync form -> YAML", key="ui_predict_yaml_sync")
    with s2:
        st.checkbox("Sync YAML -> form", key="ui_predict_yaml_to_form_sync")

    st.text_area("Edit YAML", key="ui_predict_yaml_text", height=360)

    b1, b2, b3, b4, b5 = st.columns(5)

    with b1:
        if st.button("Regenerate from form", key="predict_regenerate_yaml"):
            st.session_state["predict_yaml_pending"] = generated
            st.rerun()

    with b2:
        if st.button("Apply YAML to form", key="predict_apply_yaml_to_form"):
            ok, msg = configs.validate_predict_yaml_text(st.session_state["ui_predict_yaml_text"])
            if not ok:
                st.error(msg)
            else:
                st.session_state["predict_form_pending_from_yaml"] = configs.parse_yaml_text(
                    st.session_state["ui_predict_yaml_text"]
                )
                st.session_state["ui_predict_yaml_applied"] = str(st.session_state["ui_predict_yaml_text"])
                st.rerun()

    with b3:
        if st.button("Validate YAML", key="predict_validate_yaml"):
            ok, msg = configs.validate_predict_yaml_text(st.session_state["ui_predict_yaml_text"])
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    with b4:
        if st.button("Save predict.yaml", key="predict_save_yaml"):
            try:
                obj = configs.parse_yaml_text(st.session_state["ui_predict_yaml_text"])
                configs.write_yaml_file(predict_cfg_path, obj)
                st.success(f"Saved: {predict_cfg_path}")
            except Exception as exc:
                st.error(str(exc))

    job = st.session_state.get("ui_jobs", {}).get("predict")
    running = bool(job is not None and getattr(job, "status", "") == "running")

    with b5:
        if st.button("Run Predict", key="predict_run", disabled=running):
            ok, msg = configs.validate_predict_yaml_text(st.session_state["ui_predict_yaml_text"])
            if not ok:
                st.error(msg)
            else:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    suffix=".yaml",
                    prefix="counter_ui_predict_",
                    dir="/tmp",
                    delete=False,
                ) as f:
                    f.write(st.session_state["ui_predict_yaml_text"])
                    tmp_cfg = Path(f.name)

                cmd = [
                    sys.executable,
                    "-m",
                    "counter.predict",
                    "--config",
                    str(tmp_cfg),
                    "--models",
                    str(models_cfg_path),
                ]
                handle = jobs.start_job(kind="predict", command=cmd, cwd=project_root())
                st.session_state["ui_jobs"]["predict"] = handle
                st.success("Predict job started.")
