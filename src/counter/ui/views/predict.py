from __future__ import annotations

import json
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from counter.core.config import load_models_registry
from counter.core.types import CanonicalClass
from counter.predict.mapping.pretrained import INTERMEDIATE_NAMES
from counter.ui.services import configs, discovery, jobs, line_picker, uploads
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
_CROSSING_EVENTS = {
    "Line crossed IN": "Dovnitř",
    "Line crossed OUT": "Ven",
}
_DEFAULT_MODEL_ID = "yolo_pretrained/yolo11n"
_DEFAULT_VIDEO_NAME = "vid_test.mp4"
_VIDEO_FORMATS = {
    ".mp4": "video/mp4",
    ".mov": "video/mp4",
    ".avi": "video/avi",
    ".mkv": "video/mp4",
}
_DEFAULT_LINE_COORDS = [727, 459, 1133, 549]
_DEFAULT_LINE_RESOLUTION = [1920, 1080]
_canvas_compat_checked = False
_canvas_compat_available = False


# ---------------------------------------------------------------------------
# Job status helpers (shared with render_page)
# ---------------------------------------------------------------------------

def _status_label(status: str) -> str:
    return {
        "running": "běží",
        "completed": "dokončeno",
        "failed": "selhalo",
        "cancelled": "zrušeno",
    }.get(status, status or "neznámý")


def _poll_predict_job() -> None:
    state_jobs = st.session_state.get("ui_jobs", {})
    job = state_jobs.get("predict")
    if job is None:
        return

    jobs.poll_job(job)
    if job.output_path is None and job.status in {"completed", "failed", "cancelled"}:
        job.output_path = jobs.guess_output_path_from_logs(job.logs, cwd=job.cwd)

    state_jobs["predict"] = job


def _progress_from_logs(logs: list[str], *, status: str) -> tuple[float, str]:
    if status == "completed":
        return 1.0, "Predikce dokončena."

    current_frame = 0
    total_frames = 0
    video_name = ""

    for line in logs:
        try:
            rec = json.loads(line)
        except Exception:
            continue

        event = str(rec.get("event", ""))
        if event == "video_start":
            current_frame = 0
            total_frames = int(rec.get("frames", 0) or 0)
            video_name = str(rec.get("video", "")).strip()
        elif event == "frame_sample":
            current_frame = max(current_frame, int(rec.get("frame", 0) or 0))
            video_name = str(rec.get("video", video_name)).strip()
        elif event == "video_done" and total_frames > 0:
            current_frame = total_frames
            video_name = str(rec.get("video", video_name)).strip()

    if total_frames > 0:
        progress = min(max(current_frame / float(total_frames), 0.0), 1.0)
        label = f"{video_name or 'Video'}: {current_frame} / {total_frames} snímků"
        return progress, label

    if status == "running":
        return 0.0, "Inicializace predikce..."
    if status == "failed":
        return 0.0, "Predikce selhala."
    if status == "cancelled":
        return 0.0, "Predikce byla zrušena."
    return 0.0, "Predikce čeká na spuštění."


def _job_summary(job: Any) -> list[str]:
    status = str(getattr(job, "status", ""))
    logs = list(getattr(job, "logs", []) or [])

    current_frame = 0
    total_frames = 0
    video_name = ""
    last_event = ""
    device_line = ""

    for line in logs:
        try:
            rec = json.loads(line)
        except Exception:
            continue

        event = str(rec.get("event", ""))
        if event == "device_selected":
            device_line = f"Zařízení: `{rec.get('device', '?')}`"
        elif event == "device_fallback":
            device_line = (
                f"Zařízení: `{rec.get('effective_device', '?')}` "
                f"(místo `{rec.get('requested_device', '?')}`; {rec.get('reason', '')})"
            )
        elif event == "video_start":
            current_frame = 0
            total_frames = int(rec.get("frames", 0) or 0)
            video_name = str(rec.get("video", "")).strip()
            last_event = f"Zpracovává se video `{video_name}`."
        elif event == "frame_sample":
            current_frame = max(current_frame, int(rec.get("frame", 0) or 0))
            video_name = str(rec.get("video", video_name)).strip()
        elif event == "video_done":
            video_name = str(rec.get("video", video_name)).strip()
            last_event = f"Video `{video_name}` bylo dokončeno."
        elif event in {"Line crossed IN", "Line crossed OUT"}:
            direction = "dovnitř" if event.endswith("IN") else "ven"
            cls = rec.get("class_id", "?")
            frame_idx = int(rec.get("frame_idx", 0) or 0)
            last_event = f"Poslední průchod: {direction}, třída `{cls}`, snímek {frame_idx}."

    if status == "completed" and total_frames > 0:
        current_frame = total_frames
        if not last_event:
            last_event = "Predikce byla dokončena."
    elif status == "failed" and not last_event:
        last_event = "Predikce skončila chybou."
    elif status == "cancelled" and not last_event:
        last_event = "Predikce byla zrušena."

    lines = [f"Stav: **{_status_label(status)}**"]
    if device_line:
        lines.append(device_line)
    if video_name:
        lines.append(f"Video: `{video_name}`")
    if total_frames > 0:
        percent = int(round((current_frame / float(total_frames)) * 100))
        lines.append(f"Průběh: {current_frame} / {total_frames} snímků ({percent} %)")
    if last_event:
        lines.append(last_event)

    if status == "running" and getattr(job, "started_at", None):
        elapsed_s = max(0, int(time.time() - float(job.started_at)))
        minutes, seconds = divmod(elapsed_s, 60)
        hours, minutes = divmod(minutes, 60)
        lines.append(f"Uplynulý čas: {hours:02d}:{minutes:02d}:{seconds:02d}")

    return lines


def _render_predict_job_status() -> None:
    job = st.session_state.get("ui_jobs", {}).get("predict")
    st.subheader("Stav predikce")

    if job is None:
        st.info("Predikce zatím nebyla spuštěna.")
        return

    status = str(getattr(job, "status", ""))
    progress_value, progress_text = _progress_from_logs(job.logs, status=status)

    info_col, action_col = st.columns([5, 1])
    with info_col:
        st.progress(progress_value, text=progress_text)
        for line in _job_summary(job):
            st.write(line)
    with action_col:
        if status == "running":
            st.write("")
            if st.button("Zastavit", key="cancel_predict", use_container_width=True):
                jobs.cancel_job(job)
                st.session_state["ui_jobs"]["predict"] = job

    with st.expander("Logy", expanded=False):
        logs_text = "\n".join(job.logs[-200:]) if job.logs else ""
        st.code(logs_text or "(zatím bez logů)", language="text")


@st.fragment(run_every=1)
def _render_predict_job_status_live() -> None:
    _poll_predict_job()
    _render_predict_job_status()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_videos(videos_dir: Path) -> List[str]:
    if not videos_dir.exists():
        return []
    return [p.name for p in sorted(videos_dir.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTS]


def _effective_videos_dir() -> str:
    raw = str(st.session_state.get("predict_videos_dir", "")).strip()
    if raw:
        return raw
    fallback = str(st.session_state.get("ui_videos_dir", "data/videos")).strip()
    return fallback or "data/videos"


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _class_label(class_id: Any) -> str:
    try:
        return CanonicalClass(int(class_id)).name.lower().replace("_", " ")
    except Exception:
        try:
            return str(INTERMEDIATE_NAMES[int(class_id)])
        except Exception:
            return str(class_id)


def _format_timecode(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"

    total = max(0.0, float(seconds))
    minutes, secs = divmod(total, 60.0)
    hours, minutes = divmod(int(minutes), 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes:02d}:{secs:05.2f}"


def _ensure_defaults(predict_cfg_path: Path, model_options: List[str], video_options: List[str]) -> None:
    current = str(predict_cfg_path)
    loaded = str(st.session_state.get("ui_predict_minimal_loaded_from", ""))

    if loaded != current:
        cfg = configs.load_yaml_dict(predict_cfg_path)
        initial_model_id = str(cfg.get("model_id", _DEFAULT_MODEL_ID))
        if _DEFAULT_MODEL_ID in model_options:
            initial_model_id = _DEFAULT_MODEL_ID
        st.session_state["predict_model_id"] = initial_model_id

        videos_dir = str(cfg.get("videos_dir", "")).strip() or str(st.session_state.get("ui_videos_dir", "data/videos"))
        st.session_state["predict_videos_dir"] = videos_dir

        configured_videos = cfg.get("videos", [])
        first_video = (
            configured_videos[0]
            if isinstance(configured_videos, list) and configured_videos
            else _DEFAULT_VIDEO_NAME
        )
        if _DEFAULT_VIDEO_NAME in video_options:
            first_video = _DEFAULT_VIDEO_NAME
        st.session_state["predict_selected_video"] = str(first_video)
        st.session_state["predict_render_every_n_frames"] = int(cfg.get("render_every_n_frames", 5) or 5)
        line_cfg = cfg.get("line", {}) or {}
        coords = line_cfg.get("coords") or _DEFAULT_LINE_COORDS
        resolution = line_cfg.get("default_resolution") or _DEFAULT_LINE_RESOLUTION
        st.session_state["predict_line_x1"] = int(coords[0])
        st.session_state["predict_line_y1"] = int(coords[1])
        st.session_state["predict_line_x2"] = int(coords[2])
        st.session_state["predict_line_y2"] = int(coords[3])
        st.session_state["predict_line_default_w"] = int(resolution[0])
        st.session_state["predict_line_default_h"] = int(resolution[1])
        st.session_state["ui_predict_minimal_loaded_from"] = current

    if model_options:
        if _DEFAULT_MODEL_ID in model_options and st.session_state.get("predict_model_id") not in model_options:
            st.session_state["predict_model_id"] = _DEFAULT_MODEL_ID
        elif st.session_state.get("predict_model_id") not in model_options:
            st.session_state["predict_model_id"] = model_options[0]

    if video_options:
        if _DEFAULT_VIDEO_NAME in video_options and st.session_state.get("predict_selected_video") not in video_options:
            st.session_state["predict_selected_video"] = _DEFAULT_VIDEO_NAME
        elif st.session_state.get("predict_selected_video") not in video_options:
            st.session_state["predict_selected_video"] = video_options[0]


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


def _build_predict_dict(predict_cfg_path: Path) -> Dict[str, Any]:
    base_cfg = deepcopy(configs.load_yaml_dict(predict_cfg_path))
    model_id = str(st.session_state.get("predict_model_id", "")).strip()
    video_name = str(st.session_state.get("predict_selected_video", "")).strip()
    runs_dir = str(st.session_state.get("ui_runs_predict_dir", project_root() / "runs" / "predict"))

    out: Dict[str, Any] = dict(base_cfg)
    out["run_id"] = f"ui_{model_id.replace('/', '_')}__{Path(video_name).stem}"
    out["model_id"] = model_id
    out["output_dir"] = runs_dir
    out["videos_dir"] = _effective_videos_dir()
    out["videos"] = [video_name]
    out["save_video"] = True
    out["debug"] = False
    out["render_every_n_frames"] = max(1, int(st.session_state.get("predict_render_every_n_frames", 5) or 5))

    preview = dict(out.get("preview") or {})
    preview["enabled"] = False
    out["preview"] = preview

    export = dict(out.get("export") or {})
    export["out_dir"] = runs_dir
    export["save_video"] = True
    export["save_counts_json"] = True
    export.setdefault("save_raw", True)
    out["export"] = export
    out["line"] = {
        "name": str((out.get("line") or {}).get("name", "main_line")),
        "coords": [
            int(st.session_state.get("predict_line_x1", _DEFAULT_LINE_COORDS[0])),
            int(st.session_state.get("predict_line_y1", _DEFAULT_LINE_COORDS[1])),
            int(st.session_state.get("predict_line_x2", _DEFAULT_LINE_COORDS[2])),
            int(st.session_state.get("predict_line_y2", _DEFAULT_LINE_COORDS[3])),
        ],
        "default_resolution": [
            int(st.session_state.get("predict_line_default_w", _DEFAULT_LINE_RESOLUTION[0])),
            int(st.session_state.get("predict_line_default_h", _DEFAULT_LINE_RESOLUTION[1])),
        ],
    }

    return out


def _render_video_upload(videos_dir: Path) -> None:
    with st.expander("Nahrát nové video", expanded=False):
        st.caption(
            "Nahrajte vlastní video, které se přidá do složky s videi a ihned bude dostupné k predikci."
        )
        uploaded = st.file_uploader(
            "Vyberte video soubor",
            type=["mp4", "avi", "mov", "mkv"],
            key="predict_video_upload_file",
        )
        if uploaded is not None:
            col_btn, col_info = st.columns([1, 3])
            with col_btn:
                if st.button("Nahrát a vybrat", key="predict_video_upload_btn", use_container_width=True):
                    try:
                        res = uploads.save_video_upload(
                            bytes(uploaded.getbuffer()),
                            uploaded.name,
                            videos_dir,
                            overwrite=True,
                        )
                        st.session_state["predict_selected_video"] = res.path.name
                        st.session_state["_predict_upload_success"] = res.path.name
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Nahrávání selhalo: {exc}")
            with col_info:
                size_mb = round(uploaded.size / (1024 * 1024), 1)
                st.caption(f"Soubor: `{uploaded.name}` ({size_mb} MB)")

        success_name = st.session_state.pop("_predict_upload_success", None)
        if success_name:
            st.success(f"Video `{success_name}` bylo nahráno a vybráno.")


def _render_model_upload(models_cfg_path: Path) -> None:
    with st.expander("Nahrát vlastní váhy modelu", expanded=False):
        st.caption(
            "Nahrajte soubor s váhami modelu (.pt / .pth / .onnx). "
            "Model se uloží do složky `models/` a zaregistruje v `models.yaml`."
        )
        root = project_root()
        models_root = Path(st.session_state.get("ui_models_root", str(root / "models/ui")))

        c1, c2 = st.columns(2)
        with c1:
            st.selectbox("Backend", options=["yolo", "rfdetr"], key="ui_upload_model_backend")
            st.text_input(
                "ID modelu (např. yolo_tuned/muj_model)",
                key="ui_upload_model_id",
                help="Bude použito jako klíč v models.yaml a v dropdownu výběru modelu.",
            )
        with c2:
            st.selectbox("Varianta", options=["tuned", "pretrained"], key="ui_upload_model_variant")
            if st.session_state.get("ui_upload_model_backend") == "rfdetr":
                st.selectbox(
                    "Velikost RF-DETR *",
                    options=["nano", "small", "medium", "large", "xlarge", "2xlarge"],
                    key="ui_upload_model_rfdetr_size",
                )

        st.markdown("**Mapování tříd** (pro vlastní modely)")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.number_input("tourist", step=1, value=0, key="ui_upload_map_tourist")
        with m2:
            st.number_input("skier", step=1, value=1, key="ui_upload_map_skier")
        with m3:
            st.number_input("cyclist", step=1, value=2, key="ui_upload_map_cyclist")
        with m4:
            st.number_input("tourist_dog", step=1, value=3, key="ui_upload_map_tourist_dog")

        model_file = st.file_uploader(
            "Soubor s váhami (.pt / .pth / .onnx / .engine / .bin)",
            type=["pt", "pth", "onnx", "engine", "bin"],
            key="ui_upload_model_file",
        )

        if model_file is not None:
            col_btn, col_info = st.columns([1, 3])
            with col_btn:
                if st.button("Nahrát a registrovat", key="ui_upload_model_btn", use_container_width=True):
                    model_id = str(st.session_state.get("ui_upload_model_id", "")).strip()
                    backend = str(st.session_state.get("ui_upload_model_backend", "yolo"))
                    variant = str(st.session_state.get("ui_upload_model_variant", "tuned"))
                    rfdetr_size = str(st.session_state.get("ui_upload_model_rfdetr_size", "")).strip() or None

                    if not model_id:
                        st.error("Zadejte ID modelu.")
                    elif backend == "rfdetr" and not rfdetr_size:
                        st.error("Vyberte velikost RF-DETR modelu.")
                    else:
                        try:
                            mapping = {
                                "tourist": int(st.session_state.get("ui_upload_map_tourist", 0)),
                                "skier": int(st.session_state.get("ui_upload_map_skier", 1)),
                                "cyclist": int(st.session_state.get("ui_upload_map_cyclist", 2)),
                                "tourist_dog": int(st.session_state.get("ui_upload_map_tourist_dog", 3)),
                            }
                            up = uploads.save_model_upload(
                                bytes(model_file.getbuffer()),
                                model_file.name,
                                models_root,
                                backend=backend,
                                variant=variant,
                                model_id=model_id,
                                overwrite=True,
                            )
                            uploads.register_model_in_registry(
                                models_yaml_path=models_cfg_path,
                                project_root=root,
                                model_id=model_id,
                                backend=backend,
                                variant=variant,
                                weights_path=up.path,
                                mapping=mapping,
                                rfdetr_size=rfdetr_size,
                            )
                            st.session_state["_model_upload_success"] = model_id
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Nahrávání selhalo: {exc}")
            with col_info:
                size_mb = round(model_file.size / (1024 * 1024), 1)
                st.caption(f"Soubor: `{model_file.name}` ({size_mb} MB)")

        success_id = st.session_state.pop("_model_upload_success", None)
        if success_id:
            st.success(f"Model `{success_id}` byl nahrán a zaregistrován.")


def _render_line_editor(video_name: str) -> None:
    with st.expander("Vlastní čára", expanded=False):
        st.caption("Můžete upravit pozici čáry ručně nebo ji kliknout do prvního snímku vybraného videa.")

        videos_dir = Path(_effective_videos_dir())
        video_path = videos_dir / video_name if video_name else None

        if video_path is not None and video_name:
            if st.button("Načíst první snímek", key="predict_load_first_frame"):
                try:
                    frame_rgb, w, h = line_picker.load_first_frame_rgb(video_path)
                    st.session_state["ui_line_frame_rgb"] = frame_rgb
                    st.session_state["ui_line_frame_video"] = video_name
                    st.session_state["ui_line_frame_size"] = (int(w), int(h))
                    st.session_state["predict_line_default_w"] = int(w)
                    st.session_state["predict_line_default_h"] = int(h)
                except Exception as exc:
                    st.error(f"Nepodařilo se načíst první snímek: {exc}")

        frame_rgb = st.session_state.get("ui_line_frame_rgb")
        frame_video = str(st.session_state.get("ui_line_frame_video", ""))
        frame_size = st.session_state.get("ui_line_frame_size", (_DEFAULT_LINE_RESOLUTION[0], _DEFAULT_LINE_RESOLUTION[1]))
        max_x = max(1, int(frame_size[0]) - 1)
        max_y = max(1, int(frame_size[1]) - 1)

        if frame_rgb is not None and frame_video == video_name:
            if _ensure_canvas_compat():
                try:
                    st.caption(f"První snímek: {video_name}")
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
                        key=f"predict_line_canvas_{video_name}",
                    )
                    points = line_picker.extract_two_points(canvas.json_data)
                    if len(points) >= 2:
                        (x1, y1), (x2, y2) = points[:2]
                        st.caption(f"Naklikané body: ({x1}, {y1}) a ({x2}, {y2})")
                        if st.button("Použít naklikané body", key="predict_use_canvas_points"):
                            st.session_state["predict_line_x1"] = int(x1)
                            st.session_state["predict_line_y1"] = int(y1)
                            st.session_state["predict_line_x2"] = int(x2)
                            st.session_state["predict_line_y2"] = int(y2)
                except Exception as exc:
                    st.image(frame_rgb, caption=f"První snímek: {video_name}", use_container_width=True)
                    st.caption(f"Klikací výběr čáry selhal ({type(exc).__name__}), použijte ruční zadání.")
            else:
                st.image(frame_rgb, caption=f"První snímek: {video_name}", use_container_width=True)
                st.caption("Klikací výběr čáry není v tomto prostředí dostupný, použijte ruční zadání.")

        coord_col1, coord_col2 = st.columns(2)
        with coord_col1:
            st.number_input("x1", min_value=0, max_value=max_x, step=1, key="predict_line_x1")
            st.number_input("y1", min_value=0, max_value=max_y, step=1, key="predict_line_y1")
        with coord_col2:
            st.number_input("x2", min_value=0, max_value=max_x, step=1, key="predict_line_x2")
            st.number_input("y2", min_value=0, max_value=max_y, step=1, key="predict_line_y2")

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.number_input("Šířka referenčního rozlišení", min_value=1, step=1, key="predict_line_default_w")
        with res_col2:
            st.number_input("Výška referenčního rozlišení", min_value=1, step=1, key="predict_line_default_h")


def _find_latest_matching_run(runs_root: Path, *, model_id: str, video_name: str) -> Optional[Dict[str, Any]]:
    stem = Path(video_name).stem
    candidates: List[Dict[str, Any]] = []

    for run in discovery.discover_predict_runs(runs_root):
        run_model_id = str(run.get("model_id", ""))
        if model_id and run_model_id != model_id:
            continue

        predict_dir = Path(run["predict_dir"])
        counts_path = predict_dir / f"{stem}.counts.json"
        if not counts_path.exists():
            continue

        try:
            mtime = Path(run["run_dir"]).stat().st_mtime
        except Exception:
            mtime = 0.0

        candidate = dict(run)
        candidate["counts_path"] = counts_path
        candidate["mtime"] = mtime
        candidates.append(candidate)

    if not candidates:
        return None

    candidates.sort(key=lambda item: float(item["mtime"]), reverse=True)
    return candidates[0]


def _resolve_pred_video_path(predict_dir: Path, counts_obj: Dict[str, Any], video_name: str) -> Optional[Path]:
    meta = counts_obj.get("meta", {}) or {}
    outputs = meta.get("outputs", {}) or {}

    explicit = str(outputs.get("pred_video", "")).strip()
    if explicit:
        explicit_path = Path(explicit)
        if explicit_path.exists():
            return explicit_path

    stem = Path(video_name).stem
    candidates = list(predict_dir.glob(f"{stem}.pred.*"))
    if not candidates:
        return None

    def _priority(path: Path) -> tuple[int, str]:
        ext = path.suffix.lower()
        preferred = {
            ".mp4": 0,
            ".mov": 1,
            ".mkv": 2,
            ".avi": 3,
        }.get(ext, 99)
        return preferred, path.name

    candidates.sort(key=_priority)
    return candidates[0]


def _render_video_player(video_path: Path, *, start_time: int) -> None:
    try:
        video_bytes = video_path.read_bytes()
    except Exception as exc:
        st.warning(f"Nepodařilo se načíst video {video_path}: {exc}")
        return

    video_format = _VIDEO_FORMATS.get(video_path.suffix.lower(), "video/mp4")
    st.caption(f"Soubor videa: `{video_path.name}`")
    if video_path.suffix.lower() != ".mp4":
        st.warning("Tento formát videa nemusí jít v prohlížeči přehrát korektně.")
    st.video(video_bytes, format=video_format, start_time=start_time)


def _load_crossing_events(predict_dir: Path, *, video_name: str) -> List[Dict[str, Any]]:
    log_files = sorted(predict_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)

    for log_path in log_files:
        rows: List[Dict[str, Any]] = []
        active_video = ""
        active_fps: Optional[float] = None

        try:
            with log_path.open("r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line:
                        continue

                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue

                    event = str(rec.get("event", ""))
                    if event == "video_start":
                        active_video = str(rec.get("video", "")).strip()
                        active_fps = _safe_float(rec.get("fps"))
                        continue

                    if event == "video_done":
                        active_video = ""
                        active_fps = None
                        continue

                    direction = _CROSSING_EVENTS.get(event)
                    if direction is None:
                        continue

                    current_video = active_video or video_name
                    if current_video != video_name:
                        continue

                    frame_idx = int(rec.get("frame_idx", 0) or 0)
                    seconds = None
                    if active_fps and active_fps > 0:
                        seconds = frame_idx / active_fps

                    rows.append(
                        {
                            "timecode": _format_timecode(seconds),
                            "seconds": seconds,
                            "frame_idx": frame_idx,
                            "direction": direction,
                            "class": _class_label(rec.get("class_id")),
                            "track_id": int(rec.get("track_id", 0) or 0),
                            "logged_at": str(rec.get("t", "")),
                        }
                    )
        except Exception:
            continue

        if rows:
            return rows

    return []


def _selected_dataframe_row(table_state: Any, rows_df: pd.DataFrame) -> Optional[pd.Series]:
    if rows_df.empty or table_state is None:
        return None

    selection = getattr(table_state, "selection", None)
    if selection is None and isinstance(table_state, dict):
        selection = table_state.get("selection")

    if selection is None:
        return None

    selected_rows = getattr(selection, "rows", None)
    if selected_rows is None and isinstance(selection, dict):
        selected_rows = selection.get("rows", [])

    if not selected_rows:
        return None

    try:
        idx = int(selected_rows[0])
    except Exception:
        return None

    if idx < 0 or idx >= len(rows_df):
        return None
    return rows_df.iloc[idx]


def _counts_table(counts_obj: Dict[str, Any]) -> pd.DataFrame:
    in_count = counts_obj.get("in_count", {}) or {}
    out_count = counts_obj.get("out_count", {}) or {}

    rows: List[Dict[str, Any]] = []
    for class_id in [int(c.value) for c in CanonicalClass]:
        rows.append(
            {
                "Třída": _class_label(class_id),
                "Dovnitř": int(in_count.get(str(class_id), in_count.get(class_id, 0)) or 0),
                "Ven": int(out_count.get(str(class_id), out_count.get(class_id, 0)) or 0),
            }
        )

    return pd.DataFrame(rows)


def _render_result_detail(
    *,
    run: Dict[str, Any],
    counts_path: Path,
    counts_obj: Dict[str, Any],
    video_name: str,
    header: str,
    section: str = "latest",
) -> None:
    st.subheader(header)

    predict_dir = Path(run["predict_dir"])
    pred_video_path = _resolve_pred_video_path(predict_dir, counts_obj, video_name)
    events = _load_crossing_events(predict_dir, video_name=video_name)
    in_total = sum(int(v) for v in (counts_obj.get("in_count", {}) or {}).values())
    out_total = sum(int(v) for v in (counts_obj.get("out_count", {}) or {}).values())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", str(run.get("model_id", "")))
    c2.metric("Video", video_name)
    c3.metric("Dovnitř", in_total)
    c4.metric("Ven", out_total)

    st.caption(f"Běh: {run['run_id']}")

    left, right = st.columns([1, 1.4])
    with left:
        st.markdown("**Počty podle tříd**")
        st.dataframe(_counts_table(counts_obj), hide_index=True, use_container_width=True)

    selected_event = None
    with right:
        st.markdown("**Detekované průchody**")
        if events:
            events_df = pd.DataFrame(events)
            display_df = events_df[["timecode", "direction", "class", "frame_idx"]].rename(
                columns={
                    "timecode": "Čas",
                    "direction": "Směr",
                    "class": "Třída",
                    "frame_idx": "Snímek",
                }
            )
            table_state = st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row",
                key=f"predict_events_{section}_{run['run_id']}_{Path(video_name).stem}",
            )
            selected_event = _selected_dataframe_row(table_state, events_df)
        else:
            st.caption("V logu běhu nebyly nalezeny žádné průchody.")

    if pred_video_path is None:
        st.info("Predikované video pro tento běh nebylo nalezeno.")
        return

    start_time = 0
    if selected_event is not None:
        seconds = selected_event.get("seconds")
        if seconds is not None:
            start_time = max(0, int(float(seconds)))
        st.caption(
            "Video je posunuté na "
            f"{selected_event['timecode']} ({selected_event['direction']}, {selected_event['class']})."
        )

    _render_video_player(pred_video_path, start_time=start_time)


def _history_rows(runs_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for counts_path in runs_root.rglob("*.counts.json"):
        if counts_path.name == "aggregate.counts.json":
            continue

        counts_obj = discovery.load_json(counts_path)
        if not counts_obj:
            continue

        in_count = counts_obj.get("in_count", {}) or {}
        out_count = counts_obj.get("out_count", {}) or {}
        meta = counts_obj.get("meta", {}) or {}

        try:
            modified_at = counts_path.stat().st_mtime
        except Exception:
            modified_at = 0.0

        predict_dir = counts_path.parent
        rows.append(
            {
                "label": (
                    f"{meta.get('run_id', predict_dir.parent.name)} | "
                    f"{meta.get('model_id', '')} | {counts_obj.get('video', '')} | "
                    f"IN {sum(int(v) for v in in_count.values())} / OUT {sum(int(v) for v in out_count.values())}"
                ),
                "run_id": str(meta.get("run_id", predict_dir.parent.name)),
                "model_id": str(meta.get("model_id", "")),
                "video": str(counts_obj.get("video", "")),
                "in_total": sum(int(v) for v in in_count.values()),
                "out_total": sum(int(v) for v in out_count.values()),
                "status": "completed",
                "predict_dir": predict_dir,
                "run_dir": predict_dir.parent,
                "counts_path": counts_path,
                "modified_at": float(modified_at),
            }
        )

    rows.sort(key=lambda item: float(item["modified_at"]), reverse=True)
    return rows


def _run_from_job_logs(job: Any, *, model_id: str, video_name: str) -> Optional[Dict[str, Any]]:
    if job is None:
        return None

    run_id = ""
    for raw_line in getattr(job, "logs", []) or []:
        try:
            rec = json.loads(raw_line)
        except Exception:
            continue

        event = str(rec.get("event", ""))
        if event == "run_start":
            run_id = str(rec.get("run_id", ""))

        if event != "video_done":
            continue
        if str(rec.get("video", "")) != video_name:
            continue

        counts_path_raw = str(rec.get("counts_path", "")).strip()
        if not counts_path_raw:
            continue

        counts_path = Path(counts_path_raw)
        if not counts_path.is_absolute():
            counts_path = (Path(getattr(job, "cwd", ".")) / counts_path).resolve()
        if not counts_path.exists():
            continue

        return {
            "run_id": run_id or str(counts_path.parent.parent.name),
            "model_id": model_id,
            "predict_dir": counts_path.parent,
            "counts_path": counts_path,
        }

    return None


def _render_results(runs_root: Path, *, model_id: str, video_name: str, running: bool) -> None:
    if not model_id or not video_name:
        st.subheader("Poslední výsledek")
        st.caption("Vyberte model a video. Potom se zde zobrazí poslední výsledek.")
        return

    job = st.session_state.get("ui_jobs", {}).get("predict")
    run = _run_from_job_logs(job, model_id=model_id, video_name=video_name)
    if run is None:
        run = _find_latest_matching_run(runs_root, model_id=model_id, video_name=video_name)
    if run is None:
        st.subheader("Poslední výsledek")
        if running:
            st.info("Predikce právě běží. Výsledek se zobrazí po dokončení běhu.")
        else:
            st.info("Pro aktuální výběr zatím neexistuje dokončený běh.")
        return

    counts_path = Path(run["counts_path"])
    counts_obj = discovery.load_json(counts_path)
    if not counts_obj:
        st.subheader("Poslední výsledek")
        st.warning(f"Nepodařilo se načíst soubor counts: {counts_path}")
        return

    _render_result_detail(
        run=run,
        counts_path=counts_path,
        counts_obj=counts_obj,
        video_name=video_name,
        header="Poslední výsledek",
        section="latest",
    )


def _render_history(runs_root: Path, *, model_id: str, video_name: str) -> None:
    st.subheader("Historie predikcí")

    rows = _history_rows(runs_root)
    if not rows:
        st.info("V historii zatím nejsou žádné dokončené predikce.")
        return

    filtered = rows
    only_current = st.checkbox(
        "Pouze pro aktuálně vybraný model a video",
        value=False,
        key="predict_history_only_current",
    )
    if only_current:
        filtered = [row for row in rows if row["model_id"] == model_id and row["video"] == video_name]

    if not filtered:
        st.info("Pro aktuální filtr nebyly nalezeny žádné záznamy.")
        return

    history_df = pd.DataFrame(
        [
            {
                "Běh": row["run_id"],
                "Model": row["model_id"],
                "Video": row["video"],
                "Dovnitř": row["in_total"],
                "Ven": row["out_total"],
            }
            for row in filtered[:20]
        ]
    )
    st.dataframe(history_df, hide_index=True, use_container_width=True)

    options = {row["label"]: row for row in filtered}
    default_label = filtered[0]["label"]
    selected_label = st.selectbox(
        "Prohlédnout výsledek z historie",
        options=list(options.keys()),
        index=0,
        key="predict_history_selected",
    )
    selected = options.get(selected_label, options[default_label])

    counts_obj = discovery.load_json(Path(selected["counts_path"]))
    if not counts_obj:
        st.warning(f"Nepodařilo se načíst soubor counts: {selected['counts_path']}")
        return

    run = {
        "run_id": selected["run_id"],
        "model_id": selected["model_id"],
        "predict_dir": selected["predict_dir"],
    }
    _render_result_detail(
        run=run,
        counts_path=Path(selected["counts_path"]),
        counts_obj=counts_obj,
        video_name=str(selected["video"]),
        header="Vybraný výsledek z historie",
        section="history",
    )


@st.fragment(run_every=1)
def _render_results_live() -> None:
    job = st.session_state.get("ui_jobs", {}).get("predict")
    if job is not None:
        jobs.poll_job(job)
        st.session_state["ui_jobs"]["predict"] = job
    running = bool(job is not None and getattr(job, "status", "") == "running")

    _render_results(
        Path(st.session_state["ui_runs_predict_dir"]),
        model_id=str(st.session_state.get("predict_model_id", "")),
        video_name=str(st.session_state.get("predict_selected_video", "")),
        running=running,
    )


def render() -> None:
    st.header("Predikce jednoho videa")

    predict_cfg_path = Path(st.session_state["ui_predict_config_path"])
    models_cfg_path = Path(st.session_state["ui_models_config_path"])

    model_options: List[str] = []
    registry = None
    if not models_cfg_path.exists():
        models_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        default_path = project_root() / "configs" / "models.default.yaml"
        if default_path.exists():
            import shutil
            shutil.copy(default_path, models_cfg_path)
        else:
            models_cfg_path.write_text("models: {}\n", encoding="utf-8")
    try:
        registry = load_models_registry(models_cfg_path)
        model_options = sorted(registry.models.keys())
    except Exception as exc:
        st.error(f"Nepodařilo se načíst registr modelů: {exc}")

    _ensure_defaults(predict_cfg_path, model_options, [])
    videos_dir = Path(_effective_videos_dir())
    video_options = _list_videos(videos_dir)
    _ensure_defaults(predict_cfg_path, model_options, video_options)

    # --- Uploads ---
    _render_video_upload(videos_dir)
    _render_model_upload(models_cfg_path)

    if not video_options:
        st.warning(f"Ve složce nejsou žádná videa: {videos_dir}")
    else:
        st.caption(f"Složka s videi: `{videos_dir}`")

    # --- Model variant filter + model selector ---
    if registry is not None:
        models_by_variant: Dict[str, List[str]] = {"tuned": [], "pretrained": []}
        for mid, spec in registry.models.items():
            bucket = "tuned" if spec.variant == "tuned" else "pretrained"
            models_by_variant[bucket].append(mid)
        for bucket in models_by_variant:
            models_by_variant[bucket] = sorted(models_by_variant[bucket])

        if "predict_model_variant" not in st.session_state:
            current_model = st.session_state.get("predict_model_id", "")
            if current_model and registry.models.get(current_model):
                st.session_state["predict_model_variant"] = registry.models[current_model].variant
            else:
                st.session_state["predict_model_variant"] = "tuned"

        variant_options = [v for v in ["tuned", "pretrained"] if models_by_variant[v]]
        variant_index = variant_options.index(st.session_state["predict_model_variant"]) if st.session_state["predict_model_variant"] in variant_options else 0

        selected_variant = st.radio(
            "Varianta modelu",
            options=variant_options,
            index=variant_index,
            horizontal=True,
            key="predict_model_variant",
        )

        filtered_model_options = models_by_variant.get(selected_variant, []) or model_options
    else:
        filtered_model_options = model_options
        selected_variant = None

    # Keep selected model consistent when variant changes
    current_model_id = st.session_state.get("predict_model_id", "")
    if filtered_model_options and current_model_id not in filtered_model_options:
        st.session_state["predict_model_id"] = filtered_model_options[0]

    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Model", options=filtered_model_options or [""], key="predict_model_id")
    with col2:
        st.selectbox("Video", options=video_options or [""], key="predict_selected_video")

    st.number_input(
        "Vykreslit každý n-tý snímek",
        min_value=1,
        step=1,
        key="predict_render_every_n_frames",
    )
    _render_line_editor(str(st.session_state.get("predict_selected_video", "")))

    job = st.session_state.get("ui_jobs", {}).get("predict")
    running = bool(job is not None and getattr(job, "status", "") == "running")
    disabled = running or not model_options or not video_options

    if st.button("Spustit predikci", key="predict_run", disabled=disabled, use_container_width=True):
        cfg = _build_predict_dict(predict_cfg_path)
        yaml_text = configs.dump_yaml_text(cfg)
        ok, msg = configs.validate_predict_yaml_text(yaml_text)
        if not ok:
            st.error(msg)
        else:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".yaml",
                prefix="counter_ui_predict_",
                dir=tempfile.gettempdir(),
                delete=False,
            ) as fh:
                fh.write(yaml_text)
                tmp_cfg = Path(fh.name)

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
            st.success("Predikce byla spuštěna.")

    _render_results_live()
    _render_history(
        Path(st.session_state["ui_runs_predict_dir"]),
        model_id=str(st.session_state.get("predict_model_id", "")),
        video_name=str(st.session_state.get("predict_selected_video", "")),
    )


def render_page() -> None:
    """Top-level page entry point for multipage navigation."""
    st.title("Counter UI")
    st.caption("Predikce průchodů čárou pomocí YOLO / RF-DETR modelů.")
    _render_predict_job_status_live()
    render()
