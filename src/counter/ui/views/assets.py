from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st

from counter.ui.services import uploads
from counter.ui.state import project_root



def _mapping_from_state() -> Optional[Dict[str, int]]:
    enabled = bool(st.session_state.get("assets_model_mapping_enabled", False))
    if not enabled:
        return None

    try:
        return {
            "tourist": int(st.session_state.get("assets_map_tourist", 0)),
            "skier": int(st.session_state.get("assets_map_skier", 1)),
            "cyclist": int(st.session_state.get("assets_map_cyclist", 2)),
            "tourist_dog": int(st.session_state.get("assets_map_tourist_dog", 3)),
        }
    except Exception:
        return None



def render() -> None:
    st.header("Assets")

    root = project_root()
    videos_dir = Path(st.session_state["ui_videos_dir"])
    gt_dir = Path(st.session_state["ui_gt_dir"])
    models_cfg_path = Path(st.session_state["ui_models_config_path"])
    models_root = root / "models"

    st.subheader("Upload Videos")
    uploaded_videos = st.file_uploader(
        "Select video files",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=True,
        key="assets_video_upload",
    )
    if st.button("Upload Videos", key="assets_upload_videos"):
        if not uploaded_videos:
            st.warning("No videos selected.")
        else:
            for uf in uploaded_videos:
                res = uploads.save_video_upload(bytes(uf.getbuffer()), uf.name, videos_dir, overwrite=True)
                st.success(f"{res.path} ({res.message})")

    st.subheader("Upload Ground Truth")
    uploaded_gt = st.file_uploader(
        "Select .counts.json or .zip",
        type=["json", "zip"],
        accept_multiple_files=True,
        key="assets_gt_upload",
    )
    if st.button("Upload Ground Truth", key="assets_upload_gt"):
        if not uploaded_gt:
            st.warning("No files selected.")
        else:
            for uf in uploaded_gt:
                try:
                    results = uploads.save_gt_upload(bytes(uf.getbuffer()), uf.name, gt_dir, overwrite=True)
                    for res in results:
                        st.success(f"{res.path} ({res.message})")
                except Exception as exc:
                    st.error(f"{uf.name}: {exc}")

    st.subheader("Upload Model + Auto-register")
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("backend", options=["yolo", "rfdetr"], key="assets_model_backend")
        st.selectbox("variant", options=["tuned", "pretrained"], key="assets_model_variant")
        st.text_input("model_id", key="assets_model_id")
        st.text_input("rfdetr_size (optional)", key="assets_model_size")

    with c2:
        st.checkbox("Set mapping", value=False, key="assets_model_mapping_enabled")
        st.number_input("mapping.tourist", step=1, value=0, key="assets_map_tourist")
        st.number_input("mapping.skier", step=1, value=1, key="assets_map_skier")
        st.number_input("mapping.cyclist", step=1, value=2, key="assets_map_cyclist")
        st.number_input("mapping.tourist_dog", step=1, value=3, key="assets_map_tourist_dog")

    model_file = st.file_uploader(
        "Model weights (.pt/.pth/.onnx/.engine/.bin)",
        type=["pt", "pth", "onnx", "engine", "bin"],
        key="assets_model_upload",
    )

    if st.button("Upload Model + Register", key="assets_upload_model"):
        model_id = str(st.session_state.get("assets_model_id", "")).strip()
        backend = str(st.session_state.get("assets_model_backend", "yolo"))
        variant = str(st.session_state.get("assets_model_variant", "tuned"))

        if not model_id:
            st.error("model_id is required.")
        elif model_file is None:
            st.error("Please select a model file.")
        else:
            try:
                up = uploads.save_model_upload(
                    bytes(model_file.getbuffer()),
                    model_file.name,
                    models_root,
                    backend=backend,
                    variant=variant,
                    model_id=model_id,
                    overwrite=True,
                )
                mapping = _mapping_from_state()
                uploads.register_model_in_registry(
                    models_yaml_path=models_cfg_path,
                    project_root=root,
                    model_id=model_id,
                    backend=backend,
                    variant=variant,
                    weights_path=up.path,
                    mapping=mapping,
                    rfdetr_size=str(st.session_state.get("assets_model_size", "")).strip() or None,
                )
                st.success(f"Uploaded: {up.path}")
                st.success(f"Registered model_id='{model_id}' in {models_cfg_path}")
            except Exception as exc:
                st.error(str(exc))

    st.subheader("Asset Manager")
    assets = uploads.list_assets(videos_dir=videos_dir, gt_dir=gt_dir, models_root=models_root)

    if assets:
        df = pd.DataFrame(
            [
                {
                    "kind": a.kind,
                    "path": str(a.path),
                    "size_mb": round(a.size_bytes / (1024 * 1024), 3),
                    "modified_at": a.modified_at.strftime("%Y-%m-%d %H:%M:%S"),
                }
                for a in assets
            ]
        )
        st.dataframe(df, use_container_width=True)

        selected_path = st.selectbox("Preview asset", options=[str(a.path) for a in assets], key="assets_preview_path")
        p = Path(selected_path)

        if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            st.video(str(p))
        elif p.suffix.lower() == ".json":
            try:
                import json

                st.json(json.loads(p.read_text(encoding="utf-8")))
            except Exception as exc:
                st.error(f"Failed to parse JSON: {exc}")
        else:
            st.caption(p.name)

        st.checkbox("I understand delete is permanent", key="assets_delete_confirm")
        if st.button("Delete selected asset", key="assets_delete_button"):
            if not st.session_state.get("assets_delete_confirm", False):
                st.warning("Please confirm deletion first.")
            else:
                try:
                    uploads.delete_asset(
                        p,
                        allowed_roots=[videos_dir, gt_dir, models_root],
                    )
                    st.success(f"Deleted: {p}")
                except Exception as exc:
                    st.error(str(exc))
    else:
        st.info("No assets found in configured directories.")
