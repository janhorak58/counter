from __future__ import annotations

from pathlib import Path
from typing import Dict

import streamlit as st



def project_root() -> Path:
    return Path(__file__).resolve().parents[3]



def ensure_state_defaults() -> None:
    root = project_root()

    defaults: Dict[str, object] = {
        "ui_project_root": str(root),
        "ui_predict_config_path": str(root / "configs/predict_ui.yaml"),
        "ui_eval_config_path": str(root / "configs/eval.yaml"),
        "ui_models_config_path": str(root / "configs/models.yaml"),
        "ui_models_root": str(root / "models"),
        "ui_videos_dir": str(root / "data/videos"),
        "ui_gt_dir": str(root / "data/counts_gt"),
        "ui_runs_predict_dir": str(root / "runs/predict"),
        "ui_runs_eval_dir": str(root / "runs/eval"),
        "ui_jobs": {"predict": None, "eval": None},
        "ui_predict_yaml_text": "",
        "ui_eval_yaml_text": "",
        "ui_predict_yaml_sync": True,
        "ui_eval_yaml_sync": True,
        "ui_predict_yaml_to_form_sync": False,
        "ui_eval_yaml_to_form_sync": False,
        "ui_predict_yaml_applied": "",
        "ui_eval_yaml_applied": "",
        "ui_predict_coords_text": "",
        "ui_line_frame_rgb": None,
        "ui_line_frame_video": "",
        "ui_line_frame_size": (1920, 1080),
        "ui_predict_loaded_from": "",
        "ui_eval_loaded_from": "",
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
