from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from counter.ui.services import configs, jobs
from counter.ui.state import project_root



def _seed_eval_form_from_dict(cfg: Dict[str, Any]) -> None:
    st.session_state["eval_gt_dir"] = str(cfg.get("gt_dir", st.session_state.get("ui_gt_dir", "data/counts_gt")))
    st.session_state["eval_runs_dir"] = str(cfg.get("runs_dir", st.session_state.get("ui_runs_predict_dir", "runs/predict")))
    st.session_state["eval_out_dir"] = str(cfg.get("out_dir", st.session_state.get("ui_runs_eval_dir", "runs/eval")))
    st.session_state["eval_only_completed"] = bool(cfg.get("only_completed", True))
    st.session_state["eval_videos_dir"] = str(cfg.get("videos_dir", ""))

    filters = cfg.get("filters") or {}
    st.session_state["eval_filter_backends"] = list(filters.get("backends", []))
    st.session_state["eval_filter_variants"] = list(filters.get("variants", []))
    st.session_state["eval_filter_model_ids"] = ",".join(filters.get("model_ids", []))
    st.session_state["eval_filter_run_ids"] = ",".join(filters.get("run_ids", []))

    charts = cfg.get("charts") or {}
    st.session_state["eval_charts_enabled"] = bool(charts.get("enabled", True))



def _seed_eval_form_from_yaml(path: Path) -> None:
    _seed_eval_form_from_dict(configs.load_yaml_dict(path))



def _csv_to_list(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]



def _build_eval_dict() -> Dict[str, Any]:
    return {
        "gt_dir": str(st.session_state.get("eval_gt_dir", st.session_state.get("ui_gt_dir", "data/counts_gt"))),
        "runs_dir": str(st.session_state.get("eval_runs_dir", st.session_state.get("ui_runs_predict_dir", "runs/predict"))),
        "out_dir": str(st.session_state.get("eval_out_dir", st.session_state.get("ui_runs_eval_dir", "runs/eval"))),
        "only_completed": bool(st.session_state.get("eval_only_completed", True)),
        "videos_dir": str(st.session_state.get("eval_videos_dir", "")),
        "filters": {
            "run_ids": _csv_to_list(st.session_state.get("eval_filter_run_ids", "")),
            "backends": list(st.session_state.get("eval_filter_backends", [])),
            "variants": list(st.session_state.get("eval_filter_variants", [])),
            "model_ids": _csv_to_list(st.session_state.get("eval_filter_model_ids", "")),
        },
        "charts": {
            "enabled": bool(st.session_state.get("eval_charts_enabled", True)),
        },
    }



def _apply_pending_updates() -> None:
    if "eval_yaml_pending" in st.session_state:
        text = str(st.session_state.pop("eval_yaml_pending"))
        st.session_state["ui_eval_yaml_text"] = text
        st.session_state["ui_eval_yaml_applied"] = text

    if "eval_form_pending_from_yaml" in st.session_state:
        cfg = st.session_state.pop("eval_form_pending_from_yaml")
        if isinstance(cfg, dict):
            _seed_eval_form_from_dict(cfg)



def _ensure_loaded_from_config(eval_cfg_path: Path) -> None:
    current = str(eval_cfg_path)
    loaded = str(st.session_state.get("ui_eval_loaded_from", ""))
    initialized = bool(st.session_state.get("eval_form_initialized", False))

    if not initialized or loaded != current:
        _seed_eval_form_from_yaml(eval_cfg_path)
        st.session_state["eval_form_initialized"] = True
        st.session_state["ui_eval_loaded_from"] = current
        st.session_state["ui_eval_yaml_text"] = ""
        st.session_state["ui_eval_yaml_applied"] = ""



def _sync_yaml_to_form_if_needed() -> None:
    if not bool(st.session_state.get("ui_eval_yaml_to_form_sync", True)):
        return

    # Prevent sync tug-of-war: when form->YAML auto-sync is enabled,
    # form edits are the source of truth.
    if bool(st.session_state.get("ui_eval_yaml_sync", True)):
        return

    text = str(st.session_state.get("ui_eval_yaml_text", "")).strip()
    if not text:
        return

    if text == str(st.session_state.get("ui_eval_yaml_applied", "")):
        return

    ok, _ = configs.validate_eval_yaml_text(text)
    if not ok:
        return

    cfg = configs.parse_yaml_text(text)
    _seed_eval_form_from_dict(cfg)
    st.session_state["ui_eval_yaml_applied"] = text



def render() -> None:
    st.header("Evaluation")

    eval_cfg_path = Path(st.session_state["ui_eval_config_path"])

    _apply_pending_updates()
    _ensure_loaded_from_config(eval_cfg_path)
    _sync_yaml_to_form_if_needed()

    if st.button("Reload from eval.yaml", key="eval_reload_yaml"):
        _seed_eval_form_from_yaml(eval_cfg_path)
        st.session_state["ui_eval_loaded_from"] = str(eval_cfg_path)
        st.session_state["ui_eval_yaml_text"] = ""
        st.session_state["ui_eval_yaml_applied"] = ""
        st.success("Eval form reloaded from YAML.")
        st.rerun()

    col1, col2 = st.columns(2)

    with col1:
        st.text_input("gt_dir", key="eval_gt_dir")
        st.text_input("runs_dir", key="eval_runs_dir")
        st.text_input("out_dir", key="eval_out_dir")
        st.checkbox("only_completed", key="eval_only_completed")
        st.text_input("videos_dir (optional)", key="eval_videos_dir")

    with col2:
        st.multiselect("filters.backends", options=["yolo", "rfdetr"], key="eval_filter_backends")
        st.multiselect("filters.variants", options=["tuned", "pretrained"], key="eval_filter_variants")
        st.text_input("filters.model_ids (comma-separated)", key="eval_filter_model_ids")
        st.text_input("filters.run_ids (comma-separated)", key="eval_filter_run_ids")
        st.checkbox("charts.enabled", key="eval_charts_enabled")

    generated = configs.dump_yaml_text(_build_eval_dict())
    if st.session_state.get("ui_eval_yaml_sync", True) or not st.session_state.get("ui_eval_yaml_text", ""):
        st.session_state["ui_eval_yaml_text"] = generated
        st.session_state["ui_eval_yaml_applied"] = generated

    st.subheader("Eval YAML Editor")
    s1, s2 = st.columns(2)
    with s1:
        st.checkbox("Sync form -> YAML", key="ui_eval_yaml_sync")
    with s2:
        st.checkbox("Sync YAML -> form", key="ui_eval_yaml_to_form_sync")

    st.text_area("Edit YAML", key="ui_eval_yaml_text", height=280)

    b1, b2, b3, b4, b5 = st.columns(5)

    with b1:
        if st.button("Regenerate from form", key="eval_regenerate_yaml"):
            st.session_state["eval_yaml_pending"] = generated
            st.rerun()

    with b2:
        if st.button("Apply YAML to form", key="eval_apply_yaml_to_form"):
            ok, msg = configs.validate_eval_yaml_text(st.session_state["ui_eval_yaml_text"])
            if not ok:
                st.error(msg)
            else:
                st.session_state["eval_form_pending_from_yaml"] = configs.parse_yaml_text(
                    st.session_state["ui_eval_yaml_text"]
                )
                st.session_state["ui_eval_yaml_applied"] = str(st.session_state["ui_eval_yaml_text"])
                st.rerun()

    with b3:
        if st.button("Validate YAML", key="eval_validate_yaml"):
            ok, msg = configs.validate_eval_yaml_text(st.session_state["ui_eval_yaml_text"])
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    with b4:
        if st.button("Save eval.yaml", key="eval_save_yaml"):
            try:
                obj = configs.parse_yaml_text(st.session_state["ui_eval_yaml_text"])
                configs.write_yaml_file(eval_cfg_path, obj)
                st.success(f"Saved: {eval_cfg_path}")
            except Exception as exc:
                st.error(str(exc))

    job = st.session_state.get("ui_jobs", {}).get("eval")
    running = bool(job is not None and getattr(job, "status", "") == "running")

    with b5:
        if st.button("Run Eval", key="eval_run", disabled=running):
            ok, msg = configs.validate_eval_yaml_text(st.session_state["ui_eval_yaml_text"])
            if not ok:
                st.error(msg)
            else:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    suffix=".yaml",
                    prefix="counter_ui_eval_",
                    dir="/tmp",
                    delete=False,
                ) as f:
                    f.write(st.session_state["ui_eval_yaml_text"])
                    tmp_cfg = Path(f.name)

                cmd = [sys.executable, "-m", "counter.eval", "--config", str(tmp_cfg)]
                handle = jobs.start_job(kind="eval", command=cmd, cwd=project_root())
                st.session_state["ui_jobs"]["eval"] = handle
                st.success("Eval job started.")
