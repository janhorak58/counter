from __future__ import annotations

import streamlit as st

from counter.ui.views import assets, browse, eval as eval_page, predict
from counter.ui.services import jobs
from counter.ui.state import ensure_state_defaults, project_root



def _poll_jobs() -> None:
    state_jobs = st.session_state.get("ui_jobs", {})

    for kind in ["predict", "eval"]:
        job = state_jobs.get(kind)
        if job is None:
            continue

        jobs.poll_job(job)
        if job.output_path is None and job.status in {"completed", "failed", "cancelled"}:
            job.output_path = jobs.guess_output_path_from_logs(job.logs, cwd=job.cwd)

        state_jobs[kind] = job



def _render_job_status(kind: str) -> None:
    job = st.session_state.get("ui_jobs", {}).get(kind)
    title = f"{kind.capitalize()} Job"

    with st.expander(title, expanded=bool(job is not None and getattr(job, "status", "") == "running")):
        if job is None:
            st.caption("No job launched yet.")
            return

        st.write(f"Status: **{job.status}**")
        st.write(f"Command: `{ ' '.join(job.command) }`")

        if job.output_path:
            st.write(f"Detected output: `{job.output_path}`")

        if job.status == "running":
            if st.button(f"Cancel {kind}", key=f"cancel_{kind}"):
                jobs.cancel_job(job)
                st.session_state["ui_jobs"][kind] = job

        logs_text = "\n".join(job.logs[-500:]) if job.logs else ""
        st.code(logs_text or "(no logs yet)", language="text")



def _has_running_jobs() -> bool:
    state_jobs = st.session_state.get("ui_jobs", {})
    for kind in ["predict", "eval"]:
        job = state_jobs.get(kind)
        if job is not None and getattr(job, "status", "") == "running":
            return True
    return False



def _render_jobs_static() -> None:
    _render_job_status("predict")
    _render_job_status("eval")



@st.fragment(run_every=1)
def _render_jobs_live() -> None:
    _poll_jobs()
    _render_job_status("predict")
    _render_job_status("eval")



def _render_sidebar() -> None:
    st.sidebar.header("Paths")

    st.sidebar.text_input("Predict config", key="ui_predict_config_path")
    st.sidebar.text_input("Eval config", key="ui_eval_config_path")
    st.sidebar.text_input("Models config", key="ui_models_config_path")

    st.sidebar.text_input("Videos dir", key="ui_videos_dir")
    st.sidebar.text_input("GT dir", key="ui_gt_dir")

    st.sidebar.text_input("Runs/predict dir", key="ui_runs_predict_dir")
    st.sidebar.text_input("Runs/eval dir", key="ui_runs_eval_dir")

    st.sidebar.caption(f"Project root: {project_root()}")



def main() -> None:
    st.set_page_config(page_title="Counter UI", layout="wide")
    ensure_state_defaults()
    _poll_jobs()

    st.title("Counter Streamlit UI")
    _render_sidebar()

    page = st.sidebar.radio("Page", options=["Predict", "Evaluation", "Assets", "Browse"], key="ui_page")

    if _has_running_jobs():
        _render_jobs_live()
    else:
        _render_jobs_static()

    if page == "Predict":
        predict.render()
    elif page == "Evaluation":
        eval_page.render()
    elif page == "Assets":
        assets.render()
    elif page == "Browse":
        browse.render()


if __name__ == "__main__":
    main()
