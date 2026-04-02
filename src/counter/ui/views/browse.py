from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from counter.ui.services import discovery



def _render_predict_runs(runs_root: Path) -> None:
    st.subheader("Predict Runs")
    runs = discovery.discover_predict_runs(runs_root)

    if not runs:
        st.info(f"No predict runs found in: {runs_root}")
        return

    df = pd.DataFrame(
        [
            {
                "run_id": r["run_id"],
                "model_id": r["model_id"],
                "backend": r["backend"],
                "variant": r["variant"],
                "status": r["status"],
                "run_dir": str(r["run_dir"]),
            }
            for r in runs
        ]
    )
    st.dataframe(df, use_container_width=True)

    run_options = {f"{r['run_id']} | {r['model_id']}": r for r in runs}
    selected_key = st.selectbox("Select predict run", options=list(run_options.keys()), key="browse_predict_selected")
    run = run_options[selected_key]

    run_json = discovery.load_json(Path(run["run_dir"]) / "run.json")
    if run_json:
        st.markdown("**run.json**")
        st.json(run_json)

    rows = discovery.load_counts_rows(Path(run["predict_dir"]))
    if rows:
        st.markdown("**Counts by video**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        counts_file = st.selectbox(
            "Open counts JSON",
            options=[r["path"] for r in rows],
            key="browse_counts_file",
        )
        obj = discovery.load_json(Path(counts_file))
        if obj:
            st.json(obj)

    pred_videos = discovery.list_pred_videos(Path(run["predict_dir"]))
    if pred_videos:
        vid = st.selectbox("Open predicted video", options=[str(p) for p in pred_videos], key="browse_pred_video")
        st.video(vid)



def _render_eval_runs(eval_root: Path) -> None:
    st.subheader("Evaluation Runs")

    eval_runs = discovery.discover_eval_runs(eval_root)
    if not eval_runs:
        st.info(f"No eval runs found in: {eval_root}")
        return

    selected = st.selectbox(
        "Select eval output folder",
        options=[str(x.eval_dir) for x in eval_runs],
        key="browse_eval_selected",
    )

    eval_dir = Path(selected)
    tables = discovery.load_eval_tables(eval_dir)

    per_run = tables.get("per_run", pd.DataFrame())
    per_video = tables.get("per_video", pd.DataFrame())
    per_class = tables.get("per_class", pd.DataFrame())

    if per_run.empty:
        st.warning("per_run_metrics.csv missing or empty.")
        return

    score_cols = [c for c in per_run.columns if str(c).startswith("score_")]
    default_score = "score_total_micro_wape"
    score_index = score_cols.index(default_score) if default_score in score_cols else 0
    score_col = st.selectbox(
        "Leaderboard metric",
        options=score_cols or [per_run.columns[0]],
        index=score_index if score_cols else 0,
        key="browse_eval_score_col",
    )

    leader = per_run.copy()
    if score_col in leader.columns:
        leader = leader.sort_values(score_col, ascending=True, na_position="last")

    st.markdown("**Leaderboard**")
    st.dataframe(leader, use_container_width=True)

    if score_col in leader.columns and "model_id" in leader.columns:
        chart_df = leader[["model_id", score_col]].dropna().head(20)
        if not chart_df.empty:
            fig = px.bar(chart_df, x="model_id", y=score_col, title=f"Top models by {score_col}")
            st.plotly_chart(fig, use_container_width=True)

    if "run_id" in leader.columns:
        selected_run = st.selectbox("Run drilldown", options=list(leader["run_id"].astype(str).unique()), key="browse_eval_run")

        if not per_video.empty and "run_id" in per_video.columns:
            st.markdown("**Per-video metrics**")
            pv = per_video[per_video["run_id"].astype(str) == str(selected_run)]
            st.dataframe(pv, use_container_width=True)

        if not per_class.empty and "run_id" in per_class.columns:
            st.markdown("**Per-class metrics**")
            pc = per_class[per_class["run_id"].astype(str) == str(selected_run)]
            st.dataframe(pc, use_container_width=True)

    charts = discovery.list_chart_images(eval_dir)
    if charts:
        st.markdown("**Charts**")
        chart_opts = [str(p) for p in charts]
        selected_charts = st.multiselect(
            "Select chart images",
            options=chart_opts,
            default=chart_opts[:6],
            key="browse_eval_charts",
        )
        for p in selected_charts:
            st.image(p, caption=Path(p).name, use_container_width=True)



def render() -> None:
    st.header("Browse")

    runs_predict_root = Path(st.session_state["ui_runs_predict_dir"])
    runs_eval_root = Path(st.session_state["ui_runs_eval_dir"])

    tab1, tab2 = st.tabs(["Predict Runs", "Eval Runs"])

    with tab1:
        _render_predict_runs(runs_predict_root)

    with tab2:
        _render_eval_runs(runs_eval_root)
