from __future__ import annotations

import streamlit as st

from counter.ui.state import ensure_state_defaults
from counter.ui.views import predict


def main() -> None:
    st.set_page_config(page_title="Counter UI", layout="wide")
    ensure_state_defaults()
    predict.render_page()


if __name__ == "__main__":
    main()
