import os
os.environ["MEDIAPIPE_CACHE_DIR"] = "/tmp"
import streamlit as st

st.set_page_config(layout="wide", page_title="PoseSync Suite")

st.sidebar.title("Navigation")
st.sidebar.info("Select a module from the list below.")

pg = st.navigation([
    st.Page("process_karana.py", title="Reference Processor (Admin)", icon="🕉️"),
    st.Page("interactive_coach.py", title="Interactive Coach (Smart)", icon="🤖"),
    st.Page("live_corrector.py", title="Visual Corrector (Classic)", icon="👁️"),
    # st.Page("benchmark_page.py", title="Model Benchmark (Research)", icon="⚔️"),
    st.Page("image_comparison.py", title="Static Analyzer", icon="📷"),
])

pg.run()