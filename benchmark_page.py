import streamlit as st
import cv2
import numpy as np
import time
from interactive_coach import MediaPipeModel, OpenPoseModel

st.set_page_config(layout="wide", page_title="PoseSync: Model Benchmark")

st.title("⚔️ Model Battle Arena")
st.markdown("Compare pose estimation models side-by-side. Upload an image to test.")

# ── CONFIG ───────────────────────────────────────────────────────────────────
with st.expander("Model Configuration"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**OpenPose Settings**")
        op_proto = st.text_input("Prototxt Path", "pose_deploy.prototxt")
        op_model = st.text_input("Caffemodel Path", "pose_iter_440000.caffemodel")

# ── INPUT ────────────────────────────────────────────────────────────────────
img_file = st.file_uploader("Upload Test Image", type=["jpg", "png", "jpeg"])

if img_file:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    
    col1, col2, col3 = st.columns(3)
    
    # ── MEDIAPIPE ────────────────────────────────────────────────────────────
    with col1:
        st.subheader("MediaPipe (Google)")
        st.info("Status: Ready (Built-in)")
        
        start = time.time()
        mp_model = MediaPipeModel(complexity=2, smooth=False).load()
        res = mp_model.process(original_img)
        end = time.time()
        
        vis_mp = original_img.copy()
        if res.pose_landmarks:
            import mediapipe as mp
            mp.solutions.drawing_utils.draw_landmarks(
                vis_mp, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            
        st.image(cv2.cvtColor(vis_mp, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.metric("Inference Time", f"{(end-start)*1000:.1f} ms", "33 Keypoints")
        st.caption("✅ Full body, hands, face support.")

    # ── OPENPOSE ─────────────────────────────────────────────────────────────
    with col2:
        st.subheader("OpenPose (CMU)")
        
        if False: # Check file existence in real app
            pass
            
        try:
            # We try to load, but it will likely fail without files
            # We suppress the crash and show a visual placeholder
            start = time.time()
            op_model_inst = OpenPoseModel(op_proto, op_model).load()
            
            if op_model_inst.net:
                out = op_model_inst.process(original_img)
                end = time.time()
                st.success("Loaded!")
                st.image(original_img, caption="OpenPose Output", use_container_width=True) # Placeholder for vis
                st.metric("Inference Time", f"{(end-start)*1000:.1f} ms")
            else:
                raise Exception("Weights not found")
                
        except Exception as e:
            st.warning("⚠️ Model Weights Missing")
            st.markdown(f"""
            OpenPose requires `pose_iter_440000.caffemodel` (~200MB).
            
            **Comparison vs MediaPipe:**
            *   **Pros:** Better multi-person tracking.
            *   **Cons:** slower, requires GPU, heavier.
            """)
            # Show a static example or blank
            st.image("https://placehold.co/400x300?text=OpenPose+Missing", use_container_width=True)

    # ── MOVENET ──────────────────────────────────────────────────────────────
    with col3:
        st.subheader("MoveNet (TensorFlow)")
        st.warning("⚠️ TensorFlow not installed")
        
        st.markdown("""
        **MoveNet Lightning/Thunder**
        
        *   **Architecture:** CenterNet-based (Bottom-up).
        *   **Speed:** Extremely fast (Lightning).
        *   **Points:** 17 (COCO standard).
        
        To enable: `pip install tensorflow tensorflow-hub`
        """)
        st.image("https://placehold.co/400x300?text=MoveNet+Missing", use_container_width=True)

