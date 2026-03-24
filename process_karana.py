import cv2
import mediapipe as mp
import numpy as np
import json
import os
import streamlit as st
from interactive_coach import MediaPipeModel, get_limb_angles

# ── CONFIG ───────────────────────────────────────────────────────────────────
SOURCE_DIR = "karana-images"
DATA_DIR = "karana_data"
MARKED_DIR = "karana_marked"

st.set_page_config(page_title="Karana Data Processor", layout="wide")
st.title("🕉️ Karana Reference Processor")
st.markdown("Use this tool to extract and approve baseline data for the Karana library.")

# Initialize Model
@st.cache_resource
def load_processor():
    return MediaPipeModel(complexity=2).load()

processor = load_processor()

# ── PROCESSOR LOGIC ───────────────────────────────────────────────────────────
def save_karana_data(name, landmarks, angles, img_marked):
    # Save JSON Data
    data_path = os.path.join(DATA_DIR, f"{name}.json")
    with open(data_path, "w") as f:
        json.dump({
            "name": name,
            "landmarks": landmarks,
            "angles": angles
        }, f, indent=4)
    
    # Save Marked Image
    img_path = os.path.join(MARKED_DIR, f"{name}.jpg")
    cv2.imwrite(img_path, cv2.cvtColor(img_marked, cv2.COLOR_RGB2BGR))
    return data_path, img_path

# ── UI ────────────────────────────────────────────────────────────────────────
files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not files:
    st.error(f"No images found in `{SOURCE_DIR}` folder. Please add images numbered 1.jpg, 2.jpg, etc.")
else:
    target_file = st.selectbox("Select Karana Image to Process", sorted(files))
    img_path = os.path.join(SOURCE_DIR, target_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        raw_img = cv2.imread(img_path)
        raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        st.image(raw_img_rgb, use_container_width=True)

    # Extraction
    res = processor.process(raw_img)
    lms = processor.get_landmarks(res)
    
    if lms:
        angles = get_limb_angles(lms)
        
        # Draw for approval
        marked_img = raw_img_rgb.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            marked_img, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 140), thickness=2, circle_radius=4),
            mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2)
        )
        
        with col2:
            st.subheader("Marked & Analyzed")
            st.image(marked_img, use_container_width=True)
            st.write("**Extracted Angles:**")
            st.json(angles)

        # Approval Step
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Approve & Save Data", use_container_width=True):
                d_p, i_p = save_karana_data(target_file.split('.')[0], lms, angles, marked_img)
                st.success(f"Saved to {d_p} and {i_p}!")
        with c2:
            if st.button("❌ Redo (Reprocess)", use_container_width=True):
                st.warning("Model re-processing triggered. If detection is consistently wrong, check lighting or pose visibility.")
    else:
        st.error("Pose detection failed for this image. Please ensure the full body is visible.")
