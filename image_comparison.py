import streamlit as st
import cv2
import numpy as np
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(page_title="Pose Comparison Tool", layout="wide")
st.title("Pose Feedback System")

# Initialize MediaPipe Pose engine
@st.cache_resource
def load_pose():
    return mp_pose.Pose(
        static_image_mode=True, 
        model_complexity=2, 
        min_detection_confidence=0.5
    )

pose_tracker = load_pose()

def calculate_angle_3d(a, b, c):
    """Calculates the 3D angle at point B using vector dot product."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0:
        return 0.0
    cosine_angle = np.dot(ba, bc) / norm
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def process_frame(uploaded_file):
    if uploaded_file is None:
        return None, None, None
    
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = pose_tracker.process(image_rgb)
    
    if not results.pose_landmarks:
        return None, image_rgb, None

    lm = results.pose_landmarks.landmark
    def get_pt_3d(idx): return [lm[idx].x, lm[idx].y, lm[idx].z]

    # Calculate 8 specific joint angles in 3D
    angles = [
        calculate_angle_3d(get_pt_3d(12), get_pt_3d(14), get_pt_3d(16)), # Right Elbow
        calculate_angle_3d(get_pt_3d(11), get_pt_3d(13), get_pt_3d(15)), # Left Elbow
        calculate_angle_3d(get_pt_3d(24), get_pt_3d(26), get_pt_3d(28)), # Right Knee
        calculate_angle_3d(get_pt_3d(23), get_pt_3d(25), get_pt_3d(27)), # Left Knee
        calculate_angle_3d(get_pt_3d(14), get_pt_3d(12), get_pt_3d(24)), # Right Shoulder
        calculate_angle_3d(get_pt_3d(13), get_pt_3d(11), get_pt_3d(23)), # Left Shoulder
        calculate_angle_3d(get_pt_3d(12), get_pt_3d(24), get_pt_3d(26)), # Right Hip
        calculate_angle_3d(get_pt_3d(11), get_pt_3d(23), get_pt_3d(25))  # Left Hip
    ]
    
    # Draw skeleton for visual confirmation
    annotated = image_rgb.copy()
    mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    return np.array(angles).reshape(1, -1), annotated, angles

# UI Layout
col_target, col_attempt = st.columns(2)

with col_target:
    target_upload = st.file_uploader("Upload Target Pose", type=['png', 'jpg', 'jpeg'])

with col_attempt:
    attempt_upload = st.file_uploader("Upload Your Attempt", type=['png', 'jpg', 'jpeg'])

if target_upload and attempt_upload:
    vec_t, img_t, raw_t = process_frame(target_upload)
    vec_a, img_a, raw_a = process_frame(attempt_upload)

    if vec_t is not None and vec_a is not None:
        # Calculate Similarity
        sim_score = cosine_similarity(vec_t, vec_a)[0][0]
        
        st.subheader(f"Overall Match Score: {sim_score * 100:.2f}%")
        
        res_col1, res_col2 = st.columns(2)
        res_col1.image(img_t, caption="Target Landmarks")
        res_col2.image(img_a, caption="Attempt Landmarks")

        # Feedback Table
        labels = [
            "Right Elbow", "Left Elbow", 
            "Right Knee", "Left Knee", 
            "Right Shoulder", "Left Shoulder",
            "Right Hip", "Left Hip"
        ]
        diffs = np.abs(np.array(raw_t) - np.array(raw_a))
        
        st.write("### Analysis Breakdown")
        table_data = []
        for i in range(len(labels)):
            if diffs[i] < 15:
                status = "✅ Perfect"
            elif diffs[i] < 30:
                status = "⚠️ Minor Adjustment"
            else:
                status = "❌ Needs Correction"
                
            table_data.append({
                "Joint": labels[i],
                "Target Angle": f"{raw_t[i]:.1f}°",
                "Attempt Angle": f"{raw_a[i]:.1f}°",
                "Difference": f"{diffs[i]:.1f}°",
                "Status": status
            })
        st.table(table_data)
    else:
        st.error("Could not detect pose landmarks in one or both images.")