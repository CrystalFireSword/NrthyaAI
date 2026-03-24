import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from mediapipe.python.solutions import pose as mp_pose

st.set_page_config(layout="wide")
st.title("Live Pose Trainer: Directional Guidance")

@st.cache_resource
def load_pose_engine():
    return mp_pose.Pose(
        model_complexity=2, 
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.7
    )

pose_tracker = load_pose_engine()

def get_angle_3d(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0: return 0
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)))

# --- TARGET CALIBRATION ---
target_file = st.sidebar.file_uploader("Upload Target Image", type=['jpg', 'jpeg', 'png'])
target_lms = None

if target_file:
    file_bytes = np.asarray(bytearray(target_file.read()), dtype=np.uint8)
    t_img = cv2.imdecode(file_bytes, 1)
    t_res = pose_tracker.process(cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB))
    if t_res.pose_landmarks:
        target_lms = t_res.pose_landmarks.landmark
        st.sidebar.success("Target Pose Calibrated")

run = st.checkbox('Start Live Trainer')
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        results = pose_tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks and target_lms:
            u = results.pose_landmarks.landmark
            
            # 1. VISIBILITY GUARD (Shoulders and Hips)
            key_pts = [u[11], u[12], u[23], u[24]]
            if any(pt.visibility < 0.6 for pt in key_pts):
                cv2.putText(frame, "STAND BACK: DETECTING TORSO...", (50, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # 2. SCALING CALCULATIONS (Based on Torso Height)
                u_torso_h = abs((u[11].y + u[12].y)/2 - (u[23].y + u[24].y)/2)
                t_torso_h = abs((target_lms[11].y + target_lms[12].y)/2 - (target_lms[23].y + target_lms[24].y)/2)
                scale = u_torso_h / t_torso_h if t_torso_h != 0 else 1

                # 3. ANCHOR: Shoulder Midpoint
                u_mid_sh = [(u[11].x + u[12].x)/2, (u[11].y + u[12].y)/2]
                t_mid_sh = [(target_lms[11].x + target_lms[12].x)/2, (target_lms[11].y + target_lms[12].y)/2]

                def get_projected_pt(idx):
                    """Maps target points onto the user's current scale and position."""
                    px = (target_lms[idx].x - t_mid_sh[0]) * scale + u_mid_sh[0]
                    py = (target_lms[idx].y - t_mid_sh[1]) * scale + u_mid_sh[1]
                    return (int(px * w), int(py * h))

                # 4. DRAW TARGET SKELETON (Green)
                for conn in mp_pose.POSE_CONNECTIONS:
                    p1, p2 = get_projected_pt(conn[0]), get_projected_pt(conn[1])
                    cv2.line(frame, p1, p2, (0, 255, 0), 1)

                # 5. ANGLE ANALYSIS & DIRECTIONAL LINES
                angle_indices = [
                    (12, 14, 16, "R-Elbow"), (11, 13, 15, "L-Elbow"),
                    (24, 26, 28, "R-Knee"), (23, 25, 27, "L-Knee"),
                    (14, 12, 24, "R-Shoulder"), (13, 11, 23, "L-Shoulder"),
                    (12, 24, 26, "R-Waist"), (11, 23, 25, "L-Waist")
                ]

                for idxs in angle_indices:
                    def xyz(lms, i): return [lms[i].x, lms[i].y, lms[i].z]
                    u_ang = get_angle_3d(xyz(u, idxs[0]), xyz(u, idxs[1]), xyz(u, idxs[2]))
                    t_ang = get_angle_3d(xyz(target_lms, idxs[0]), xyz(target_lms, idxs[1]), xyz(target_lms, idxs[2]))
                    
                    u_pos = (int(u[idxs[1]].x * w), int(u[idxs[1]].y * h))
                    t_pos = get_projected_pt(idxs[1]) # The 'correct' green point

                    if abs(u_ang - t_ang) < 15:
                        cv2.circle(frame, u_pos, 8, (0, 255, 0), -1)
                    else:
                        # Draw Red Point on user
                        cv2.circle(frame, u_pos, 8, (0, 0, 255), -1)
                        # Draw DIRECTIONAL LINE to the Green target point
                        cv2.line(frame, u_pos, t_pos, (0, 255, 255), 2) # Yellow guidance line
                        cv2.circle(frame, t_pos, 5, (0, 255, 0), -1) # Mini green goal point

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()