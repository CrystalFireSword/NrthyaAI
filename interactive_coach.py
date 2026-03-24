import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
from abc import ABC, abstractmethod

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="PoseSync: Interactive Coach")

# ── STYLE ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
  
  .stApp { background: #0a0a0f; color: #e8e8e8; font-family: 'Syne', sans-serif; }
  
  h1 {
    font-family: 'Syne', sans-serif;
    background: linear-gradient(90deg, #00ff9d, #00c6ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  
  .instruction-box {
    background: #111118;
    border-left: 4px solid #00ff9d;
    padding: 20px;
    border-radius: 0 12px 12px 0;
    margin-bottom: 20px;
  }
  
  .instruction-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #fff;
  }
  
  .sub-instruction {
    font-family: 'Space Mono', monospace;
    color: #888;
    font-size: 0.9rem;
  }
  
  .metric-card {
    background: #1a1a24;
    border-radius: 8px;
    padding: 12px;
    margin: 4px;
    text-align: center;
  }
  .metric-val { font-size: 1.5rem; font-weight: bold; color: #00ff9d; }
  .metric-label { font-size: 0.7rem; color: #aaa; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ── ABSTRACT MODEL INTERFACE ──────────────────────────────────────────────────
class PoseModel(ABC):
    @abstractmethod
    def load(self): pass
    
    @abstractmethod
    def process(self, frame): pass
    
    @abstractmethod
    def get_landmarks(self, results): pass  # Returns list of (x, y, z, visibility)

# ── MEDIAPIPE IMPLEMENTATION ──────────────────────────────────────────────────
class MediaPipeModel(PoseModel):
    def __init__(self, complexity=1, smooth=True):
        self.mp_pose = mp.solutions.pose
        self.pose = None
        self.complexity = complexity
        self.smooth = smooth

    def load(self):
        if not self.pose:
            self.pose = self.mp_pose.Pose(
                model_complexity=self.complexity,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                smooth_landmarks=self.smooth
            )
        return self

    def process(self, frame):
        return self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def get_landmarks(self, results):
        if not results.pose_landmarks:
            return None
        # MediaPipe has 33 landmarks
        lms = []
        for lm in results.pose_landmarks.landmark:
            lms.append((lm.x, lm.y, lm.z, lm.visibility))
        return lms

# ── OPENPOSE IMPLEMENTATION (OPENCV DNN) ──────────────────────────────────────
class OpenPoseModel(PoseModel):
    def __init__(self, proto_path, model_path):
        self.net = None
        self.proto = proto_path
        self.model = model_path
        self.points = 18 # COCO (usually) or 25 (Body_25)

    def load(self):
        try:
            self.net = cv2.dnn.readNetFromCaffe(self.proto, self.model)
        except Exception as e:
            st.error(f"Failed to load OpenPose model: {e}")
            self.net = None
        return self

    def process(self, frame):
        if not self.net: return None
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(blob)
        return self.net.forward()

    def get_landmarks(self, output):
        if output is None: return None
        # Parse Heatmaps to Keypoints (simplified)
        H = output.shape[2]
        W = output.shape[3]
        points = []
        for i in range(self.points):
            probMap = output[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            x = (frame.shape[1] * point[0]) / W
            y = (frame.shape[0] * point[1]) / H
            
            if prob > 0.1:
                # Normalize to 0-1 for compatibility with MediaPipe logic
                points.append((x / frame.shape[1], y / frame.shape[0], 0.0, prob)) 
            else:
                points.append((0, 0, 0, 0)) # Not visible
        return points

# ── GEOMETRY ENGINE ───────────────────────────────────────────────────────────
def calculate_angle_3d(a, b, c):
    """3D angle between vectors BA and BC."""
    a, b, c = np.array(a[:3]), np.array(b[:3]), np.array(c[:3])
    ba = a - b
    bc = c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0: return 0.0
    cosine = np.dot(ba, bc) / norm
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

def get_limb_angles(landmarks):
    """
    Extracts key angles from landmarks.
    Indices (MediaPipe):
    11=L_Sho, 13=L_Elb, 15=L_Wri
    12=R_Sho, 14=R_Elb, 16=R_Wri
    23=L_Hip, 25=L_Kne, 27=L_Ank
    24=R_Hip, 26=R_Kne, 28=R_Ank
    """
    if not landmarks or len(landmarks) < 33: return {}
    
    lm = landmarks
    return {
        "Left Elbow": calculate_angle_3d(lm[11], lm[13], lm[15]),
        "Right Elbow": calculate_angle_3d(lm[12], lm[14], lm[16]),
        "Left Knee": calculate_angle_3d(lm[23], lm[25], lm[27]),
        "Right Knee": calculate_angle_3d(lm[24], lm[26], lm[28]),
        "Left Shoulder": calculate_angle_3d(lm[13], lm[11], lm[23]),
        "Right Shoulder": calculate_angle_3d(lm[14], lm[12], lm[24]),
        "Left Hip": calculate_angle_3d(lm[11], lm[23], lm[25]),
        "Right Hip": calculate_angle_3d(lm[12], lm[24], lm[26]),
    }

# ── COACH LOGIC ───────────────────────────────────────────────────────────────
class InteractiveCoach:
    def __init__(self, target_landmarks):
        self.target_angles = get_limb_angles(target_landmarks)
        self.threshold = 15.0
        # Correction sequence: Base -> Core -> Extremities
        self.sequence = [
            ("Left Knee", "Legs"), ("Right Knee", "Legs"),
            ("Left Hip", "Hips"), ("Right Hip", "Hips"),
            ("Left Shoulder", "Torso"), ("Right Shoulder", "Torso"),
            ("Left Elbow", "Arms"), ("Right Elbow", "Arms")
        ]

    def analyze(self, user_landmarks):
        user_angles = get_limb_angles(user_landmarks)
        feedback = []
        status = "Perfect"
        
        # Sequential Check
        primary_correction = None
        
        for joint, group in self.sequence:
            u_a = user_angles.get(joint, 0)
            t_a = self.target_angles.get(joint, 0)
            diff = u_a - t_a
            
            if abs(diff) > self.threshold:
                direction = "Extend/Straighten" if diff < 0 else "Bend/Flex"
                primary_correction = {
                    "joint": joint,
                    "diff": diff,
                    "msg": f"{direction} your {joint}",
                    "group": group
                }
                status = "Correcting"
                break # Stop at first major error for step-by-step coaching
        
        return status, primary_correction, user_angles

# ── UI COMPONENTS ─────────────────────────────────────────────────────────────
st.sidebar.title("🛠️ Model Settings")
model_choice = st.sidebar.selectbox("Pose Model", ["MediaPipe (Recommended)", "OpenPose (Requires Weights)", "MoveNet (Experimental)"])

# Load Model
model = None
if model_choice.startswith("MediaPipe"):
    model = MediaPipeModel(complexity=1).load()
elif model_choice.startswith("OpenPose"):
    st.sidebar.warning("OpenPose requires .prototxt and .caffemodel files.")
    proto = st.sidebar.text_input("Path to .prototxt", "pose_deploy.prototxt")
    weights = st.sidebar.text_input("Path to .caffemodel", "pose_iter_440000.caffemodel")
    if st.sidebar.button("Load OpenPose"):
        model = OpenPoseModel(proto, weights).load()
else:
    st.sidebar.info("MoveNet support coming soon.")
    model = MediaPipeModel(complexity=0).load() # Fallback

# ── MAIN APP ──────────────────────────────────────────────────────────────────
st.title("PoseSync: Interactive Coach")

col_ref, col_live = st.columns([1, 2])

with col_ref:
    st.subheader("Reference Pose")
    ref_file = st.file_uploader("Upload Target", type=["jpg", "png", "jpeg"])
    coach = None
    target_vis = None
    
    if ref_file and model:
        file_bytes = np.asarray(bytearray(ref_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        # Process Reference
        res = model.process(img)
        t_lms = model.get_landmarks(res)
        
        if t_lms:
            coach = InteractiveCoach(t_lms)
            st.success("Reference Pose Analyzed ✅")
            
            # Visualise Reference
            vis_ref = img.copy()
            mp.solutions.drawing_utils.draw_landmarks(
                vis_ref, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            st.image(cv2.cvtColor(vis_ref, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.error("No pose detected in reference.")

with col_live:
    st.subheader("Live Correction")
    run = st.checkbox("Start Camera")
    frame_window = st.empty()
    feedback_window = st.empty()
    
    if run and coach:
        cap = cv2.VideoCapture(0)
        
        while run:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # 1. Process
            res = model.process(frame)
            u_lms = model.get_landmarks(res)
            
            overlay = frame.copy()
            
            if u_lms:
                # 2. Analyze
                status, correction, angles = coach.analyze(u_lms)
                
                # 3. Visual Feedback
                # Draw Skeleton
                mp.solutions.drawing_utils.draw_landmarks(
                    overlay, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                
                # 4. Textual Feedback (The "Coach")
                if correction:
                    # Highlight the specific joint
                    joint_name = correction['joint']
                    # Map name to index (simplified)
                    idx_map = {"Right Knee": 26, "Left Knee": 25, "Right Elbow": 14, "Left Elbow": 13}
                    if joint_name in idx_map:
                        idx = idx_map[joint_name]
                        px, py = int(u_lms[idx][0] * w), int(u_lms[idx][1] * h)
                        cv2.circle(overlay, (px, py), 20, (0, 0, 255), -1)
                        cv2.circle(overlay, (px, py), 30, (0, 0, 255), 2)
                    
                    feedback_html = f"""
                    <div class="instruction-box">
                        <div class="instruction-text">{correction['msg']}</div>
                        <div class="sub-instruction">Focus on your {correction['group']} • Deviation: {abs(correction['diff']):.1f}°</div>
                    </div>
                    """
                else:
                    feedback_html = """
                    <div class="instruction-box" style="border-color: #00ff00;">
                        <div class="instruction-text">Perfect Pose! Hold it! 🌟</div>
                    </div>
                    """
                
                feedback_window.markdown(feedback_html, unsafe_allow_html=True)
                
            else:
                feedback_window.warning("No user detected. Step into frame.")
            
            # Blend overlay
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        cap.release()
