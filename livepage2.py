import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from mediapipe.python.solutions import pose as mp_pose
import time

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="PoseSync – Real-Time Trainer")

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: #0a0a0f;
    color: #e8e8e8;
  }
  .stApp { background: #0a0a0f; }

  h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #00ff9d, #00c6ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
  }
  .subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #555;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 2px;
    margin-bottom: 24px;
  }

  /* Score badge */
  .score-box {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 16px 24px;
    text-align: center;
    margin-bottom: 12px;
  }
  .score-val {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
  }
  .score-label {
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #555;
    margin-top: 4px;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0d0d14 !important;
    border-right: 1px solid #1a1a28;
  }

  /* Streamlit checkbox */
  .stCheckbox label { font-family: 'Space Mono', monospace; font-size: 0.85rem; }

  /* Angle table rows */
  .angle-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 12px;
    margin: 3px 0;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
  }
  .angle-row.good  { background: #00ff9d18; color: #00ff9d; }
  .angle-row.bad   { background: #ff334418; color: #ff6677; }
  .angle-name { font-weight: 700; }
  .angle-delta { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("<h1>PoseSync</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-Time Pose Correction Engine</p>', unsafe_allow_html=True)

# ── POSE ENGINE ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_pose_engine():
    return mp_pose.Pose(
        model_complexity=1,  # Reduced for better real-time latency
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True
    )

pose_engine = load_pose_engine()

# ── STATE MANAGEMENT ─────────────────────────────────────────────────────────
if 'angle_history' not in st.session_state:
    st.session_state.angle_history = {}

def get_smoothed_angle(name, new_angle, window=5):
    """Exponential Moving Average to reduce jitter."""
    if name not in st.session_state.angle_history:
        st.session_state.angle_history[name] = new_angle
    
    alpha = 2 / (window + 1)
    smoothed = alpha * new_angle + (1 - alpha) * st.session_state.angle_history[name]
    st.session_state.angle_history[name] = smoothed
    return smoothed

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_angle_3d(a, b, c):
    """3-D joint angle at vertex b."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / norm, -1.0, 1.0))))


def lm_xyz(lms, i):
    return [lms[i].x, lms[i].y, lms[i].z]


# All joints to analyse: (joint_A, vertex, joint_C, display_name)
JOINT_DEFS = [
    (12, 14, 16, "R-Elbow"),
    (11, 13, 15, "L-Elbow"),
    (24, 26, 28, "R-Knee"),
    (23, 25, 27, "L-Knee"),
    (14, 12, 24, "R-Shoulder"),
    (13, 11, 23, "L-Shoulder"),
    (12, 24, 26, "R-Hip"),
    (11, 23, 25, "L-Hip"),
    (24, 12, 11, "R-Upper-Back"),
    (23, 11, 12, "L-Upper-Back"),
    (26, 24, 23, "R-Pelvis"),
    (25, 23, 24, "L-Pelvis"),
]

THRESHOLD = 15  # degrees

# Colour palette
COL_TARGET   = (0, 255, 140)    # neon green – ghost skeleton
COL_GOOD     = (0, 255, 140)    # green dot  – joint OK
COL_BAD      = (60, 60, 255)    # red dot    – joint off (BGR)
COL_USER_SK  = (180, 180, 180)  # grey – user skeleton
COL_TEXT_G   = (0, 255, 140)
COL_TEXT_R   = (80, 80, 255)


def draw_skeleton(frame, lms, w, h, color, thickness=1, alpha=0.55):
    """Draw all POSE_CONNECTIONS for a landmark set onto frame."""
    overlay = frame.copy()
    for conn in mp_pose.POSE_CONNECTIONS:
        i1, i2 = conn
        if lms[i1].visibility < 0.3 or lms[i2].visibility < 0.3:
            continue
        p1 = (int(lms[i1].x * w), int(lms[i1].y * h))
        p2 = (int(lms[i2].x * w), int(lms[i2].y * h))
        cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_arrow(frame, src, dst, error_deg, thickness=4):
    """Draw an arrowhead from src to dst with color based on error."""
    # Color interpolation: Green (0,255,140) -> Yellow (0,255,255) -> Red (60,60,255)
    if error_deg < 15:
        color = (0, 255, 140)
    elif error_deg < 30:
        color = (0, 255, 255)
    else:
        color = (60, 60, 255)
    
    cv2.arrowedLine(frame, src, dst, color, thickness, cv2.LINE_AA, tipLength=0.2)


def put_label(frame, text, pos, color, scale=0.5, thickness=1):
    x, y = pos
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_DUPLEX,
                scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_DUPLEX,
                scale, color, thickness, cv2.LINE_AA)


def overlay_hud(frame, score, joint_results, w, h):
    """Draw the score HUD and per-joint deltas onto the frame."""
    # Score bar top-left
    bar_w = 300
    bar_h = 60
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + bar_w, 10 + bar_h), (15, 15, 25), -1)
    fill = int(bar_w * score / 100)
    bar_color = COL_GOOD if score >= 70 else (0, 165, 255) if score >= 40 else COL_BAD
    cv2.rectangle(overlay, (10, 10), (10 + fill, 10 + bar_h), bar_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    put_label(frame, f"SCORE  {score:.0f}%", (18, 50), (255, 255, 255), 1.0, 2)

    # Joint deltas – right side column
    x_col = w - 240
    y_start = 18
    for name, u_ang, t_ang, ok in joint_results:
        delta = u_ang - t_ang
        sign  = "+" if delta > 0 else ""
        color = COL_TEXT_G if ok else COL_TEXT_R
        label = f"{name:<14} {sign}{delta:+.0f}"
        put_label(frame, label, (x_col, y_start), color, 0.65, 2)
        y_start += 26


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Target Pose")
    target_file = st.file_uploader("Upload reference image", type=["jpg", "jpeg", "png"])
    target_lms  = None
    target_preview = None

    if target_file:
        file_bytes = np.asarray(bytearray(target_file.read()), dtype=np.uint8)
        t_img_bgr  = cv2.imdecode(file_bytes, 1)
        t_img_rgb  = cv2.cvtColor(t_img_bgr, cv2.COLOR_BGR2RGB)
        t_res      = pose_engine.process(t_img_rgb)

        if t_res.pose_landmarks:
            target_lms = t_res.pose_landmarks.landmark
            # Draw skeleton preview
            preview = t_img_rgb.copy()
            mp.solutions.drawing_utils.draw_landmarks(
                preview,
                t_res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 200, 100), thickness=2, circle_radius=3),
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 200, 100), thickness=2)
            )
            st.image(preview, caption="Target Pose Detected ✅", use_container_width=True)
        else:
            st.error("No pose detected in the uploaded image. Try a clearer full-body photo.")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    threshold_deg = st.slider("Correction threshold (°)", 5, 30, 15, 1)
    show_user_skeleton = st.checkbox("Show user skeleton", value=True)
    show_target_ghost  = st.checkbox("Show target ghost skeleton", value=True)
    show_arrows        = st.checkbox("Show correction arrows", value=True)
    show_angle_labels  = st.checkbox("Show angle delta labels", value=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:0.7rem;color:#444;line-height:1.8'>
    🟢 Green dot = joint within target<br>
    🔴 Red dot = joint needs correction<br>
    ➡️ Arrow = direction to move<br>
    Ghost = target pose overlay
    </div>
    """, unsafe_allow_html=True)

# ── MAIN LAYOUT ───────────────────────────────────────────────────────────────
col_feed, col_stats = st.columns([3, 1])

with col_feed:
    run = st.checkbox("▶  Start Live Camera", value=False)
    FRAME_WINDOW = st.empty()

with col_stats:
    st.markdown("### Joint Analysis")
    score_placeholder  = st.empty()
    joints_placeholder = st.empty()

# ── LIVE LOOP ─────────────────────────────────────────────────────────────────
if run:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Cannot read camera frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_engine.process(rgb)

        joint_results = []
        score = 0.0

        if results.pose_landmarks:
            u = results.pose_landmarks.landmark

            # Visibility guard – need torso to be visible
            key_pts = [u[11], u[12], u[23], u[24]]
            if any(pt.visibility < 0.5 for pt in key_pts):
                put_label(frame, "STEP BACK – DETECTING TORSO...",
                          (w // 2 - 250, h // 2), (0, 80, 255), 0.9, 2)
            else:
                # ── DRAW USER SKELETON ──────────────────────────────────────
                if show_user_skeleton:
                    draw_skeleton(frame, u, w, h, COL_USER_SK, thickness=6, alpha=0.8)

                # ── TARGET GHOST PROJECTION ────────────────────────────────
                if target_lms is not None:
                    # Anisotropic scale: X uses shoulder-width ratio, Y uses torso-height ratio
                    # This stops the ghost from expecting wider/narrower shoulders than the user has
                    u_sh_w  = abs(u[11].x - u[12].x)
                    t_sh_w  = abs(target_lms[11].x - target_lms[12].x)
                    scale_x = (u_sh_w / t_sh_w) if t_sh_w > 0 else 1.0

                    u_torso = abs((u[11].y + u[12].y) / 2 - (u[23].y + u[24].y) / 2)
                    t_torso = abs((target_lms[11].y + target_lms[12].y) / 2 -
                                  (target_lms[23].y + target_lms[24].y) / 2)
                    scale_y = (u_torso / t_torso) if t_torso > 0 else 1.0

                    # Anchor: shoulder midpoint
                    u_mid = np.array([(u[11].x + u[12].x) / 2,
                                      (u[11].y + u[12].y) / 2])
                    t_mid = np.array([(target_lms[11].x + target_lms[12].x) / 2,
                                      (target_lms[11].y + target_lms[12].y) / 2])

                    def project(idx):
                        """Map a target landmark to user-space pixel coords with per-axis scaling."""
                        px = (target_lms[idx].x - t_mid[0]) * scale_x + u_mid[0]
                        py = (target_lms[idx].y - t_mid[1]) * scale_y + u_mid[1]
                        return (int(np.clip(px * w, 0, w - 1)),
                                int(np.clip(py * h, 0, h - 1)))

                    # Draw ghost skeleton
                    if show_target_ghost:
                        ghost_overlay = frame.copy()
                        for conn in mp_pose.POSE_CONNECTIONS:
                            i1, i2 = conn
                            if (target_lms[i1].visibility < 0.3 or
                                    target_lms[i2].visibility < 0.3):
                                continue
                            cv2.line(ghost_overlay, project(i1), project(i2),
                                     COL_TARGET, 6, cv2.LINE_AA)
                        cv2.addWeighted(ghost_overlay, 0.6, frame, 0.4, 0, frame)

                    # ── JOINT ANALYSIS ──────────────────────────────────────
                    good_count = 0
                    for (a_idx, v_idx, c_idx, name) in JOINT_DEFS:
                        # Skip if any of the 3 landmarks are not visible enough
                        vis_ok = (u[a_idx].visibility > 0.4 and
                                  u[v_idx].visibility > 0.4 and
                                  u[c_idx].visibility > 0.4)
                        if not vis_ok:
                            continue

                        u_ang_raw = get_angle_3d(lm_xyz(u, a_idx),
                                                 lm_xyz(u, v_idx),
                                                 lm_xyz(u, c_idx))
                        u_ang = get_smoothed_angle(name, u_ang_raw)
                        
                        t_ang = get_angle_3d(lm_xyz(target_lms, a_idx),
                                             lm_xyz(target_lms, v_idx),
                                             lm_xyz(target_lms, c_idx))

                        error_deg = abs(u_ang - t_ang)
                        ok    = error_deg < threshold_deg
                        joint_results.append((name, u_ang, t_ang, ok))
                        if ok:
                            good_count += 1

                        # Pixel positions
                        u_pos = (int(u[v_idx].x * w), int(u[v_idx].y * h))
                        t_pos = project(v_idx)

                        if ok:
                            cv2.circle(frame, u_pos, 18, COL_GOOD, -1, cv2.LINE_AA)
                            cv2.circle(frame, u_pos, 18, (255, 255, 255), 3, cv2.LINE_AA)
                        else:
                            cv2.circle(frame, u_pos, 18, COL_BAD, -1, cv2.LINE_AA)
                            cv2.circle(frame, u_pos, 18, (255, 255, 255), 3, cv2.LINE_AA)

                            if show_arrows:
                                draw_arrow(frame, u_pos, t_pos, error_deg, 6)
                            cv2.circle(frame, t_pos, 12, COL_TARGET, -1, cv2.LINE_AA)
                            cv2.circle(frame, t_pos, 12, (255, 255, 255), 3, cv2.LINE_AA)

                        if show_angle_labels:
                            delta = u_ang - t_ang
                            sign  = "+" if delta >= 0 else ""
                            label = f"{sign}{delta:.0f}"
                            color = COL_TEXT_G if ok else COL_TEXT_R
                            put_label(frame, label,
                                      (u_pos[0] + 20, u_pos[1] - 8),
                                      color, 0.9, 2)

                    # Score
                    visible_joints = len(joint_results)
                    score = (good_count / visible_joints * 100) if visible_joints else 0.0
                    overlay_hud(frame, score, joint_results, w, h)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                           channels="RGB", use_container_width=True)

        # ── SIDEBAR STATS UPDATE ────────────────────────────────────────────
        if target_lms is not None and joint_results:
            score_color = "#00ff9d" if score >= 70 else "#ffaa00" if score >= 40 else "#ff4455"
            score_placeholder.markdown(f"""
            <div class="score-box">
              <div class="score-val" style="color:{score_color}">{score:.0f}%</div>
              <div class="score-label">Pose Match</div>
            </div>
            """, unsafe_allow_html=True)

            rows_html = ""
            for name, u_ang, t_ang, ok in joint_results:
                delta = u_ang - t_ang
                sign  = "+" if delta >= 0 else ""
                cls   = "good" if ok else "bad"
                rows_html += f"""
                <div class="angle-row {cls}">
                  <span class="angle-name">{name}</span>
                  <span class="angle-delta">{sign}{delta:.0f}°</span>
                </div>"""
            joints_placeholder.markdown(rows_html, unsafe_allow_html=True)

    cap.release()

elif not run:
    if target_lms is None:
        FRAME_WINDOW.markdown("""
        <div style='
          background:#0d0d14;
          border:1px dashed #1e1e2e;
          border-radius:16px;
          padding:80px 40px;
          text-align:center;
          font-family:Space Mono,monospace;
          color:#333;
        '>
          <div style='font-size:3rem;margin-bottom:16px'>📷</div>
          <div style='font-size:1rem;color:#555'>Upload a target pose image in the sidebar,<br>then start the live camera.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        FRAME_WINDOW.markdown("""
        <div style='
          background:#0d0d14;
          border:1px dashed #00ff9d44;
          border-radius:16px;
          padding:80px 40px;
          text-align:center;
          font-family:Space Mono,monospace;
        '>
          <div style='font-size:3rem;margin-bottom:16px'>✅</div>
          <div style='font-size:1rem;color:#00ff9d'>Target pose loaded!<br>
          <span style='color:#555'>Check "Start Live Camera" above to begin.</span></div>
        </div>
        """, unsafe_allow_html=True)