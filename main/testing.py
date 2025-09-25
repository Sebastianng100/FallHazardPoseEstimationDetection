import cv2, math
import mediapipe as mp
import gradio as gr
import numpy as np

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# ------------ helpers ------------
def pt(lm, idx): return lm[idx].x, lm[idx].y, lm[idx].visibility
def mid(a, b):   return ((a[0]+b[0])/2, (a[1]+b[1])/2, min(a[2], b[2]))

def angle_from_vertical(p1, p2):
    # angle (deg) between vector p1->p2 and vertical axis (0 = vertical, 90 = horizontal)
    vx, vy = (p2[0]-p1[0], p2[1]-p1[1])
    vlen = math.hypot(vx, vy) + 1e-6
    # vertical axis (0,-1) in image coords (y down), so flip sign
    cos = (-vy) / vlen
    ang = math.degrees(math.acos(max(-1,min(1,cos))))
    return ang  # 0 upright, 90 horizontal

def bbox_of_points(pts):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))

# ------------ core logic ------------
def classify_fall(lm, img_h):
    """Return ('Fall'|'No Fall', debug dict). lm coords are normalized [0..1]."""
    Pose = mp_pose.PoseLandmark
    # key joints (x,y,vis)
    nose = pt(lm, Pose.NOSE)
    lhip = pt(lm, Pose.LEFT_HIP); rhip = pt(lm, Pose.RIGHT_HIP)
    lsho = pt(lm, Pose.LEFT_SHOULDER); rsho = pt(lm, Pose.RIGHT_SHOULDER)
    lank = pt(lm, Pose.LEFT_ANKLE); rank = pt(lm, Pose.RIGHT_ANKLE)
    lknee = pt(lm, Pose.LEFT_KNEE); rknee = pt(lm, Pose.RIGHT_KNEE)

    # gate low-visibility frames
    vis_ok = [nose, lhip, rhip, lsho, rsho]
    if sum(v for _,_,v in vis_ok)/len(vis_ok) < 0.5:
        return "No Person Detected", {"reason":"low visibility"}

    # mid points
    mid_sho = mid(lsho, rsho)
    mid_hip = mid(lhip, rhip)

    # 1) torso angle (0=vertical, 90=horizontal)
    torso_ang = angle_from_vertical(mid_sho, mid_hip)   # deg

    # 2) head below hips?
    head_below_hips = nose[1] > (lhip[1] + rhip[1]) / 2

    # 3) body COM (avg of shoulders+hips+knees+ankles) near bottom of frame?
    core = [lsho, rsho, lhip, rhip, lknee, rknee, lank, rank]
    com_y = sum(p[1] for p in core) / len(core)         # normalized 0..1
    com_low = com_y > 0.75                              # tune (0.70–0.80)

    # 4) pose bbox aspect ratio wide?
    key_pts = [lsho, rsho, lhip, rhip, lknee, rknee]
    x1,y1,x2,y2 = bbox_of_points(key_pts)
    w, h = (x2-x1), (y2-y1)
    wide_body = (w / (h+1e-6)) > 1.2

    # vote
    votes = int(torso_ang > 55) + int(head_below_hips) + int(com_low) + int(wide_body)
    status = "Fall" if votes >= 2 else "No Fall"
    dbg = dict(torso_ang=round(torso_ang,1), head_below_hips=bool(head_below_hips),
               com_y=round(com_y,2), wide_body=bool(wide_body), votes=votes)
    return status, dbg

def detect_fall_from_image(image_rgb):
    if image_rgb is None: return "No Image", None
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        res = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    if not res.pose_landmarks:
        return "No Person Detected", image_rgb

    status, dbg = classify_fall(res.pose_landmarks.landmark, img_bgr.shape[0])

    vis = img_bgr.copy()
    mp_draw.draw_landmarks(vis, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # draw status + a couple of debug values
    txt = f"{status} | torso={dbg['torso_ang']}° votes={dbg['votes']}"
    cv2.rectangle(vis, (10,10), (10+320, 10+30), (0,0,0), -1)
    cv2.putText(vis, txt, (18,33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return status, cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

# ------------ Gradio UI ------------
with gr.Blocks() as demo:
    gr.Markdown("## 🧍‍♂️ Fake Fall Detection (Pose-based, improved rules)")
    img = gr.Image(type="numpy", label="Upload Image")
    run = gr.Button("▶ Run Inference", variant="primary")
    out_txt = gr.Textbox(label="Result")
    out_img = gr.Image(label="Pose Visualization")
    run.click(detect_fall_from_image, inputs=img, outputs=[out_txt, out_img])

if __name__ == "__main__":
    demo.launch(share=True)   # gets you a public link
