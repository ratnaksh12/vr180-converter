import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import subprocess
import shutil

# ---------------- Depth Stub (fast CPU fallback) ----------------
def fake_depth_map(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (31, 31), 0)

# ---------------- Stereo (DIBR-like) ----------------
def dibr_stereo(frame, depth, ipd=10):
    h, w = frame.shape[:2]
    disparity = (255 - depth) / 255.0 * ipd
    left = np.zeros_like(frame)
    right = np.zeros_like(frame)
    for y in range(h):
        for x in range(w):
            dx = int(disparity[y, x])
            if x - dx >= 0:
                left[y, x] = frame[y, x - dx]
            if x + dx < w:
                right[y, x] = frame[y, x + dx]
    return left, right

# ---------------- Optical Effects ----------------
def apply_vignette(frame, strength=2.0):
    rows, cols = frame.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols / strength)
    kernel_y = cv2.getGaussianKernel(rows, rows / strength)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    vignette = np.copy(frame)
    for i in range(3):
        vignette[:, :, i] = vignette[:, :, i] * mask
    return vignette

def apply_barrel_distortion(frame, k=-0.05):
    h, w = frame.shape[:2]
    fx, fy = w, h
    K = np.array([[fx, 0, w/2],
                  [0, fy, h/2],
                  [0,  0,   1]], dtype=np.float32)
    D = np.array([k, 0, 0, 0], dtype=np.float32)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

def apply_panini(frame, d=1.0, k=0.0):
    h, w = frame.shape[:2]
    f = w / np.pi
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    x = (map_x - w/2) / f
    y = (map_y - h/2) / f
    denom = (d - k) + (k * np.cosh(x))
    x_out = np.sinh(x) / denom
    y_out = y * (d*np.cosh(x) - k) / denom
    map_x = (x_out * f + w/2).astype(np.float32)
    map_y = (y_out * f + h/2).astype(np.float32)
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def apply_foveated_blur(frame, start_radius=0.85):
    h, w = frame.shape[:2]
    center = (w//2, h//2)
    mask = np.zeros((h, w), np.float32)
    cv2.circle(mask, center, int(min(h, w) * start_radius / 2), 1, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 50)
    blurred = cv2.GaussianBlur(frame, (21, 21), 50)
    mask_3ch = cv2.merge([mask]*3)
    return (frame * mask_3ch + blurred * (1-mask_3ch)).astype(np.uint8)

def apply_disc_containment(frame, blur_strength=12, feather=0.35):
    h, w = frame.shape[:2]
    center = (w//2, h//2)
    radius = int(min(h, w) * 0.495)

    mask = np.zeros((h, w), np.float32)
    cv2.circle(mask, center, radius, 1, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), int(min(h, w) * feather))

    blurred = cv2.GaussianBlur(frame, (0, 0), blur_strength)
    blurred = cv2.convertScaleAbs(blurred, alpha=1.1, beta=10)

    mask_3ch = cv2.merge([mask]*3)
    contained = (frame * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
    return contained

# ---------------- Main Pipeline ----------------
def process_video_pipeline(input_path, output_path, fps=30, eye_w=480, eye_h=480):
    barrel_k = -0.05
    vignette_strength = 2.0
    fovea_radius = 0.85

    # temp silent video
    temp_video = output_path.replace(".mp4", "_silent.mp4")

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (eye_w*2, eye_h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (eye_w, eye_h))

        depth = fake_depth_map(frame)
        left_eye, right_eye = dibr_stereo(frame, depth, ipd=10)

        left_eye = apply_disc_containment(
            apply_foveated_blur(
                apply_vignette(
                    apply_barrel_distortion(
                        apply_panini(left_eye), k=barrel_k),
                    strength=vignette_strength),
                start_radius=fovea_radius)
        )

        right_eye = apply_disc_containment(
            apply_foveated_blur(
                apply_vignette(
                    apply_barrel_distortion(
                        apply_panini(right_eye), k=barrel_k),
                    strength=vignette_strength),
                start_radius=fovea_radius)
        )

        # --- Side by Side (SBS) ---
        stereo_frame = np.hstack((left_eye, right_eye))
        out.write(stereo_frame)

    cap.release()
    out.release()

    # ---- Add Audio Back with ffmpeg ----
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_video,
            "-i", input_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            output_path
        ]
        subprocess.run(cmd, check=True)
        os.remove(temp_video)  # cleanup
    except Exception as e:
        print("⚠️ ffmpeg audio merge failed:", e)
        shutil.copy(temp_video, output_path)  # fallback copy
        try:
            os.remove(temp_video)
        except:
            pass

# ---------------- Streamlit UI ----------------
st.title("🎥 Palace – 2D → VR180 Converter")

uploaded_file = st.file_uploader("Upload your 2D video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(uploaded_file.read())
        input_path = tmp_in.name

    output_path = os.path.join(tempfile.gettempdir(), "output_vr180_sbs.mp4")

    if st.button("🚀 Start Conversion"):
        with st.spinner("Processing with fisheye, vignette, disc containment…"):
            process_video_pipeline(input_path, output_path)

        st.success("✅ Conversion completed!")
        st.video(output_path)
        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download VR180 SBS Video", f, file_name="vr180_sbs.mp4")
