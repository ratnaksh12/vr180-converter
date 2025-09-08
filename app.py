import sys
import subprocess
import streamlit as st
import os
# Ensure uploads folder exists
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
from convert_vr180 import convert_to_vr180

st.set_page_config(page_title="VR180 3D Converter", layout="wide")

st.title("🎥 VR180 3D Video Converter")
st.markdown("""
Upload your video and convert it to **VR180 (Top-Bottom Stereo)** format.

- Works with VR headsets (Meta Quest, Pico, YouTube VR)
- Injects proper metadata for VR playback
""")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file:
    input_path = os.path.join("uploads", uploaded_file.name)
    output_path = os.path.join("outputs", f"VR180_{uploaded_file.name}")

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(input_path)

    if st.button("Convert to VR180"):
        with st.spinner("Converting to VR180..."):
            try:
                converted_file = convert_to_vr180(input_path, output_path)
                st.success("Conversion complete ✅")
                st.video(converted_file)
                with open(converted_file, "rb") as f:
                    st.download_button("⬇️ Download VR180 Video", f, file_name=os.path.basename(converted_file))
            except Exception as e:
                st.error(f"Error: {e}")
