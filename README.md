# 🎥 VR180 3D Video Converter

A Streamlit app to convert regular videos into **VR180 (Top-Bottom Stereo)** format with proper metadata for playback in VR headsets like **Meta Quest, Pico, or YouTube VR**.

## 🚀 Features
- Upload `.mp4`, `.mov`, `.avi`, `.mkv`
- Convert to **VR180 top-bottom stereo**
- Inject VR metadata using Google’s `spatial-media` tool
- Download the final VR180-ready video

## ⚙️ Setup
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/VR180-Converter.git
cd VR180-Converter

# Create venv
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

# Install deps
pip install -r requirements.txt

# Install spatial-media (manual step)
git clone https://github.com/google/spatial-media.git
cd spatial-media
python setup.py install
cd ..

# Run the app
streamlit run app.py
