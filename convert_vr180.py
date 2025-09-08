import os
import subprocess
import sys

def convert_to_vr180(input_path, output_path):
    """
    Convert a normal video into VR180 top-bottom format and inject metadata.
    """
    temp_path = output_path.replace(".mp4", "_stereo.mp4")

    # Step 1: Duplicate video into top-bottom layout
    cmd1 = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter_complex", "[0:v]split=2[v1][v2];[v1][v2]vstack=2[vout]",
        "-map", "[vout]", "-map", "0:a?",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        temp_path
    ]
    subprocess.run(cmd1, check=True)

    # Step 2: Inject VR180 metadata
    cmd2 = [
        sys.executable, "-m", "spatialmedia",
        "-i", "--stereo=top-bottom", "--180",
        temp_path, output_path
    ]
    subprocess.run(cmd2, check=True)

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return output_path
