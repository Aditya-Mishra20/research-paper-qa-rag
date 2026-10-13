"""Poster + bake poster as frame 0 (replace, don't add) + verify."""
import os, shutil, subprocess
from imageio_ffmpeg import get_ffmpeg_exe

FF = get_ffmpeg_exe()
WORK = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(WORK)

# Poster: strongest settled frame (CTA + sub settled, t=19.0)
shutil.copy(os.path.join(WORK, "still_19.00.png"), os.path.join(OUT, "brag.jpg"))

def run(*args):
    subprocess.run([FF, "-y", *args], check=True, capture_output=True, text=True)

# 599 frames (drop original frame 0)
run("-i", os.path.join(WORK, "brag_silent.mp4"),
    "-vf", "select=gte(n\\,1),setpts=N/FRAME_RATE/TB",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", os.path.join(WORK, "rest.mp4"))
# 1 poster frame, same size/codec
run("-loop", "1", "-i", os.path.join(OUT, "brag.jpg"),
    "-frames:v", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "-s", "1920x1080", os.path.join(WORK, "head.mp4"))
# concat -> 600 frames, then mux score
run("-i", os.path.join(WORK, "head.mp4"), "-i", os.path.join(WORK, "rest.mp4"),
    "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", os.path.join(WORK, "silent2.mp4"))
run("-i", os.path.join(WORK, "silent2.mp4"), "-i", os.path.join(WORK, "score.wav"),
    "-c:v", "copy", "-c:a", "aac", "-shortest", os.path.join(OUT, "brag.mp4"))

# verify: streams, duration, size
p = subprocess.run([FF, "-i", os.path.join(OUT, "brag.mp4")],
                   capture_output=True, text=True)
print([l.strip() for l in p.stderr.splitlines() if "Duration" in l or "Stream" in l])
print("bytes:", os.path.getsize(os.path.join(OUT, "brag.mp4")))
print("poster bytes:", os.path.getsize(os.path.join(OUT, "brag.jpg")))
print("FINALIZE_OK")
