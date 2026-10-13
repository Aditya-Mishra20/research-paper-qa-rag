"""brag-slim renderer: Research Paper QA launch video. Every frame is a pure function of t."""
from __future__ import annotations
import math, os, subprocess, sys, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, DUR = 1920, 1080, 30, 20
BG = (14, 17, 23); PANEL = (22, 27, 34); ACCENT = (255, 75, 75)
GOLD = (255, 209, 102); WHITE = (255, 255, 255); DIM = (154, 164, 178)
GREEN = (63, 185, 80); BLUE = (88, 166, 255)

def font(sz):
    for p in ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            try: return ImageFont.truetype(p, sz)
            except Exception: pass
    return ImageFont.load_default()

F96, F72, F56, F44, F36, F30 = font(96), font(72), font(56), font(44), font(36), font(30)

def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ease(x): x = clamp(x); return 1 - (1 - x) ** 3
def center(d, y, text, f, fill):  # settled centered text
    l, t, r, b = d.textbbox((0, 0), text, font=f)
    d.text(((W - (r - l)) / 2, y), text, font=f, fill=fill)

def pdf_rain(img, t):
    d = ImageDraw.Draw(img)
    for i in range(26):
        x = (i * 373) % (W - 120)
        y = (i * 197 + t * (260 + (i % 5) * 90)) % (H + 300) - 200
        d.rounded_rectangle([x, y, x + 110, y + 140], 12, fill=(46, 54, 70), outline=(96, 106, 128), width=2)
        d.rectangle([x + 18, y + 22, x + 92, y + 34], fill=ACCENT)
        for k in range(3):
            d.rectangle([x + 18, y + 48 + k * 20, x + 92 - (i * 13 + k * 29) % 40, y + 58 + k * 20], fill=(120, 130, 150))

def scene1(t):  # 0-2.5 hook
    img = Image.new("RGB", (W, H), BG); pdf_rain(img, t); d = ImageDraw.Draw(img)
    s = ease(t / 0.6)
    d.rounded_rectangle([W/2-560*s, 300, W/2+560*s, 470], 24, fill=(22, 27, 34))
    if t > 0.35: center(d, 340, "DROWNING IN PDFs?", F96, WHITE)
    n = int(394 * ease((t - 0.6) / 1.6)); l, _, r, _ = d.textbbox((0, 0), f"{n} chunks", font=F44)
    d.rounded_rectangle([(W-(r-l))/2 - 28, 505, (W+(r-l))/2 + 28, 575], 20, fill=(14, 17, 23))
    d.text(((W-(r-l))/2, 512), f"{n} chunks", font=F44, fill=GOLD)
    return img

def app_window(d, built):
    d.rounded_rectangle([230, 150, 1690, 940], 28, fill=PANEL, outline=(48, 56, 68), width=3)
    d.rectangle([230, 150, 650, 940], fill=(18, 22, 29))  # sidebar
    d.text((270, 180), "Papers", font=F44, fill=WHITE)

def scene2(t):  # 2.5-6 reveal (t local)
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img); app_window(d, t)
    if t > 0.3:
        d.rounded_rectangle([270, 260, 610, 360], 14, outline=ACCENT, width=3)
        d.text((295, 290), "Upload PDF or TXT", font=F30, fill=WHITE)
    if t > 0.9:
        d.rounded_rectangle([270, 390, 610, 480], 14, outline=(60, 68, 82), width=2)
        d.text((295, 415), "arXiv ID  2401.12345", font=F30, fill=DIM)
    if t > 1.5:
        d.text((700, 200), "Research Paper QA", font=F72, fill=WHITE)
        d.text((700, 290), "upload. index. ask.", font=F44, fill=GOLD)
    if t > 1.8:
        for i, chip in enumerate(["FastAPI", "Chroma", "MiniLM", "Gemini"]):
            x = 700 + i * 230
            d.rounded_rectangle([x, 380, x + 200, 440], 20, fill=(30, 36, 46))
            l, _, r, _ = d.textbbox((0, 0), chip, font=F30)
            d.text((x + (200-(r-l))/2, 392), chip, font=F30, fill=BLUE)
    return img

def scene3(t):  # 6-10 ingest
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    nb = min(24, int(t * 8))
    for i in range(24):
        x = 360 + i * 64
        h = 40 + ((i * 53) % 90) if i < nb else 12
        d.rectangle([x, 880 - h, x + 44, 880], fill=GREEN if i < nb else (48, 56, 68))
    d.polygon([(760, 620), (1160, 620), (1060, 760), (860, 760)], fill=(30, 36, 46), outline=ACCENT, width=4)
    fy = 180 + ease(t / 1.2) * 260  # file card falls into funnel
    d.rounded_rectangle([790, fy, 1130, fy + 150], 14, fill=(240, 240, 240))
    d.rectangle([818, fy + 22, 890, fy + 38], fill=ACCENT)
    d.text((818, fy + 52), "2505.10468v4", font=F36, fill=(40, 40, 40))
    d.text((818, fy + 100), "arXiv PDF", font=F30, fill=(110, 110, 110))
    if t > 1.6:
        center(d, 920, "Indexed  (394 chunks)", F72, WHITE)
    if t > 2.4:
        txt = "sha256 dedup  •  local MiniLM embeddings  •  0 quota"
        l, _, r, _ = d.textbbox((0, 0), txt, font=F36)
        d.rounded_rectangle([(W-(r-l))/2 - 30, 1005, (W+(r-l))/2 + 30, 1062], 24, fill=(22, 60, 38))
        d.text(((W-(r-l))/2, 1012), txt, font=F36, fill=(140, 230, 170))
    return img

Q = "what is ai agent?"
A_LINES = ["An AI agent perceives, reasons and acts to complete goals",
           "across tools and documents, citing every claim as it goes."]
PAGES = ["p.6", "p.3", "p.21", "p.23", "p.32"]

def scene4(t):  # 10-15 ask
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    d.rounded_rectangle([460, 140, 1460, 250], 24, fill=(38, 60, 110))
    nq = min(len(Q), int(t * 14))
    d.text((510, 170), Q[:nq] + ("▍" if nq < len(Q) else ""), font=F44, fill=WHITE)
    if t > 1.4:
        at = t - 1.4
        d.rounded_rectangle([460, 300, 1460, 560], 24, fill=(30, 42, 30))
        shown = []
        for i, ln in enumerate(A_LINES):
            part = ln[:max(0, int(at * 40 - i * 45))]
            if part: shown.append((part, i))
        y = 330
        for part, i in shown:
            d.text((510, y), part, font=F36, fill=(230, 255, 230)); y += 60
        if at > 1.2:
            d.text((510, y), "[2505.10468v4, p.6]  [p.3]", font=F36, fill=GOLD)
        if at > 1.8:
            for i, pg in enumerate(PAGES):
                x = 500 + i * 190
                d.rounded_rectangle([x, 620, x + 165, 690], 20, outline=GOLD, width=3)
                l, _, r, _ = d.textbbox((0, 0), pg, font=F36)
                d.text((x + (165-(r-l))/2, 632), pg, font=F36, fill=GOLD)
        if at > 2.4: center(d, 730, "Every claim cites its page.", F44, WHITE)
    return img

def scene5(t):  # 15-20 outro
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    for i, pg in enumerate(PAGES):
        a = t * 0.9 + i * 2 * math.pi / 5
        x, y = W/2 + math.cos(a) * (470 - t * 40), 420 + math.sin(a) * (175 - t * 10)
        d.rounded_rectangle([x - 80, y - 34, x + 80, y + 34], 18, outline=GOLD, width=3)
        l, _, r, _ = d.textbbox((0, 0), pg, font=F36)
        d.text((x - (r-l)/2, y - 24), pg, font=F36, fill=GOLD)
    r = 60 + 10 * math.sin(t * 4)
    d.ellipse([W/2 - r, 400 - r, W/2 + r, 400 + r], fill=(40, 30, 30), outline=ACCENT, width=5)
    l, _, rr, _ = d.textbbox((0, 0), "QA", font=F72)
    d.text(((W-(rr-l))/2, 362), "QA", font=F72, fill=WHITE)
    if t > 0.8: center(d, 600, "Stop reading. Start interrogating.", F72, WHITE)
    if t > 1.6: center(d, 710, "localhost:8501  •  free-tier RAG that cites its sources", F36, DIM)
    return img

BOUNDS = [(0, 2.5, scene1), (2.5, 6.0, scene2), (6.0, 10.0, scene3), (10.0, 15.0, scene4), (15.0, 20.0, scene5)]
def frame_at(t):
    t = max(0, min(DUR - 1e-3, t))
    for s, e, fn in BOUNDS:
        if s <= t < e:
            lt = t - s
            img = fn(lt)
            dip = clamp(lt / 0.25)  # dip-through-bg entrance (no muddy crossfade)
            if dip < 1 and s > 0:
                base = Image.new("RGB", (W, H), BG)
                return Image.blend(base, img, ease(dip))
            return img
    return scene5(DUR - 15.0)

# ---- audio: one 20s synth piece, A minor, 100bpm ----
def score(sr=44100):
    n = int(DUR * sr); t = np.arange(n) / sr; y = np.zeros(n)
    step = 60 / 100 / 2
    bass = [110, 0, 130.8, 0, 164.8, 0, 130.8, 98] * 14
    for i, f in enumerate(bass[:int(DUR / step)]):
        if not f: continue
        s0 = int(i * step * sr); s1 = min(n, s0 + int(step * sr * 0.95))
        tt = np.arange(s1 - s0) / sr
        y[s0:s1] += 0.22 * np.sin(2*np.pi*f*tt) * np.exp(-tt*3)
    kick = np.zeros(n)
    for b in range(int(DUR / (60/100))):
        s0 = int(b * (60/100) * sr); s1 = min(n, s0 + int(0.18*sr))
        tt = np.arange(s1-s0)/sr
        kick[s0:s1] += np.sin(2*np.pi*(120*np.exp(-tt*22)+45)*tt) * np.exp(-tt*16)
    y += 0.5 * kick
    rng = np.random.default_rng(7)
    hat = rng.standard_normal(n) * 0.05 * ((t * 4) % 1 < 0.12)
    y += hat
    for ev, f0 in [(2.5, 440), (6.0, 660), (8.4, 880), (11.8, 990), (15.0, 523)]:
        s0 = int(ev*sr); s1 = min(n, s0+int(0.4*sr)); tt = np.arange(s1-s0)/sr
        y[s0:s1] += 0.25*np.sin(2*np.pi*f0*tt)*np.exp(-tt*8)
    ris = (t > 0.4) & (t < 2.5); y[ris] += 0.10*rng.standard_normal(ris.sum())*(t[ris]-0.4)
    s0 = int(18.6*sr); s1 = min(n, s0+int(1.2*sr)); tt = np.arange(s1-s0)/sr
    y[s0:s1] += 0.4*np.sin(2*np.pi*220*tt)*np.exp(-tt*2.5)
    y = y / max(1e-6, np.abs(y).max()) * 0.89
    return (y * 32767).astype(np.int16)

def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    os.makedirs(os.path.join(out, "work"), exist_ok=True)
    if "--stills" in sys.argv:
        for tt in (1.2, 2.7, 4.5, 7.5, 9.0, 12.5, 14.0, 17.5, 19.0, 6.05):
            frame_at(tt).save(os.path.join(out, "work", f"still_{tt:.2f}.png"))
        print("stills ok"); return
    import imageio.v2 as imageio
    from imageio_ffmpeg import get_ffmpeg_exe
    silent = os.path.join(out, "work", "brag_silent.mp4")
    with imageio.get_writer(silent, fps=FPS, codec="libx264", quality=8, macro_block_size=8) as w:
        for i in range(int(FPS * DUR)):
            w.append_data(np.asarray(frame_at(i / FPS)))
    wav = os.path.join(out, "work", "score.wav")
    with wave.open(wav, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(44100)
        f.writeframes(score().tobytes())
    final = os.path.join(out, "brag.mp4")
    subprocess.run([get_ffmpeg_exe(), "-y", "-i", silent, "-i", wav,
                    "-c:v", "copy", "-c:a", "aac", "-shortest", final], check=True)
    print("video ok:", final)

if __name__ == "__main__":
    main()
