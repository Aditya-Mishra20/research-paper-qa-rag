# brag-plan: Research Paper QA RAG

## Answers (Inspect)
- **What is it (one sentence)?** Chat with your research PDFs — upload or paste an arXiv ID, ask anything, get answers with page citations.
- **Who is it for / what does it do for them?** Students/researchers drowning in papers; turns a 30-page PDF into a 5-second cited answer.
- **What sets it apart?** Free-tier stack that actually works: local MiniLM embeddings (zero quota), Chroma persistence, sha256 dedup, strict `[Title, p.N]` citations, FastAPI + Streamlit split.
- **Most impressive claim?** `Indexed: 2505.10468v4 Ai agents and agentic ai.pdf (394 chunks)` — then every answer cites exact pages.
- **Visual hook?** The real UI beats: sidebar upload → "Indexed (394 chunks)" toast → chat question → answer + expandable page-cited sources.
- **Real UI/flow shown?** Streamlit app (title 📚 Research Paper QA, sidebar Papers/uploader/arXiv, chat bubbles), FastAPI `/docs` badges, terminal `uvicorn` + ingest log.
- **Tone?** `default` — punchy, playful, clean.
- **Share caption?** "I stopped reading papers line-by-line — now I interrogate them. Upload PDF → 394 chunks → cited answers. 📚"

## Angle
"Drowning in PDFs? Interrogate them." Skeptic-to-believer in 20s: pile of PDFs → one upload → cited chat answer.

## Storyboard (20s, 1920x1080, 30fps)
| # | Scene | Time | Visual | Text on screen | Audio |
|---|-------|------|--------|----------------|-------|
| 1 | Hook | 0-2.5s | PDF icons raining onto dark bg, counter ticking 0→394 | "DROWNING IN PDFs?" | kick + riser in |
| 2 | Reveal | 2.5-6s | App window builds itself: sidebar + uploader + arXiv box; toast pops | "Research Paper QA — upload. index. ask." | downbeat, arp starts |
| 3 | Highlight 1 — ingest | 6-10s | File card flies into funnel → chunk bars → "Indexed (394 chunks)" + dedup chip "sha256: skipped duplicate" | "394 chunks. deduped. local embeddings." | typing ticks → success blip |
| 4 | Highlight 2 — ask | 10-15s | Chat: user bubble "what is ai agent?" types out; assistant streams answer; 5 source expanders fan out p.6/p.3/p.21/p.23/p.32 | "Every claim cites its page." | message pop, soft shimmer |
| 5 | Punchline/outro | 15-20s | Citations orbit into logo mark 📚; stack chips: FastAPI • Chroma • MiniLM • Gemini; CTA | "Stop reading. Start interrogating." + localhost:8501 | final hit + tail |

Total: 2.5+3.5+4+5+5 = 20s. Every claim is real (394 chunks, page numbers, stack names from pyproject).

## Visual identity
- bg `#0E1117`, panel `#161B22`, accent `#FF4B4B`, gold `#FFD166`, text `#FFFFFF`, dim `#9AA4B2`
- Font: system bold sans (DejaVu Sans Bold if present, else Arial Bold) — big, centered, readable ≥0.3s/word
- Motion: ease-out entrances, staggered lists, simulated typing, no flat pans; stagger crossfades (out→in)

## Sound
One 20s synth piece, A-minor-ish, 100bpm: kick/hat pulse, bass arp, riser into scene 2, blips for toast/index/chat, final hit. SFX in-key, ducked under music.

## Build
- `work/render.py`: PIL frames (pure function of t) + numpy WAV score → `imageio-ffmpeg` binary for mp4 mux.
- Stills check every scene + mid-transitions before full render. Poster = strongest settled frame (scene 5 CTA), baked as frame 0.
