# CLAUDE.md

## Purpose
Extract person-level metadata from videos in `testVideos/`: detect and track people, estimate PETA/Market-1501-style attributes (gender, age group, clothing colour, accessories), and compute appearance ReID embeddings so the same person gets the same `person_id`. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design.

## Constraints
- **CPU only.** No NVIDIA GPU or driver on this machine (8 cores, ~15 GB RAM). Never assume CUDA; use `device="cpu"`.
- Input videos can be 4K (the test clip is 3840x2160, 60 s). Sample frames (`--fps 3` default) and downscale for detection; crop from the full-resolution frame.
- Appearance-based ReID and attributes only. **No face recognition.**
- Market-1501 and PETA are research datasets; check licences of pretrained weights before any non-research use.

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_models.py
```

## Run
```bash
.venv/bin/python extract.py testVideos/ --fps 3 --out output/
```
Options: `--fps N` sampling rate, `--det-width W` detection width (default 1280), `--save-crops` write best crops, `--max-frames N` for quick tests, `--threshold T` for track merging (default 0.4).

## Layout
- `extract.py` CLI entry point
- `src/probe.py` ffprobe file info
- `src/detect_track.py` YOLO person detection + ByteTrack
- `src/reid.py` OSNet ReID embeddings
- `src/attributes.py` Market-1501 attribute model (12 attributes)
- `scripts/download_models.py` fetches weights into `models/`
- `src/cluster.py` cross-track identity clustering
- `testVideos/` input, `output/` results (`<video>.json`, `summary.csv`, `crops/`)

## Conventions
- One JSON per video in `output/`, schema in ARCHITECTURE.md. Timestamps in seconds.
- Keep each module small, with pure functions that take arrays/paths and return plain dicts/lists.
- Don't commit `.venv/`, model weights, or `output/`.
