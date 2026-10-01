# metadata-extract

Command-line tool that extracts person-level metadata from video files: it detects and tracks people, estimates the 12 Market-1501 attributes (gender, age, hair length, sleeve length, lower clothing length and type, hat, backpack, bag, handbag, upper and lower colour), and groups tracks into stable person IDs with OSNet ReID embeddings. It also stores the full technical file metadata (complete ffprobe dump, SHA-256, codec, bitrate, tags). Runs on CPU, no GPU needed.

## Requirements
- Linux, macOS or Windows with Python 3.10+
- `ffmpeg` (provides `ffprobe`), for example `sudo apt install ffmpeg`
- About 4 GB of disk space for dependencies and model weights (about 240 MB of weights)

## Installation
```bash
git clone <this-repo> && cd metadata-extract    # or just cd into the folder
python3 -m venv .venv
source .venv/bin/activate                        # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/download_models.py                # OSNet ReID + Market-1501 attribute weights into models/
```

## Usage
Run on a single video:
```bash
python extract.py /path/to/video.mp4
```

Run on every video in a folder:
```bash
python extract.py testVideos/
```

Common examples:
```bash
# Quick test on the first 30 sampled frames
python extract.py video.mp4 --max-frames 30

# Higher sampling rate, save person crops, custom output folder
python extract.py video.mp4 --fps 5 --save-crops --out results/

# Stricter or looser merging of tracks into the same person
python extract.py video.mp4 --threshold 0.5
```
Without activating the venv, call it directly: `.venv/bin/python extract.py video.mp4`.

### Options
| Option | Default | Description |
|---|---|---|
| `input` | required | A video file or a folder of videos (`.mp4 .mov .avi .mkv .webm .m4v`) |
| `--out DIR` | `output` | Where results are written |
| `--fps N` | `3` | Frames per second to sample. Higher is more accurate and slower |
| `--det-width W` | `1280` | Width frames are scaled to for detection (crops use full resolution) |
| `--max-frames N` | all | Stop after N sampled frames, useful for quick tests |
| `--crops-per-track K` | `5` | Best crops kept per track for ReID and attributes |
| `--threshold T` | `0.4` | Maximum cosine distance to merge tracks into one person; lower means fewer merges |
| `--save-crops` | off | Save person crops to `OUT/crops/<video>/<person_id>/` |

## Output
For each video `name.ext`:

- `output/name.json`: full metadata (file info incl. raw ffprobe, persons, tracks, per-frame detections, attributes)
- `output/name_embeddings.npz`: one 512-d OSNet embedding per track (`track_ids`, `embeddings`)
- `output/summary.csv`: one row per person per video (shared by all videos in the run)
- `output/crops/...`: only with `--save-crops`

Example `name.json`:
```json
{
  "file": {"name": "video.mp4", "duration_s": 59.988, "width": 3840, "height": 2160,
           "fps": 29.97, "codec": "h264", "has_audio": true, "creation_time": "2024-03-12T04:48:36Z"},
  "params": {"sample_fps": 3.0, "det_width": 1280, "max_frames": null},
  "persons": [
    {
      "person_id": "P001",
      "tracks": [{"track_id": 1, "first_seen_s": 0.0, "last_seen_s": 6.34, "n_detections": 19,
                  "mean_bbox_conf": 0.572,
                  "detections": [{"t": 0.0, "bbox": [812, 340, 1010, 790], "conf": 0.61}]}],
      "attributes": {"gender": {"label": "male", "conf": 0.66}, "age": {"label": "adult", "conf": 0.79},
                     "hair_length": {"label": "short", "conf": 0.96}, "sleeve_length": {"label": "short", "conf": 0.93},
                     "lower_clothing_length": {"label": "long", "conf": 0.92}, "lower_clothing_type": {"label": "pants", "conf": 0.98},
                     "hat": {"label": "no", "conf": 0.99}, "backpack": {"label": "no", "conf": 0.98},
                     "bag": {"label": "no", "conf": 0.96}, "handbag": {"label": "no", "conf": 0.91},
                     "upper_color": {"label": "white", "conf": 0.91}, "lower_color": {"label": "black", "conf": 0.8}}
    }
  ]
}
```

## Performance tips
- 4K and long videos are slow on CPU. Lower `--fps` (for example `1`) or use `--max-frames` first.
- Detection runs on a downscaled frame. Lower `--det-width` for more speed, raise it to find smaller or more distant people.
- Rough speed: 20 sampled frames of a 4K clip take about 30 seconds on 8 cores.

## Limitations
- Attributes use a ResNet50 trained on Market-1501, so they work best on full-body, front/back views at reasonable size. Market-1501 has only 8 upper and 9 lower colour classes (for example no beige), so odd colours get a low `conf`. Filter on `conf`.
- Person IDs are appearance-based. Clothing changes can split one person into two, and similar outfits can merge two people. Tracks that overlap in time are never merged. Tune `--threshold` for your footage.
- ReID weights are trained on MSMT17 (OSNet x1.0); the attribute weights come from a third-party Market-1501 project. Check their licences before commercial use.
- No face recognition is done. Only run it on videos you have the right to analyse.

## Troubleshooting
- `ffprobe: command not found`: install ffmpeg.
- `FileNotFoundError ... models/...pth`: run `python scripts/download_models.py`.
- First run prints a download message: YOLO (`yolo11n.pt`) is fetched once.
- Out of memory on long 4K videos: lower `--fps` and `--crops-per-track`.

## Project docs
[CLAUDE.md](CLAUDE.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [IMPLEMENTATION.md](IMPLEMENTATION.md) · [TASKS.md](TASKS.md)
