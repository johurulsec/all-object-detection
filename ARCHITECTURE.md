# Architecture

## Goal
Turn a video into structured per-person metadata: when each person appears, their attributes, and a stable identity across tracks and videos.

## Pipeline
```
video
  |
  v
probe (ffprobe) ---------------------------------> file_info
  |
  v
sample frames (OpenCV, N fps)
  |
  v
downscale copy --> YOLO person detect + ByteTrack --> tracks (id, bbox, t)
  |                                                       |
  | full-res frame <-------- map bbox back ---------------+
  v
crop persons (skip small/blurry, keep best K per track)
  |
  +--> ReID embedding (OSNet x1.0, 512-d)  --> mean per track
  |
  +--> attributes (Market-1501 ResNet50, 30 heads -> 12 attributes) --> mean per track
  |
  v
cluster track embeddings (cosine) --> person_id
  |
  v
output/<video>.json, output/summary.csv, output/crops/<person_id>/
```

## Modules
| Module | Responsibility |
|---|---|
| `extract.py` | CLI, loops over videos, wires the stages, writes outputs |
| `src/probe.py` | `probe(path) -> dict`: summary fields, SHA-256 and the full raw ffprobe JSON |
| `src/detect_track.py` | Yields per-frame person boxes with track ids; handles downscale and bbox mapping to full resolution |
| `src/reid.py` | `embed(crops) -> ndarray[N, 512]`, L2-normalised OSNet features |
| `src/attributes.py` | `predict(crops) -> {attr: {label, conf}}` for the 12 Market-1501 attributes |
| `src/cluster.py` | Agglomerative clustering on track embeddings, cosine threshold, returns `person_id` per track |

## Output schema (per video)
```json
{
  "file": {"name": "...", "duration_s": 60.0, "width": 3840, "height": 2160, "fps": 29.97, "codec": "h264"},
  "params": {"sample_fps": 3, "det_width": 1280},
  "persons": [
    {
      "person_id": "P001",
      "tracks": [
        {"track_id": 4, "first_seen_s": 2.3, "last_seen_s": 17.0, "n_detections": 41, "mean_bbox_conf": 0.82}
      ],
      "attributes": {"gender": {"label": "male", "conf": 0.71}, "upper_color": {"label": "blue", "conf": 0.64}}
    }
  ]
}
```
`summary.csv` has one row per person per video with the same fields flattened.

## Model choices (CPU)
- Detection/tracking: `yolo11n` with built-in ByteTrack, class `person`.
- ReID: OSNet x1.0 (torchreid), MSMT17 weights, 512-d L2-normalised embeddings.
- Attributes: ResNet50 with 30 binary heads trained on Market-1501, decoded into the 12 Market-1501 attributes (age and colours by argmax, the rest by threshold 0.5), each with a confidence.
- Weights live in `models/` and are fetched by `scripts/download_models.py`.

## Performance notes
- 4K decode is the main cost: sample at 3 fps, run detection on a ~1280 px-wide copy, crop from the full frame.
- Process one video at a time; torch threads set to the core count.
- Expect a 60 s 4K clip to take a few minutes on CPU.

## Limits
- Small, distant, or occluded people give weak attributes; low-confidence values are kept with their score so callers can filter.
- Market-1501 attributes cover only its 8 upper and 9 lower colours and 12 attribute types; PETA's 61 attributes are not covered.
- ReID identity is appearance-based: clothing changes break matches, and a similar outfit can merge two people.
