# Implementation

Step-by-step build order. Design is in [ARCHITECTURE.md](ARCHITECTURE.md); progress is in [TASKS.md](TASKS.md).

1. **probe.py**: run `ffprobe -v error -show_format -show_streams -of json`, return duration, fps, width, height, codec, size, creation time.
2. **detect_track.py**: open the video with OpenCV, step through frames at the sampling fps, resize to `det_width`, run `YOLO("yolo11n.pt").track(persist=True, classes=[0], tracker="bytetrack.yaml", device="cpu")`, scale boxes back to full resolution, yield `(t, track_id, bbox, conf, full_frame_crop)`.
3. **Crop selection**: keep the K largest, sharpest crops per track (Laplacian variance as sharpness), drop crops under 40x80 px.
4. **reid.py**: CLIP image encoder (open_clip ViT-B/32) gives L2-normalised embeddings, averaged per track. The module has one `embed()` function so OSNet can replace it later.
5. **attributes.py**: CLIP zero-shot prompts for gender, age group, hat, backpack; HSV dominant colour on the upper and lower halves of the crop for clothing colour. Votes are averaged across a track's crops.
6. **cluster.py**: agglomerative clustering (cosine, distance threshold 0.25) over track embeddings gives `person_id`.
7. **extract.py**: CLI that loops over videos, runs the stages, writes `output/<video>.json` and `output/summary.csv`, optionally saves crops.
8. **Check**: run on `testVideos/test-vid.mp4` with `--max-frames 30`, then a full run, then review the saved crops.
