import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from src import attributes, cluster, reid
from src.detect_track import best_crops, detect_and_track
from src.probe import probe

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def process(video, args, out_dir):
    info = probe(video)
    tracks = detect_and_track(video, args.fps, args.det_width, args.max_frames)

    track_ids, embeddings, spans, track_attrs, track_crops = [], [], [], [], {}
    for tid, tr in tracks.items():
        crops = best_crops(tr, args.crops_per_track)
        if not crops:
            continue
        track_ids.append(tid)
        embeddings.append(reid.track_embedding(reid.embed(crops)))
        track_attrs.append(attributes.predict(crops))
        spans.append((tracks[tid]["dets"][0]["t"], tracks[tid]["dets"][-1]["t"]))
        track_crops[tid] = crops

    labels = cluster.assign_person_ids(embeddings, spans, args.threshold)
    persons = {}
    for tid, label, attrs in zip(track_ids, labels, track_attrs):
        dets = tracks[tid]["dets"]
        p = persons.setdefault(label, {"person_id": f"P{label + 1:03d}", "tracks": [], "attributes": attrs, "_n": 0})
        p["tracks"].append({
            "track_id": tid,
            "first_seen_s": dets[0]["t"],
            "last_seen_s": dets[-1]["t"],
            "n_detections": len(dets),
            "mean_bbox_conf": round(sum(d["conf"] for d in dets) / len(dets), 3),
            "detections": dets,  # per sampled frame: t (s), bbox [x1,y1,x2,y2] (full-res px), conf
        })
        # keep attributes from the track with the most detections
        if len(dets) > p["_n"]:
            p["attributes"], p["_n"] = attrs, len(dets)
        if args.save_crops:
            d = out_dir / "crops" / Path(video).stem / p["person_id"]
            d.mkdir(parents=True, exist_ok=True)
            for i, c in enumerate(track_crops[tid]):
                cv2.imwrite(str(d / f"track{tid}_{i}.jpg"), c)

    result = {
        "file": info,
        "params": {"sample_fps": args.fps, "det_width": args.det_width, "max_frames": args.max_frames},
        "persons": [{k: v for k, v in p.items() if k != "_n"} for p in persons.values()],
    }
    (out_dir / f"{Path(video).stem}.json").write_text(json.dumps(result, indent=2))
    if embeddings:
        np.savez_compressed(out_dir / f"{Path(video).stem}_embeddings.npz",
                            track_ids=np.array(track_ids), embeddings=np.stack(embeddings))
    return result


def main():
    ap = argparse.ArgumentParser(description="Extract person attributes + ReID metadata from videos.")
    ap.add_argument("input", help="video file or folder")
    ap.add_argument("--out", default="output")
    ap.add_argument("--fps", type=float, default=3.0, help="sampling rate")
    ap.add_argument("--det-width", type=int, default=1280)
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N sampled frames")
    ap.add_argument("--crops-per-track", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.4, help="max cosine distance to merge tracks into one person")
    ap.add_argument("--save-crops", action="store_true")
    args = ap.parse_args()

    src = Path(args.input)
    videos = sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXT) if src.is_dir() else [src]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for v in videos:
        print(f"Processing {v.name} ...", flush=True)
        res = process(v, args, out_dir)
        for p in res["persons"]:
            row = {"video": v.name, "person_id": p["person_id"], "n_tracks": len(p["tracks"]),
                   "first_seen_s": min(t["first_seen_s"] for t in p["tracks"]),
                   "last_seen_s": max(t["last_seen_s"] for t in p["tracks"])}
            row.update({k: a["label"] for k, a in p["attributes"].items()})
            rows.append(row)
        print(f"  {len(res['persons'])} person(s)")

    if rows:
        with open(out_dir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
