import cv2
from ultralytics import YOLO

MIN_W, MIN_H = 40, 80


def sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def detect_and_track(video_path, sample_fps=3.0, det_width=1280, max_frames=None, model_name="yolo11n.pt"):
    """Return {track_id: {"dets": [...], "crops": [(score, crop)]}}.

    Detection runs on a downscaled frame; crops are taken from the full-resolution frame.
    """
    model = YOLO(model_name)
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / sample_fps))
    tracks = {}
    idx = used = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            scale = det_width / w if w > det_width else 1.0
            small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
            res = model.track(small, persist=True, classes=[0], tracker="bytetrack.yaml",
                              device="cpu", verbose=False)[0]
            t = idx / src_fps
            if res.boxes is not None and res.boxes.id is not None:
                for box, tid, conf in zip(res.boxes.xyxy.tolist(), res.boxes.id.int().tolist(),
                                          res.boxes.conf.tolist()):
                    x1, y1, x2, y2 = [v / scale for v in box]
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(w, int(x2)), min(h, int(y2))
                    tr = tracks.setdefault(tid, {"dets": [], "crops": []})
                    tr["dets"].append({"t": round(t, 3), "bbox": [x1, y1, x2, y2], "conf": round(conf, 3)})
                    if x2 - x1 >= MIN_W and y2 - y1 >= MIN_H:
                        crop = frame[y1:y2, x1:x2].copy()
                        tr["crops"].append((sharpness(crop) * (x2 - x1) * (y2 - y1), crop))
            used += 1
            if max_frames and used >= max_frames:
                break
        idx += 1
    cap.release()
    return tracks


def best_crops(track, k=5):
    """Keep the k crops with the highest sharpness x area score."""
    return [c for _, c in sorted(track["crops"], key=lambda x: -x[0])[:k]]
