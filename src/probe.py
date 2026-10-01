import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path


def ffprobe_raw(path):
    """Complete ffprobe output: format, every stream, chapters, programs, tags."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-show_chapters",
         "-show_programs", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def probe(path):
    """Summary fields plus the full raw ffprobe dump and file hash."""
    path = Path(path)
    raw = ffprobe_raw(path)
    video = next(s for s in raw["streams"] if s["codec_type"] == "video")
    fmt = raw["format"]
    st = path.stat()
    return {
        "name": path.name,
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": int(fmt["size"]),
        "modified_time": st.st_mtime,
        "container": fmt.get("format_long_name"),
        "duration_s": round(float(fmt["duration"]), 3),
        "bit_rate": int(fmt["bit_rate"]) if "bit_rate" in fmt else None,
        "width": video["width"],
        "height": video["height"],
        "fps": round(float(Fraction(video["r_frame_rate"])), 3),
        "codec": video["codec_name"],
        "pix_fmt": video.get("pix_fmt"),
        "rotation": next((sd.get("rotation") for sd in video.get("side_data_list", []) if "rotation" in sd), 0),
        "has_audio": any(s["codec_type"] == "audio" for s in raw["streams"]),
        "creation_time": fmt.get("tags", {}).get("creation_time"),
        "ffprobe": raw,
    }
