import os
import re
from pathlib import Path

SEED = 42

# --- Project layout -------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
SPLITS = ROOT / "splits"
for _d in (OUTPUTS, MODELS, REPORTS, SPLITS):
    _d.mkdir(parents=True, exist_ok=True)

# --- Dataset (read-only, outside the repo, do not copy) -------------------
# Override on another machine with env var MAGNUM_DATA_ROOT.
DATA_ROOT = Path(
    os.environ.get(
        "MAGNUM_DATA_ROOT",
        r"C:\Users\zhoma\OneDrive\Рабочий стол\archive",
    )
)
DATA_ALL = DATA_ROOT / "Dataset" / "Dataset"  # <class>/<file>.jpg
DATA_OFFICIAL = DATA_ROOT / "final_dataset" / "final_dataset"  # {train,val,test}/<class>/

OFFICIAL_SPLITS = ("train", "val", "test")
CLASSES = ["Capsicum", "Carrot", "Cucumber", "Potato", "Raddish", "Squash"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# --- Filename parsing -----------------------------------------------------
# Names look like YYYYMMDD_HHMMSS.jpg (standalone photo) or
# YYYYMMDD_HHMMSS_frame_N.jpg (frame extracted from a video).
# group id = "YYYYMMDD_HHMMSS" (frames of one video share it).
_NAME_RE = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{6})(?:_frame_(?P<frame>\d+))?$")


def parse_name(filename: str) -> dict:
    """Return dict(video_id, is_frame, frame_idx) or video_id=None if the
    name does not match the expected pattern (then the caller must decide)."""
    stem = Path(filename).stem
    m = _NAME_RE.match(stem)
    if not m:
        return {"video_id": None, "is_frame": False, "frame_idx": None}
    return {
        "video_id": f"{m['date']}_{m['time']}",
        "is_frame": m["frame"] is not None,
        "frame_idx": int(m["frame"]) if m["frame"] is not None else None,
    }
