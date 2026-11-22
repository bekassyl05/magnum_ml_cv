"""Contact sheet of misclassified test frames (to count frames where no vegetable is visible = label noise).
  python src/tools_error_sheet.py              (all errors, sorted by confidence)
Output: outputs/gradcam/error_sheet.jpg  (small JPEG, 10 columns, index numbers on every tile)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from PIL import Image, ImageDraw

from src import config as C

err = pd.read_csv(C.OUTPUTS / "errors.csv")
cols, tile = 10, 110
rows = (len(err) + cols - 1) // cols
sheet = Image.new("RGB", (cols * tile, rows * tile), "white")
for i, r in enumerate(err.itertuples()):
    with Image.open(r.path) as im:
        im = im.convert("RGB").resize((tile - 2, tile - 2))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 52, 11], fill="black")
    d.text((2, 0), f"{i}:{r.cls[:3]}>{r.pred[:3]}", fill="white")
    sheet.paste(im, ((i % cols) * tile, (i // cols) * tile))
out = C.OUTPUTS / "gradcam" / "error_sheet.jpg"
sheet.save(out, quality=80)
print(f"{len(err)} errors -> {out}")
