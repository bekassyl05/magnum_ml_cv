"""Step 8a: build the small out-of-distribution (OOD) set for the 'unknown' bonus. DEV ONLY: nothing here is needed
at inference time and data/ is git-ignored.
    python src/08_get_ood.py
Creates:
  data/ood/other_veg/        ~150 photos of OTHER vegetables/fruits/food/shelves from Wikimedia Commons (search by keyword,
                             JPEG thumbnails, 320 px). Review outputs/ood_sheet.jpg and delete wrong images by hand.
  data/ood/no_veg_frames/    test frames in which no vegetable is visible (empty counter / hand only), taken from
                             outputs/errors.csv by the tile numbers seen on outputs/gradcam/error_sheet.jpg
  data/ood/own/              (optional) put your own photos here: empty shelf, tomatoes, packaged goods ...
Caveat: Commons photos look different from the dataset (studio/web style), which makes them easier to detect than
real shop photos. Own photos in data/ood/own/ make the threshold choice more realistic.
"""
import json
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from PIL import Image

from src import config as C

OOD = C.ROOT / "data" / "ood"
API = "https://commons.wikimedia.org/w/api.php"
UA = "magnum-test-ood/1.0 (educational test task; python-urllib)"
TERMS = ["tomato", "onion", "garlic", "broccoli", "cabbage", "cauliflower", "eggplant", "apple", "banana", "lemon",
         "orange fruit", "eggs", "bread", "supermarket shelf", "kitchen counter", "empty table"]
PER_TERM = 12
# tile numbers on error_sheet.jpg (= row numbers of outputs/errors.csv) that show no vegetable (counter / hand only)
NO_VEG_IDX = [0, 1, 2, 4, 5, 7, 9, 15, 17, 19, 25, 26, 27, 31, 37, 39, 40, 44, 46, 48, 49, 50]


def open_url(req, tries=5):
    for k in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=30)
        except Exception:  # HTTP 429 (rate limit) or a DNS/network hiccup: wait and retry
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def api(params):
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers={"User-Agent": UA})
    with open_url(req) as r:
        return json.load(r)


def fetch(term, n):
    d = api(dict(action="query", format="json", generator="search", gsrnamespace=6, gsrlimit=n * 2,
                 gsrsearch=f"{term} filetype:bitmap", prop="imageinfo", iiprop="url|mime", iiurlwidth=320))
    out = []
    for p in d.get("query", {}).get("pages", {}).values():
        ii = (p.get("imageinfo") or [{}])[0]
        if ii.get("mime") == "image/jpeg" and ii.get("thumburl"):
            out.append(ii["thumburl"])
    return out[:n]


def main():
    d = OOD / "other_veg"
    d.mkdir(parents=True, exist_ok=True)
    total = 0
    for term in TERMS:
        k = 0
        try:
            for i, url in enumerate(fetch(term, PER_TERM)):
                dst = d / f"{term.replace(' ', '_')}_{i:02d}.jpg"
                if not dst.exists():
                    req = urllib.request.Request(url, headers={"User-Agent": UA})
                    with open_url(req) as r:
                        dst.write_bytes(r.read())
                    time.sleep(1.0)
                k += 1
        except Exception as e:  # network / rate limit: continue with the other terms
            print(f"  {term}: error {e!r}")
        print(f"{term}: {k} images")
        total += k
    print(f"other_veg: {total} images in {d}")

    nv = OOD / "no_veg_frames"
    nv.mkdir(parents=True, exist_ok=True)
    err = pd.read_csv(C.OUTPUTS / "errors.csv")
    for i in NO_VEG_IDX:
        if i < len(err):
            shutil.copy(err.path.iloc[i], nv / Path(err.path.iloc[i]).name)
    (OOD / "own").mkdir(exist_ok=True)
    print(f"no_veg_frames: {len(list(nv.glob('*.jpg')))} images; put own photos into {OOD / 'own'}")

    files = sorted(d.glob("*.jpg"))
    cols, tile = 15, 80
    rows = (len(files) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, max(rows, 1) * tile), "white")
    for i, f in enumerate(files):
        try:
            with Image.open(f) as im:
                sheet.paste(im.convert("RGB").resize((tile - 2, tile - 2)), ((i % cols) * tile, (i // cols) * tile))
        except Exception:
            pass
    sheet.save(C.OUTPUTS / "ood_sheet.jpg", quality=80)
    print("review outputs/ood_sheet.jpg and delete images that are not what the file name says")


if __name__ == "__main__":
    main()
