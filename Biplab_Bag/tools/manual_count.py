"""Manual counting tool for validating the automatic counts.

Two modes
---------
blind   (independent ground truth)  - shows the RAW photo.  Click every item
        once (screw head / nut centre).  Count = number of clicks.
review  (check a method's output)   - shows the method's OVERLAY.
        Left-click  = item the method MISSED            (+1)
        Right-click = item the method COUNTED TOO OFTEN (-1)
        Count = method count + missed - extra.

The image is shown tile by tile (zoomed) so dense piles stay readable.
Only clicks inside the tile (normal colours) are accepted; the faded border
is context from the neighbouring tiles.  Rule for items crossing a tile edge:
click the item where its head / centre is - it is then counted exactly once.

Keys:  n / SPACE = next tile   b = previous tile   + / - = zoom in / out
       u = undo last click
       s = save                q = save and go to the next image   ESC = close without saving

Progress is stored per image in tools/manual/<image>_<mode>.json, so you can
stop and resume (use --restart to start an image again).  The counts are
written to tools/manual/manual_counts.csv.  Clicks are stored in working
coordinates (image resized to a 2000 px long side).

Examples (run from Biplab_Bag/)
--------
  # independent count of a raw photo
  python tools/manual_count.py blind -i <photo folder>/20240713_194200.jpg
  # review the Non-AI result
  python tools/manual_count.py review -i <photo folder>/20240713_194200.jpg \
         --overlays Non_AI/outputs/overlays --counts Non_AI/outputs/counts.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

WORK_SIDE = 2000      # same working resolution as the solutions / overlays
VIEW = 900            # on-screen size of one tile (px) - set from the screen size at start-up
CTX = 40              # context margin shown around each tile (working px)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(paths):
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXT)
        elif p.suffix.lower() in IMG_EXT:
            out.append(p)
    return out


def screen_size() -> tuple[int, int]:
    """Real screen size in pixels (DPI-aware on Windows), with a safe fallback."""
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
        u = ctypes.windll.user32
        return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
    except Exception:
        return 1600, 900


def resize_long_side(img, side):
    h, w = img.shape[:2]
    s = min(1.0, side / max(h, w))
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


class Session:
    def __init__(self, img, tile, mode, state):
        self.img, self.mode = img, mode
        self.set_tile(state.get("tile_size", tile))
        self.t = min(state.get("tile", 0), len(self.tiles) - 1)
        self.clicks = [tuple(c) for c in state.get("clicks", [])]   # (x, y, sign) in working px
        self.scale = 1.0
        self.origin = (0, 0)
        self.bar = 0

    def set_tile(self, tile):
        H, W = self.img.shape[:2]
        self.tile = int(max(150, min(tile, max(H, W))))
        self.tiles = [(x, y) for y in range(0, H, self.tile) for x in range(0, W, self.tile)]
        self.t = 0

    def zoom(self, factor):
        """Change tile size, staying at the same image position."""
        cx, cy = self.tiles[self.t]
        self.set_tile(self.tile * factor)
        best = min(range(len(self.tiles)), key=lambda i: abs(self.tiles[i][0] - cx) + abs(self.tiles[i][1] - cy))
        self.t = best

    def render(self):
        H, W = self.img.shape[:2]
        x, y = self.tiles[self.t]
        x0, y0 = max(0, x - CTX), max(0, y - CTX)
        x1, y1 = min(W, x + self.tile + CTX), min(H, y + self.tile + CTX)
        view = self.img[y0:y1, x0:x1].copy()
        # fade the context margin
        inner = np.zeros(view.shape[:2], bool)
        inner[y - y0:y - y0 + self.tile, x - x0:x - x0 + self.tile] = True
        view[~inner] = (0.35 * view[~inner] + 0.65 * 255).astype(np.uint8)
        self.scale = VIEW / max(view.shape[:2])
        self.origin = (x0, y0)
        view = cv2.resize(view, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        for cx, cy, sgn in self.clicks:
            px, py = int((cx - x0) * self.scale), int((cy - y0) * self.scale)
            if 0 <= px < view.shape[1] and 0 <= py < view.shape[0]:
                col = (0, 200, 0) if sgn > 0 else (0, 0, 255)
                r = max(6, VIEW // 120)
                cv2.circle(view, (px, py), r, col, -1)
                cv2.circle(view, (px, py), r, (255, 255, 255), 2)
        plus = sum(1 for c in self.clicks if c[2] > 0)
        minus = sum(1 for c in self.clicks if c[2] < 0)
        txt = (f"tile {self.t + 1}/{len(self.tiles)}  clicks={plus}" if self.mode == "blind"
               else f"tile {self.t + 1}/{len(self.tiles)}  missed=+{plus}  extra=-{minus}")
        fs = max(0.5, VIEW / 1500)
        bar = int(34 * fs / 0.6)
        head = np.full((bar, view.shape[1], 3), 255, np.uint8)       # status bar ABOVE the image
        cv2.putText(head, txt + "   [n]ext [b]ack [+/-]zoom [u]ndo [s]ave [q]uit", (6, int(bar * 0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), max(1, int(fs * 2)), cv2.LINE_AA)
        self.bar = bar
        return np.vstack([head, view])

    def on_mouse(self, ev, px, py, *_):
        if ev not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        if self.mode == "blind" and ev == cv2.EVENT_RBUTTONDOWN:
            return
        cx = self.origin[0] + px / self.scale
        cy = self.origin[1] + (py - self.bar) / self.scale
        x, y = self.tiles[self.t]
        if not (x <= cx < x + self.tile and y <= cy < y + self.tile):
            return  # click in the faded context margin -> belongs to another tile
        self.clicks.append((round(cx, 1), round(cy, 1), 1 if ev == cv2.EVENT_LBUTTONDOWN else -1))


def save(out_dir, name, sess, mode, base):
    plus = sum(1 for c in sess.clicks if c[2] > 0)
    minus = sum(1 for c in sess.clicks if c[2] < 0)
    count = plus if mode == "blind" else base + plus - minus
    (out_dir / f"{Path(name).stem}_{mode}.json").write_text(json.dumps(
        {"image": name, "mode": mode, "tile": sess.t, "tile_size": sess.tile, "clicks": sess.clicks,
         "method_count": base, "missed": plus, "extra": minus, "count": count}, indent=1))
    csv_path = out_dir / "manual_counts.csv"
    rows = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if not (r["image"] == name and r["mode"] == mode)]
    rows.append({"image": name, "mode": mode, "count": count, "method_count": base if mode == "review" else "",
                 "missed": plus if mode == "review" else "", "extra": minus if mode == "review" else ""})
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["image", "mode", "count", "method_count", "missed", "extra"])
        wr.writeheader()
        wr.writerows(sorted(rows, key=lambda r: (r["image"], r["mode"])))
    print(f"  saved: {name} [{mode}] count={count}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["blind", "review"])
    ap.add_argument("-i", "--input", nargs="+", required=True, help="raw image files / folders")
    ap.add_argument("-o", "--out", default=str(Path(__file__).resolve().parent / "manual"),
                    help="output folder (default: tools/manual, independent of the current directory)")
    ap.add_argument("--overlays", help="review: folder with <stem>_overlay.jpg")
    ap.add_argument("--counts", help="review: counts.csv of the method being checked")
    ap.add_argument("--tile", type=int, default=0, help="tile size in working px (0 = auto)")
    ap.add_argument("--restart", action="store_true", help="ignore saved clicks and count the image again")
    args = ap.parse_args()

    global VIEW
    sw, sh = screen_size()
    VIEW = int(max(500, min(0.95 * sw, sh - 190)))   # largest square that fits above the taskbar
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"results will be saved to: {out_dir / 'manual_counts.csv'}  (press s or q to save)")
    method_counts = {}
    if args.mode == "review":
        if not (args.overlays and args.counts):
            raise SystemExit("review mode needs --overlays and --counts")
        with open(args.counts, newline="") as f:
            method_counts = {r["image"]: int(r["count"]) for r in csv.DictReader(f)}

    images = list_images(args.input)
    if not images:
        raise SystemExit(f"no images found in {args.input} - check the path (it is relative to {Path.cwd()})")
    for p in images:
        if args.mode == "review":
            img = cv2.imread(str(Path(args.overlays) / f"{p.stem}_overlay.jpg"))
            if img is None:
                print(f"no overlay for {p.name}, skipped")
                continue
            base = method_counts.get(p.name, 0)
        else:
            img = cv2.imread(str(p))
            base = 0
            if img is None:
                print(f"cannot read {p.resolve()}, skipped")
                continue
        img = resize_long_side(img, WORK_SIDE)
        tile = args.tile or (500 if max(img.shape[:2]) > 1500 else 700)
        js = out_dir / f"{p.stem}_{args.mode}.json"
        state = json.loads(js.read_text()) if (js.exists() and not args.restart) else {}
        sess = Session(img, tile, args.mode, state)
        win = f"{args.mode}: {p.name}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)   # resizable
        cv2.resizeWindow(win, VIEW, VIEW)
        cv2.moveWindow(win, 20, 10)
        cv2.setMouseCallback(win, sess.on_mouse)
        print(f"{p.name}: {len(sess.tiles)} tiles")
        while True:
            cv2.imshow(win, sess.render())
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("n"), ord(" ")):
                sess.t = min(sess.t + 1, len(sess.tiles) - 1)
            elif k == ord("b"):
                sess.t = max(sess.t - 1, 0)
            elif k in (ord("+"), ord("=")):
                sess.zoom(0.75)
            elif k in (ord("-"), ord("_")):
                sess.zoom(1.33)
            elif k == ord("u") and sess.clicks:
                sess.clicks.pop()
            elif k == ord("s"):
                save(out_dir, p.name, sess, args.mode, base)
            elif k == ord("q"):
                save(out_dir, p.name, sess, args.mode, base)
                break
            elif k == 27:
                break
        cv2.destroyWindow(win)


if __name__ == "__main__":
    main()
