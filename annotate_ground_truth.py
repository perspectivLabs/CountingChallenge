"""Point-based local ground-truth annotation using original images only."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parent


def discover_images(root: Path) -> list[Path]:
    images = sorted(
        path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTS
    )
    names = [path.name for path in images]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"Image filenames must be unique for CSV matching: {duplicates}")
    if not images:
        raise FileNotFoundError(f"No supported images found under {root}")
    return images


def read_count_rows(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not {"image", "count"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must have image,count columns")
        rows = {}
        for row in reader:
            name = Path(row["image"]).name
            count = int(row["count"])
            if count < 0:
                raise ValueError(f"Negative ground-truth count for {name}")
            if name in rows:
                raise ValueError(f"Duplicate image in {path}: {name}")
            rows[name] = count
    return rows


def read_plan_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not {"image", "count", "status"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must have image,count,status columns")
        rows = {}
        for row in reader:
            name = Path(row["image"]).name
            count = int(row["count"])
            status = row["status"].strip().lower()
            if status not in {"verified", "provisional"}:
                raise ValueError(f"Invalid label status for {name}: {status}")
            if count < 0:
                raise ValueError(f"Negative ground-truth count for {name}")
            if name in rows:
                raise ValueError(f"Duplicate image in {path}: {name}")
            rows[name] = {"image": name, "count": str(count), "status": status}
    return rows


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class GroundTruthAnnotator:
    def __init__(
        self,
        images: list[Path],
        ground_truth_path: Path,
        plan_path: Path,
        progress_path: Path,
    ) -> None:
        self.images = images
        self.ground_truth_path = ground_truth_path
        self.plan_path = plan_path
        self.progress_path = progress_path
        self.ground_truth = read_count_rows(ground_truth_path)
        self.plan_rows = read_plan_rows(plan_path)
        for name, row in self.plan_rows.items():
            if row["status"] == "verified":
                if name not in self.ground_truth:
                    raise ValueError(
                        f"{name} is marked verified in {plan_path} but is missing "
                        f"from {ground_truth_path}"
                    )
                if int(row["count"]) != self.ground_truth[name]:
                    raise ValueError(
                        f"Verified count mismatch for {name} between {plan_path} "
                        f"and {ground_truth_path}"
                    )
        for name, count in self.ground_truth.items():
            self.plan_rows[name] = {
                "image": name,
                "count": str(count),
                "status": "verified",
            }

        self.annotations: dict[str, list[list[float]]] = {}
        self.index = 0
        if progress_path.exists():
            saved = json.loads(progress_path.read_text(encoding="utf-8"))
            self.annotations = saved.get("points", {})
            self.index = int(saved.get("current_index", 0))
        self.index = min(max(self.index, 0), len(images) - 1)

        self.root = tk.Tk()
        self.root.title("Ground-truth point annotator")
        self.root.geometry("1400x900")
        self.root.minsize(800, 600)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.status = tk.StringVar()
        header = ttk.Label(self.root, textvariable=self.status, padding=8)
        header.pack(fill=tk.X)

        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(body, background="#202020", highlightthickness=0)
        xbar = ttk.Scrollbar(body, orient=tk.HORIZONTAL, command=self.canvas.xview)
        ybar = ttk.Scrollbar(body, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill=tk.X)
        self._button(controls, "Previous", self.previous).pack(side=tk.LEFT, padx=3)
        self._button(controls, "Next", self.next).pack(side=tk.LEFT, padx=3)
        self._button(controls, "Undo point", self.undo).pack(side=tk.LEFT, padx=3)
        self._button(controls, "Clear points", self.clear).pack(side=tk.LEFT, padx=3)
        self._button(controls, "Save provisional", self.save_provisional).pack(
            side=tk.RIGHT, padx=3
        )
        self._button(controls, "Save verified", self.save_verified).pack(
            side=tk.RIGHT, padx=3
        )

        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-4>", lambda event: self.zoom_step(event, 1.25))
        self.canvas.bind("<Button-5>", lambda event: self.zoom_step(event, 0.8))
        self.canvas.bind("<ButtonPress-2>", self.pan_start)
        self.canvas.bind("<B2-Motion>", self.pan_move)
        self.canvas.bind("<ButtonPress-3>", self.pan_start)
        self.canvas.bind("<B3-Motion>", self.pan_move)
        self.root.bind("<Left>", lambda _event: self.previous())
        self.root.bind("<Right>", lambda _event: self.next())
        self.root.bind("<BackSpace>", lambda _event: self.undo())
        self.root.bind("<Control-s>", lambda _event: self.save_verified())
        self.root.bind("+", lambda _event: self.zoom_step(None, 1.25))
        self.root.bind("-", lambda _event: self.zoom_step(None, 0.8))

        self.original: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.image_item: int | None = None
        self.scale = 1.0
        self.base_scale = 1.0
        self.zoom_factor = 1.0
        self.canvas.bind("<Configure>", self.on_resize)
        self.load_image()

    @staticmethod
    def _button(parent: ttk.Frame, text: str, command):
        return ttk.Button(parent, text=text, command=command)

    @property
    def current_path(self) -> Path:
        return self.images[self.index]

    @property
    def current_points(self) -> list[list[float]]:
        return self.annotations.setdefault(self.current_path.name, [])

    def saved_label(self) -> tuple[str, str]:
        name = self.current_path.name
        if name in self.ground_truth:
            return "VERIFIED", str(self.ground_truth[name])
        row = self.plan_rows.get(name)
        if row and row["status"] == "provisional":
            return "PROVISIONAL", row["count"]
        return "UNVERIFIED", "—"

    def update_status(self) -> None:
        label_status, saved_count = self.saved_label()
        self.status.set(
            f"{self.index + 1}/{len(self.images)}  |  {self.current_path.name}  |  "
            f"{label_status}: {saved_count}  |  Clicked points: {len(self.current_points)}"
            "  |  Wheel: zoom  |  Middle/right drag: pan  |  Arrows: navigate"
        )

    def load_image(self) -> None:
        with Image.open(self.current_path) as image:
            self.original = image.convert("RGB")
        self.zoom_factor = 1.0
        self.root.update_idletasks()
        self.fit_image()
        self.update_status()

    def fit_image(self) -> None:
        if self.original is None:
            return
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.base_scale = min(
            1.0, width / self.original.width, height / self.original.height
        )
        self.scale = self.base_scale * self.zoom_factor
        self.redraw()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def redraw(self) -> None:
        if self.original is None:
            return
        width = max(1, round(self.original.width * self.scale))
        height = max(1, round(self.original.height * self.scale))
        resized = self.original.resize((width, height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.image_item = self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)
        self.canvas.configure(scrollregion=(0, 0, width, height))
        radius = max(4, min(10, round(6 * self.zoom_factor)))
        for number, (x, y) in enumerate(self.current_points, start=1):
            px, py = x * self.scale, y * self.scale
            self.canvas.create_oval(
                px - radius,
                py - radius,
                px + radius,
                py + radius,
                fill="#ff3030",
                outline="white",
                width=2,
            )
            self.canvas.create_text(
                px + radius + 2,
                py - radius - 2,
                text=str(number),
                fill="white",
                anchor=tk.SW,
                font=("Segoe UI", 9, "bold"),
            )

    def on_resize(self, _event) -> None:
        if self.original is not None and self.zoom_factor == 1.0:
            self.fit_image()

    def add_point(self, event) -> None:
        if self.original is None:
            return
        x = self.canvas.canvasx(event.x) / self.scale
        y = self.canvas.canvasy(event.y) / self.scale
        if 0 <= x < self.original.width and 0 <= y < self.original.height:
            self.current_points.append([round(x, 2), round(y, 2)])
            self.save_progress()
            self.redraw()
            self.update_status()

    def undo(self) -> None:
        if self.current_points:
            self.current_points.pop()
            self.save_progress()
            self.redraw()
            self.update_status()

    def clear(self) -> None:
        if self.current_points and messagebox.askyesno(
            "Clear points", "Remove all points for this image?"
        ):
            self.annotations[self.current_path.name] = []
            self.save_progress()
            self.redraw()
            self.update_status()

    def save_progress(self) -> None:
        data = {"current_index": self.index, "points": self.annotations}
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    def save_label(self, status: str) -> None:
        name = self.current_path.name
        count = len(self.current_points)
        if count == 0:
            messagebox.showwarning("No points", "Place one point per visible object first.")
            return

        if status == "provisional" and name in self.ground_truth:
            if not messagebox.askyesno(
                "Change verified label",
                "This image is currently verified. Remove it from formal evaluation?",
            ):
                return
            self.ground_truth.pop(name)
        if status == "verified" and name in self.ground_truth:
            previous = self.ground_truth[name]
            if previous != count and not messagebox.askyesno(
                "Replace verified count",
                f"Replace the existing verified count {previous} with {count}?",
            ):
                return

        self.plan_rows[name] = {
            "image": name,
            "count": str(count),
            "status": status,
        }
        if status == "verified":
            self.ground_truth[name] = count
        self.write_labels()
        self.update_status()
        messagebox.showinfo(
            "Saved",
            f"{name}: {count} {status} objects.\n\n"
            "Only verified labels are written to ground_truth.csv.",
        )

    def ordered_rows(self, rows: dict[str, dict[str, str]]) -> list[dict[str, str]]:
        order = {path.name: index for index, path in enumerate(self.images)}
        return sorted(rows.values(), key=lambda row: (order.get(row["image"], len(order)), row["image"]))

    def write_labels(self) -> None:
        gt_rows = [
            {"image": name, "count": str(count)}
            for name, count in self.ground_truth.items()
        ]
        gt_rows.sort(
            key=lambda row: next(
                (index for index, path in enumerate(self.images) if path.name == row["image"]),
                len(self.images),
            )
        )
        write_rows(self.ground_truth_path, ["image", "count"], gt_rows)
        write_rows(
            self.plan_path,
            ["image", "count", "status"],
            self.ordered_rows(self.plan_rows),
        )

    def navigate(self, offset: int) -> None:
        target = self.index + offset
        if 0 <= target < len(self.images):
            self.save_progress()
            self.index = target
            self.load_image()
            self.save_progress()

    def previous(self) -> None:
        self.navigate(-1)

    def next(self) -> None:
        self.navigate(1)

    def zoom(self, event) -> None:
        self.zoom_step(event, 1.25 if event.delta > 0 else 0.8)

    def zoom_step(self, event, factor: float) -> None:
        if self.original is None:
            return
        if event is None:
            anchor_x = self.canvas.winfo_width() / 2
            anchor_y = self.canvas.winfo_height() / 2
        else:
            anchor_x, anchor_y = event.x, event.y
        old_scale = self.scale
        image_x = self.canvas.canvasx(anchor_x) / old_scale
        image_y = self.canvas.canvasy(anchor_y) / old_scale
        self.zoom_factor = min(8.0, max(0.25, self.zoom_factor * factor))
        self.scale = self.base_scale * self.zoom_factor
        self.redraw()
        self.root.update_idletasks()
        width = max(1, round(self.original.width * self.scale))
        height = max(1, round(self.original.height * self.scale))
        self.canvas.xview_moveto(
            max(0.0, min(1.0, (image_x * self.scale - anchor_x) / width))
        )
        self.canvas.yview_moveto(
            max(0.0, min(1.0, (image_y * self.scale - anchor_y) / height))
        )

    def pan_start(self, event) -> None:
        self.canvas.scan_mark(event.x, event.y)

    def pan_move(self, event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def save_verified(self) -> None:
        self.save_label("verified")

    def save_provisional(self) -> None:
        self.save_label("provisional")

    def close(self) -> None:
        self.save_progress()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT.parent / "data",
        help="Dataset root containing the original photos (searched recursively).",
    )
    parser.add_argument(
        "--ground-truth", type=Path, default=PROJECT_ROOT / "ground_truth.csv"
    )
    parser.add_argument(
        "--plan", type=Path, default=PROJECT_ROOT / "ground_truth_plan.csv"
    )
    parser.add_argument(
        "--progress", type=Path, default=PROJECT_ROOT / "annotation_progress.json"
    )
    args = parser.parse_args()
    images = discover_images(args.input)
    GroundTruthAnnotator(
        images, args.ground_truth, args.plan, args.progress
    ).run()


if __name__ == "__main__":
    main()
