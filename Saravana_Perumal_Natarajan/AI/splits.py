"""Image-level data split: one image per group in validation and in the held-out test set."""

TEST = {
    "img2.jpg",
    "20240713_193659.jpg",
    "20240713_193907.jpg",
    "20240713_194316.jpg",
    "20240713_194630.jpg",
}
VAL = {"img5.jpg", "20240713_193650.jpg", "20240713_193839.jpg", "20240713_194256.jpg", "20240713_194621.jpg"}


def split_of(file_name: str) -> str:
    """Return ``"test"``, ``"val"`` or ``"train"`` for an image file name."""
    return "test" if file_name in TEST else "val" if file_name in VAL else "train"
