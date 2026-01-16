from pathlib import Path
from typing import Optional
import re

from PIL import Image


def _next_index_in_dir(folder: Path) -> int:
    """
    Looks for files like 000.png, 001.png, ... and returns next available index.
    """
    folder.mkdir(parents=True, exist_ok=True)
    pat = re.compile(r"^(\d+)\.png$")
    max_i = -1
    for p in folder.glob("*.png"):
        m = pat.match(p.name)
        if m:
            max_i = max(max_i, int(m.group(1)))
    return max_i + 1


def make_sample_path(base_dir: Path, label: str, index: Optional[int] = None) -> Path:
    """
    data/<label>/<index>.png
    If index is None, auto-increment based on existing files in that folder.
    """
    out_dir = base_dir / label
    if index is None:
        index = _next_index_in_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{int(index):03d}.png"


def save_axis_free_png(rgb_uint8, out_path: Path) -> None:
    """
    Save an already-rendered RGB image (no axes) as a PNG.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(rgb_uint8, mode="RGB")
    img.save(out_path)