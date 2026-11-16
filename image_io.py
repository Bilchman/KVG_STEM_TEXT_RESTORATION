"""
Модуль операцій введення-виведення, конвертація файл та збереження CSV.
"""

from pathlib import Path
import csv
from typing import Optional, List
import re
import numpy as np
from PIL import Image, UnidentifiedImageError


class ImageIO:
    """
    Клас для читання зображень у NumPy-масиви, збереження на диск та роботи з CSV.
    """

    EXTENSIONS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

    @staticmethod
    def luminance(path: Path) -> Optional[np.ndarray]:
        """
        Відкриває фото, переводить у LAB і повертає L-канал як 2D масив.
        При пошкодженому або порожньому файлі повертає None.
        """
        if not path.exists() or path.stat().st_size == 0:
            return None
        try:
            with Image.open(path) as img:
                if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                    rgb_img = Image.alpha_composite(bg, img.convert("RGBA")).convert("RGB")
                else:
                    rgb_img = img.convert("RGB")

                return np.array(rgb_img.convert("LAB").split()[0], dtype=np.uint8)
        except (UnidentifiedImageError, OSError, ValueError):
            return None

    @staticmethod
    def save(arr: np.ndarray, path: Path) -> None:
        """
        Конвертує 2D масив NumPy у фото та зберігає.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr.astype(np.uint8), mode="L").save(path)

    @staticmethod
    def write_csv(rows: list, path: Path) -> None:
        """
        Записує список даних у CSV файл.
        """
        if not rows:
            return

        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def list_images(directory: Path) -> List[Path]:
        """
        Повертає відсортований список графічних файлів у директорії.
        """
        images = []
        if not directory.exists():
            return images

        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in ImageIO.EXTENSIONS:
                images.append(file_path)

        images.sort(
            key = lambda p:
                [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", p.name)]
        )
        return images

    @staticmethod
    def find_gt(gt_dir: Path, stem: str) -> Optional[Path]:
        """
        Шукає еталонний файл у папці gt_dir за іменем stem.
        """
        if not gt_dir.exists():
            return None

        for file_path in ImageIO.list_images(gt_dir):
            if file_path.stem == stem:
                return file_path
            if file_path.stem == f"{stem}_gt":
                return file_path

        return None
