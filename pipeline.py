'''
Логіка пайплайну та налаштування параметрів запуску відповідно до конфігу
'''

import json
import time
import shutil
from pathlib import Path
import numpy as np

from algorithms import Algorithms
from image_io import ImageIO
from metrics import Metrics


class Pipeline:
    """
    Містить 5-стадійний пайплайн за моделю Гатоса та формує таблиці метрик.
    """

    def __init__(self, config_path: str = "config.json"):
        """
        Завантаження конфігурації за замовчуванням та з файлу config.json.
        """
        self.cfg = {
            "data_dir": "dataset",
            "out_dir": "processed",
            "gt_dir": "gt",
            "clahe_threshold": 1.0,
            "clahe_grid": 8,
            "sauvola_frame": 25,
            "sauvola_k": 0.2,
            "sauvola_r": 128.0,
            "bg_radius": 20,
            "contrast_q": 0.6,
            "contrast_p1": 0.5,
            "contrast_p2": 0.8,
            "morph_n": 3
        }
        config = Path(config_path)
        if config.exists():
            with open(config, "r", encoding="utf-8") as f:
                self.cfg.update(json.load(f))

        self.data_dir = Path(self.cfg["data_dir"])
        self.out_dir = Path(self.cfg["out_dir"])
        self.gt_dir = Path(self.cfg["gt_dir"])

    def process(self, lum: np.ndarray) -> tuple:
        """
        Головний пайплайн обробки.
        Приймає масив яскравості (lum) і повертає кортеж із 5 проміжних результатів обробки.
        """
        # 1. CLAHE
        st1 = Algorithms.clahe(
            lum,
            threshold = self.cfg["clahe_threshold"], grid_size = self.cfg["clahe_grid"]
        )

        # 2. Саувола
        st2 = Algorithms.sauvola(
            st1,
            frame = self.cfg["sauvola_frame"], k = self.cfg["sauvola_k"], r = self.cfg["sauvola_r"]
        )

        # 3. Інтерполяція фону
        st3 = Algorithms.background(st1, st2, radius = self.cfg["bg_radius"])

        # 4. Контрастна класифікація
        st4 = Algorithms.contrast_classification(
            st1, st3, st2,
            consts = (self.cfg["contrast_q"], self.cfg["contrast_p1"], self.cfg["contrast_p2"])
        )

        # 5. Морфологія (Розширення та звуження)
        st5 = Algorithms.shrink_swell(st4, n = self.cfg["morph_n"])

        return st1, st2, st3, st4, st5

    def _process_file(self, img_path: Path) -> dict:
        """
        Відкриття файлу в масив, копіювання оригіналу, запуск пайплайну та збереження.
        """
        lum = ImageIO.luminance(img_path)
        if lum is None:
            return {"File": img_path.name, "Status": "Error: Corrupted or empty file"}

        folder = self.out_dir / img_path.stem
        folder.mkdir(parents=True, exist_ok=True)
        t_start = time.time()

        shutil.copy(img_path, folder / f"original{img_path.suffix}")
        ImageIO.save(lum, folder / "stage0_lum.png")

        stages = self.process(lum)

        stage_names = [
            "stage1_clahe.png",
            "stage2_sauvola.png",
            "stage3_background.png",
            "stage4_contrast.png",
            "stage5_shrink_swell.png"
        ]

        for name, st_arr in zip(stage_names, stages):
            ImageIO.save(st_arr, folder / name)

        # Інверсія щоб був чорний текст на білому фоні
        final = 255 - stages[-1]
        ImageIO.save(final, folder / "final.png")

        return self._create_report(img_path, [lum, *stages], t_start)

    def _create_report(self, img_path: Path, stages: list, t_start: float) -> dict:
        """
        Обчислення поетапних та глобальних метрик, запис локального CSV.
        """
        lum = stages[0]
        final = 255 - stages[-1]
        folder = self.out_dir / img_path.stem

        # 1. Локальний metrics.csv (поетапні метрики для стадій 1..5)
        local_rows = [{
            "Stage": f"Stage {i}",
            "MSE": round(Metrics.mse(st, lum), 2),
            "PSNR": round(Metrics.psnr(st, lum), 2),
            "SSIM": round(Metrics.ssim(st, lum), 4),
            "EPI": round(Metrics.edge_preservation_index(lum, st), 4)
        } for i, st in enumerate(stages[1:], 1)]
        ImageIO.write_csv(local_rows, folder / "metrics.csv")

        # 2. Базові та шумові метрики для загального звіту
        noisy = self._run_noise(lum)
        summary = {
            "File": img_path.name,
            "Time_s": round(time.time() - t_start, 3),
            "MSE": round(Metrics.mse(final, lum), 2),
            "PSNR": round(Metrics.psnr(final, lum), 2),
            "SSIM": round(Metrics.ssim(final, lum), 4),
            "EPI": round(Metrics.edge_preservation_index(lum, final), 4),
            "NSI": round(Metrics.noise_stability_index(final, noisy), 4),
            "DIBCO_FM_%": -1.0,
            "DIBCO_Fps_%": -1.0,
            "DIBCO_PSNR": -1.0,
            "DIBCO_DRD": -1.0,
            "DIBCO_Avg_Score": -1.0,
            "GT_NRM": -1.0,
            "Detail_Loss_FN_%": -1.0,
            "Noise_FP_%": -1.0,
            "Status": "Success"
        }

        # 3. Метрики DIBCO та NRM за наявності Ground Truth
        gt_file = ImageIO.find_gt(self.gt_dir, img_path.stem)
        if gt_file:
            summary.update(self._gt_metrics(folder, final, gt_file))

        return summary

    def _gt_metrics(self, folder: Path, final: np.ndarray, gt_file: Path) -> dict:
        """
        Копіює еталон GT та обчислює всі еталонні метрики стандарту DIBCO та NRM.
        """
        gt = ImageIO.luminance(gt_file)
        if gt is None or gt.shape != final.shape:
            return {}

        shutil.copy(gt_file, folder / f"gt{gt_file.suffix}")

        fm = round(Metrics.f_measure(final, gt), 2)
        fps = round(Metrics.pseudo_f_measure(final, gt), 2)
        psnr_gt = round(Metrics.psnr(final, gt), 2)
        drd_val = round(Metrics.drd(final, gt), 2)
        avg = round(Metrics.avg_score(fm, fps, psnr_gt, drd_val), 2)
        nrm_val, fn_loss, fp_noise = Metrics.nrm(final, gt)

        return {
            "DIBCO_FM_%": fm,
            "DIBCO_Fps_%": fps,
            "DIBCO_PSNR": psnr_gt,
            "DIBCO_DRD": drd_val,
            "DIBCO_Avg_Score": avg,
            "GT_NRM": round(nrm_val, 4),
            "Detail_Loss_FN_%": round(fn_loss * 100, 2),
            "Noise_FP_%": round(fp_noise * 100, 2),
        }

    def _run_noise(self, lum: np.ndarray) -> np.ndarray:
        """
        Накладає шум на масив яскравості й проганяє через той самий process.
        """
        np.random.seed(42)
        noisy_lum = np.clip(
            lum.astype(np.float64) + np.random.normal(0, 5, lum.shape), 0, 255).astype(np.uint8)

        _, _, _, _, st5 = self.process(noisy_lum)
        return 255 - st5

    def run(self):
        """
        Запуск пайплайну для всієї вибірки з папки dataset.
        """
        files = ImageIO.list_images(self.data_dir)
        print(f"Знайдено {len(files)} фото. Починаємо обробку...")

        all_summary = []
        for idx, img_path in enumerate(files, 1):
            res = self._process_file(img_path)
            all_summary.append(res)
            print(
                f"[{idx}/{len(files)}] {img_path.name} -> "
                f"{res.get('Status')} ({res.get('Time_s', 0)} с)"
            )

        ImageIO.write_csv(all_summary, self.out_dir / "summary_metrics.csv")
        print("Готово! Результати збережено в processed/summary_metrics.csv")


if __name__ == "__main__":
    Pipeline().run()
