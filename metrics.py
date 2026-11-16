"""
Мість клас для розрахунку метрик якості та стабільності.
"""

from typing import Tuple
import numpy as np


class Metrics:
    """
    Клас розрахунку метрик якості та стабільності.
    """

    # Базові метрики

    @staticmethod
    def mse(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Середньоквадратична помилка (MSE).
        Чим менше значення MSE, тим менша помилка.
        """
        return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))

    @staticmethod
    def psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
        """
        Пікове співвідношення сигнал/шум (PSNR).
        Чим вище значення PSNR, тим менше стороннього шуму додано в результаті обробки.
        """
        err = Metrics.mse(img1, img2)
        if err == 0.0:
            return 999.0
        return float(10.0 * np.log10((max_val ** 2) / err))

    @staticmethod
    def ssim(img1: np.ndarray, img2: np.ndarray, k1: float = 0.01, k2: float = 0.03) -> float:
        """
        Індекс структурної подібності (SSIM).
        Лежить у діапазоні [-1.0, 1.0]. Значення 1.0 свідчить про 100% збереження структури.
        """
        c1 = (k1 * 255.0) ** 2
        c2 = (k2 * 255.0) ** 2

        i1 = img1.astype(np.float64)
        i2 = img2.astype(np.float64)

        mu1 = np.mean(i1)
        mu2 = np.mean(i2)
        var1 = np.var(i1)
        var2 = np.var(i2)
        cov = np.mean((i1 - mu1) * (i2 - mu2))

        num = (2.0 * mu1 * mu2 + c1) * (2.0 * cov + c2)
        den = (mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2)
        return float(num / den)

    # Метрики DIBCO

    @staticmethod
    def f_measure(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
        """
        F-Measure (FM) за стандартом DIBCO.
        Передбачає вигляд документа: текст = 0 (чорний), фон = 255 (білий).
        Що вищий показник FM, то точніше відновлено документ без викривлення гліфів.
        """
        pred_text = pred_binary == 0
        gt_text = gt_binary == 0

        tp = np.sum(pred_text & gt_text)
        fp = np.sum(pred_text & (~gt_text))
        fn = np.sum((~pred_text) & gt_text)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0.0:
            return 0.0
        return float(2.0 * (precision * recall) / (precision + recall) * 100.0)

    @staticmethod
    def _recall_weights(gt_text: np.ndarray) -> np.ndarray:
        """
        Побудова карти ваг pseudo-Recall у діапазоні [0.0, 1.0]
        Скелет символу і тонкі лінії 1px отримує вагу 1.0, зовнішній край товстих літер 0.0.
        """
        dist = np.zeros(gt_text.shape, dtype=np.float64)
        curr = gt_text.copy()
        step = 1
        while np.any(curr):
            pad = np.pad(curr, 1, mode="constant", constant_values=False)
            eroded = (
                pad[:-2, :-2] & pad[:-2, 1:-1] & pad[:-2, 2:] &
                pad[1:-1, :-2] & pad[1:-1, 1:-1] & pad[1:-1, 2:] &
                pad[2:, :-2] & pad[2:, 1:-1] & pad[2:, 2:]
            )
            dist[curr & (~eroded)] = float(step)
            curr = eroded
            step += 1

        pad_d = np.pad(dist, 1, mode="constant", constant_values=0.0)
        max_nb = np.maximum.reduce([
            pad_d[:-2, :-2], pad_d[:-2, 1:-1], pad_d[:-2, 2:],
            pad_d[1:-1, :-2], pad_d[1:-1, 1:-1], pad_d[1:-1, 2:],
            pad_d[2:, :-2], pad_d[2:, 1:-1], pad_d[2:, 2:]
        ])

        is_skeleton = gt_text & (dist == max_nb)
        return np.where(
            is_skeleton,
            1.0,
            np.where(gt_text, (dist - 1.0) / np.maximum(max_nb - 1.0, 1.0), 0.0)
        )

    @staticmethod
    def pseudo_f_measure(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
        """
        Pseudo F-Measure (Fps) за стандартом DIBCO.
        Використовує вагові коефіцієнти відносно контуру символів:
        суворіше штрафує за втрату центрального скелета літери та шум біля меж.
        Що вищий показник, то краще збережена топологічна структура символів
        для подальшого розпізнавання тексту.
        """
        pred_text = pred_binary == 0
        gt_text = gt_binary == 0

        r_weights = Metrics._recall_weights(gt_text)
        r_tp = float(np.sum(r_weights[pred_text & gt_text]))
        r_fn = float(np.sum(r_weights[(~pred_text) & gt_text]))
        pseudo_recall = r_tp / (r_tp + r_fn) if (r_tp + r_fn) > 0.0 else 0.0

        pad_gt = np.pad(gt_text.astype(np.float64), 1, mode="constant")
        density = (
            pad_gt[:-2, 1:-1] + pad_gt[2:, 1:-1] +
            pad_gt[1:-1, :-2] + pad_gt[1:-1, 2:] +
            pad_gt[:-2, :-2] + pad_gt[:-2, 2:] +
            pad_gt[2:, :-2] + pad_gt[2:, 2:]
        ) / 8.0

        p_weights = 1.0 + density
        p_tp = float(np.sum(pred_text & gt_text))
        p_fp = float(np.sum(p_weights[pred_text & (~gt_text)]))
        pseudo_precision = p_tp / (p_tp + p_fp) if (p_tp + p_fp) > 0.0 else 0.0

        if pseudo_precision + pseudo_recall == 0.0:
            return 0.0
        return float(
            2.0 * (pseudo_precision * pseudo_recall) / (pseudo_precision + pseudo_recall) * 100.0
        )

    @staticmethod
    def _count_nubn(gt_bin: np.ndarray) -> int:
        """
        Підрахунок кількості неоднорідних блоків 8x8 (NUBN) в еталоні GT.
        """
        height, width = gt_bin.shape
        h_trim = (height // 8) * 8
        w_trim = (width // 8) * 8

        blocks = gt_bin[:h_trim, :w_trim].reshape(height // 8, 8, width // 8, 8)
        block_sums = blocks.sum(axis=(1, 3))
        nubn = int(np.sum((block_sums > 0) & (block_sums < 64)))
        return max(1, nubn)

    @staticmethod
    def _drd_weights() -> np.ndarray:
        """
        Побудова нормованої вагової матриці 5x5.
        """
        y_grid, x_grid = np.ogrid[-2:3, -2:3]
        dist = np.sqrt(x_grid ** 2 + y_grid ** 2).astype(np.float64)
        dist[2, 2] = 1.0

        weights = 1.0 / dist
        weights[2, 2] = 0.0
        return weights / np.sum(weights)

    @staticmethod
    def drd(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
        """
        Distance Reciprocal Distortion (DRD) за стандартом DIBCO.
        Вимірює візуальні спотворення бінарного зображення на основі оберненої
        відстані від помилкових пікселів до найближчих пікселів символу.
        Чим ближче до 0.0, тим чистіший документ і тим менше візуально помітних розривів чи плям.
        """
        pred_bin = (pred_binary == 0).astype(np.float64)
        gt_bin = (gt_binary == 0).astype(np.float64)
        height, width = pred_bin.shape

        nubn = Metrics._count_nubn(gt_bin)
        weights = Metrics._drd_weights()

        pad_gt = np.pad(gt_bin, 2, mode="edge")
        gt_conv = np.zeros((height, width), dtype=np.float64)
        for dy in range(5):
            for dx in range(5):
                if weights[dy, dx] > 0.0:
                    gt_conv += pad_gt[dy:dy + height, dx:dx + width] * weights[dy, dx]

        flipped_mask = pred_bin != gt_bin
        distortion = float(np.sum(np.abs(pred_bin[flipped_mask] - gt_conv[flipped_mask])))

        return float(distortion / nubn)

    @staticmethod
    def avg_score(fm: float, fps: float, psnr_val: float, drd_val: float) -> float:
        """
        Агрегований показник DIBCO.
        Об'єднує чотири основні метрики (FM, Fps, PSNR, DRD) у загальний рейтинг.
        """
        return float((fm + fps + psnr_val + (100.0 - drd_val)) / 4.0)

    @staticmethod
    def nrm(pred_binary: np.ndarray, gt_binary: np.ndarray) -> Tuple[float, float, float]:
        """
        Negative Rate Metric (NRM).
        Повертає: (NRM_загальний, Втрата_деталей_FN, Помилковий_шум_FP).
        - FN_rate: відсоток втрачених штрихів літер.
        - FP_rate: відсоток бруду на фоні.
        Діапазон [0.0, 1.0]. Ідеальне значення 0.0.
        Дозволяє точно налаштувати поріг q у стадії 4 для досягнення балансу.
        """
        pred_text = pred_binary == 0
        gt_text = gt_binary == 0

        tp = np.sum(pred_text & gt_text)
        fp = np.sum(pred_text & (~gt_text))
        fn = np.sum((~pred_text) & gt_text)
        tn = np.sum((~pred_text) & (~gt_text))

        fn_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        total_nrm = (fn_rate + fp_rate) / 2.0
        return float(total_nrm), float(fn_rate), float(fp_rate)

    # Індекс збереження країв

    @staticmethod
    def edge_preservation_index(orig_lum: np.ndarray, proc_lum: np.ndarray) -> float:
        """
        Коефіцієнт збереження меж (EPI). Оцінює збереження різкості контурів букв.
        Діапазон від 0 до 1, що ближче до 1, то менше розмито краї.
        """
        if orig_lum.shape[0] < 2 or orig_lum.shape[1] < 2:
            return 0.0

        i1 = orig_lum.astype(np.float64)
        i2 = proc_lum.astype(np.float64)

        delta1 = np.abs(i1[1:, :-1] - i1[:-1, :-1]) + np.abs(i1[:-1, 1:] - i1[:-1, :-1])
        delta2 = np.abs(i2[1:, :-1] - i2[:-1, :-1]) + np.abs(i2[:-1, 1:] - i2[:-1, :-1])

        d1_mean = np.mean(delta1)
        d2_mean = np.mean(delta2)

        cov = np.sum((delta1 - d1_mean) * (delta2 - d2_mean))
        var1 = np.sum((delta1 - d1_mean) ** 2)
        var2 = np.sum((delta2 - d2_mean) ** 2)

        denom = np.sqrt(var1 * var2)
        if denom == 0.0:
            return 0.0
        return float(cov / denom)

    # Індекс стабільності до шуму

    @staticmethod
    def noise_stability_index(clean_binary: np.ndarray, noisy_binary: np.ndarray) -> float:
        """
        Індекс стабільності до шуму (NSI) на основі метрики Жаккара (IoU).
        оцінює, наскільки сильно руйнується вихідний текст при появі вхідного шуму.
        Чим ближче до 1.0 тим краще.
        """
        t_clean = clean_binary == 0
        t_noisy = noisy_binary == 0

        intersection = np.sum(t_clean & t_noisy)
        union = np.sum(t_clean | t_noisy)

        if union == 0:
            return 1.0
        return float(intersection / union)
