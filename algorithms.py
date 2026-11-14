"""
Класичні алгоритми обробки та реставрації документів.
"""

import numpy as np

class Algorithms:
    """
    Реалізація алгоритмів альтернативного пайплайну адаптивної бінаризації за моделлю Гатоса.
    """

    @staticmethod
    def clahe(lum: np.ndarray, threshold: float = 1.0, grid_size: int = 8) -> np.ndarray:
        """
        Адаптивне локальне вирівнювання гістограми (CLAHE, Zuiderveld 1994).
        Розбиває зображення на сітку блоків, обмежує контраст і інтерполює межі.
        """
        height, width = lum.shape
        grid_size = max(1, min(grid_size, height, width))
        cell_height = height // grid_size
        cell_width = width // grid_size

        mapping = np.zeros((grid_size, grid_size, 256), dtype=np.uint8)

        for i in range(grid_size):
            for j in range(grid_size):
                cell = lum[
                    i * cell_height:(i + 1) * cell_height if i < grid_size - 1 else height,
                    j * cell_width:(j + 1) * cell_width if j < grid_size - 1 else width,
                ]
                clip = max(1, int(threshold * (cell.size / 256.0)))
                hist = Algorithms._clip(Algorithms._histogram(cell), clip)
                mapping[i, j] = Algorithms._create_mapping(hist, cell.size)

        return Algorithms._interpolation(lum, mapping, grid_size, cell_height, cell_width)

    @staticmethod
    def _histogram(cell: np.ndarray) -> np.ndarray:
        """
        Обчислює 256-бінову гістограму інтенсивностей для окремого блоку сітки.
        """
        hist, _ = np.histogram(cell, bins=256, range=(0, 256))
        return hist

    @staticmethod
    def _clip(hist: np.ndarray, clip_limit: int) -> np.ndarray:
        """
        Обрізає піки гістограми вище clip_limit і рівномірно перерозподіляє надлишок.
        """
        excess = np.maximum(0, hist - clip_limit)
        total = np.sum(excess)

        clipped = np.minimum(hist, clip_limit) + total // 256
        clipped[:total % 256] += 1
        return clipped

    @staticmethod
    def _create_mapping(hist: np.ndarray, cell_size: int) -> np.ndarray:
        """
        Формує таблицю перекодування через нормовану кумулятивну суму.
        """
        cdf = np.cumsum(hist)
        mapping = (cdf / float(cell_size)) * 255.0
        return np.clip(np.round(mapping), 0, 255).astype(np.uint8)

    @staticmethod
    def _blend_block(sub: np.ndarray, corners: tuple) -> np.ndarray:
        """
        Білінійне змішування пікселів однієї ділянки за 4 сусідніми таблицями.
        """
        h, w = sub.shape
        x = np.linspace(0.0, 1.0, w, endpoint=False).reshape(1, -1)
        y = np.linspace(0.0, 1.0, h, endpoint=False).reshape(-1, 1)

        tl, tr, bl, br = corners
        top = (1.0 - x) * tl[sub] + x * tr[sub]
        bottom = (1.0 - x) * bl[sub] + x * br[sub]
        return np.round((1.0 - y) * top + y * bottom)

    @staticmethod
    def _interpolation(lum: np.ndarray, mapping: np.ndarray,
        grid_size: int, cell_height: int, cell_width: int) -> np.ndarray:
        """
        Виконує білінійну інтерполяцію між сусідніми блоками для усунення швів.
        """
        height, width = lum.shape
        output = np.zeros_like(lum)

        pad = np.pad(mapping, ((1, 1), (1, 1), (0, 0)), mode="edge")

        y = [0] + [int((i + 0.5) * cell_height) for i in range(grid_size)] + [height]
        x = [0] + [int((j + 0.5) * cell_width) for j in range(grid_size)] + [width]

        for i in range(grid_size + 1):
            for j in range(grid_size + 1):
                corners = (pad[i, j], pad[i, j + 1], pad[i + 1, j], pad[i + 1, j + 1])
                output[y[i]:y[i + 1], x[j]:x[j + 1]] = Algorithms._blend_block(lum[y[i]:y[i + 1],
                    x[j]:x[j + 1]], corners)
        return output

    @staticmethod
    def sauvola(lum: np.ndarray, frame: int = 25, k: float = 0.2, r: float = 128.0) -> np.ndarray:
        """
        Адаптивна локальна бінаризація Сауволи (Sauvola, 1997).
        Повертає первинну маску переднього плану: 255 — текст, 0 — фон.
        """
        radius = frame // 2
        intg, intg_sq = Algorithms._get_integrals(lum)
        bounds = Algorithms._window_bounds(lum.shape, radius)

        count = Algorithms._window_area(bounds)
        sum_win = Algorithms._window_sum(intg, bounds)
        sum_sq = Algorithms._window_sum(intg_sq, bounds)

        mean = sum_win / count
        deviation = np.sqrt(np.maximum(0.0, (sum_sq / count) - (mean ** 2)))

        threshold = mean * (1.0 + k * ((deviation / r) - 1.0))

        binary = np.where(lum <= threshold, 255, 0).astype(np.uint8)
        return binary

    @staticmethod
    def _window_bounds(shape: tuple, radius: int) -> tuple:
        """
        Повертає границі (y1, y2, x1, x2) для вікон навколо кожного пікселя.
        """
        h, w = shape
        y = np.arange(h)
        x = np.arange(w)
        y1 = np.clip(y - radius, 0, h)[:, None]
        y2 = np.clip(y + radius + 1, 0, h)[:, None]
        x1 = np.clip(x - radius, 0, w)[None, :]
        x2 = np.clip(x + radius + 1, 0, w)[None, :]
        return y1, y2, x1, x2

    @staticmethod
    def _window_sum(integral: np.ndarray, bounds: tuple) -> np.ndarray:
        """
        Обчислює суму значень у вікні за інтегральною матрицею та границями.
        """
        y1, y2, x1, x2 = bounds
        return integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]

    @staticmethod
    def _window_area(bounds: tuple) -> np.ndarray:
        """
        Повертає кількість пікселів у кожному вікні.
        """
        y1, y2, x1, x2 = bounds
        return (y2 - y1) * (x2 - x1)

    @staticmethod
    def _integral(image: np.ndarray) -> np.ndarray:
        """
        Обчислює двовимірний інтегральний масив для однієї матриці.
        """
        h, w = image.shape
        intg = np.zeros((h + 1, w + 1), dtype=np.float64)
        intg[1:, 1:] = np.cumsum(np.cumsum(image.astype(np.float64), axis=0), axis=1)
        return intg

    @staticmethod
    def _get_integrals(image: np.ndarray):
        """
        Обчислює двовимірні інтегральні масиви значень пікселів та їх квадратів.
        """
        img_f = image.astype(np.float64)
        return Algorithms._integral(img_f), Algorithms._integral(img_f ** 2)

    @staticmethod
    def background(lum: np.ndarray, binary_mask: np.ndarray, radius: int = 20) -> np.ndarray:
        """
        Оцінка поверхні чистого фону B(x, y).
        Інтерполює значення тла під літерами через навколишні фонові пікселі.
        """
        mask = (binary_mask == 0).astype(np.float64)
        bg = lum.astype(np.float64) * mask

        intg_bg = Algorithms._integral(bg)
        intg_pnts = Algorithms._integral(mask)
        bounds = Algorithms._window_bounds(lum.shape, radius)

        sum_win = Algorithms._window_sum(intg_bg, bounds)
        points = Algorithms._window_sum(intg_pnts, bounds)

        points = np.maximum(points, 1.0)
        interpolated = sum_win / points

        background = np.where(binary_mask == 0, lum, interpolated)
        return np.clip(np.round(background), 0, 255).astype(np.uint8)

    @staticmethod
    def contrast_classification(lum: np.ndarray, background: np.ndarray, binary_mask: np.ndarray,
        consts: tuple = (0.6, 0.5, 0.8)) -> np.ndarray:
        """
        Контрастна класифікація пікселів за сигмоїдою.
        Порівнює різницю (B - I) із динамічним порогом для виділення тексту.
        """
        q, p1, p2 = consts
        bg = background.astype(np.float64)
        diff = bg - lum.astype(np.float64)

        text_pixels = binary_mask > 0
        delta = np.sum(diff * text_pixels) / np.maximum(np.sum(text_pixels), 1.0)

        bg_avg = np.sum(bg * (~text_pixels)) / np.maximum(np.sum(~text_pixels), 1.0)
        bg_avg = np.maximum(bg_avg, 1e-6)

        exp = (-4.0 * bg) / (bg_avg * (1.0 - p1)) + (2.0 * (1.0 + p1)) / (1.0 - p1)
        sigmoid = (1.0 - p2) / (1.0 + np.exp(np.clip(exp, -50.0, 50.0))) + p2
        d = q * delta * sigmoid

        return np.where(diff > d, 255, 0).astype(np.uint8)

    @staticmethod
    def shrink_swell(binary: np.ndarray, n: int = 5) -> np.ndarray:
        """
        Морфологічна постобробка звуженням і розширенням.
        Прибирає поодинокий шум та заповнює розриви штрихів.
        """
        area = n * n
        fg = (binary > 0).astype(np.float64)

        intg = Algorithms._integral(fg)
        bounds = Algorithms._window_bounds(binary.shape, n // 2)
        count = Algorithms._window_sum(intg, bounds)

        # Shrink
        cleaned = np.where((fg == 1) & (count < 0.10 * area), 0, fg)

        # Swell
        final = np.where((cleaned == 0) & (count > 0.35 * area), 1, cleaned)

        return (final * 255).astype(np.uint8)
