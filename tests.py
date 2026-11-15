"""
Модульні автотести для перевірки коректності методів обробки зображень.
Усі очікувані значення прописані як константні матриці.
"""

import unittest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

from algorithms import Algorithms
from image_io import ImageIO
from pipeline import Pipeline


class TestAlgorithms5x5(unittest.TestCase):
    """
    Тестування 5 методів алгоритмічного конвеєра на матрицях 5x5
    із жорстко зафіксованими ручними матрицями-еталонами.
    """

    def test_clahe_equalization_5x5(self):
        """
        Тест 1 (CLAHE): перевірка лінійного вирівнювання 25 унікальних рівнів яскравості.
        За ручним розрахунком для вхідних значень від 0 до 240 з кроком 10 кожен
        піксель отримує ранг k від 1 до 25. Значення кумулятивної функції перекодування
        обчислюється за формулою round(k / 25 * 255), перетворюючи перший піксель на 10,
        центральний на 133, а максимальний на 255. Очікуваний результат задано літеральною
        константною матрицею чисел.
        """
        lum = np.array([
            [  0,  10,  20,  30,  40],
            [ 50,  60,  70,  80,  90],
            [100, 110, 120, 130, 140],
            [150, 160, 170, 180, 190],
            [200, 210, 220, 230, 240]
        ], dtype=np.uint8)

        res = Algorithms.clahe(lum, threshold=1.0, grid_size=1)

        expected = np.array([
            [ 10,  20,  31,  41,  51],
            [ 61,  71,  82,  92, 102],
            [112, 122, 133, 143, 153],
            [163, 173, 184, 194, 204],
            [214, 224, 235, 245, 255]
        ], dtype=np.uint8)

        np.testing.assert_array_equal(res, expected)

    def test_sauvola_threshold_5x5(self):
        """
        Тест 2 (Саувола): бінаризація одиночної точки чорнила 50 на однорідному фоні 200.
        За ручним розрахунком для вікна 3x3 дає суму 1650 і середнє mu = 1650 / 9 = 183.33,
        дисперсію (50^2 + 8*200^2)/9 - 183.33^2 = 2222.22 та відхилення
        sigma = 47.14. Локальний поріг становить T = 183.33 * (1 + 0.2 * (47.14 / 128 - 1)) =
        160.17. Оскільки для центру 50 <= 160.17, він позначається як текст 255, а для решти
        пікселів 200 > 160.17, тому вони переходять у 0.
        """
        lum = np.array([
            [200, 200, 200, 200, 200],
            [200, 200, 200, 200, 200],
            [200, 200,  50, 200, 200],
            [200, 200, 200, 200, 200],
            [200, 200, 200, 200, 200]
        ], dtype=np.uint8)

        res = Algorithms.sauvola(lum, frame=3, k=0.2, r=128.0)

        expected = np.array([
            [  0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0],
            [  0,   0, 255,   0,   0],
            [  0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0]
        ], dtype=np.uint8)

        np.testing.assert_array_equal(res, expected)

    def test_background_interpolation_5x5(self):
        """
        Тест 3 (Поверхня фону): реконструкція пергаменту під розпізнаним символом.
        За ручним розрахунком у вікні 3x3 для центрального пікселя тексту є 8 фонових сусідів
        із яскравістю 200, що дає інтерпольоване середнє значення (8 * 200) / 8 = 200.
        У результаті інтерполяції символ стирається, а вихідна матриця відновлюється до
        ідеально однорідного фонового поля зі значенням 200.
        """
        lum = np.array([
            [200, 200, 200, 200, 200],
            [200, 200, 200, 200, 200],
            [200, 200,  50, 200, 200],
            [200, 200, 200, 200, 200],
            [200, 200, 200, 200, 200]
        ], dtype=np.uint8)

        binary_mask = np.array([
            [  0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0],
            [  0,   0, 255,   0,   0],
            [  0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0]
        ], dtype=np.uint8)

        res = Algorithms.background(lum, binary_mask, radius=1)

        expected = np.array([
            [200, 200, 200, 200, 200],
            [200, 200, 200, 200, 200],
            [200, 200, 200, 200, 200],
            [200, 200, 200, 200, 200],
            [200, 200, 200, 200, 200]
        ], dtype=np.uint8)

        np.testing.assert_array_equal(res, expected)

    def test_contrast_classification_5x5(self):
        """
        Тест 4 (Контрастна класифікація): перевірка динамічного відсікання за сигмоїдою Гатоса.
        За ручним розрахунком різниця у центрі становить 200 - 50 = 150,
        середній контраст delta = 150.0, а середня фонова яскравість 200.0. При bg = 200
        показник експоненти дорівнює (-4 + 2 * 1.5) / 0.5 = -2.0, сигмоїдне насичення становить
        0.2 / (1 + exp(-2)) + 0.8 = 0.9762, а адаптивний поріг дорівнює d = 0.6 * 150 * 0.9762 =
        87.85. Оскільки різниця у центрі 150 > 87.85, піксель класифікується як текст 255, а фонові
        пікселі 0 <= 87.85 відсікаються в 0.
        """
        lum = np.array([
            [200, 200, 200, 200, 200],
            [200, 200, 200, 200, 200],
            [200, 200,  50, 200, 200],
            [200, 200, 200, 200, 200],
            [200, 200, 200, 200, 200]
        ], dtype=np.uint8)

        bg = np.array([
            [200, 200, 200, 200, 200],
            [200, 200, 200, 200, 200],
            [200, 200, 200, 200, 200],
            [200, 200, 200, 200, 200],
            [200, 200, 200, 200, 200]
        ], dtype=np.uint8)

        binary_mask = np.array([
            [  0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0],
            [  0,   0, 255,   0,   0],
            [  0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0]
        ], dtype=np.uint8)

        res = Algorithms.contrast_classification(
            lum, bg, binary_mask, consts=(0.6, 0.5, 0.8)
        )

        expected = np.array([
            [  0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0],
            [  0,   0, 255,   0,   0],
            [  0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0]
        ], dtype=np.uint8)

        np.testing.assert_array_equal(res, expected)

    def test_shrink_noise_removal_5x5(self):
        """
        Тест 5 (Морфологічне звуження): усунення одиночного пікселя шуму 255 у точці (2, 2).
        За ручним розрахунком при вікні 5 на 5 граничний поріг для
        звуження становить count < 0.10 * 25 = 2.5. Оскільки центральний піксель має лише
        1 одиничне значення в усьому вікні, умова 1 < 2.5 виконується, і шумова цятка
        обнуляється, утворюючи повністю очищену нульову матрицю.
        """
        binary = np.array([
            [  0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0],
            [  0,   0, 255,   0,   0],
            [  0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0]
        ], dtype=np.uint8)

        res = Algorithms.shrink_swell(binary, n=5)

        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)

        np.testing.assert_array_equal(res, expected)

    def test_swell_hole_filling_5x5(self):
        """
        Тест 6 (Морфологічне розширення): заповнення внутрішнього розриву контуру. За ручним
        розрахунком при вікні 3 на 3 для чорного пікселя в центрі кільця 3 на 3
        поріг розширення становить count > 0.35 * 9 = 3.15. Оскільки центр оточений 8
        сусідніми пікселями тексту, умова 8 > 3.15 виконується, розрив склеюється до 255,
        а зовнішні нулі не розростаються бо 3 <= 3.15, утворюючи суцільний квадрат.
        """
        binary = np.array([
            [  0,   0,   0,   0,   0],
            [  0, 255, 255, 255,   0],
            [  0, 255,   0, 255,   0],
            [  0, 255, 255, 255,   0],
            [  0,   0,   0,   0,   0]
        ], dtype=np.uint8)

        res = Algorithms.shrink_swell(binary, n=3)

        expected = np.array([
            [  0,   0,   0,   0,   0],
            [  0, 255, 255, 255,   0],
            [  0, 255, 255, 255,   0],
            [  0, 255, 255, 255,   0],
            [  0,   0,   0,   0,   0]
        ], dtype=np.uint8)

        np.testing.assert_array_equal(res, expected)

class TestEdgeCases(unittest.TestCase):
    """
    Перевірка обробки крайових значень без аварійного завершення.
    Покриває: порожні файли, пошкоджені файли, колірні формати,
    нестандартні прямокутні розміри та монохромні зображення.
    """

    def test_monochrome_zeros(self):
        """
        Крайній випадок: повністю чорне зображення (ділення на 0, дисперсія = 0).
        """
        zeros = np.zeros((5, 5), dtype=np.uint8)
        st1 = Algorithms.clahe(zeros, grid_size=2)
        st2 = Algorithms.sauvola(st1, frame=3)
        st3 = Algorithms.background(st1, st2, radius=1)
        st4 = Algorithms.contrast_classification(st1, st3, st2)
        st5 = Algorithms.shrink_swell(st4, n=3)

        self.assertEqual(st5.shape, (5, 5))

    def test_monochrome_ones(self):
        """
        Крайній випадок: повністю біле зображення.
        """
        ones = np.full((5, 5), 255, dtype=np.uint8)
        st1 = Algorithms.clahe(ones, grid_size=2)
        st2 = Algorithms.sauvola(st1, frame=3)
        st3 = Algorithms.background(st1, st2, radius=1)
        st4 = Algorithms.contrast_classification(st1, st3, st2)
        st5 = Algorithms.shrink_swell(st4, n=3)

        self.assertEqual(st5.shape, (5, 5))

    def test_non_square_dimensions(self):
        """
        Крайній випадок: неквадратні розміри зображення (наприклад 4x13).
        Перевіряє коректність роботи ковзних вікон і сіток CLAHE.
        """
        rect_img = np.random.randint(0, 256, (4, 13), dtype=np.uint8)
        st1 = Algorithms.clahe(rect_img, grid_size=2)
        st2 = Algorithms.sauvola(st1, frame=3)
        st3 = Algorithms.background(st1, st2, radius=2)
        st4 = Algorithms.contrast_classification(st1, st3, st2)
        st5 = Algorithms.shrink_swell(st4, n=3)

        self.assertEqual(st5.shape, (4, 13))

    def test_empty_file_io(self):
        """
        Крайній випадок: порожній файл розміром 0 байт.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty_file = Path(tmp_dir) / "empty.png"
            empty_file.touch()

            result = ImageIO.luminance(empty_file)
            self.assertIsNone(result)

    def test_corrupted_file_io(self):
        """
        Крайній випадок: пошкоджений файл зі сміттям замість коректного заголовка.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            corrupt_file = Path(tmp_dir) / "corrupt.png"
            corrupt_file.write_bytes(b"INVALID_HEADER_NOT_AN_IMAGE_BYTES_123456789")

            result = ImageIO.luminance(corrupt_file)
            self.assertIsNone(result)

    def test_different_color_formats(self):
        """
        Крайній випадок: зображення у форматі RGBA з альфа-каналом прозорості.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            rgba_file = Path(tmp_dir) / "transparent.png"
            img = Image.new("RGBA", (10, 10), (120, 150, 180, 128))
            img.save(rgba_file)

            lum = ImageIO.luminance(rgba_file)
            self.assertIsNotNone(lum)
            assert lum is not None
            self.assertEqual(lum.shape, (10, 10))
            self.assertEqual(lum.dtype, np.uint8)

    def test_pipeline_graceful_error_handling(self):
        """
        Крайній випадок: робота пайплайну на пошкодженому файлі
        повинна повертати статус помилки без аварійного переривання циклу.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            corrupt_file = Path(tmp_dir) / "bad.jpg"
            corrupt_file.write_bytes(b"CORRUPTED_JPG_CONTENT")

            pipe = Pipeline()
            report = pipe.process_file(corrupt_file)

            self.assertEqual(report.get("File"), "bad.jpg")
            self.assertIn("Error", report.get("Status", ""))


if __name__ == "__main__":
    unittest.main()
