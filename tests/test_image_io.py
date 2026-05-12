from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.face_recognition_system.image_io import GrayImage, read_image, write_pgm


class ImageIoTests(unittest.TestCase):
    def test_write_and_read_binary_pgm(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.pgm"
            image = GrayImage(width=2, height=2, pixels=[[0, 128], [200, 255]])

            write_pgm(path, image)
            loaded = read_image(path)

            self.assertEqual(loaded.width, 2)
            self.assertEqual(loaded.height, 2)
            self.assertEqual(loaded.pixels, image.pixels)

    def test_read_ascii_ppm_as_grayscale(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.ppm"
            path.write_text("P3\n2 1\n255\n255 0 0 0 255 0\n", encoding="ascii")

            loaded = read_image(path)

            self.assertEqual(loaded.pixels[0][0], 76)
            self.assertEqual(loaded.pixels[0][1], 150)


if __name__ == "__main__":
    unittest.main()
