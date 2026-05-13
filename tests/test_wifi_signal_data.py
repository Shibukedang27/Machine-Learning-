from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.wifi_motion_radar.signal_data import build_windows, generate_demo_dataset, read_wifi_csv


class WifiSignalDataTests(unittest.TestCase):
    def test_demo_data_builds_signal_windows(self) -> None:
        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "wifi.csv"

            generate_demo_dataset(csv_path, samples_per_label=60)
            samples = read_wifi_csv(csv_path)
            windows = build_windows(samples)

            self.assertEqual(len(samples), 240)
            self.assertGreater(len(windows), 10)
            self.assertEqual(len(windows[-1].features), 7)


if __name__ == "__main__":
    unittest.main()
