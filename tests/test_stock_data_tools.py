from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.stock_pattern_analysis.data_tools import build_feature_rows, generate_demo_stock, read_price_csv


class StockDataToolTests(unittest.TestCase):
    def test_demo_stock_generates_feature_rows(self) -> None:
        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "demo.csv"

            generate_demo_stock(csv_path, days=80)
            rows = read_price_csv(csv_path)
            features = build_feature_rows(rows)

            self.assertEqual(len(rows), 80)
            self.assertGreater(len(features), 40)
            self.assertEqual(len(features[-1].features), 7)


if __name__ == "__main__":
    unittest.main()
