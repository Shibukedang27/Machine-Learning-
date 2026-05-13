from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.stock_pattern_analysis.data_tools import generate_demo_stock
from src.stock_pattern_analysis.model import analyze, load_model, train


class StockModelTests(unittest.TestCase):
    def test_train_and_analyze_demo_stock(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "demo.csv"
            model_path = root / "model.json"

            generate_demo_stock(csv_path, days=150)
            model = train(csv_path, model_path)
            result = analyze(csv_path, load_model(model_path))

            self.assertTrue(model_path.exists())
            self.assertGreater(model["training_rows"], 100)
            self.assertGreaterEqual(result.probability_up, 0.0)
            self.assertLessEqual(result.probability_up, 1.0)
            self.assertTrue(result.pattern)
            self.assertTrue(result.notes)


if __name__ == "__main__":
    unittest.main()
