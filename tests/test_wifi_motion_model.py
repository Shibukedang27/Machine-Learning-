from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.wifi_motion_radar.model import load_model, predict, train, training_accuracy
from src.wifi_motion_radar.signal_data import generate_demo_dataset


class WifiMotionModelTests(unittest.TestCase):
    def test_train_and_predict_demo_signals(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "wifi.csv"
            model_path = root / "wifi_model.json"

            generate_demo_dataset(csv_path, samples_per_label=90)
            model = train(csv_path, model_path)
            prediction = predict(csv_path, load_model(model_path))

            self.assertTrue(model_path.exists())
            self.assertGreaterEqual(training_accuracy(csv_path, model), 0.8)
            self.assertIn(prediction.label, model["labels"])
            self.assertGreaterEqual(prediction.confidence, 0.0)
            self.assertLessEqual(prediction.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
