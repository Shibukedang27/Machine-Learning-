from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.face_recognition_system.demo_data import generate_demo_dataset
from src.face_recognition_system.recognizer import load_model, recognize, train


class RecognizerTests(unittest.TestCase):
    def test_training_and_recognition_on_demo_faces(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "people"
            model_path = root / "model.json"

            generate_demo_dataset(dataset, samples_per_person=4)
            model = train(dataset, model_path)
            prediction = recognize(model, dataset / "Ada" / "ada_01.pgm")

            self.assertTrue(model_path.exists())
            self.assertEqual(set(load_model(model_path)["profiles"]), {"Ada", "Ben"})
            self.assertEqual(prediction.label, "Ada")
            self.assertGreater(prediction.confidence, 0.85)


if __name__ == "__main__":
    unittest.main()
