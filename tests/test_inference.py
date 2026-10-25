import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from fer2013_emotion.inference import predict_image


class FakeModel:
    def __init__(self, expected_shape: tuple[int, ...]) -> None:
        self.expected_shape = expected_shape

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        if batch.shape != self.expected_shape:
            raise AssertionError(f"Unexpected model input shape: {batch.shape}")
        probabilities = np.zeros((1, 7), dtype=np.float32)
        probabilities[0, 3] = 1.0
        return probabilities


class InferenceTests(unittest.TestCase):
    def test_custom_cnn_receives_resized_grayscale_image(self) -> None:
        class_names = [f"class_{index}" for index in range(7)]
        metadata = {"variant": "custom-cnn", "class_names": class_names}
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "face.png"
            Image.new("RGB", (12, 20), color="red").save(image_path)
            model = FakeModel((1, 48, 48, 1))

            result = predict_image(model, metadata, image_path)

        self.assertEqual(result["class_name"], "class_3")
        self.assertEqual(result["confidence"], 1.0)
        self.assertEqual(result["probabilities"]["class_3"], 1.0)

    def test_resnet_receives_resized_rgb_image(self) -> None:
        metadata = {"variant": "resnet50v2", "class_names": [f"class_{i}" for i in range(7)]}
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "face.png"
            Image.new("L", (12, 20), color=128).save(image_path)
            model = FakeModel((1, 224, 224, 3))

            result = predict_image(model, metadata, image_path)

        self.assertEqual(result["class_name"], "class_3")


if __name__ == "__main__":
    unittest.main()
