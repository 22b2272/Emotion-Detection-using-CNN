import unittest

import numpy as np

from fer2013_emotion.evaluation import classification_metrics


class ClassificationMetricsTests(unittest.TestCase):
    def test_metrics_use_probability_argmax_and_class_order(self) -> None:
        result = classification_metrics(
            true_labels=[0, 1, 2, 2],
            probabilities=np.asarray(
                [
                    [0.9, 0.05, 0.05],
                    [0.2, 0.7, 0.1],
                    [0.6, 0.3, 0.1],
                    [0.1, 0.2, 0.7],
                ]
            ),
            class_names=["angry", "happy", "sad"],
        )

        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [1, 0, 1]])
        self.assertEqual(result["per_class"]["sad"]["support"], 2)
        self.assertAlmostEqual(result["macro_f1"], 7 / 9)

    def test_rejects_probability_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            classification_metrics([0, 1], np.ones((2, 2)), ["angry", "happy", "sad"])


if __name__ == "__main__":
    unittest.main()
