import tempfile
import unittest
from pathlib import Path

from PIL import Image

from fer2013_emotion.data import stratified_split, validate_dataset


class DatasetValidationTests(unittest.TestCase):
    def make_dataset(self, root: Path, *, image_count: int = 2) -> None:
        for split in ("train", "test"):
            for class_index in range(7):
                class_dir = root / split / f"class_{class_index}"
                class_dir.mkdir(parents=True)
                count = image_count if split == "train" else 1
                for image_index in range(count):
                    Image.new("L", (8, 8), color=image_index).save(
                        class_dir / f"image_{image_index}.png"
                    )

    def test_validates_dataset_without_modifying_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root)
            before = sorted(path.relative_to(root) for path in root.rglob("*.png"))

            paths = validate_dataset(root)

            after = sorted(path.relative_to(root) for path in root.rglob("*.png"))
            self.assertEqual(paths.class_names, tuple(f"class_{index}" for index in range(7)))
            self.assertEqual(before, after)

    def test_rejects_mismatched_train_and_test_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root)
            (root / "test" / "class_6" / "image_0.png").unlink()
            (root / "test" / "class_6").rmdir()

            with self.assertRaisesRegex(ValueError, "classes do not match"):
                validate_dataset(root)

    def test_reports_corrupt_images_instead_of_deleting_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root)
            corrupt = root / "train" / "class_0" / "broken.png"
            corrupt.write_text("not an image", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "invalid image files"):
                validate_dataset(root)
            self.assertTrue(corrupt.exists())


class StratifiedSplitTests(unittest.TestCase):
    def test_split_is_reproducible_and_contains_each_class_on_both_sides(self) -> None:
        class_paths = [
            [Path(f"class_{class_index}/image_{image_index}.png") for image_index in range(10)]
            for class_index in range(7)
        ]

        first = stratified_split(class_paths, validation_fraction=0.2, seed=17)
        second = stratified_split(class_paths, validation_fraction=0.2, seed=17)

        self.assertEqual(first, second)
        train_paths, train_labels, validation_paths, validation_labels = first
        self.assertEqual(len(train_paths), 56)
        self.assertEqual(len(validation_paths), 14)
        self.assertEqual(set(train_labels), set(range(7)))
        self.assertEqual(set(validation_labels), set(range(7)))
        self.assertFalse(set(train_paths) & set(validation_paths))


if __name__ == "__main__":
    unittest.main()
