"""Regression checks for genotype validation. No research data needed."""
import csv
import tempfile
import unittest
from pathlib import Path
from repro_io import load_genotype_dataset


class InputChecks(unittest.TestCase):
    def check_value(self, value):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["snp_a", "class"])
                writer.writerows([[value, 0], [2, 1]])
            return load_genotype_dataset(path)

    def test_fraction_rejected(self):
        with self.assertRaises(ValueError):
            self.check_value(0.5)

    def test_overflow_rejected(self):
        with self.assertRaises(ValueError):
            self.check_value(65536)

    def test_valid_integer(self):
        x, y, names, label = self.check_value(1)
        self.assertEqual(x[:, 0].tolist(), [1, 2])
        self.assertEqual(y.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
