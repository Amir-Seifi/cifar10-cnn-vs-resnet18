"""Regression tests for the offline audit of the recorded experiments."""

import contextlib
import csv
import io
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.verify_results import ROOT, verify


class TestReportedMetrics(unittest.TestCase):
    def test_verify_published_outputs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            verify()

    def test_detect_swapped_validation_image_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results"
            shutil.copytree(ROOT, results)
            path = results / "augmentation/no_aug_validation_predictions.csv"
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                fields = reader.fieldnames
            first = rows[0]
            other = next(row for row in rows if row["true_id"] != first["true_id"])
            first["image_id"], other["image_id"] = other["image_id"], first["image_id"]
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Ablation label"):
                verify(results)
