"""Full tests of training and prediction.

This runs five epochs of training over a small toy data set, attempting to
overfit, then compares the resubstitution predictions on this set to
previously computed results. As such this is essentially a change-detector
test.
"""

import contextlib
import difflib
import os
import tempfile
import unittest

from parameterized import parameterized

from yoyodyne_pretrained.cli import main

# Directory the unit test is located in, relative to the working directory.
DIR = os.path.relpath(os.path.dirname(__file__), os.getcwd())
CONFIG_PATH = os.path.join(DIR, "testdata/mbert.yaml")
TESTDATA_DIR = os.path.join(DIR, "testdata")


class YoyodynePretrainedTest(unittest.TestCase):
    def assertNonEmptyFileExists(self, path: str):
        self.assertTrue(os.path.exists(path), msg=f"file {path} not found")
        self.assertGreater(
            os.stat(path).st_size, 0, msg="file {path} is empty"
        )

    def assertFileIdentity(self, actual_path: str, expected_path: str):
        with (
            open(actual_path, "r") as actual,
            open(expected_path, "r") as expected,
        ):
            diff = list(
                difflib.unified_diff(
                    actual.readlines(),
                    expected.readlines(),
                    fromfile=actual_path,
                    tofile=expected_path,
                    n=1,
                )
            )
        self.assertEqual(diff, [], f"Prediction differences found:\n{diff}")

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(
            prefix="yoyodyne_pretrained_test-"
        )
        self.assertNonEmptyFileExists(CONFIG_PATH)

    def tearDown(self):
        self.tempdir.cleanup()

    # Only English is included right now but we leave this a possibility.
    @parameterized.expand(["en"])
    def test_model(self, langcode: str):
        # Fits model.
        train_path = os.path.join(TESTDATA_DIR, f"{langcode}_train.tsv")
        main.python_interface(
            [
                "fit",
                f"--config={CONFIG_PATH}",
                f"--data.train={train_path}",
                # We are trying to overfit on the training data.
                f"--data.val={train_path}",
            ]
        )
        # Confirms a checkpoint was created.
        checkpoint_path = "lightning_logs/version_0/checkpoints/last.ckpt"
        self.assertNonEmptyFileExists(checkpoint_path)
        # Predicts on "expected" data.
        predicted_path = os.path.join(
            self.tempdir.name, f"{langcode}_predicted.txt"
        )
        expected_path = os.path.join(TESTDATA_DIR, f"{langcode}_expected.txt")
        main.python_interface(
            [
                "predict",
                f"--ckpt_path={checkpoint_path}",
                f"--config={CONFIG_PATH}",
                f"--data.predict={train_path}",
                f"--prediction.path={predicted_path}",
            ]
        )
        self.assertNonEmptyFileExists(predicted_path)
        self.assertFileIdentity(predicted_path, expected_path)
        # Evaluates on "expected" data.
        evaluated_path = os.path.join(
            self.tempdir.name,
            f"{langcode}_evaluated.test",
        )
        expected_path = os.path.join(TESTDATA_DIR, f"{langcode}_expected.test")
        with open(evaluated_path, "w") as sink:
            with contextlib.redirect_stdout(sink):
                main.python_interface(
                    [
                        "test",
                        f"--ckpt_path={checkpoint_path}",
                        f"--config={CONFIG_PATH}",
                        f"--data.test={train_path}",
                    ]
                )
        self.assertNonEmptyFileExists(evaluated_path)
        self.assertFileIdentity(evaluated_path, expected_path)
