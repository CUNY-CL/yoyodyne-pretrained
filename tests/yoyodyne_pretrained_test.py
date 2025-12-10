"""Full trests of training and prediction.

See testdata/data for code for regenerating data and accuracy/loss statistics.
"""

import contextlib
import difflib
import os
import re
import tempfile
import unittest

from parameterized import parameterized

from yoyodyne_pretrained import cli


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
            difflines = "".join(
                difflib.unified_diff(
                    [self._normalize(line) for line in actual],
                    [self._normalize(line) for line in expected],
                    fromfile=actual_path,
                    tofile=expected_path,
                    n=1,
                )
            )
            if difflines:
                self.fail(f"Prediction differences found:\n{difflines}")

    @staticmethod
    def _normalize(line: str) -> str:
        return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", line)

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(
            prefix="yoyodyne_pretrained_test-"
        )

    def tearDown(self):
        self.tempdir.cleanup()

    DIR = os.path.relpath(os.path.dirname(__file__), os.getcwd())
    CONFIG_DIR = os.path.join(DIR, "testdata/configs")
    TESTDATA_DIR = os.path.join(DIR, "testdata/data")
    DATA_CONFIG_PATH = os.path.join(CONFIG_DIR, "data.yaml")
    CHECKPOINT_CONFIG_PATH = os.path.join(CONFIG_DIR, "checkpoint.yaml")
    TRAINER_CONFIG_PATH = os.path.join(CONFIG_DIR, "trainer.yaml")
    SEED = 49

    @parameterized.expand(["byt5", "mbert_tied"])
    def test_model(self, arch: str):
        train_path = os.path.join(self.TESTDATA_DIR, "train.tsv")
        self.assertNonEmptyFileExists(train_path)
        dev_path = os.path.join(self.TESTDATA_DIR, "dev.tsv")
        self.assertNonEmptyFileExists(dev_path)
        test_path = os.path.join(self.TESTDATA_DIR, "test.tsv")
        self.assertNonEmptyFileExists(test_path)
        model_dir = os.path.join(self.tempdir.name, "models")
        # Gets config paths.
        model_config_path = os.path.join(self.CONFIG_DIR, f"{arch}.yaml")
        self.assertNonEmptyFileExists(model_config_path)
        # Fits and confirms creation of the checkpoint.
        cli.python_interface(
            [
                "fit",
                f"--checkpoint={self.CHECKPOINT_CONFIG_PATH}",
                f"--data={self.DATA_CONFIG_PATH}",
                f"--data.train={train_path}",
                f"--data.val={dev_path}",
                f"--data.model_dir={model_dir}",
                f"--model={model_config_path}",
                f"--seed_everything={self.SEED}",
                f"--trainer={self.TRAINER_CONFIG_PATH}",
            ]
        )
        checkpoint_path = (
            f"{model_dir}/lightning_logs/version_0/checkpoints/last.ckpt"
        )
        self.assertNonEmptyFileExists(checkpoint_path)
        # Predicts on test data.
        predicted_path = os.path.join(
            self.tempdir.name, f"_{arch}_predicted.txt"
        )
        cli.python_interface(
            [
                "predict",
                f"--ckpt_path={checkpoint_path}",
                f"--data={self.DATA_CONFIG_PATH}",
                f"--data.model_dir={model_dir}",
                f"--data.predict={test_path}",
                f"--model={model_config_path}",
                f"--prediction.path={predicted_path}",
            ]
        )
        self.assertNonEmptyFileExists(predicted_path)
        evaluation_path = os.path.join(
            self.tempdir.name, f"{arch}_evaluated.test"
        )
        # Evaluates on test data and compares with result.
        with open(evaluation_path, "w") as sink:
            with contextlib.redirect_stdout(sink):
                cli.python_interface(
                    [
                        "test",
                        f"--ckpt_path={checkpoint_path}",
                        f"--data={self.DATA_CONFIG_PATH}",
                        f"--data.test={test_path}",
                        f"--data.model_dir={model_dir}",
                        f"--model={model_config_path}",
                        "--trainer.enable_progress_bar=false",
                    ]
                )
        self.assertNonEmptyFileExists(evaluation_path)
        expected_path = os.path.join(
            self.TESTDATA_DIR, f"{arch}_expected.test"
        )
        self.assertFileIdentity(evaluation_path, expected_path)


if __name__ == "__main__":
    unittest.main()
