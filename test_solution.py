
import math
import os
import tempfile
import unittest

import pandas as pd

from main import (
    DataLoadError,
    MappingError,
    Dataset,
    TrainingDataset,
    IdealDataset,
    TestDataset,
    FunctionAnalyser,
)


# Helpers 
def _write_csv(df: pd.DataFrame, directory: str, name: str) -> str:
    path = os.path.join(directory, name)
    df.to_csv(path, index=False)
    return path


def _make_train_df(n: int = 10) -> pd.DataFrame:
    x = [float(i) for i in range(n)]
    return pd.DataFrame(
        {
            "x":  x,
            "y1": [v * 2      for v in x],
            "y2": [v * 3      for v in x],
            "y3": [v ** 2     for v in x],
            "y4": [math.sin(v) for v in x],
        }
    )


def _make_ideal_df(n: int = 10) -> pd.DataFrame:
    x = [float(i) for i in range(n)]
    data = {"x": x}
    # 50 ideal columns: y1 is a perfect match for train y1 (y=2x)
    for k in range(1, 51):
        data[f"y{k}"] = [v * (k % 5 + 1) for v in x]  # varied functions
    # Make y1 a near-perfect match for train y1
    data["y1"] = [v * 2 for v in x]
    return pd.DataFrame(data)


def _make_test_df() -> pd.DataFrame:
    return pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 2.1, 4.05]})


#  Test cases 
class TestDataLoadError(unittest.TestCase):
    """Dataset raises DataLoadError for bad inputs."""

    def test_missing_file_raises(self):
        with self.assertRaises(DataLoadError):
            Dataset("/nonexistent/path/file.csv")

    def test_training_missing_columns_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({"x": [1, 2], "z": [3, 4]})  # wrong columns
            path = _write_csv(df, tmpdir, "bad_train.csv")
            with self.assertRaises(DataLoadError):
                TrainingDataset(path)

    def test_ideal_too_few_columns_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({"x": [1, 2], "y1": [3, 4]})  # only 1 ideal col
            path = _write_csv(df, tmpdir, "bad_ideal.csv")
            with self.assertRaises(DataLoadError):
                IdealDataset(path)

    def test_test_missing_y_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({"x": [1, 2], "z": [3, 4]})
            path = _write_csv(df, tmpdir, "bad_test.csv")
            with self.assertRaises(DataLoadError):
                TestDataset(path)


class TestDatasetLoading(unittest.TestCase):
    """Valid CSVs load without error."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_training_loads(self):
        path = _write_csv(_make_train_df(), self.tmpdir, "train.csv")
        ds = TrainingDataset(path)
        self.assertEqual(list(ds.df.columns), ["x", "y1", "y2", "y3", "y4"])

    def test_ideal_loads(self):
        path = _write_csv(_make_ideal_df(), self.tmpdir, "ideal.csv")
        ds = IdealDataset(path)
        self.assertIn("y50", ds.df.columns)

    def test_test_loads(self):
        path = _write_csv(_make_test_df(), self.tmpdir, "test.csv")
        ds = TestDataset(path)
        self.assertIn("y", ds.df.columns)


class TestFunctionAnalyser(unittest.TestCase):
    """Core analysis logic."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.train_path  = _write_csv(_make_train_df(), self.tmpdir, "train.csv")
        self.ideal_path  = _write_csv(_make_ideal_df(), self.tmpdir, "ideal.csv")
        self.test_path   = _write_csv(_make_test_df(),  self.tmpdir, "test.csv")
        self.db_path     = os.path.join(self.tmpdir, "test.db")

    def _make_analyser(self) -> FunctionAnalyser:
        return FunctionAnalyser(
            self.train_path, self.ideal_path, self.test_path, self.db_path
        )

    def test_load_to_db_creates_tables(self):
        fa = self._make_analyser()
        fa.load_to_db()
        from sqlalchemy import inspect
        insp = inspect(fa.engine)
        tables = insp.get_table_names()
        self.assertIn("training_data", tables)
        self.assertIn("ideal_functions", tables)

    def test_select_returns_four_functions(self):
        fa = self._make_analyser()
        fa.load_to_db()
        chosen = fa.select_ideal_functions()
        self.assertEqual(len(chosen), 4)
        for t_col in ["y1", "y2", "y3", "y4"]:
            self.assertIn(t_col, chosen)
            self.assertTrue(chosen[t_col].startswith("y"))

    def test_train_y1_matches_ideal_y1(self):
        """train y1 = 2x should match ideal y1 = 2x (SSE = 0)."""
        fa = self._make_analyser()
        fa.load_to_db()
        fa.select_ideal_functions()
        self.assertEqual(fa.chosen_ideal_cols["y1"], "y1")

    def test_map_test_data_before_select_raises(self):
        fa = self._make_analyser()
        with self.assertRaises(MappingError):
            fa.map_test_data()

    def test_map_test_data_returns_dataframe(self):
        fa = self._make_analyser()
        fa.load_to_db()
        fa.select_ideal_functions()
        result = fa.map_test_data()
        self.assertIsInstance(result, pd.DataFrame)
        for col in ["x", "y", "delta_y", "ideal_func_no"]:
            self.assertIn(col, result.columns)

    def test_mapped_delta_within_threshold(self):
        """All mapped deltas must be ≤ max_dev * sqrt(2)."""
        fa = self._make_analyser()
        fa.load_to_db()
        fa.select_ideal_functions()
        result = fa.map_test_data()
        for _, row in result.iterrows():
            # Find corresponding max_deviation
            i_col = row["ideal_func_no"]
            t_col = {v: k for k, v in fa.chosen_ideal_cols.items()}[i_col]
            threshold = fa.max_deviations[t_col] * math.sqrt(2)
            self.assertLessEqual(row["delta_y"], threshold + 1e-9)

    def test_sqrt2_threshold_math(self):
        """Pure math: verify sqrt(2) constant."""
        self.assertAlmostEqual(math.sqrt(2), 1.41421356, places=6)


class TestSqrt2Threshold(unittest.TestCase):
    """Isolated threshold logic tests."""

    def test_point_within_threshold_accepted(self):
        max_dev   = 1.0
        threshold = max_dev * math.sqrt(2)
        delta     = 1.4
        self.assertLessEqual(delta, threshold)

    def test_point_outside_threshold_rejected(self):
        max_dev   = 1.0
        threshold = max_dev * math.sqrt(2)
        delta     = 1.5
        self.assertGreater(delta, threshold)


if __name__ == "__main__":
    unittest.main(verbosity=2)