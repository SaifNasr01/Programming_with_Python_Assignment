
import math
import os
import pandas as pd
from sqlalchemy import create_engine, Column, Float, String, Integer
from sqlalchemy.orm import declarative_base, Session
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Legend

#  SQLAlchemy ORM base 
Base = declarative_base()


#  Custom Exceptions 
class DataLoadError(Exception):
    """Raised when a CSV file cannot be loaded or has unexpected structure."""
    pass


class MappingError(Exception):
    """Raised when test-data mapping encounters an unrecoverable problem."""
    pass


#  ORM Table Models 
class TrainingData(Base):
    """ORM model for the training-data table (x, y1-y4)."""
    __tablename__ = "training_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    x  = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    y3 = Column(Float, nullable=False)
    y4 = Column(Float, nullable=False)


class IdealFunctions(Base):
    """ORM model for the ideal-functions table (x, y1 … y50)."""
    __tablename__ = "ideal_functions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    x  = Column(Float, nullable=False)
    # y1 … y50 added dynamically below
    

# Add y1–y50 columns to IdealFunctions dynamically
for _i in range(1, 51):
    setattr(IdealFunctions, f"y{_i}", Column(Float))


class TestResults(Base):
    """ORM model for the test-results table (x, y, delta_y, ideal_func_no)."""
    __tablename__ = "test_results"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    x             = Column(Float,  nullable=False)
    y             = Column(Float,  nullable=False)
    delta_y       = Column(Float,  nullable=False)
    ideal_func_no = Column(String, nullable=False)


#  Base Dataset class 
class Dataset:
    """
    Base class for all datasets.

    Provides common CSV-loading logic and basic validation.
    """

    def __init__(self, filepath: str):
        """
        Initialise and load a CSV file.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.

        Raises
        ------
        DataLoadError
            If the file is missing or cannot be parsed.
        """
        if not os.path.exists(filepath):
            raise DataLoadError(f"File not found: {filepath}")
        try:
            self.df: pd.DataFrame = pd.read_csv(filepath)
        except Exception as exc:
            raise DataLoadError(f"Failed to parse {filepath}: {exc}") from exc
        self.filepath = filepath

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filepath={self.filepath!r}, shape={self.df.shape})"


#  Specialised Dataset subclasses 
class TrainingDataset(Dataset):
    """
    Dataset subclass for the four training functions.

    Inherits CSV loading from :class:`Dataset`.
    """

    def __init__(self, filepath: str):
        super().__init__(filepath)
        expected = {"x", "y1", "y2", "y3", "y4"}
        if not expected.issubset(set(self.df.columns)):
            raise DataLoadError(
                f"Training CSV must contain columns {expected}; "
                f"got {list(self.df.columns)}"
            )


class IdealDataset(Dataset):
    """
    Dataset subclass for the 50 ideal functions.

    Inherits CSV loading from :class:`Dataset`.
    """

    def __init__(self, filepath: str):
        super().__init__(filepath)
        if "x" not in self.df.columns or len(self.df.columns) < 51:
            raise DataLoadError(
                "Ideal CSV must contain 'x' plus 50 y-columns."
            )


class TestDataset(Dataset):
    

    def __init__(self, filepath: str):
        super().__init__(filepath)
        expected = {"x", "y"}
        if not expected.issubset(set(self.df.columns)):
            raise DataLoadError(
                f"Test CSV must contain columns {expected}; "
                f"got {list(self.df.columns)}"
            )


#  Core Analyser 
class FunctionAnalyser:
    

    def __init__(
        self,
        train_path: str,
        ideal_path: str,
        test_path: str,
        db_path: str = "results.db",
    ):
        self.train_ds = TrainingDataset(train_path)
        self.ideal_ds = IdealDataset(ideal_path)
        self.test_ds  = TestDataset(test_path)

        engine_url = f"sqlite:///{db_path}"
        self.engine = create_engine(engine_url, echo=False)
        Base.metadata.create_all(self.engine)

        # Populated during analysis
        self.chosen_ideal_cols: dict[str, str] = {}   # train col → ideal col
        self.max_deviations:    dict[str, float] = {}  # train col → max |Δy|
        self.test_results_df:   pd.DataFrame | None = None

    #  Step 1: persist raw data
    def load_to_db(self) -> None:
        """Load training data and ideal functions into the SQLite database."""
        # Training
        train_df = self.train_ds.df.copy()
        train_df.rename(columns={"y1": "y1", "y2": "y2", "y3": "y3", "y4": "y4"}, inplace=True)
        train_df.to_sql("training_data", self.engine, if_exists="replace", index=False)
        print(f"[DB] Training data loaded: {len(train_df)} rows.")

        # Ideal functions
        self.ideal_ds.df.to_sql("ideal_functions", self.engine, if_exists="replace", index=False)
        print(f"[DB] Ideal functions loaded: {len(self.ideal_ds.df)} rows × {len(self.ideal_ds.df.columns)-1} functions.")

    # ── Step 2: least-squares selection ─
    def select_ideal_functions(self) -> dict[str, str]:
        
        train_df = self.train_ds.df.set_index("x")
        ideal_df = self.ideal_ds.df.set_index("x")

        ideal_y_cols = [c for c in ideal_df.columns]  # y1 … y50

        for t_col in ["y1", "y2", "y3", "y4"]:
            best_col   = None
            best_sse   = math.inf
            best_max_dev = 0.0

            t_vals = train_df[t_col]

            for i_col in ideal_y_cols:
                # Align on common x-values without column-name collision
                i_vals = ideal_df[i_col]
                common_x = t_vals.index.intersection(i_vals.index)
                residuals = t_vals[common_x] - i_vals[common_x]
                sse = (residuals ** 2).sum()
                if sse < best_sse:
                    best_sse     = sse
                    best_col     = i_col
                    best_max_dev = residuals.abs().max()

            self.chosen_ideal_cols[t_col] = best_col
            self.max_deviations[t_col]    = best_max_dev
            print(
                f"  Training {t_col} → Ideal {best_col}  "
                f"(SSE={best_sse:.4f}, max|Δy|={best_max_dev:.4f})"
            )

        return self.chosen_ideal_cols

    #  Step 3: map test data ─
    def map_test_data(self) -> pd.DataFrame:
        """
        For each test point decide whether it can be assigned to one of the
        four chosen ideal functions using the sqrt(2) threshold rule.

        Returns
        -------
        pd.DataFrame
            Rows that were successfully mapped, with columns
            [x, y, delta_y, ideal_func_no].

        Raises
        ------
        MappingError
            If ``select_ideal_functions`` has not been called first.
        """
        if not self.chosen_ideal_cols:
            raise MappingError("Call select_ideal_functions() before map_test_data().")

        ideal_df = self.ideal_ds.df.set_index("x")
        records  = []

        for _, row in self.test_ds.df.iterrows():
            x_val = row["x"]
            y_val = row["y"]

            best_match    = None
            best_delta    = math.inf
            best_ideal_no = None

            for t_col, i_col in self.chosen_ideal_cols.items():
                threshold = self.max_deviations[t_col] * math.sqrt(2)

                if x_val not in ideal_df.index:
                    continue  # x not in ideal table → skip

                ideal_y = ideal_df.loc[x_val, i_col]
                delta   = abs(y_val - ideal_y)

                if delta <= threshold and delta < best_delta:
                    best_delta    = delta
                    best_match    = i_col
                    best_ideal_no = i_col  # e.g. "y42"

            if best_match is not None:
                records.append(
                    {
                        "x":             x_val,
                        "y":             y_val,
                        "delta_y":       best_delta,
                        "ideal_func_no": best_ideal_no,
                    }
                )

        self.test_results_df = pd.DataFrame(records)

        # Persist to DB
        self.test_results_df.to_sql("test_results", self.engine, if_exists="replace", index=False)
        print(f"[DB] Test results saved: {len(self.test_results_df)} mapped points.")
        return self.test_results_df

    #  Step 4: visualisation ─
    def visualise(self, output_html: str = "visualisation.html") -> None:
        """
        Generate an interactive Bokeh HTML visualisation containing:
          - Training data overlaid with chosen ideal functions
          - Test points colour-coded by assigned ideal function

        Parameters
        ----------
        output_html : str
            Path of the output HTML file.
        """
        output_file(output_html)

        train_df = self.train_ds.df.set_index("x").sort_index().reset_index()
        ideal_df = self.ideal_ds.df.set_index("x").sort_index().reset_index()

        PALETTE = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
        plots    = []

        for idx, (t_col, i_col) in enumerate(self.chosen_ideal_cols.items()):
            color = PALETTE[idx]
            p = figure(
                title=f"Training {t_col}  →  Ideal {i_col}",
                width=500,
                height=350,
                x_axis_label="x",
                y_axis_label="y",
            )
            # Training scatter
            p.circle(
                train_df["x"], train_df[t_col],
                color=color, alpha=0.5, size=5,
                legend_label=f"Train {t_col}",
            )
            # Ideal line
            p.line(
                ideal_df["x"], ideal_df[i_col],
                color=color, line_width=2, line_dash="dashed",
                legend_label=f"Ideal {i_col}",
            )

            # Overlay matched test points for this ideal function
            if self.test_results_df is not None and not self.test_results_df.empty:
                sub = self.test_results_df[
                    self.test_results_df["ideal_func_no"] == i_col
                ]
                if not sub.empty:
                    p.triangle(
                        sub["x"], sub["y"],
                        color="black", size=8,
                        legend_label="Test (matched)",
                    )

            p.legend.location = "top_left"
            p.legend.click_policy = "hide"
            plots.append(p)

        grid = gridplot([plots[:2], plots[2:]])
        save(grid)
        print(f"[VIS] Visualisation saved to {output_html}")

    #  Full pipeline 
    def run(self) -> None:
        """Execute the complete pipeline end-to-end."""
        print("\n=== Step 1: Loading data into SQLite ===")
        self.load_to_db()

        print("\n=== Step 2: Selecting ideal functions (Least-Squares) ===")
        self.select_ideal_functions()

        print("\n=== Step 3: Mapping test data ===")
        results = self.map_test_data()
        print(results.to_string(index=False))

        print("\n=== Step 4: Generating visualisation ===")
        self.visualise()

        print("\n=== Done! ===")


#  Entry point 
if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    analyser = FunctionAnalyser(
        train_path=os.path.join(_here, "train.csv"),
        ideal_path=os.path.join(_here, "ideal.csv"),
        test_path=os.path.join(_here, "test.csv"),
        db_path=os.path.join(_here, "results.db"),
    )
    analyser.run()