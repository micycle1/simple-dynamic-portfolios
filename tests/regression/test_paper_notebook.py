"""The paper example notebook still reproduces the paper's headline numbers.

Runs the notebook's own code, in process, through the evaluation table and the
forecast-accuracy computation, on the cached downloads in ``data/`` (or
``$SBG_DATA``). Skipped when those files are absent, so it never downloads.

On the paper's data snapshot the numbers must match ``paper_headline.json``
to solver precision. On any other download (Yahoo restates adjusted closes
after each distribution) they must match to within 0.005, as a smoke check.

After an intentional change to the numbers, regenerate the reference with::

    uv run python tests/regression/test_paper_notebook.py --update
"""

import contextlib
import importlib.util
import io
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "paper_example.ipynb"
EXPECTED = Path(__file__).with_name("paper_headline.json")
DATA_FILES = ["closes", "volumes", "closes_eval", "ffr", "cpi_econ", "macro_raw", "fama_french_raw"]
SNAPSHOT_ATOL = 1e-6
FRESH_ATOL = 5e-3


def data_dir() -> Path:
    """Where the notebook looks for its cached downloads."""
    path = Path(os.environ.get("SBG_DATA", "data"))
    return path if path.is_absolute() else ROOT / path


def fingerprint(data: Path) -> dict[str, str]:
    """Content hashes of the cached downloads (values and index, not file bytes)."""
    return {name: str(pd.util.hash_pandas_object(pd.read_parquet(data / f"{name}.parquet")).sum())
            for name in DATA_FILES}


def run_notebook_through_evaluation() -> dict:
    """Execute the notebook's code cells up to the forecast-accuracy cell; return its namespace."""
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    sources = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    stop = next(i for i, s in enumerate(sources) if "def cosine_similarity" in s)
    assert any("table[columns]" in s for s in sources[:stop]), "evaluation table cell moved"
    namespace: dict = {}
    cwd = os.getcwd()
    try:
        # The cells print and show figures (plotly JSON outside Jupyter); keep that quiet.
        with contextlib.redirect_stdout(io.StringIO()):
            for source in sources[:stop + 1]:
                exec(compile(source, str(NOTEBOOK), "exec"), namespace)  # noqa: S102
    finally:
        os.chdir(cwd)  # the setup cell moves to the repository root
    return namespace


def headline(namespace: dict) -> dict:
    """The evaluation table and the mean forecast/realized cosine similarity."""
    table = namespace["table"][namespace["columns"]]
    return {"metrics": {name: row.to_dict() for name, row in table.iterrows()},
            "mean_cosine": namespace["cosine"].mean().to_dict()}


@unittest.skipUnless(all((data_dir() / f"{n}.parquet").exists() for n in DATA_FILES),
                     "paper data not downloaded (run notebooks/paper_example.ipynb once)")
@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ("plotly", "yfinance", "pyarrow")),
                     "notebook dependencies not installed (uv sync installs the dev group)")
class TestPaperNotebook(unittest.TestCase):
    """The notebook's pipeline on the cached paper data."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.expected = json.loads(EXPECTED.read_text())
        cls.snapshot = fingerprint(data_dir()) == cls.expected["fingerprint"]
        cls.atol = SNAPSHOT_ATOL if cls.snapshot else FRESH_ATOL
        cls.actual = headline(run_notebook_through_evaluation())

    def test_evaluation_table(self) -> None:
        """Return, volatility, Sharpe ratios, drawdown, turnover, and consistency, 2006-2026."""
        expected = pd.DataFrame(self.expected["metrics"]).T
        actual = pd.DataFrame(self.actual["metrics"]).T
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=0, atol=self.atol)

    def test_forecast_accuracy(self) -> None:
        """Mean daily cosine similarity of forecasts and realized 100-day returns (Figure 7)."""
        for name, value in self.expected["mean_cosine"].items():
            self.assertTrue(np.isclose(self.actual["mean_cosine"][name], value, rtol=0, atol=self.atol),
                            (name, self.actual["mean_cosine"][name], value))


def update() -> None:
    """Rewrite the reference from the current data, which becomes the snapshot."""
    reference = {"fingerprint": fingerprint(data_dir()), **headline(run_notebook_through_evaluation())}
    EXPECTED.write_text(json.dumps(reference, indent=2) + "\n")
    print(f"wrote {EXPECTED.relative_to(ROOT)}")


if __name__ == "__main__":
    if "--update" in sys.argv:
        update()
    else:
        unittest.main()
