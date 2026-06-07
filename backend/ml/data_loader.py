from pathlib import Path

import pandas as pd

FEATURES = ["AT", "V", "AP", "RH"]
TARGET = "PE"
TRAIN_SHEETS = ["Sheet1", "Sheet2", "Sheet3"]
TEST_SHEETS = ["Sheet4", "Sheet5"]

DATA_PATH = Path(__file__).resolve().parents[2] / "CCPP" / "Folds5x2_pp.xlsx"


def load_fold_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load 3 sheets for training and 2 sheets for testing."""
    train_frames = [pd.read_excel(DATA_PATH, sheet_name=s) for s in TRAIN_SHEETS]
    test_frames = [pd.read_excel(DATA_PATH, sheet_name=s) for s in TEST_SHEETS]
    return pd.concat(train_frames, ignore_index=True), pd.concat(test_frames, ignore_index=True)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES], df[TARGET]
