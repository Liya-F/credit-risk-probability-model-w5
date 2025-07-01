import pytest
import pandas as pd
import sys
import os

# Ensure src is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from train import ModelTrainer

SAMPLE_CSV = """CustomerId,Feature1,Feature2,AnyFraud
1,10,100,0
2,20,200,1
3,30,300,0
4,40,400,1
"""

@pytest.fixture
def trainer(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text(SAMPLE_CSV)
    return ModelTrainer(str(file))

def test_load_data(trainer):
    trainer.load_data()
    assert trainer.df is not None
    assert trainer.df.shape[0] == 4
    assert "Feature1" in trainer.df.columns

def test_split_data(trainer):
    trainer.load_data()
    trainer.split_data(target_column="AnyFraud")
    assert trainer.X_train.shape[0] > 0
    assert "AnyFraud" not in trainer.X_train.columns
