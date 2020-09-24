from pathlib import Path
import pandas as pd
import numpy as np
import pymc3 as pm
import arviz as az
import json
import ast
from loguru import logger
import dask.dataframe as dd

PARENT_DIR = Path(__file__).parents[1]


class RecEngine:
    def __init__(self):
        pass

    def read_data(self, file_name):
        return pd.read_csv(file_name)

    def pm_model(self):
        with pm.Model() as model:
            pass
