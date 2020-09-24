import numpy as np
import json
import ast
from loguru import logger
from rec_engine.lib.rec_engine import RecEngine
from pathlib import Path
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 500)
PARENT_DIR = Path(__file__).parents[0]


def run(infile: str):
    rec = RecEngine()
    row = []
    file_name = PARENT_DIR / 'data' / 'google_local_review' / infile
    with open(file_name, 'r') as f:
        lines = f.readlines()
        i = 1
        for line in lines:
            y = json.loads(json.dumps(ast.literal_eval(line)))
            row.append(y)
            i = i + 1
            if i % 10000 == 0:
                logger.info(f"i={i}")

    res = pd.DataFrame(row)
    print(res[:10])
    return res


if __name__ == '__main__':
    places = run('first.1000.places.clean.json')
    reviews = run('first.1000.reviews.clean.json')
    users = run('first.1000.users.clean.json')
