import numpy as np
import json
import ast
from loguru import logger
from rec_engine.lib.rec_engine import RecEngine
from pathlib import Path
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
PARENT_DIR = Path(__file__).parents[0]


def run(infile, field_name, literal_eval=True):
    rec = RecEngine()
    row = []
    file_name = PARENT_DIR / 'data' / 'google_local_review' / infile
    with open(file_name, 'r') as f:
        lines = f.readlines()
        i = 1
        for line in lines:
            try:
                if literal_eval:
                    y = json.loads(json.dumps(ast.literal_eval(line)))
                else:
                    y = json.loads(line)

                for field in list(y.keys()):
                    if field not in field_name:
                        del y[field]
                    elif isinstance(y[field], list) and isinstance(y[field][0], str):
                        try:
                            # y[field] = ','.join(y[field])
                            y[field] = y[field][len(y[field]) - 1]
                        except:
                            y[field] = y[field][0]
                    elif isinstance(y[field], list) and isinstance(y[field][0], list):
                        y[field] = y[field][0][1]

            except ValueError:
                continue
            row.append(y)
            if i % 10000 == 0:
                logger.info(f"i={i}")
            i = i + 1

    res = pd.DataFrame(row)
    print(res[:10])
    return res

def row_transform():

    pass

if __name__ == '__main__':
    places = run('places.clean.json', ['gPlusPlaceId', 'gps', 'price', 'closed'], True)
    reviews = run('reviews.clean.json', ['gPlusPlaceId', 'reviewTime', 'rating', 'categories', 'gPlusUserId'], True)
    reviews.reviewTime = pd.to_datetime(reviews.reviewTime, format='%b %d, %Y')
    reviews = reviews.sort_values(['reviewTime'], ascending=[True])

    users = run('users.clean.json', ['gPlusUserId', 'jobs', 'currentPlace', 'previousPlace'], False)


    dat = reviews.merge(places, on='gPlusPlaceId').merge(users, on='gPlusUserId')
    dat.to_csv('google_local_review.csv')

