from typing import IO
import json
import requests

import pandas as pd
import numpy as np



def predict_car(data: dict, url: str='http://185.87.51.170:8000') -> float:
    url_car = url + '/predict_item'
    data_json = json.dumps(data)
    result = requests.post(url_car, data=data_json).json()

    return result

def predict_cars_csv(data: IO, url: str='http://185.87.51.170:8000'):
    url_cars = url + '/predict_items'
    data_req = data.to_dict(orient='records')

    data_prepare = json.dumps(data_req)
    response = requests.post(url_cars, data=data_prepare).json()

    return pd.DataFrame(response)