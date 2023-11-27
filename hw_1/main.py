from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import re
import json

import pandas as pd
import numpy as np
import pickle

with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pickle', 'rb') as f:
    scaler = pickle.load(f)

with open('col.pickle', 'rb') as f:
    col = pickle.load(f)    

with open('col_scaler.pickle', 'rb') as f:
    col_scaler = pickle.load(f)    


app = FastAPI()



def prepare_data(data):
    sign_list = [0 for i in range(len(col))] #Cписок - строка в датафрейме
    brand = 'brand_' + data.name.split()[0] #Название бренда
    seats = 'seats_' + str(int(data.seats))
    fuel = 'feul_' + data.fuel
    transmission = 'transmission_' + data.transmission
    owner = 'owner_' + data.owner
    seller_type = 'seller_type_' + data.seller_type

    list_categor = [brand, seats, fuel, transmission, owner, seller_type]

    y_true = data.selling_price


    for i in range(len(col)):

        for cat in list_categor: #Добавляем категории
            if cat == col[i]:
                sign_list[i] = 1
                continue
        

        for fet in ['mileage', 'engine', 'max_power']:
            if fet == col[i]:
                sign_list[i] = float(re.match(r"[0-9.]{1,}", str(data.model_dump()[fet])).group(0)) if re.match(r"[0-9.]{1,}", str(data.model_dump()[fet])) else 0
                continue
           
        for num in ['year', 'km_driven',]: #Добавляем численные признаки
            if num == col[i] and num in data.model_dump():
                sign_list[i] = data.model_dump()[num]
                continue


    #create dataframe
    df = pd.DataFrame(columns=col)
    df.loc[len(df)] = sign_list

    df['year_sq'] = df['year']**2


    #scaler
    df[col_scaler] = scaler.transform(df[col_scaler])

    pred = model.predict(df)

    return np.round(np.expm1(pred), 2), y_true


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    pred, true  = prepare_data(item)
    return pred


@app.post("/predict_items")
def predict_items(items: List[Item]):
    pred = []
    true = []
    df = {i: [] for i in list(items[0].model_dump().keys())}
    col_dict = df.keys()

    for i in items:

        t_pred, t_true = prepare_data(i)
        pred.append(float(t_pred))
        for key in col_dict:
            df[key].append(i.model_dump()[key])

    df['predict'] = pred

    return df