from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import torch
import pandas as pd

app = FastAPI()
# gmm_pipe = torch.load('../models/pytorch_classification_v1.pt')


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/health', status_code=200)
def healthcheck():
    return 'NN classification is all ready to go!'

@app.get("/mall/customers/segmentation")
def predict(genre: str,	age: int, income: int, spending: int):
    features = format_features(genre,	age, income, spending)
    # obs = pd.DataFrame(features)
    # pred = gmm_pipe.predict(obs)
    # return JSONResponse(pred.tolist())
    return JSONResponse(features)

def format_features(genre: str,	age: int, income: int, spending: int):
    return {
        'Gender': [genre],
        'Age': [age],
        'Annual Income (k$)': [income],
        'Spending Score (1-100)': [spending]
    }