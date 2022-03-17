from tkinter import E
from unittest import result
from fastapi import FastAPI
from starlette.responses import JSONResponse
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi import Request
from joblib import load
import pandas as pd
import torch
import torch.nn as nn
from importlib.machinery import SourceFileLoader
create_dataset = SourceFileLoader('create_dataset', '../src/data/create_dataset.py').load_module()
py_torch = SourceFileLoader('pytorch', '../src/models/pytorch.py').load_module()

app = FastAPI()
model = torch.load('../models/pytorch_classification_v2.pt')

@app.get('/health', status_code=200)
def healthcheck():
    return 'NN classification is all ready to go!'

# Change to POST later
@app.post("/beer/type")
async def predict_beer(info : Request):
    features = await info.json()
    print("Received features:")
    print(features)
    features = convert_single_response_arrays(features)
    print("Converted features:")
    print(features)
    features_df = pd.DataFrame(features)
    X = create_dataset.normalize_features(features_df)
    print("Normalized features:")
    print(X)
    predictions = py_torch.predict(X, model)
    print("Predictions:")
    print(predictions)
    return JSONResponse({"beer_style": predictions})

# Change to POST later
@app.post("/beers/type")
async def predict_beers(info : Request):
    features = await info.json()    
    print("Received features:")
    print(features)
    features_df = pd.DataFrame(features)
    X = create_dataset.normalize_features(features_df)
    print("Normalized features:")
    print(X)
    predictions = py_torch.predict(X, model)
    print("Predictions:")
    print(predictions)
    return JSONResponse({"beer_style": predictions})

@app.get("/model/architecture")
def info():
    architecture = {
        "layer_1_type": "Linear",
        "layer_1_neuron_counts": "6, 15",
        "layer_2_type": "Linear",
        "layer_2_neuron_counts": "15, 104"
    }
    return JSONResponse(architecture)

@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <html>
        <head>
            <title>Beer type predictor</title>
        </head>
        <body>
            <h1>Project Description</h1>
            <p>The project objective is to train a custom neural networks model that will accurately predict a type of beer based on some rating criteria such as appearance, aroma, palate or taste. Additionally, develop a web app and deploy it online to serve the trained model for real-time predictions.<br /><br />Github model training and testing: <a href="https://github.com/nuwanprabhath/beer-type-predictor">https://github.com/nuwanprabhath/beer-type-predictor</a>&nbsp;<br />Github API: <a href="https://github.com/nuwanprabhath/nn_api">https://github.com/nuwanprabhath/nn_api</a><br />Heroku host: <a href="https://intense-atoll-50811.herokuapp.com">https://intense-atoll-50811.herokuapp.com</a></p>
            <p>&nbsp;</p>
            <h1>List of endpoints</h1>
            <p>Visit /docs for a detailed explanation of endpoints. A brief explanation of endpoints can be found below. Deployed host of Heroku is: <a href="https://intense-atoll-50811.herokuapp.com">https://intense-atoll-50811.herokuapp.com</a></p>
            <p>1. Home<br /><strong>Path:</strong> /<br /><strong>Type:</strong> GET<br /><strong>Description:</strong> Displaying a brief description of the project objectives, list of endpoints, expected input parameters and output format of the model, link to the Github repo related to this project.<br /><br />2. Health<br /><strong>Path:</strong> /health<br /><strong>Type:</strong> GET<br /><strong>Description:</strong> Returning status code 200 with a string with a welcome message</p>
            <p>3. Single beer type prediction<br /><strong>Path:</strong> /beer/type<br /><strong>Type:</strong> POST<br /><strong>Description:</strong> Returning prediction for a single input only<br /><strong>Body parameters: </strong>brewery_name, review_aroma, review_appearance, review_palate, review_taste, beer_abv<br /><strong>Body format:</strong> application/json<br /><strong>Output:</strong> {beer_style: ["style"]}<br /><strong>Sample body:&nbsp; </strong>{<br />"brewery_name": "Vecchio Birraio",<br />"review_aroma": 2.0,<br />"review_appearance": 2.5,<br />"review_palate": 1.5,<br />"review_taste": 1.5,<br />"beer_abv": 5<br />}<br /><strong>Sample output:&nbsp;</strong>{"beer_style": ["American IPA"]}<br /><strong>Sample CURL command:</strong><br />curl -X POST \<br />https://intense-atoll-50811.herokuapp.com/beer/type \<br />-H 'cache-control: no-cache' \<br />-H 'content-type: application/json' \<br />-d '{<br />"brewery_name": "Vecchio Birraio",<br />"review_aroma": 2.0,<br />"review_appearance": 2.5,<br />"review_palate": 1.5,<br />"review_taste": 1.5,<br />"beer_abv": 5<br />}'</p>
            <p>4. Multiple beer type prediction<br /><strong>Path:</strong> /beers/type<br /><strong>Type:</strong> POST<br /><strong>Description:</strong> Returning predictions for multiple inputs<br /><strong>Body parameters: </strong>brewery_name, review_aroma, review_appearance, review_palate, review_taste, beer_abv<br /><strong>Body format:</strong> application/json<br /><strong>Output:</strong> {beer_style: ["style 1", "style 2", ...]}<br /><strong>Sample body:&nbsp;</strong>{<br />"brewery_name": ["Vecchio Birraio", "Vecchio Birraio"],<br />"review_aroma": [2.0, 2.5],<br />"review_appearance": [2.5, 3.0],<br />"review_palate": [1.5, 3.0],<br />"review_taste": [1.5, 3.0],<br />"beer_abv": [5, 6.2]<br />}<br /><strong>Sample output:</strong>{"beer_style":["Cream Ale","Cream Ale"]}<br /><strong>Sample CURL command:&nbsp;<br /></strong>curl -X POST \<br />https://intense-atoll-50811.herokuapp.com/beers/type \<br />-H 'cache-control: no-cache' \<br />-H 'content-type: application/json' \<br />-d '{<br />"brewery_name": ["Vecchio Birraio", "Vecchio Birraio"],<br />"review_aroma": [2.0, 2.5],<br />"review_appearance": [2.5, 3.0],<br />"review_palate": [1.5, 3.0],<br />"review_taste": [1.5, 3.0],<br />"beer_abv": [5, 6.2]<br />}'<strong><br /><br /></strong></p>
            <p>5. Display architecture of the neural network<br /><strong>Path:</strong> /beers/type<br /><strong>Type:</strong> GET<br /><strong>Description:</strong> Displaying the architecture of your Neural Networks (listing of all layers with their types)</p>
            <p>&nbsp;</p>
            <p>&nbsp;</p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

def convert_single_response_arrays(feature):
    """
    Convert single JSON request to array so it can be converted to pandas dataframe

    Parameters
    ----------
    feature: JSON object with following keys

    Returns
    -------
    results: JSON object with values in arrays
    """

    return {
    "brewery_name": [feature['brewery_name']],
    "review_aroma": [feature['review_aroma']],
    "review_appearance": [feature['review_appearance']],
    "review_palate": [feature['review_palate']],
    "review_taste": [feature['review_taste']],
    "beer_abv": [feature['beer_abv']]
    }