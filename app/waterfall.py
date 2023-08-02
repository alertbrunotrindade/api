import numpy as np
import pandas as pd
import xgboost
import shap
import warnings
from warnings import filterwarnings
import io
import random
import base64
from io import BytesIO
from matplotlib.figure import Figure
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG
filterwarnings(action='ignore', category=DeprecationWarning)
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')
from warnings import filterwarnings
import matplotlib.pyplot as plt
import base64
from io import BytesIO
warnings.filterwarnings(action="ignore")

def reader_dataframe(path_arquivos):
    base_construcao_waterfall = pd.read_csv(rf'{path_arquivos}\base_construcao_usar.csv', delimiter=';')
    data = pd.read_csv(base_construcao_waterfall['base_construcao_usar'][0], delimiter=';', encoding='utf-8')
    new_data = data.replace(np.nan, None)
    new_data = new_data.select_dtypes(exclude=['object'])
    return new_data


def select_columns(path_arquivos):
    data = reader_dataframe(path_arquivos)
    X = data.drop("VENDAS", axis=1)
    y = data['VENDAS']
    model = xgboost.XGBClassifier().fit(X, y)
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    item = shap.plots.waterfall(shap_values[0], max_display=20)

    # Save the figure in the static directory


    return item


def gen_html(path_arquivos):
    #fig = Figure()
    fig = select_columns(path_arquivos)
    
    ax = fig.subplots()

    buf = BytesIO()
    fig.savefig(buf, format="png")

    data = base64.b64encode(buf.getbuffer()).decode("ascii")

    graficos  = f"<img src='data:image/png;base64,{data}'/>"

    return graficos


