from fastapi import FastAPI, Request, Form, Body, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
import pickle
from prophet import Prophet
import mysql.connector
from datetime import datetime, timedelta
import json
import math
from fastapi.responses import JSONResponse

import statsmodels.api as sm
# from pandas import datetime
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


app = FastAPI()

# Mysql Connection
cnx = mysql.connector.connect(user='root', password='',
                              host='localhost', database='gempa')

# cnx = mysql.connector.connect("202.73.26.166", "starkapl_sus", "4jrR]-JGYsM!", "starkapl_gempa")
'''model = MyModel()

class Request(BaseModel):
    province_name: str
    date: str'''

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    context = {'request': request}
    # return templates.TemplateResponse("index.html", context)
    return templates.TemplateResponse("dashboard.html", context)
    # return templates.TemplateResponse("coba.html", context)

@app.get("/about", response_class=HTMLResponse)
def about(request: Request):
    context = {'request': request}
    return templates.TemplateResponse("about.html", context)

@app.get("/penanggulangan", response_class=HTMLResponse)
def penanggulangan(request: Request):
    context = {'request': request}
    return templates.TemplateResponse("penanggulangan.html", context)

def get_province_name(province):
    province_names = {
        "ID-MU": "Maluku Utara",
        "ID-PA": "Papua",
        "ID-NT": "Nusa Tenggara Timur",
        "ID-LA": "Lampung",
        "ID-AC": "Aceh",
        "ID-MA": "Maluku",
        "ID-SU": "Sumatra Utara",
        "ID-JT": "Jawa Tengah",
        "ID-GO": "Gorontalo",
        "ID-JB": "Jawa Barat",
        "ID-BE": "Bengkulu",
        "ID-SA": "Sulawesi Utara",
        "ID-SB": "Sumatra Barat",
        "ID-NB": "Nusa Tenggara Barat",
        "ID-RI": "Riau",
        "ID-KS": "Kalimantan Selatan",
        "ID-ST": "Sulawesi Tengah",
        "ID-BT": "Banten",
        "ID-SS": "Sumatra Selatan",
        "ID-JI": "Jawa Timur",
        "ID-SN": "Sulawesi Selatan",
        "ID-BA": "Bali",
        "ID-PB": "Papua Barat",
        "ID-YO": "Daerah Istimewa Yogyakarta",
        "ID-JA": "Jambi",
        "ID-JK": "DKI Jakarta",
        "ID-KI": "Kalimantan Timur",
        "ID-SG": "Sulawesi Tenggara",
        "ID-KU": "Kalimantan Utara",
        "ID-SR": "Sulawesi Barat",
        "ID-KB": "Kalimantan Barat"
    }
    
    return province_names.get(province, "Unknown")

@app.get("/histori/{province}/log_magnitudes", response_class=HTMLResponse)
async def histori(province: str, response: Response, request: Request):
    cur = cnx.cursor()
    cur.execute("SELECT date, latitude, longitude, depth, mag, place FROM histori WHERE province = %s ORDER BY date DESC", (province,))
    results = cur.fetchall()
    cur.close()

    data = [[row[0], row[1], row[2], row[3], row[4], row[5]] for row in results]

    log_magnitudes = [math.log10(row[4]) for row in results]

    response.headers['Content-Type'] = 'text/html'

    province_name = get_province_name(province)
    context = {'response': response, 'request': request, 'data': data, 'province': province, 'log_magnitudes': log_magnitudes, 'province_name': province_name}
    return templates.TemplateResponse("histori.html", context)

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, province: str = Form(...), start_date: str = Form(...), end_date: str = Form(...)):
    # Load earthquake data
    gempa = pd.read_csv("gempa_indonesia.csv")
    gempa.rename(columns={'date': 'time'}, inplace=True)
    gempa['time'] = pd.to_datetime(gempa['time'])
    df = gempa[['time', 'mag', 'depth', 'province']]
    df.columns = ['ds', 'y', 'depth', 'provinces']
    df = pd.get_dummies(df, columns=['provinces'])

    # Train Prophet models for each province
    models = {}
    province_cols = [col for col in df.columns if col.startswith('provinces_')]
    for province_col in province_cols:
        province_name = province_col.replace('provinces_', '')
        province_df = df[df[province_col] == 1]
        if len(province_df) < 2:
            print(f"{province_name} has less than 2 rows, skipping.")
            continue
        m = Prophet()
        m.fit(province_df)
        models[province_name] = m

    # Generate forecast for the specified province and date range
    forecast_data = None
    if province in models:
        model = models[province]
        future = pd.date_range(start=start_date, end=end_date)
        forecast = model.predict(pd.DataFrame({'ds': future}))[['ds', 'yhat']]
        forecast = forecast.round({'yhat': 1})
        forecast_data = forecast.rename(columns={'yhat': 'Forecast'})

    return templates.TemplateResponse("dashboard.html", {"request": request, "forecast_data": forecast_data, "province": province})

@app.get("/coba", response_class=HTMLResponse)
def coba(request: Request):
    try:
        cnx = mysql.connector.connect(user='root', password='', host='localhost', database='gempa')
        cursor = cnx.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        print("MySQL connection is working.")
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
    finally:
        cursor.close()
        cnx.close()

    context = {'request': request}
    return templates.TemplateResponse("coba.html", context)
