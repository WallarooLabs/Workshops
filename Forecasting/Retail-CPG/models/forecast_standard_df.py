import json
import pandas as pd
import numpy

from statsmodels.tsa.arima.model import ARIMA

def _fit_model(dataframe):
    model = ARIMA(dataframe['count'], 
                    order=(1, 0, 1)
                    ).fit()
    return model


# def wallaroo_json(data):
#     obj = json.loads(data)
#     evaluation_frame = pd.DataFrame.from_dict(obj)

#     nforecast = 7
#     model = _fit_model(evaluation_frame)

#     forecast =  model.forecast(steps=nforecast).round().to_numpy()
#     forecast = forecast.astype(int)

#     return {"forecast": forecast.tolist()}

def wallaroo_json(data: pd.DataFrame):

    evaluation_frame = pd.DataFrame({"count": data.loc[0, 'count']})

    nforecast = 7
    model = _fit_model(evaluation_frame)

    forecast =  model.forecast(steps=nforecast).round().to_numpy()
    forecast = forecast.astype(int)

    # get the average across the week
    weekly_average = forecast.mean()

    return [
        { "forecast" : forecast.tolist(),
          "weekly_average": [weekly_average] 
        }
    ]
