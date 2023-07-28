import json
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA

def _fit_model(dataframe):
    model = ARIMA(dataframe['count'], 
                    order=(1, 0, 1)
                    ).fit()
    return model


def wallaroo_json(data):
    obj = json.loads(data)
    evaluation_frame = pd.DataFrame.from_dict(obj)

    nforecast = 7
    model = _fit_model(evaluation_frame)

    forecast =  model.forecast(steps=nforecast).round().to_numpy()
    forecast = forecast.astype(int)

    return {"forecast": forecast.tolist()}

# def wallaroo_json(data: pd.DataFrame):

#     # convert from dataframe into single list

#     # evaluation_frame = pd.DataFrame({"count": data.loc[0, 'count']})
#     # print(evaluation_frame)
#     # # obj = json.loads(evaluation_frame)
#     # # evaluation_frame = pd.DataFrame.from_dict(obj)

#     # nforecast = 7
#     # model = _fit_model(evaluation_frame)

#     # forecast =  model.forecast(steps=nforecast).round().to_numpy()
#     # forecast = forecast.astype(int)

#     # return {"forecast": forecast.tolist()}
#     # return [
#     #         { "forecast": forecast.tolist() }
#     # ]
#     return [
#         { "forecast" : 15 }
#     ]
