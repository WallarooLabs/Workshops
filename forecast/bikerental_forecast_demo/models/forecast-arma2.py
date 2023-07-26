import json
import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf, adfuller

_exogvar = ['holiday', 'weekday', 'workingday']

# I should really split on date, but I know this will work
def _split_frame(dataframe):
    historical = dataframe.loc[(dataframe['cnt'] != -1), :].reset_index(drop=True, inplace=False)
    exog = dataframe.loc[(dataframe['cnt']==-1), :].reset_index(drop=True, inplace=False)
    return historical, exog



# maybe find a better order, too
# I think they might use the one that automatically sets it?
# nope. pdarima is not part of statsmodels, so I'd have to containerize if I did that.
# (0, 1, 1) gives me a convergence error. this was the next best, and it doesn't give me a convergence error
def _fit_model(dataframe):
    model = ARIMA(dataframe['cnt'], 
                    exog = dataframe.loc[:, _exogvar],
                    order=(2, 0, 2)   
                    ).fit()
    return model


def _forecast(dataframe):
    hist, exog = _split_frame(dataframe)
    model = _fit_model(hist) # contains historical exog
    nforecast = exog.shape[0]
    cnt_forecast = model.forecast(steps=nforecast, 
                                  exog=exog.loc[:, _exogvar])
    cnt_forecast = cnt_forecast.round().to_numpy().astype(int)
    return pd.DataFrame({
        'dteday': exog['dteday'],
        'site_id': exog['site_id'],
        'forecast': cnt_forecast

    })


def wallaroo_json(data):
    obj = json.loads(data)
    input_frame = pd.DataFrame.from_dict(obj)

    result = _forecast(input_frame)

    return result.to_dict()
