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
    # numpy/pandas brittleness was throwing if int64 made it to ARIMA
    # so defend against this with explicit data cast
    model = ARIMA(np.array(dataframe['cnt'], dtype=float), 
                    exog = np.array(dataframe.loc[:, _exogvar], dtype=float),
                    order=(1, 0, 1)   
                    ).fit()
    return model


                                  
def _forecast(dataframe):
    hist, exog = _split_frame(dataframe)
 
    model = _fit_model(hist) # contains historical exog
    nforecast = exog.shape[0]

    # convert exog to np.array
    exoga = np.array(exog.loc[:, _exogvar], dtype=float)

    cnt_forecast = model.forecast(steps=nforecast, 
                                  exog=exoga)
    
    cnt_forecast = cnt_forecast.round().astype(int)

    forecast_average = cnt_forecast.mean()

    return pd.DataFrame({
        'dteday': exog['dteday'],
        'site_id': exog['site_id'],
        'forecast': cnt_forecast,
        'forecast_average' : forecast_average
    })


#
# per Jonathan
# this allows multiple sites to be sent in at once
# if each site's data frame is a single row
# of the form "input_frame.to_dict(orient="list")"
#
def wallaroo_json(data: pd.DataFrame):
    cols = data.columns.to_list()
    results = []
    for _, row in data.iterrows():
        input_frame = pd.DataFrame([row]).explode(cols)
        result = _forecast(input_frame)
        results.append(result.to_dict(orient='list'))
        
    
    return results
