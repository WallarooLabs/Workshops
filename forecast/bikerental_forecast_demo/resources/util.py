import pandas as pd
import numpy as np
import datetime

from warnings import filterwarnings
filterwarnings('ignore')

def get_forecast_days() :
    firstdate = '2011-03-01'
    days = [i*7 for i in [-1,0,1,2,3,4]]
    deltadays = pd.to_timedelta(pd.Series(days), unit='D') 

    analysis_days = (pd.to_datetime(firstdate) + deltadays).dt.date
    analysis_days = [str(day) for day in analysis_days]
    analysis_days
    seed_day = analysis_days.pop(0)

    return seed_day, analysis_days



# function to create the query to pull the historical data - counts plus exogenous variables
def mk_dt_range_query(*, tablename: str, day_of_forecast: str, site_id: str) -> str:
    assert isinstance(tablename, str)
    assert isinstance(day_of_forecast, str)
    assert isinstance(site_id, str)
    query = f'''
    select dteday, site_id, cnt, season, holiday, weekday, workingday
    from {tablename} where dteday > DATE_SUB(DATE '{day_of_forecast}', INTERVAL 1 MONTH) AND dteday <= PARSE_DATE('%F', '{day_of_forecast}')
    and site_id = '{site_id}'
    order by dteday'''
    return query



# create the query to pull the exogenous variables for the forecast days
# (table name is the same, because this is synthetic data, which is clunky, but it works....)
def mk_exog_query(*, tablename:str, day_of_forecast: str, site_id:str, nforecast=7) -> str:
    assert isinstance(tablename, str)
    assert isinstance(day_of_forecast, str)
    assert isinstance(site_id, str)
    query = f'''
    select dteday, site_id, season, holiday, weekday, workingday
    from {tablename} where dteday > PARSE_DATE('%F', '{day_of_forecast}') AND dteday <= DATE_ADD(DATE '{day_of_forecast}', INTERVAL {nforecast} day)
    and site_id = "{site_id}"
    order by dteday'''
    return query
