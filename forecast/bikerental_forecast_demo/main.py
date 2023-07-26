import json
import os
import datetime

import wallaroo
from wallaroo.object import EntityNotFoundError

import pandas as pd
import numpy as np

# for Big Query connections
from google.cloud import bigquery
from google.oauth2 import service_account
import db_dtypes


# utility functions for creating demo queries
from resources import util

#
# usual convenience functions. 
# I really should have these in util, 
# but this makes cut-n-paste from notebook safer
#

def get_workspace(name):
    workspace = None
    for ws in wl.list_workspaces():
        if ws.name() == name:
            workspace= ws
    if(workspace == None):
        workspace = wl.create_workspace(name)
    return workspace


# get a pipeline by name in the workspace
def get_pipeline(pname, create_if_absent=False):
    plist = wl.get_current_workspace().pipelines()
    pipeline = [p for p in plist if p.name() == pname]
    if len(pipeline) <= 0:
        if create_if_absent:
            pipeline = wl.build_pipeline(pname)
        else:
            raise KeyError(f"pipeline {pname} not found in this workspace")
    else:
        pipeline = pipeline[0]
    return pipeline


# the forecast function I created in the process development notebook
def do_forecast(bqclient, pipeline, in_table, out_table, forecast_day, site):
    # get input data
    query = util.mk_dt_range_query(tablename=in_table, day_of_forecast=forecast_day, site_id=site)
    xquery = util.mk_exog_query(tablename=in_table, day_of_forecast=forecast_day, site_id=site, nforecast=7)
    
    historical_data = bqclient.query(query).to_dataframe()
    exog = bqclient.query(xquery).to_dataframe()
    exog['cnt'] = -1

    input_frame = pd.concat([historical_data, exog]).reset_index(drop=True)
    input_frame['dteday'] = input_frame['dteday'].astype(str)
    
    # infer
    results = pipeline.infer(input_frame.to_dict())[0]
    resultframe = pd.DataFrame(results)

    # write to staging table
    output_table = bqclient.get_table(out_table)
    bqclient.insert_rows_from_dataframe(output_table, dataframe=resultframe)
    
    
#
# function to do the task. Assumes connection to wallaroo is "wl"
#

def do_task(arguments):    
    
    # assign the arguments
    # you could also hard-code some of these into the script,
    # if you are sure they'll never change
    
    if 'workspace_name' in arguments:
        workspace_name = arguments['workspace_name']
    else:
        workspace_name = default_args['workspace_name']
        
    if 'pipeline_name' in arguments:
        pipeline_name = arguments['pipeline_name']
    else:
        pipeline_name = default_args['pipeline_name']
        
    if 'conn_name' in arguments:
        conn_name = arguments['conn_name']
    else:
        conn_name = default_args['conn_name']    
        
    if 'dataset' in arguments:
        dataset = arguments['dataset']
    else:
        dataset = default_args['dataset']
        
    if 'input_table' in arguments:
        input_table = arguments['input_table']
    else:
        input_table = default_args['input_table']
        
    if 'output_table' in arguments:
        output_table = arguments['output_table']
    else:
        output_table = default_args['output_table']
        
    
    # go to workspace
    workspace = get_workspace(workspace_name)
    _ = wl.set_current_workspace(workspace)
    
    
    # connect to BQ
    connection = wl.get_connection(conn_name)

    # set the credentials
    bigquery_credentials = service_account.Credentials.from_service_account_info(connection.details())

    # start the client
    bigqueryclient = bigquery.Client(
    credentials=bigquery_credentials, 
    project=connection.details()['project_id']
    )
    
    # table information

    dataset = 'bikerental_forecast_demo'
    input_table = 'bikerentals'
    in_tablename = f'{dataset}.{input_table}'
    output_table = 'bikeforecasts'
    out_tablename = f'{dataset}.{output_table}'

    print(f'input data source: {in_tablename}; output staging table: {out_tablename}')
    
    # deploy the pipeline
    pipeline = get_pipeline(pipeline_name)
    pipeline.deploy()
    
    # set the day we are doing the forecasts
    # in a real life situation this would be fetched via datetime.datetime.now() or something
    today = '2011-03-01'

    print(f'Running analysis on {today}')
    
    # get all the site names
    sites = bigqueryclient.query(f"select distinct site_id from {in_tablename}").to_dataframe()
    sites = sites['site_id'].to_numpy()
    print(f'{len(sites)} rental sites')
    
    # loop over all the sites
    for site in sites:
        print(f'forecasting {site}')
        do_forecast(bigqueryclient, pipeline, in_tablename, out_tablename, today, site)

    
    print('forecast complete, results written to bikeforecast table')
    print('undeploying pipeline...')
    pipeline.undeploy()
    bigqueryclient.close()
    
    
    
#
# set default arguments
# these are hardcoded for testing
# I will also use them as fallbacks if not all args are supplied to task
#

default_args = {
    'conn_name': 'bq-wl-dev',
    'workspace_name' : 'bikerental-nbz',
    'pipeline_name' : 'bikeforecast-pipe',
    'dataset': 'bikerental_forecast_demo',
    'input_table' : 'bikerentals',
    'output_table': 'bikeforecast'
}

#
# now actually do something
#

wl = wallaroo.Client()

if wl.in_task():              # if running the orchestrated task in wallaroo
    do_task(wl.task_args())
else:                         # if testing main.py from the command line or console
    do_task(default_args)


