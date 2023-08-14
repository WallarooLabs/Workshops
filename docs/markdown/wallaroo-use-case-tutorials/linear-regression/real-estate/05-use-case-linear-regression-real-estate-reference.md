# Tutorial Notebook 4: Automation with Wallaroo Connections

Wallaroo Connections are definitions set by MLOps engineers that are used by other Wallaroo users for connection information to a data source.

This provides MLOps engineers a method of creating and updating connection information for data stores:  databases, Kafka topics, etc.  Wallaroo Connections are composed of three main parts:

* Name:  The unique name of the connection.
* Type:  A user defined string that designates the type of connection.  This is used to organize connections.
* Details:  Details are a JSON object containing the information needed to make the connection.  This can include data sources, authentication tokens, etc.

Wallaroo Connections are only used to store the connection information used by other processes to create and use external connections.  The user still has to provide the libraries and other elements to actually make and use the conneciton.

The primary advantage is Wallaroo connections allow scripts and other code to retrieve the connection details directly from their Wallaroo instance, then refer to those connection details.  They don't need to know what those details actually - they can refer to them in their code to make their code more flexible.

For this step, we will use a Google BigQuery dataset to retrieve the inference information, predict the next month of sales, then store those predictions into another table.  This will use the Wallaroo Connection feature to create a Connection, assign it to our workspace, then perform our inferences by using the Connection details to connect to the BigQuery dataset and tables.

## Prerequisites

* A Wallaroo instance version 2023.2.1 or greater.

## References

* [Wallaroo SDK Essentials Guide: Model Uploads and Registrations: Python Models](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-python/)
* [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/)
* [Wallaroo SDK Essentials Guide: Data Connections Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-dataconnections/)

## Statsmodel Forecast Connection Steps

### Import Libraries

The first step is to import the libraries that we will need.

```python
import json
import os
import datetime

import wallaroo
from wallaroo.object import EntityNotFoundError
from wallaroo.framework import Framework

# used to display dataframe information without truncating
from IPython.display import display
import pandas as pd
import numpy as np

pd.set_option('display.max_colwidth', None)

import time
import pyarrow as pa
```

```python
## convenience functions from the previous notebooks
## these functions assume your connection to wallaroo is called wl

# return the workspace called <name>, or create it if it does not exist.
# this function assumes your connection to wallaroo is called wl
def get_workspace(name):
    workspace = None
    for ws in wl.list_workspaces():
        if ws.name() == name:
            workspace= ws
    if(workspace == None):
        workspace = wl.create_workspace(name)
    return workspace

# pull a single datum from a data frame 
# and convert it to the format the model expects
def get_singleton(df, i):
    singleton = df.iloc[i,:].to_numpy().tolist()
    sdict = {'tensor': [singleton]}
    return pd.DataFrame.from_dict(sdict)

# pull a batch of data from a data frame
# and convert to the format the model expects
def get_batch(df, first=0, nrows=1):
    last = first + nrows
    batch = df.iloc[first:last, :].to_numpy().tolist()
    return pd.DataFrame.from_dict({'tensor': batch})

# Translated a column from a dataframe into a single array
# used for the Statsmodel forecast model

def get_singleton_forecast(df, field):
    singleton = pd.DataFrame({field: [df[field].values.tolist()]})
    return singleton

# Get the most recent version of a model in the workspace
# Assumes that the most recent version is the first in the list of versions.
# wl.get_current_workspace().models() returns a list of models in the current workspace

def get_model(mname):
    modellist = wl.get_current_workspace().models()
    model = [m.versions()[-1] for m in modellist if m.name() == mname]
    if len(model) <= 0:
        raise KeyError(f"model {mname} not found in this workspace")
    return model[0]

# get a pipeline by name in the workspace
def get_pipeline(pname):
    plist = wl.get_current_workspace().pipelines()
    pipeline = [p for p in plist if p.name() == pname]
    if len(pipeline) <= 0:
        raise KeyError(f"pipeline {pname} not found in this workspace")
    return pipeline[0]
```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

### Set Configurations

The following will set the workspace, model name, and pipeline that will be used for this example.  If the workspace or pipeline already exist, then they will assigned for use in this example.  If they do not exist, they will be created based on the names listed below.

Workspace names must be unique.  To allow this tutorial to run in the same Wallaroo instance for multiple users, set the `suffix` variable or share the workspace with other users.

#### Set Configurations References

* [Wallaroo SDK Essentials Guide: Workspace Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-workspace/)
* [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/)

```python
# retrieve the workspace, pipeline and model

workspace_name = "tutorial-workspace-jch"

workspace = get_workspace(workspace_name)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

# optionally, examine your current workspace
wl.get_current_workspace()

pipeline_name = 'tutorialpipeline-jch'

pipeline = get_pipeline(pipeline_name)

control_model = get_model('tutorial-model')

```

### Deploy the Pipeline

Let's set the model step to our single model pipeline, and perform a sample inference with our current data.

```python
# Set pipeline step and deploy

pipeline.undeploy()
pipeline.clear()
pipeline.add_model_step(control_model)
pipeline.deploy()
pipeline.steps()

```

    [{'ModelInference': {'models': [{'name': 'tutorial-model', 'version': '68b3f094-1b0f-4f6e-940e-4dc1cb1500b2', 'sha': 'ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a'}]}}]

```python
# sample inference from previous code here

df_from_csv = pd.read_csv('../data/test_data.csv')

singleton = get_singleton(df_from_csv, 0)
display(singleton)

single_result = pipeline.infer(singleton)
display(single_result)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-03 19:38:47.886</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[704901.9]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Create the Connection

For this demonstration, the connection set to a specific file on a GitHub repository.  The connection details can be anything that can be stored in JSON:  connection URLs, tokens, etc.

This connection will set a URL to pull a file from GitHub, then use the file contents to perform an inference.

Wallaroo connections are created through the Wallaroo Client `create_connection(name, type, details)` method.  See the [Wallaroo SDK Essentials Guide: Data Connections Management guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-dataconnections/) for full details.

Note that connection names must be unique across the Wallaroo instance - if needed, use random characters at the end to make sure your connection doesn't have the same name as a previously created connection.

Here's an example connection used to retrieve the same CSV file used in `./data/test_data.csv`:  https://raw.githubusercontent.com/WallarooLabs/Tutorials/main/Linear%20Regression/Real%20Estate/data/test_data.csv

```python
# set the connection information for other steps
# suffix is used to create a unique data connection

forecast_connection_input_name = f'house-price-data-source'
forecast_connection_input_type = "HTTP"
forecast_connection_input_argument = { 
    "url": "https://raw.githubusercontent.com/WallarooLabs/Tutorials/main/Linear%20Regression/Real%20Estate/data/test_data.csv"
    }

wl.create_connection(forecast_connection_input_name, forecast_connection_input_type, forecast_connection_input_argument)
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>house-price-data-source-john</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-08-03T19:39:04.582089+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>[]</td>
  </tr>
</table>

### List Connections

Connections for the entire Wallaroo instance are listed with Wallaroo Client `list_connections()` method.

```python
# list the connections here

wl.list_connections()
```

<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th><th>linked workspaces</th></tr><tr><td>statsmodel-bike-rentals-john</td><td>HTTP</td><td>*****</td><td>2023-08-02T20:38:34.662841+00:00</td><td>['forecast-model-tutorialjohn']</td></tr><tr><td>bike-rentals-csv-john</td><td>HTTP</td><td>*****</td><td>2023-08-03T15:40:28.804640+00:00</td><td>['forecast-model-tutorialjohn']</td></tr><tr><td>house-price-data-source-john</td><td>HTTP</td><td>*****</td><td>2023-08-03T19:39:04.582089+00:00</td><td>[]</td></tr></table>

### Get Connection by Name

To retrieve a previosly created conneciton, we can assign it to a variable with the method Wallaroo `Client.get_connection(connection_name)`.  Then we can display the connection itself.  Notice that when displaying a connection, the `details` section will be hidden, but they are retrieved with `connection.details()`.  Here's an example:

```python
myconnection = client.get_connection("My amazing connection")
display(myconnection)
display(myconnection.details()
```

Use that code to retrieve your new connection.

```python
# get the connection by name

this_connection = wl.get_connection(forecast_connection_input_name)
this_connection
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>house-price-data-source-john</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-08-03T19:39:04.582089+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>[]</td>
  </tr>
</table>

### Add Connection to Workspace

We'll now add the connection to our workspace so it can be retrieved by other workspace users.  The method Workspace `add_connection(connection_name)` adds a Data Connection to a workspace.  The method Workspace `list_connections()` displays a list of connections attached to the workspace.

```python
workspace.add_connection(forecast_connection_input_name)
workspace.list_connections()
```

<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th><th>linked workspaces</th></tr><tr><td>house-price-data-source-john</td><td>HTTP</td><td>*****</td><td>2023-08-03T19:39:04.582089+00:00</td><td>['tutorial-workspace-jch']</td></tr></table>

### Retrieve Connection from Workspace

To simulate a data scientist's procedural flow, we'll now retrieve the connection from the workspace.  Specific connections are retrieved by specifying their position in the returned list.

For example, if we have two connections in a workspace and we want the second one, we can assign it to a variable with `list_connections[1]`.

Create a new variable and retrieve the connection we just assigned to the workspace.

```python
forecast_connection = workspace.list_connections()[-1]
display(forecast_connection)
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>house-price-data-source-john</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-08-03T19:39:04.582089+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>['tutorial-workspace-jch']</td>
  </tr>
</table>

### Run Inference with Connection

We'll now retrieve sample data through the Wallaroo connection, and perform a sample inference.  The connection details are retrieved through the Connection `details()` method.  Use them to retrieve the CSV file and convert it to a DataFrame, and use it with our sample model.

Or create a new connection with your own data.  Here's some sample code for retrieving a CSV file from a URL.

```python
response = requests.get('https://myurl.com/csvsample.csv')

csv_text = response.text
csv_new = csv_text.replace('\\n', '\n')

from io import StringIO
csvStringIO = StringIO(csv_new)
df = pd.read_csv(csvStringIO)
```

```python
display(forecast_connection.details()['url'])

import requests

response = requests.get(
                    forecast_connection.details()['url']
                )

csv_text = response.text
csv_new = csv_text.replace('\\n', '\n')

from io import StringIO
csvStringIO = StringIO(csv_new)

# print(csv_new)

df_from_csv = pd.read_csv(csvStringIO)

singleton = get_singleton(df_from_csv, 0)
display(singleton)

single_result = pipeline.infer(singleton)
display(single_result)

multiple_batch = get_batch(df_from_csv, nrows=5)
multiple_result = pipeline.infer(multiple_batch)
display(multiple_result)
```

    'https://raw.githubusercontent.com/WallarooLabs/Tutorials/main/Linear%20Regression/Real%20Estate/data/test_data.csv'

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-03 19:39:36.583</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[704901.9]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-03 19:39:36.964</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[704901.9]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-08-03 19:39:36.964</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[695994.44]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-08-03 19:39:36.964</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[416164.8]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-08-03 19:39:36.964</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[655277.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-08-03 19:39:36.964</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[426854.66]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

You have now walked through setting up a basic assay and running it over historical data.

## Congratulations!
In this tutorial you have
* Deployed a single step house price prediction pipeline and sent data to it.
* Create a new Wallaroo connection
* Assigned the connection to a workspace
* Retrieved the connection from the workspace
* Used the data connection to retrieve information from outside of Wallaroo, and use it for an inference

Great job! 

### Cleaning up.

Now that the tutorial is complete, don't forget to undeploy your pipeline to free up the resources.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>tutorialpipeline-jch</td></tr><tr><th>created</th> <td>2023-08-03 19:36:31.732163+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-03 19:37:32.733734+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>1a73448b-9012-4258-bb2c-a4d25a1e6f19, d1d0cafe-78a9-4193-84af-cec1b3ed608b, 70438418-4802-4ced-a295-65ef78706fd4</td></tr><tr><th>steps</th> <td>tutorial-model</td></tr></table>

