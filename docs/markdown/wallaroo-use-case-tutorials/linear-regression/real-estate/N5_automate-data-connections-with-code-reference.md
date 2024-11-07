# Tutorial Notebook 5: Automation with Wallaroo Connections

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

## Preliminaries

In the blocks below we will preload some required libraries.

For convenience, the following `helper functions` are defined to retrieve previously created workspaces, models, and pipelines:

* `get_workspace(name, client)`: This takes in the name and the Wallaroo client being used in this session, and returns the workspace matching `name`.  If no workspaces are found matching the name, raises a `KeyError` and returns `None`.
* `get_model_version(model_name, workspace)`: Retrieves the most recent model version from the model matching the `model_name` within the provided `workspace`.  If no model matches that name, raises a `KeyError` and returns `None`.
* `get_pipeline(pipeline_name, workspace)`: Retrieves the most pipeline from the workspace matching the `pipeline_name` within the provided `workspace`.  If no model matches that name, raises a `KeyError` and returns `None`.

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

# return the workspace called <name> through the Wallaroo client.
def get_workspace(name, client):
    workspace = None
    for ws in client.list_workspaces():
        if ws.name() == name:
            workspace= ws
            return workspace
    # if no workspaces were found
    if workspace==None:
        raise KeyError(f"Workspace {name} was not found.")
    return workspace

# returns the most recent model version in a workspace for the matching `model_name`
def get_model_version(model_name, workspace):
    modellist = workspace.models()
    model_version = [m.versions()[-1] for m in modellist if m.name() == model_name]
    # if no models match, return None
    if len(modellist) <= 0:
        raise KeyError(f"Model {mname} not found in this workspace")
        return None
    return model_version[0]

# get a pipeline by name in the workspace
def get_pipeline(pipeline_name, workspace):
    plist = workspace.pipelines()
    pipeline = [p for p in plist if p.name() == pipeline_name]
    if len(pipeline) <= 0:
        raise KeyError(f"Pipeline {pipeline_name} not found in this workspace")
        return None
    return pipeline[0]

```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
## blank space to log in 

wl = wallaroo.Client()
```

### Set Configurations

Set the workspace, pipeline, and model used from Notebook 1.  The helper functions will make this task easier.

#### Set Configurations References

* [Wallaroo SDK Essentials Guide: Workspace Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-workspace/)
* [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/)

```python
# retrieve the previous workspace, model, and pipeline version

workspace_name = "tutorial-workspace-john-05"

workspace = get_workspace(workspace_name, wl)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

# optionally, examine your current workspace
wl.get_current_workspace()

model_name = 'house-price-prime'

prime_model_version = get_model_version(model_name, workspace)

pipeline_name = 'houseprice-estimator'

pipeline = get_pipeline(pipeline_name, workspace)

display(workspace)
display(prime_model_version)
display(pipeline)

```

    {'name': 'tutorial-workspace-john-05', 'id': 261, 'archived': False, 'created_by': 'd1704c38-2016-4b1d-9407-85e7e6875e6d', 'created_at': '2023-09-11T21:37:02.871776+00:00', 'models': [{'name': 'house-price-prime', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 9, 11, 21, 40, 27, 608874, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 9, 11, 21, 40, 27, 608874, tzinfo=tzutc())}], 'pipelines': [{'name': 'houseprice-estimator', 'create_time': datetime.datetime(2023, 9, 11, 21, 40, 29, 871251, tzinfo=tzutc()), 'definition': '[]'}]}

<table>
        <tr>
          <td>Name</td>
          <td>house-price-prime</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>f42b66d3-ed47-4571-910c-5e7184b2b4ad</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>xgb_model.onnx</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c</td>
        </tr>
        <tr>
          <td>Status</td>
          <td>ready</td>
        </tr>
        <tr>
          <td>Image Path</td>
          <td>None</td>
        </tr>
        <tr>
          <td>Updated At</td>
          <td>2023-11-Sep 21:40:27</td>
        </tr>
      </table>

<table><tr><th>name</th> <td>houseprice-estimator</td></tr><tr><th>created</th> <td>2023-09-11 21:40:29.871251+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-11 21:41:50.764182+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6f788da6-17ee-4df6-b120-354737212559, 7ae39ec3-1486-4bd7-9fa5-6874ee79f245, 11dd78db-9461-4ec0-9003-29538d4c242d, 8ca7a6bf-6529-434f-a3f6-bbc3bcc255d9</td></tr><tr><th>steps</th> <td>house-price-prime</td></tr></table>

## Deploy the Pipeline with the Model Version Step

As per the other tutorials:

1. Clear the pipeline of all steps.
1. Add the model version as a pipeline step.
1. Deploy the pipeline with the following deployment configuration:

```python
deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
```

```python
pipeline.clear()
pipeline.add_model_step(prime_model_version)

deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
pipeline.deploy(deployment_config=deploy_config)
```

<table><tr><th>name</th> <td>houseprice-estimator</td></tr><tr><th>created</th> <td>2023-09-11 21:40:29.871251+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-13 16:48:39.767241+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>4d3230a4-32be-4f47-90b5-d8d3b90cc2a9, 6f788da6-17ee-4df6-b120-354737212559, 7ae39ec3-1486-4bd7-9fa5-6874ee79f245, 11dd78db-9461-4ec0-9003-29538d4c242d, 8ca7a6bf-6529-434f-a3f6-bbc3bcc255d9</td></tr><tr><th>steps</th> <td>house-price-prime</td></tr></table>

## Create the Connection

For this demonstration, the connection set to a specific file on a GitHub repository.  The connection details can be anything that can be stored in JSON:  connection URLs, tokens, etc.

This connection will set a URL to pull a file from GitHub, then use the file contents to perform an inference.

Wallaroo connections are created through the Wallaroo Client `create_connection(name, type, details)` method.  See the [Wallaroo SDK Essentials Guide: Data Connections Management guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-dataconnections/) for full details.

Note that connection names must be unique across the Wallaroo instance - if needed, use random characters at the end to make sure your connection doesn't have the same name as a previously created connection.

Here's an example connection used to retrieve the same CSV file used in `./data/test_data.df.json`:  https://raw.githubusercontent.com/WallarooLabs/Tutorials/main/Linear%20Regression/Real%20Estate/data/test_data.df.json

### Create the Connection Exercise

```python
# set the connection information for other steps
# suffix is used to create a unique data connection

forecast_connection_input_name = f'house-price-data'
forecast_connection_input_type = "HTTP"
forecast_connection_input_argument = { 
    "url": "https://raw.githubusercontent.com/WallarooLabs/Tutorials/main/Linear%20Regression/Real%20Estate/data/test_data.df.json"
    }

wl.create_connection(forecast_connection_input_name, forecast_connection_input_type, forecast_connection_input_argument)
```

```python
# set the connection information for other steps
# suffix is used to create a unique data connection

forecast_connection_input_name = f'house-price-data-source-john05'
forecast_connection_input_type = "HTTP"
forecast_connection_input_argument = { 
    "url": "https://raw.githubusercontent.com/WallarooLabs/Tutorials/main/Linear%20Regression/Real%20Estate/data/test_data.df.json"
    }

wl.create_connection(forecast_connection_input_name, forecast_connection_input_type, forecast_connection_input_argument)
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>house-price-data-source-john05</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-09-13T16:44:48.872567+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>[]</td>
  </tr>
</table>

## List Connections

Connections for the entire Wallaroo instance are listed with Wallaroo Client `list_connections()` method.

## List Connections Exercise

Here's an example of listing the connections when the Wallaroo client is `wl`.

```python
wl.list_connections()
```

```python
# list the connections here

wl.list_connections()
```

<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th><th>linked workspaces</th></tr><tr><td>real-estate-connection-ex</td><td>HTTPFile</td><td>*****</td><td>2023-08-18T00:12:26.864937+00:00</td><td>['crystal-workspaceamuj']</td></tr><tr><td>real-estate-connection-ex-2</td><td>HTTP</td><td>*****</td><td>2023-08-18T00:41:18.580028+00:00</td><td>['crystal-workspaceamuj']</td></tr><tr><td>real-estate-connection-ex-3</td><td>HTTP</td><td>*****</td><td>2023-08-18T00:46:10.368793+00:00</td><td>['crystal-workspaceamuj']</td></tr><tr><td>house-price-data-source</td><td>HTTP</td><td>*****</td><td>2023-08-18T01:57:15.707226+00:00</td><td>['tutorial-workspacebiad']</td></tr><tr><td>bike-rentals-csvjohn</td><td>HTTP</td><td>*****</td><td>2023-08-30T18:06:06.979995+00:00</td><td>['multiple-replica-forecast-tutorial-john']</td></tr><tr><td>house-price-data-source-john05</td><td>HTTP</td><td>*****</td><td>2023-09-13T16:44:48.872567+00:00</td><td>[]</td></tr></table>

## Get Connection by Name

To retrieve a previosly created conneciton, we can assign it to a variable with the method Wallaroo `Client.get_connection(connection_name)`.  Then we can display the connection itself.  Notice that when displaying a connection, the `details` section will be hidden, but they are retrieved with `connection.details()`.  Here's an example:

```python
myconnection = client.get_connection("My amazing connection")
display(myconnection)
display(myconnection.details()
```

Use that code to retrieve your new connection.

### Get Connection by Name Example

Here's an example based on the Wallaroo client saved as `wl`.

```python
wl.get_connection(forecast_connection_input_name)
```

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
    <td>Name</td><td>house-price-data-source-john05</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-09-13T16:44:48.872567+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>[]</td>
  </tr>
</table>

## Add Connection to Workspace

We'll now add the connection to our workspace so it can be retrieved by other workspace users.  The method Workspace `add_connection(connection_name)` adds a Data Connection to a workspace.  The method Workspace `list_connections()` displays a list of connections attached to the workspace.

### Add Connection to Workspace Exercise

Use the connection we just created, and add it to the sample workspace.  Here's a code example where the workspace is saved to the variable `workspace` and the connection is saved as `forecast_connection_input_name`.

```python
workspace.add_connection(forecast_connection_input_name)
```

```python
workspace.add_connection(forecast_connection_input_name)
workspace.list_connections()
```

<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th><th>linked workspaces</th></tr><tr><td>house-price-data-source-john05</td><td>HTTP</td><td>*****</td><td>2023-09-13T16:44:48.872567+00:00</td><td>['tutorial-workspace-john-05']</td></tr></table>

## Retrieve Connection from Workspace

To simulate a data scientist's procedural flow, we'll now retrieve the connection from the workspace.  Specific connections are retrieved by specifying their position in the returned list.

For example, if we have two connections in a workspace and we want the second one, we can assign it to a variable with `list_connections[1]`.

Create a new variable and retrieve the connection we just assigned to the workspace.

### Retrieve Connection from Workspace Exercise

Retrieve the connection that was just associated with the workspace.  You'll use the `list_connections` method, then assign a variable to the connection.  Here's an example if the connection is the most recently one added to the workspace `workspace`.

```python
forecast_connection = workspace.list_connections()[-1]
```

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
    <td>Name</td><td>house-price-data-source-john05</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-09-13T16:44:48.872567+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>['tutorial-workspace-john-05']</td>
  </tr>
</table>

## Run Inference with Connection

Connections can be used for different purposes:  uploading new models, engine configurations - any place that data is needed.  This exercise will use the data connection to perform an inference through our deployed pipeline.

### Run Inference with Connection Exercise

We'll now retrieve sample data through the Wallaroo connection, and perform a sample inference.  The connection details are retrieved through the Connection `details()` method.  Use them to retrieve the pandas record file and convert it to a DataFrame, and use it with our sample model.

Here's a code example that uses the Python `requests` library to retrieve the file information, then turns it into a DataFrame for the inference request.

```python
display(forecast_connection.details()['url'])

import requests

response = requests.get(
                    forecast_connection.details()['url']
                )

# display(response.json())

df = pd.DataFrame(response.json())

pipeline.infer(df)
```

```python
display(forecast_connection.details()['url'])

import requests

response = requests.get(
                    forecast_connection.details()['url']
                )

# display(response.json())

df = pd.DataFrame(response.json())
display(df)

multiple_result = pipeline.infer(df)
display(multiple_result)
```

    'https://raw.githubusercontent.com/WallarooLabs/Tutorials/main/Linear%20Regression/Real%20Estate/data/test_data.df.json'

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
    <tr>
      <th>1</th>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>3995</th>
      <td>[4.0, 2.25, 2620.0, 98881.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1820.0, 800.0, 47.4662, -122.453, 1728.0, 95832.0, 63.0, 0.0, 0.0]</td>
    </tr>
    <tr>
      <th>3996</th>
      <td>[3.0, 2.5, 2244.0, 4079.0, 2.0, 0.0, 0.0, 3.0, 7.0, 2244.0, 0.0, 47.2606, -122.254, 2077.0, 4078.0, 3.0, 0.0, 0.0]</td>
    </tr>
    <tr>
      <th>3997</th>
      <td>[3.0, 1.75, 1490.0, 5000.0, 1.0, 0.0, 1.0, 3.0, 8.0, 1250.0, 240.0, 47.5257, -122.392, 1980.0, 5000.0, 61.0, 0.0, 0.0]</td>
    </tr>
    <tr>
      <th>3998</th>
      <td>[4.0, 2.5, 2740.0, 5700.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2740.0, 0.0, 47.3535, -122.026, 3010.0, 5281.0, 8.0, 0.0, 0.0]</td>
    </tr>
    <tr>
      <th>3999</th>
      <td>[5.0, 2.5, 2240.0, 7770.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1340.0, 900.0, 47.7198, -122.171, 1820.0, 7770.0, 36.0, 0.0, 0.0]</td>
    </tr>
  </tbody>
</table>
<p>4000 rows × 1 columns</p>

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
      <td>2023-09-13 16:50:14.425</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[659806.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-09-13 16:50:14.425</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[732883.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-09-13 16:50:14.425</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[419508.84]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-09-13 16:50:14.425</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[634028.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-09-13 16:50:14.425</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[427209.47]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3995</th>
      <td>2023-09-13 16:50:14.425</td>
      <td>[4.0, 2.25, 2620.0, 98881.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1820.0, 800.0, 47.4662, -122.453, 1728.0, 95832.0, 63.0, 0.0, 0.0]</td>
      <td>[436151.13]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3996</th>
      <td>2023-09-13 16:50:14.425</td>
      <td>[3.0, 2.5, 2244.0, 4079.0, 2.0, 0.0, 0.0, 3.0, 7.0, 2244.0, 0.0, 47.2606, -122.254, 2077.0, 4078.0, 3.0, 0.0, 0.0]</td>
      <td>[284810.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3997</th>
      <td>2023-09-13 16:50:14.425</td>
      <td>[3.0, 1.75, 1490.0, 5000.0, 1.0, 0.0, 1.0, 3.0, 8.0, 1250.0, 240.0, 47.5257, -122.392, 1980.0, 5000.0, 61.0, 0.0, 0.0]</td>
      <td>[575571.44]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3998</th>
      <td>2023-09-13 16:50:14.425</td>
      <td>[4.0, 2.5, 2740.0, 5700.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2740.0, 0.0, 47.3535, -122.026, 3010.0, 5281.0, 8.0, 0.0, 0.0]</td>
      <td>[432262.38]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3999</th>
      <td>2023-09-13 16:50:14.425</td>
      <td>[5.0, 2.5, 2240.0, 7770.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1340.0, 900.0, 47.7198, -122.171, 1820.0, 7770.0, 36.0, 0.0, 0.0]</td>
      <td>[445873.13]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>4000 rows × 4 columns</p>

## Cleaning up.

Now that the tutorial is complete, don't forget to undeploy your pipeline to free up the resources.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>houseprice-estimator</td></tr><tr><th>created</th> <td>2023-09-11 21:40:29.871251+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-13 16:48:39.767241+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>4d3230a4-32be-4f47-90b5-d8d3b90cc2a9, 6f788da6-17ee-4df6-b120-354737212559, 7ae39ec3-1486-4bd7-9fa5-6874ee79f245, 11dd78db-9461-4ec0-9003-29538d4c242d, 8ca7a6bf-6529-434f-a3f6-bbc3bcc255d9</td></tr><tr><th>steps</th> <td>house-price-prime</td></tr></table>

## Congratulations!

In this tutorial you have:

* Deployed a single step house price prediction pipeline and sent data to it.
* Create a new Wallaroo connection.
* Assigned the connection to a workspace.
* Retrieved the connection from the workspace.
* Used the data connection to retrieve information from outside of Wallaroo, and use it for an inference.

Great job!
