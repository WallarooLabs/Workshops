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

## Login to Wallaroo

Retrieve the previous workspace, model versions, and pipelines used in the previous notebook.

```python
## blank space to log in 

wl = wallaroo.Client()

# retrieve the previous workspace, model, and pipeline version

workspace_name = 'tutorial-workspace-forecast'

workspace = wl.get_workspace(name=workspace_name, create_if_not_exist=True)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

model_name = "forecast-control-model"

prime_model_version = wl.get_model(model_name)

pipeline_name = 'rental-forecast'

pipeline = wl.get_pipeline(pipeline_name)

# verify the workspace/pipeline/model

display(wl.get_current_workspace())
display(prime_model_version)
display(pipeline)
```

    {'name': 'tutorial-workspace-forecast', 'id': 8, 'archived': False, 'created_by': 'fca5c4df-37ac-4a78-9602-dd09ca72bc60', 'created_at': '2024-10-29T20:52:00.744998+00:00', 'models': [{'name': 'forecast-control-model', 'versions': 3, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 10, 29, 21, 35, 59, 4303, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 10, 29, 20, 54, 24, 314662, tzinfo=tzutc())}, {'name': 'forecast-alternate01-model', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 10, 30, 19, 56, 17, 519779, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 10, 30, 19, 56, 17, 519779, tzinfo=tzutc())}, {'name': 'forecast-alternate02-model', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 10, 30, 19, 56, 43, 83456, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 10, 30, 19, 56, 43, 83456, tzinfo=tzutc())}], 'pipelines': [{'name': 'rental-forecast', 'create_time': datetime.datetime(2024, 10, 29, 21, 0, 36, 927945, tzinfo=tzutc()), 'definition': '[]'}]}

<table>
        <tr>
          <td>Name</td>
          <td>forecast-control-model</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>4c9a1678-cba3-4db9-97a5-883ce89a9a24</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>forecast_standard.zip</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>80b51818171dc1e64e61c3050a0815a68b4d14b1b37e1e18dac9e4719e074eb1</td>
        </tr>
        <tr>
          <td>Status</td>
          <td>ready</td>
        </tr>
        <tr>
          <td>Image Path</td>
          <td>proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mac-deploy:v2024.2.0-5761</td>
        </tr>
        <tr>
          <td>Architecture</td>
          <td>x86</td>
        </tr>
        <tr>
          <td>Acceleration</td>
          <td>none</td>
        </tr>
        <tr>
          <td>Updated At</td>
          <td>2024-29-Oct 21:36:20</td>
        </tr>
        <tr>
          <td>Workspace id</td>
          <td>8</td>
        </tr>
        <tr>
          <td>Workspace name</td>
          <td>tutorial-workspace-forecast</td>
        </tr>
      </table>

<table><tr><th>name</th> <td>rental-forecast</td></tr><tr><th>created</th> <td>2024-10-29 21:00:36.927945+00:00</td></tr><tr><th>last_updated</th> <td>2024-10-30 20:43:23.101933+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>workspace_id</th> <td>8</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-forecast</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>ceff9712-715b-41e6-a124-b174b62a9654, 0250f403-07c6-4b01-83bc-eebdc09bca22, 31b515bb-807f-4d64-b105-fc0ae6a582f2, 614a34e0-6024-4245-9919-1a85b7a1e5d2, 6a593faf-bea3-4f57-b9ec-5c1afe7f93a7, 4dce5be3-926c-419f-9868-3dbea7baf3c1, a601ce07-937c-436a-9735-0ac842173dfb, c0d16da5-5db7-4af1-95e4-cb0c316a4ef3, bd5eb43f-5a2b-493c-a04b-863dccccb55f, 89729096-6581-42b8-9b06-10d580d31e11, b98b86fb-5941-45b6-af5d-c33f80ba7986, aead5518-ffb2-4d18-8898-89575ba90a9f, a2a887c0-a91b-4af7-b579-506c79631fa4, b8ac836a-903b-4327-a4c9-5cc7fb382aa7, 3e18cd2d-c006-497b-a756-5ecc95aa8439, bd3f7d6a-e246-4456-98b9-35b90990b86d</td></tr><tr><th>steps</th> <td>forecast-control-model</td></tr><tr><th>published</th> <td>False</td></tr></table>

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

    Waiting for deployment - this will take up to 45s .............. ok

<table><tr><th>name</th> <td>rental-forecast</td></tr><tr><th>created</th> <td>2024-10-29 21:00:36.927945+00:00</td></tr><tr><th>last_updated</th> <td>2024-10-30 20:48:14.837079+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>workspace_id</th> <td>8</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-forecast</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>585ee8cd-2f5e-4a1e-bb0d-6c88e6d94d3e, ceff9712-715b-41e6-a124-b174b62a9654, 0250f403-07c6-4b01-83bc-eebdc09bca22, 31b515bb-807f-4d64-b105-fc0ae6a582f2, 614a34e0-6024-4245-9919-1a85b7a1e5d2, 6a593faf-bea3-4f57-b9ec-5c1afe7f93a7, 4dce5be3-926c-419f-9868-3dbea7baf3c1, a601ce07-937c-436a-9735-0ac842173dfb, c0d16da5-5db7-4af1-95e4-cb0c316a4ef3, bd5eb43f-5a2b-493c-a04b-863dccccb55f, 89729096-6581-42b8-9b06-10d580d31e11, b98b86fb-5941-45b6-af5d-c33f80ba7986, aead5518-ffb2-4d18-8898-89575ba90a9f, a2a887c0-a91b-4af7-b579-506c79631fa4, b8ac836a-903b-4327-a4c9-5cc7fb382aa7, 3e18cd2d-c006-497b-a756-5ecc95aa8439, bd3f7d6a-e246-4456-98b9-35b90990b86d</td></tr><tr><th>steps</th> <td>forecast-control-model</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Create the Connection

For this demonstration, the connection set to a specific file on a GitHub repository.  The connection details can be anything that can be stored in JSON:  connection URLs, tokens, etc.

This connection will set a URL to pull a file from GitHub, then use the file contents to perform an inference.

Wallaroo connections are created through the Wallaroo Client `create_connection(name, type, details)` method.  See the [Wallaroo SDK Essentials Guide: Data Connections Management guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-dataconnections/) for full details.

Note that connection names must be unique across the Wallaroo instance - if needed, use random characters at the end to make sure your connection doesn't have the same name as a previously created connection.

Here's an example connection used to retrieve the same CSV file used in `./data/testdata_standard.df.json`:  https://raw.githubusercontent.com/WallarooLabs/Tutorials/main/Forecasting/Retail-CPG/data/testdata_standard.df.json

### Create the Connection Exercise

```python
# set the connection information for other steps
# suffix is used to create a unique data connection

forecast_connection_input_name = f'forecast-sample-data'
forecast_connection_input_type = "HTTP"
forecast_connection_input_argument = { 
    "url": "https://raw.githubusercontent.com/WallarooLabs/Tutorials/main/Forecasting/Retail-CPG/data/testdata_standard.df.json"
    }

wl.create_connection(forecast_connection_input_name, forecast_connection_input_type, forecast_connection_input_argument)
```

```python
# set the connection information for other steps
# suffix is used to create a unique data connection

forecast_connection_input_name = f'forecast-sample-connection'
forecast_connection_input_type = "HTTP"
forecast_connection_input_argument = { 
    "url": "https://raw.githubusercontent.com/WallarooLabs/Tutorials/refs/heads/wallaroo-2024.2/Forecasting/Retail-CPG/data/testdata-standard.df.json"
    }

wl.create_connection(forecast_connection_input_name, forecast_connection_input_type, forecast_connection_input_argument)
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>forecast-sample-connection</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2024-10-30T20:53:23.926727+00:00</td>
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

<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th><th>linked workspaces</th></tr><tr><td>summary-sample-connection</td><td>HTTP</td><td>*****</td><td>2024-10-29T20:33:12.209391+00:00</td><td>['tutorial-workspace-summarization']</td></tr><tr><td>forecast-sample-data</td><td>HTTP</td><td>*****</td><td>2024-10-30T20:48:30.452574+00:00</td><td>['tutorial-workspace-forecast']</td></tr><tr><td>forecast-sample-connection</td><td>HTTP</td><td>*****</td><td>2024-10-30T20:53:23.926727+00:00</td><td>[]</td></tr></table>

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
    <td>Name</td><td>forecast-sample-connection</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2024-10-30T20:53:23.926727+00:00</td>
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

<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th><th>linked workspaces</th></tr><tr><td>forecast-sample-data</td><td>HTTP</td><td>*****</td><td>2024-10-30T20:48:30.452574+00:00</td><td>['tutorial-workspace-forecast']</td></tr><tr><td>forecast-sample-connection</td><td>HTTP</td><td>*****</td><td>2024-10-30T20:53:23.926727+00:00</td><td>['tutorial-workspace-forecast']</td></tr></table>

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
    <td>Name</td><td>forecast-sample-connection</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2024-10-30T20:53:23.926727+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>['tutorial-workspace-forecast']</td>
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

single_result = pipeline.infer(df)
display(single_result)
```

    'https://raw.githubusercontent.com/WallarooLabs/Tutorials/refs/heads/wallaroo-2024.2/Forecasting/Retail-CPG/data/testdata-standard.df.json'

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[1526, 1550, 1708, 1005, 1623, 1712, 1530, 1605, 1538, 1746, 1472, 1589, 1913, 1815, 2115, 2475, 2927, 1635, 1812, 1107, 1450, 1917, 1807, 1461, 1969, 2402, 1446, 1851]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.count</th>
      <th>out.forecast</th>
      <th>out.weekly_average</th>
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-10-30 20:53:41.226</td>
      <td>[1526, 1550, 1708, 1005, 1623, 1712, 1530, 1605, 1538, 1746, 1472, 1589, 1913, 1815, 2115, 2475, 2927, 1635, 1812, 1107, 1450, 1917, 1807, 1461, 1969, 2402, 1446, 1851]</td>
      <td>[1764, 1749, 1743, 1741, 1740, 1740, 1740]</td>
      <td>1745.2858</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

## Cleaning up.

Now that the tutorial is complete, don't forget to undeploy your pipeline to free up the resources.

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s .................................... ok

<table><tr><th>name</th> <td>rental-forecast</td></tr><tr><th>created</th> <td>2024-10-29 21:00:36.927945+00:00</td></tr><tr><th>last_updated</th> <td>2024-10-30 20:48:14.837079+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>workspace_id</th> <td>8</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-forecast</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>585ee8cd-2f5e-4a1e-bb0d-6c88e6d94d3e, ceff9712-715b-41e6-a124-b174b62a9654, 0250f403-07c6-4b01-83bc-eebdc09bca22, 31b515bb-807f-4d64-b105-fc0ae6a582f2, 614a34e0-6024-4245-9919-1a85b7a1e5d2, 6a593faf-bea3-4f57-b9ec-5c1afe7f93a7, 4dce5be3-926c-419f-9868-3dbea7baf3c1, a601ce07-937c-436a-9735-0ac842173dfb, c0d16da5-5db7-4af1-95e4-cb0c316a4ef3, bd5eb43f-5a2b-493c-a04b-863dccccb55f, 89729096-6581-42b8-9b06-10d580d31e11, b98b86fb-5941-45b6-af5d-c33f80ba7986, aead5518-ffb2-4d18-8898-89575ba90a9f, a2a887c0-a91b-4af7-b579-506c79631fa4, b8ac836a-903b-4327-a4c9-5cc7fb382aa7, 3e18cd2d-c006-497b-a756-5ecc95aa8439, bd3f7d6a-e246-4456-98b9-35b90990b86d</td></tr><tr><th>steps</th> <td>forecast-control-model</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Congratulations!

In this tutorial you have:

* Deployed a single step house price prediction pipeline and sent data to it.
* Create a new Wallaroo connection.
* Assigned the connection to a workspace.
* Retrieved the connection from the workspace.
* Used the data connection to retrieve information from outside of Wallaroo, and use it for an inference.

Great job!
