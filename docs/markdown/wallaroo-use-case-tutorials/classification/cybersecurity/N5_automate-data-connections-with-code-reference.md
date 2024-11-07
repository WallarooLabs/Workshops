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

workspace_name = "tutorial-workspace-john-cybersecurity"

workspace = wl.get_workspace(name=workspace_name)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

# optionally, examine your current workspace
wl.get_current_workspace()

model_name = 'aloha-prime'

prime_model_version = wl.get_model(model_name)

pipeline_name = 'aloha-fraud-detector'

pipeline = wl.get_pipeline(pipeline_name)

display(workspace)
display(prime_model_version)
display(pipeline)

```

    {'name': 'tutorial-workspace-john-cybersecurity', 'id': 14, 'archived': False, 'created_by': '76b893ff-5c30-4f01-bd9e-9579a20fc4ea', 'created_at': '2024-05-01T16:30:01.177583+00:00', 'models': [{'name': 'aloha-prime', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 5, 1, 16, 30, 43, 651533, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 5, 1, 16, 30, 43, 651533, tzinfo=tzutc())}, {'name': 'aloha-challenger', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 5, 1, 16, 38, 56, 600586, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 5, 1, 16, 38, 56, 600586, tzinfo=tzutc())}], 'pipelines': [{'name': 'aloha-fraud-detector', 'create_time': datetime.datetime(2024, 5, 1, 16, 30, 53, 995114, tzinfo=tzutc()), 'definition': '[]'}]}

<table>
        <tr>
          <td>Name</td>
          <td>aloha-prime</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>c719bc50-f83f-4c79-b4af-f66395a8da04</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>aloha-cnn-lstm.zip</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520</td>
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
          <td>Architecture</td>
          <td>x86</td>
        </tr>
        <tr>
          <td>Acceleration</td>
          <td>none</td>
        </tr>
        <tr>
          <td>Updated At</td>
          <td>2024-01-May 16:30:43</td>
        </tr>
      </table>

<table><tr><th>name</th> <td>aloha-fraud-detector</td></tr><tr><th>created</th> <td>2024-05-01 16:30:53.995114+00:00</td></tr><tr><th>last_updated</th> <td>2024-05-01 19:51:36.073881+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>958397f7-1d3f-472c-8654-21283b8d6d85, ea59aae6-1565-4321-9086-7f8dd2a8e1c2, 41b9260e-21a8-4f16-ad56-c6267d3bae93, 46bdd5a3-fc22-41b7-b1ea-8287c99c241e, 28f5443a-ff67-4f4f-bfc3-f3f95f3c6f83, 435b73f7-76a1-4514-bd41-2cb94d3e78ff, 551242a7-fe4c-4a61-a4c7-e7fcc97509dc, d22eb0d2-9cff-4c5f-a851-10b1a19d8c44, 262909e9-8779-4c56-a994-725ddd0b58c8, 4cdf8e11-1b9c-44ab-a16d-abb054b5e9fe, 6b3529b1-1ff1-454b-8896-460c8c90d667</td></tr><tr><th>steps</th> <td>aloha-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Deploy the Pipeline with the Model Version Step

As per the other tutorials:

1. Clear the pipeline of all steps.
1. Add the model version as a pipeline step.
1. Deploy the pipeline with the following deployment configuration:

```python
deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
```

```python
deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
pipeline.deploy(deployment_config=deploy_config)
```

<table><tr><th>name</th> <td>aloha-fraud-detector</td></tr><tr><th>created</th> <td>2024-05-01 16:30:53.995114+00:00</td></tr><tr><th>last_updated</th> <td>2024-05-01 20:03:01.505745+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>8ede9ff5-e42a-404c-b248-fb4c4efd687d, 958397f7-1d3f-472c-8654-21283b8d6d85, ea59aae6-1565-4321-9086-7f8dd2a8e1c2, 41b9260e-21a8-4f16-ad56-c6267d3bae93, 46bdd5a3-fc22-41b7-b1ea-8287c99c241e, 28f5443a-ff67-4f4f-bfc3-f3f95f3c6f83, 435b73f7-76a1-4514-bd41-2cb94d3e78ff, 551242a7-fe4c-4a61-a4c7-e7fcc97509dc, d22eb0d2-9cff-4c5f-a851-10b1a19d8c44, 262909e9-8779-4c56-a994-725ddd0b58c8, 4cdf8e11-1b9c-44ab-a16d-abb054b5e9fe, 6b3529b1-1ff1-454b-8896-460c8c90d667</td></tr><tr><th>steps</th> <td>aloha-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

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

forecast_connection_input_name = f'cybersecurity-sample-data'
forecast_connection_input_type = "HTTP"
forecast_connection_input_argument = { 
    "url": "https://raw.githubusercontent.com/WallarooLabs/Tutorials/main/Classification/Cybersecurity/data/data-1k.df.json"
    }

wl.create_connection(forecast_connection_input_name, forecast_connection_input_type, forecast_connection_input_argument)
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>cybersecurity-sample-data</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2024-05-01T20:03:35.224420+00:00</td>
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

<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th><th>linked workspaces</th></tr><tr><td>house-price-data-source-john05</td><td>HTTP</td><td>*****</td><td>2024-04-24T16:16:26.716896+00:00</td><td>['tutorial-workspace-john-05']</td></tr><tr><td>cybersecurity-sample-data</td><td>HTTP</td><td>*****</td><td>2024-05-01T20:03:35.224420+00:00</td><td>[]</td></tr></table>

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
    <td>Name</td><td>cybersecurity-sample-data</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2024-05-01T20:03:35.224420+00:00</td>
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

<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th><th>linked workspaces</th></tr><tr><td>cybersecurity-sample-data</td><td>HTTP</td><td>*****</td><td>2024-05-01T20:03:35.224420+00:00</td><td>['tutorial-workspace-john-cybersecurity']</td></tr></table>

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
    <td>Name</td><td>cybersecurity-sample-data</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2024-05-01T20:03:35.224420+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>['tutorial-workspace-john-cybersecurity']</td>
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

    'https://raw.githubusercontent.com/WallarooLabs/Tutorials/main/Classification/Cybersecurity/data/data-1k.df.json'

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text_input</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 16, 32, 23, 29, 32, 30, 19, 26, 17]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 27, 31, 29, 28, 15, 33, 29, 12, 36, 31, 12]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 25, 21, 16, 22, 20, 19, 19, 28]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 20, 22, 18, 32, 15, 12, 33, 17, 31, 14, 14, 27, 18]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 22, 12, 30, 24, 13, 19, 25, 36, 28, 13, 12, 13]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 31, 35, 13, 14, 25, 23, 14, 21, 20, 32, 14, 32, 29, 16, 33]</td>
    </tr>
    <tr>
      <th>996</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 20, 29, 23, 14, 12, 27, 35, 34]</td>
    </tr>
    <tr>
      <th>997</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 28, 19, 19, 28, 15, 12, 18]</td>
    </tr>
    <tr>
      <th>998</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 12, 19, 35, 16, 35, 27, 16]</td>
    </tr>
    <tr>
      <th>999</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 12, 28, 30, 19, 34, 26, 18, 31, 25, 13, 29, 17, 24, 29, 14, 36, 13]</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 1 columns</p>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.text_input</th>
      <th>out.banjori</th>
      <th>out.corebot</th>
      <th>out.cryptolocker</th>
      <th>out.dircrypt</th>
      <th>out.gozi</th>
      <th>out.kraken</th>
      <th>out.locky</th>
      <th>out.main</th>
      <th>out.matsnu</th>
      <th>out.pykspa</th>
      <th>out.qakbot</th>
      <th>out.ramdo</th>
      <th>out.ramnit</th>
      <th>out.simda</th>
      <th>out.suppobox</th>
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-05-01 20:03:37.993</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 16, 32, 23, 29, 32, 30, 19, 26, 17]</td>
      <td>[0.0015195814]</td>
      <td>[0.98291475]</td>
      <td>[0.012099549]</td>
      <td>[4.7591115e-05]</td>
      <td>[2.0289312e-05]</td>
      <td>[0.00031977257]</td>
      <td>[0.011029262]</td>
      <td>[0.997564]</td>
      <td>[0.010341609]</td>
      <td>[0.008038961]</td>
      <td>[0.016155047]</td>
      <td>[0.00623623]</td>
      <td>[0.0009985747]</td>
      <td>[1.7933434e-26]</td>
      <td>[1.388995e-27]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-05-01 20:03:37.993</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 27, 31, 29, 28, 15, 33, 29, 12, 36, 31, 12]</td>
      <td>[2.837503e-05]</td>
      <td>[1.2753118e-05]</td>
      <td>[0.025435215]</td>
      <td>[6.150966e-10]</td>
      <td>[2.321774e-10]</td>
      <td>[0.051351104]</td>
      <td>[0.022038758]</td>
      <td>[0.9885122]</td>
      <td>[0.023624167]</td>
      <td>[0.017496044]</td>
      <td>[0.07612714]</td>
      <td>[0.018284446]</td>
      <td>[0.00016227343]</td>
      <td>[2.9736e-26]</td>
      <td>[6.570557e-23]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-05-01 20:03:37.993</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 25, 21, 16, 22, 20, 19, 19, 28]</td>
      <td>[3.0770573e-07]</td>
      <td>[4.86675e-05]</td>
      <td>[0.036468606]</td>
      <td>[2.0693407e-15]</td>
      <td>[7.2607375e-18]</td>
      <td>[0.09667879]</td>
      <td>[0.073321395]</td>
      <td>[0.9993358]</td>
      <td>[0.0913113]</td>
      <td>[0.0527945]</td>
      <td>[2.7352993e-07]</td>
      <td>[0.041695543]</td>
      <td>[0.052203804]</td>
      <td>[4.6102867e-37]</td>
      <td>[3.6129874e-29]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-05-01 20:03:37.993</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 20, 22, 18, 32, 15, 12, 33, 17, 31, 14, 14, 27, 18]</td>
      <td>[8.8228285e-13]</td>
      <td>[3.5226062e-06]</td>
      <td>[0.100503676]</td>
      <td>[1.6081854e-09]</td>
      <td>[3.923381e-17]</td>
      <td>[0.15465459]</td>
      <td>[0.24250229]</td>
      <td>[0.99999857]</td>
      <td>[0.25655058]</td>
      <td>[0.13984609]</td>
      <td>[2.9986824e-05]</td>
      <td>[0.16115357]</td>
      <td>[0.038542073]</td>
      <td>[2.5434677e-31]</td>
      <td>[5.6750776e-37]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-05-01 20:03:37.993</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 22, 12, 30, 24, 13, 19, 25, 36, 28, 13, 12, 13]</td>
      <td>[5.4870607e-06]</td>
      <td>[0.0029785605]</td>
      <td>[0.0143616935]</td>
      <td>[1.9806076e-10]</td>
      <td>[3.051769e-10]</td>
      <td>[0.014699642]</td>
      <td>[0.03709711]</td>
      <td>[0.9984837]</td>
      <td>[0.036889926]</td>
      <td>[0.021504985]</td>
      <td>[0.0007605833]</td>
      <td>[0.017085439]</td>
      <td>[0.0009147275]</td>
      <td>[0.0]</td>
      <td>[8.360769e-30]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>2024-05-01 20:03:37.993</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 31, 35, 13, 14, 25, 23, 14, 21, 20, 32, 14, 32, 29, 16, 33]</td>
      <td>[2.0416806e-12]</td>
      <td>[7.744161e-09]</td>
      <td>[0.011983096]</td>
      <td>[8.3120476e-08]</td>
      <td>[7.3397146e-14]</td>
      <td>[0.123229906]</td>
      <td>[0.07365251]</td>
      <td>[0.9999754]</td>
      <td>[0.10082034]</td>
      <td>[0.057467848]</td>
      <td>[0.00017084264]</td>
      <td>[0.07321706]</td>
      <td>[0.0018467624]</td>
      <td>[3.284046e-36]</td>
      <td>[0.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>996</th>
      <td>2024-05-01 20:03:37.993</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 20, 29, 23, 14, 12, 27, 35, 34]</td>
      <td>[5.5039258e-11]</td>
      <td>[2.5695294e-07]</td>
      <td>[0.10195666]</td>
      <td>[6.0391613e-07]</td>
      <td>[5.471229e-18]</td>
      <td>[0.031199494]</td>
      <td>[0.15874198]</td>
      <td>[0.9999727]</td>
      <td>[0.10557272]</td>
      <td>[0.060710385]</td>
      <td>[1.0149461e-07]</td>
      <td>[0.034435842]</td>
      <td>[0.25794983]</td>
      <td>[8.326947e-27]</td>
      <td>[4.358661e-21]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>997</th>
      <td>2024-05-01 20:03:37.993</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 28, 19, 19, 28, 15, 12, 18]</td>
      <td>[0.011286308]</td>
      <td>[0.06214513]</td>
      <td>[0.09424207]</td>
      <td>[6.8554013e-16]</td>
      <td>[0.0031474212]</td>
      <td>[0.013521446]</td>
      <td>[0.0521531]</td>
      <td>[0.66066873]</td>
      <td>[0.039717037]</td>
      <td>[0.026172414]</td>
      <td>[0.015154914]</td>
      <td>[0.019907918]</td>
      <td>[0.040248252]</td>
      <td>[9.4758866e-27]</td>
      <td>[1.1339277e-21]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>998</th>
      <td>2024-05-01 20:03:37.993</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 12, 19, 35, 16, 35, 27, 16]</td>
      <td>[5.6315566e-06]</td>
      <td>[3.3642746e-06]</td>
      <td>[0.13612257]</td>
      <td>[5.6732154e-11]</td>
      <td>[2.7730684e-08]</td>
      <td>[0.0025221605]</td>
      <td>[0.05455697]</td>
      <td>[0.9998954]</td>
      <td>[0.03288219]</td>
      <td>[0.021166842]</td>
      <td>[0.000873619]</td>
      <td>[0.016495932]</td>
      <td>[0.014340238]</td>
      <td>[1.683203e-30]</td>
      <td>[4.5956004e-25]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>999</th>
      <td>2024-05-01 20:03:37.993</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 12, 28, 30, 19, 34, 26, 18, 31, 25, 13, 29, 17, 24, 29, 14, 36, 13]</td>
      <td>[1.3068625e-12]</td>
      <td>[1.1029468e-09]</td>
      <td>[0.014839977]</td>
      <td>[2.2757316e-08]</td>
      <td>[8.438438e-15]</td>
      <td>[0.30495816]</td>
      <td>[0.11627986]</td>
      <td>[0.99999803]</td>
      <td>[0.14364219]</td>
      <td>[0.10407086]</td>
      <td>[6.763152e-07]</td>
      <td>[0.12739345]</td>
      <td>[0.007844242]</td>
      <td>[2.8263536e-35]</td>
      <td>[0.0]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 18 columns</p>

## Cleaning up.

Now that the tutorial is complete, don't forget to undeploy your pipeline to free up the resources.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>aloha-fraud-detector</td></tr><tr><th>created</th> <td>2024-05-01 16:30:53.995114+00:00</td></tr><tr><th>last_updated</th> <td>2024-05-01 20:03:01.505745+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>8ede9ff5-e42a-404c-b248-fb4c4efd687d, 958397f7-1d3f-472c-8654-21283b8d6d85, ea59aae6-1565-4321-9086-7f8dd2a8e1c2, 41b9260e-21a8-4f16-ad56-c6267d3bae93, 46bdd5a3-fc22-41b7-b1ea-8287c99c241e, 28f5443a-ff67-4f4f-bfc3-f3f95f3c6f83, 435b73f7-76a1-4514-bd41-2cb94d3e78ff, 551242a7-fe4c-4a61-a4c7-e7fcc97509dc, d22eb0d2-9cff-4c5f-a851-10b1a19d8c44, 262909e9-8779-4c56-a994-725ddd0b58c8, 4cdf8e11-1b9c-44ab-a16d-abb054b5e9fe, 6b3529b1-1ff1-454b-8896-460c8c90d667</td></tr><tr><th>steps</th> <td>aloha-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Congratulations!

In this tutorial you have:

* Deployed a single step house price prediction pipeline and sent data to it.
* Create a new Wallaroo connection.
* Assigned the connection to a workspace.
* Retrieved the connection from the workspace.
* Used the data connection to retrieve information from outside of Wallaroo, and use it for an inference.

Great job!
