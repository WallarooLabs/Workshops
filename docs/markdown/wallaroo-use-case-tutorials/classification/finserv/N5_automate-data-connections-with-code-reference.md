# Tutorial Notebook 5: Automation with Wallaroo Connections

Wallaroo Connections are definitions set by MLOps engineers that are used by other Wallaroo users for connection information to a data source.

This provides MLOps engineers a method of creating and updating connection information for data stores:  databases, Kafka topics, etc.  Wallaroo Connections are composed of three main parts:

* Name:  The unique name of the connection.
* Type:  A user defined string that designates the type of connection.  This is used to organize connections.
* Details:  Details are a JSON object containing the information needed to make the connection.  This can include data sources, authentication tokens, etc.

Wallaroo Connections are only used to store the connection information used by other processes to create and use external connections.  The user still has to provide the libraries and other elements to actually make and use the conneciton.

The primary advantage is Wallaroo connections allow scripts and other code to retrieve the connection details directly from their Wallaroo instance, then refer to those connection details.  They don't need to know what those details actually - they can refer to them in their code to make their code more flexible.

For this step, we will use a link to a file stored in GitHub to retrieve the inference information, perform the inference and display the results.  This will use the Wallaroo Connection feature to create a Connection, assign it to our workspace, then perform our inferences by using the Connection details to connect and retrieve the data.

## Prerequisites

* A Wallaroo instance version 2024.2 or greater.

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

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
## blank space to log in 

wl = wallaroo.Client()
# retrieve the previous workspace, model, and pipeline version

workspace_name = "tutorial-finserv-john"

workspace = wl.get_workspace(workspace_name)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

# optionally, examine your current workspace
wl.get_current_workspace()

model_name = 'classification-finserv-prime'

prime_model_version = wl.get_model(model_name)

pipeline_name = 'ccfraud-detector'

pipeline = wl.get_pipeline(pipeline_name)

display(workspace)
display(prime_model_version)
display(pipeline)

```

    {'name': 'tutorial-finserv-john', 'id': 11, 'archived': False, 'created_by': '94016008-4e0e-45bc-b2c6-d6f06236b4f5', 'created_at': '2024-09-05T16:18:31.258882+00:00', 'models': [{'name': 'classification-finserv-prime', 'versions': 3, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 9, 5, 16, 48, 18, 348992, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 9, 5, 16, 18, 42, 470017, tzinfo=tzutc())}, {'name': 'ccfraud-xgboost-version', 'versions': 3, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 9, 5, 16, 53, 34, 714217, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 9, 5, 16, 32, 28, 543534, tzinfo=tzutc())}], 'pipelines': [{'name': 'assay-demonstration-tutorial', 'create_time': datetime.datetime(2024, 9, 5, 16, 18, 46, 604658, tzinfo=tzutc()), 'definition': '[]'}, {'name': 'ccfraud-detector', 'create_time': datetime.datetime(2024, 9, 5, 16, 18, 43, 626892, tzinfo=tzutc()), 'definition': '[]'}]}

<table>
        <tr>
          <td>Name</td>
          <td>classification-finserv-prime</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>24761304-0cb3-40e7-9462-d2a454637152</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>keras_ccfraud.onnx</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507</td>
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
          <td>2024-05-Sep 16:48:18</td>
        </tr>
        <tr>
          <td>Workspace id</td>
          <td>11</td>
        </tr>
        <tr>
          <td>Workspace name</td>
          <td>tutorial-finserv-john</td>
        </tr>
      </table>

<table><tr><th>name</th> <td>ccfraud-detector</td></tr><tr><th>created</th> <td>2024-09-05 16:18:43.626892+00:00</td></tr><tr><th>last_updated</th> <td>2024-09-05 17:41:14.081888+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>workspace_id</th> <td>11</td></tr><tr><th>workspace_name</th> <td>tutorial-finserv-john</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>9f9fc75d-9297-4de5-9962-1cd46e5006df, 93b4dd36-5e02-440d-931c-80198d1ee48a, 252d1e6f-35a5-420a-9590-d08cd76943a7, cb026715-9ced-40bb-9108-333b88c9de64, e90c10e3-ab38-43a8-a315-fb4250c09b21, 583e7a6a-3a7a-4420-abd0-c91e346a874d, cbfc4951-4d2c-41cb-89c7-934ac5bd2cbf, 32ab9ef6-2ac4-4d92-a46e-5d5c286af48c, 410d43d4-e698-4747-bbd1-cb62afee258a, 9edcf6f6-660d-470d-b8f0-24f54a335e8f, cd63b4fa-6549-41b4-af8f-576b1f0ef8b3, ff2d47a5-47c8-4fcb-8709-e95b3d0d4340, 8e74290b-4cb6-43a5-9b27-8431643438fd</td></tr><tr><th>steps</th> <td>classification-finserv-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Deploy the Pipeline with the Model Version Step

As per the other tutorials:

1. Clear the pipeline of all steps.
1. Add the model version as a pipeline step.
1. Deploy the pipeline with the following deployment configuration:

```python
deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
```

```python
# deploy model config

pipeline.clear()
pipeline.add_model_step(prime_model_version)

deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
```

```python
# deploy model

pipeline.deploy(deployment_config=deploy_config)
```

<table><tr><th>name</th> <td>ccfraud-detector</td></tr><tr><th>created</th> <td>2024-09-05 16:18:43.626892+00:00</td></tr><tr><th>last_updated</th> <td>2024-09-09 18:40:24.827905+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>workspace_id</th> <td>11</td></tr><tr><th>workspace_name</th> <td>tutorial-finserv-john</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>8cd86867-a263-4d36-b81b-ee6cd2fe7eeb, dae462d4-8290-46f1-be9c-65b4840953ea, 9f9fc75d-9297-4de5-9962-1cd46e5006df, 93b4dd36-5e02-440d-931c-80198d1ee48a, 252d1e6f-35a5-420a-9590-d08cd76943a7, cb026715-9ced-40bb-9108-333b88c9de64, e90c10e3-ab38-43a8-a315-fb4250c09b21, 583e7a6a-3a7a-4420-abd0-c91e346a874d, cbfc4951-4d2c-41cb-89c7-934ac5bd2cbf, 32ab9ef6-2ac4-4d92-a46e-5d5c286af48c, 410d43d4-e698-4747-bbd1-cb62afee258a, 9edcf6f6-660d-470d-b8f0-24f54a335e8f, cd63b4fa-6549-41b4-af8f-576b1f0ef8b3, ff2d47a5-47c8-4fcb-8709-e95b3d0d4340, 8e74290b-4cb6-43a5-9b27-8431643438fd</td></tr><tr><th>steps</th> <td>classification-finserv-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Create the Connection

For this demonstration, the connection set to a specific file on a GitHub repository.  The connection details can be anything that can be stored in JSON:  connection URLs, tokens, etc.

This connection will set a URL to pull a file from GitHub, then use the file contents to perform an inference.

Wallaroo connections are created through the Wallaroo Client `create_connection(name, type, details)` method.  See the [Wallaroo SDK Essentials Guide: Data Connections Management guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-dataconnections/) for full details.

Note that connection names must be unique across the Wallaroo instance - if needed, use random characters at the end to make sure your connection doesn't have the same name as a previously created connection.

Here's an example connection used to retrieve the same CSV file used in `./data/cc_data_1k.df.json`:  https://raw.githubusercontent.com/WallarooLabs/Tutorials/main/Classification/FinServ/data/cc_data_1k.df.json

### Create the Connection Exercise

```python
# set the connection information for other steps
# suffix is used to create a unique data connection

ccfraud_connection_input_name = f'ccfraud-test-data'
ccfraud_connection_input_type = "HTTP"
ccfraud_connection_input_argument = { 
    "url": "https://raw.githubusercontent.com/WallarooLabs/Tutorials/main/Classification/FinServ/data/cc_data_1k.df.json"
    }

wl.create_connection(ccfraud_connection_input_name, ccfraud_connection_input_type, ccfraud_connection_input_argument)
```

```python
# set the connection information for other steps
# suffix is used to create a unique data connection

suffix='john'

ccfraud_connection_input_name = f'ccfraud-test-data'
ccfraud_connection_input_type = "HTTP"
ccfraud_connection_input_argument = { 
    "url": "https://raw.githubusercontent.com/WallarooLabs/Tutorials/main/Classification/FinServ/data/cc_data_1k.df.json"
    }

wl.create_connection(ccfraud_connection_input_name, ccfraud_connection_input_type, ccfraud_connection_input_argument)
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>ccfraud-test-data</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2024-09-09T18:40:39.478062+00:00</td>
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

<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th><th>linked workspaces</th></tr><tr><td>ccfraud-test-data</td><td>HTTP</td><td>*****</td><td>2024-09-09T18:40:39.478062+00:00</td><td>[]</td></tr></table>

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
wl.get_connection(ccfraud_connection_input_name)
```

```python
# get the connection by name

this_connection = wl.get_connection(ccfraud_connection_input_name)
this_connection
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>ccfraud-test-data</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2024-09-09T18:40:39.478062+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>[]</td>
  </tr>
</table>

## Add Connection to Workspace

We'll now add the connection to our workspace so it can be retrieved by other workspace users.  The method Workspace `add_connection(connection_name)` adds a Data Connection to a workspace.  The method Workspace `list_connections()` displays a list of connections attached to the workspace.

### Add Connection to Workspace Exercise

Use the connection we just created, and add it to the sample workspace.  Here's a code example where the workspace is saved to the variable `workspace` and the connection is saved as `ccfraud_connection_input_name`.

```python
workspace.add_connection(ccfraud_connection_input_name)
```

```python
# add connection

workspace.add_connection(ccfraud_connection_input_name)
workspace.list_connections()
```

<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th><th>linked workspaces</th></tr><tr><td>ccfraud-test-data</td><td>HTTP</td><td>*****</td><td>2024-09-09T18:40:39.478062+00:00</td><td>['tutorial-finserv-john']</td></tr></table>

## Retrieve Connection from Workspace

To simulate a data scientist's procedural flow, we'll now retrieve the connection from the workspace.  Specific connections are retrieved by specifying their position in the returned list.

For example, if we have two connections in a workspace and we want the second one, we can assign it to a variable with `list_connections[1]`.

Create a new variable and retrieve the connection we just assigned to the workspace.

### Retrieve Connection from Workspace Exercise

Retrieve the connection that was just associated with the workspace.  You'll use the `list_connections` method, then assign a variable to the connection.  Here's an example if the connection is the most recently one added to the workspace `workspace`.

```python
ccfraud_connection = workspace.list_connections()[-1]
```

```python
# get connection

ccfraud_connection = workspace.list_connections()[-1]
display(ccfraud_connection)
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>ccfraud-test-data</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2024-09-09T18:40:39.478062+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>['tutorial-finserv-john']</td>
  </tr>
</table>

## Run Inference with Connection

Connections can be used for different purposes:  uploading new models, engine configurations - any place that data is needed.  This exercise will use the data connection to perform an inference through our deployed pipeline.

### Run Inference with Connection Exercise

We'll now retrieve sample data through the Wallaroo connection, and perform a sample inference.  The connection details are retrieved through the Connection `details()` method.  Use them to retrieve the pandas record file and convert it to a DataFrame, and use it with our sample model.

Here's a code example that uses the Python `requests` library to retrieve the file information, then turns it into a DataFrame for the inference request.

```python
display(ccfraud_connection.details()['url'])

import requests

response = requests.get(
                    ccfraud_connection.details()['url']
                )

# display(response.json())

df = pd.DataFrame(response.json())

pipeline.infer(df)
```

```python
# inference using connection

display(ccfraud_connection.details()['url'])

import requests

response = requests.get(
                    ccfraud_connection.details()['url']
                )

# display(response.json())

df = pd.DataFrame(response.json())
display(df)

multiple_result = pipeline.infer(df)
display(multiple_result)
```

    'https://raw.githubusercontent.com/WallarooLabs/Tutorials/main/Classification/FinServ/data/cc_data_1k.df.json'

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
      <td>[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[0.5817662108, 0.097881551, 0.1546819424, 0.4754101949, -0.1978862306, -0.4504344854, 0.0166540447, -0.0256070551, 0.0920561602, -0.2783917153, 0.0593299441, -0.0196585416, -0.4225083157, -0.1217538877, 1.5473094894, 0.2391622864, 0.3553974881, -0.7685165301, -0.7000849355, -0.1190043285, -0.3450517133, -1.1065114108, 0.2523411195, 0.0209441826, 0.2199267436, 0.2540689265, -0.0450225094, 0.1086773898, 0.2547179311]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>996</th>
      <td>[1.052355506, -0.7602601059, -0.3124601687, -0.5580714587, -0.6198353331, 0.6635428464, -1.2171685083, 0.3144529308, 0.2360632058, 0.878209955, -0.5518803042, -0.2781328417, -0.5675947058, -0.0982688053, 0.1475098349, -0.3097481612, -1.0898892231, 2.804466934, -0.4211447753, -0.7315488305, -0.5311840374, -0.9053830525, 0.5382443229, -0.68327623, -1.1848642272, 0.9872236995, -0.0260721428, -0.1405966468, 0.0759031399]</td>
    </tr>
    <tr>
      <th>997</th>
      <td>[-0.8464537996, -0.7608807925, 2.186072883, -0.1614362994, -0.4069378894, 0.734079177, -0.4611705734, 0.4751492626, 1.4952832213, -0.9349105827, -0.7654272171, 0.4362793613, -0.6623354486, -1.5326388376, -1.4311992842, -1.0573215483, 0.9304904478, -1.2836000946, -1.079419331, 0.7138847264, 0.2710369668, 1.1943291742, 0.2527110226, 0.3107779567, 0.4219366694, 2.4854295825, 0.1754876037, -0.2362979978, 0.9979986569]</td>
    </tr>
    <tr>
      <th>998</th>
      <td>[1.0046377125, 0.0343666504, -1.3512533246, 0.4160460291, 0.5910548281, -0.8187740907, 0.5840864966, -0.447623496, 1.1193896296, -0.1156579903, 0.1298919303, -2.6410683948, 1.1658091033, 2.3607999565, -0.4265055896, -0.4862102299, 0.5102253659, -0.3384745171, -0.4081285365, -0.199414607, 0.0151691668, 0.2644673476, -0.0483547565, 0.9869714364, 0.629627219, 0.8990505678, -0.3731273846, -0.2166148809, 0.6374669208]</td>
    </tr>
    <tr>
      <th>999</th>
      <td>[0.4951101913, -0.2499369449, 0.4553345161, 0.9242750451, -0.3643510229, 0.602688482, -0.3785553207, 0.3170957153, 0.7368986387, -0.1195106678, 0.4017042912, 0.7371143425, -1.2229791154, 0.0061993212, -1.3541149574, -0.5839052891, 0.1648461272, -0.1527212037, 0.2456232399, -0.1432012313, -0.0383696111, 0.0865420131, -0.284099885, -0.5027591867, 1.1117147574, -0.5666540195, 0.121220185, 0.0667640208, 0.6583281816]</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>[0.6118805301, 0.1726081102, 0.4310545502, 0.5032148221, -0.2746663262, -0.464798859, -0.1098384885, -0.0978937224, 0.9820529526, -0.2237381949, 2.3315375168, -1.5852745605, 1.6050692254, 1.9720759474, -0.4217479714, 0.5348796175, 0.0875849983, 0.3280840192, -0.0394716814, -0.1796805095, -0.4955020407, -1.1889449446, 0.246698494, 0.4185131811, 0.3026018698, 0.0812114542, -0.1557850823, 0.0171892918, -0.7236631158]</td>
    </tr>
  </tbody>
</table>
<p>1001 rows × 1 columns</p>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-09-09 18:40:41.582</td>
      <td>[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-09-09 18:40:41.582</td>
      <td>[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-09-09 18:40:41.582</td>
      <td>[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-09-09 18:40:41.582</td>
      <td>[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-09-09 18:40:41.582</td>
      <td>[0.5817662108, 0.097881551, 0.1546819424, 0.4754101949, -0.1978862306, -0.4504344854, 0.0166540447, -0.0256070551, 0.0920561602, -0.2783917153, 0.0593299441, -0.0196585416, -0.4225083157, -0.1217538877, 1.5473094894, 0.2391622864, 0.3553974881, -0.7685165301, -0.7000849355, -0.1190043285, -0.3450517133, -1.1065114108, 0.2523411195, 0.0209441826, 0.2199267436, 0.2540689265, -0.0450225094, 0.1086773898, 0.2547179311]</td>
      <td>[0.0010916889]</td>
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
      <th>996</th>
      <td>2024-09-09 18:40:41.582</td>
      <td>[1.052355506, -0.7602601059, -0.3124601687, -0.5580714587, -0.6198353331, 0.6635428464, -1.2171685083, 0.3144529308, 0.2360632058, 0.878209955, -0.5518803042, -0.2781328417, -0.5675947058, -0.0982688053, 0.1475098349, -0.3097481612, -1.0898892231, 2.804466934, -0.4211447753, -0.7315488305, -0.5311840374, -0.9053830525, 0.5382443229, -0.68327623, -1.1848642272, 0.9872236995, -0.0260721428, -0.1405966468, 0.0759031399]</td>
      <td>[0.00011596084]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>997</th>
      <td>2024-09-09 18:40:41.582</td>
      <td>[-0.8464537996, -0.7608807925, 2.186072883, -0.1614362994, -0.4069378894, 0.734079177, -0.4611705734, 0.4751492626, 1.4952832213, -0.9349105827, -0.7654272171, 0.4362793613, -0.6623354486, -1.5326388376, -1.4311992842, -1.0573215483, 0.9304904478, -1.2836000946, -1.079419331, 0.7138847264, 0.2710369668, 1.1943291742, 0.2527110226, 0.3107779567, 0.4219366694, 2.4854295825, 0.1754876037, -0.2362979978, 0.9979986569]</td>
      <td>[0.0002785325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>998</th>
      <td>2024-09-09 18:40:41.582</td>
      <td>[1.0046377125, 0.0343666504, -1.3512533246, 0.4160460291, 0.5910548281, -0.8187740907, 0.5840864966, -0.447623496, 1.1193896296, -0.1156579903, 0.1298919303, -2.6410683948, 1.1658091033, 2.3607999565, -0.4265055896, -0.4862102299, 0.5102253659, -0.3384745171, -0.4081285365, -0.199414607, 0.0151691668, 0.2644673476, -0.0483547565, 0.9869714364, 0.629627219, 0.8990505678, -0.3731273846, -0.2166148809, 0.6374669208]</td>
      <td>[0.0011070371]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>999</th>
      <td>2024-09-09 18:40:41.582</td>
      <td>[0.4951101913, -0.2499369449, 0.4553345161, 0.9242750451, -0.3643510229, 0.602688482, -0.3785553207, 0.3170957153, 0.7368986387, -0.1195106678, 0.4017042912, 0.7371143425, -1.2229791154, 0.0061993212, -1.3541149574, -0.5839052891, 0.1648461272, -0.1527212037, 0.2456232399, -0.1432012313, -0.0383696111, 0.0865420131, -0.284099885, -0.5027591867, 1.1117147574, -0.5666540195, 0.121220185, 0.0667640208, 0.6583281816]</td>
      <td>[0.0008533001]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>2024-09-09 18:40:41.582</td>
      <td>[0.6118805301, 0.1726081102, 0.4310545502, 0.5032148221, -0.2746663262, -0.464798859, -0.1098384885, -0.0978937224, 0.9820529526, -0.2237381949, 2.3315375168, -1.5852745605, 1.6050692254, 1.9720759474, -0.4217479714, 0.5348796175, 0.0875849983, 0.3280840192, -0.0394716814, -0.1796805095, -0.4955020407, -1.1889449446, 0.246698494, 0.4185131811, 0.3026018698, 0.0812114542, -0.1557850823, 0.0171892918, -0.7236631158]</td>
      <td>[0.0012498498]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1001 rows × 4 columns</p>

## Cleaning up.

Now that the tutorial is complete, don't forget to undeploy your pipeline to free up the resources.

```python
# undeploy the pipeline

pipeline.undeploy()
```

<table><tr><th>name</th> <td>ccfraud-detector</td></tr><tr><th>created</th> <td>2024-09-05 16:18:43.626892+00:00</td></tr><tr><th>last_updated</th> <td>2024-09-09 18:40:24.827905+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>workspace_id</th> <td>11</td></tr><tr><th>workspace_name</th> <td>tutorial-finserv-john</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>8cd86867-a263-4d36-b81b-ee6cd2fe7eeb, dae462d4-8290-46f1-be9c-65b4840953ea, 9f9fc75d-9297-4de5-9962-1cd46e5006df, 93b4dd36-5e02-440d-931c-80198d1ee48a, 252d1e6f-35a5-420a-9590-d08cd76943a7, cb026715-9ced-40bb-9108-333b88c9de64, e90c10e3-ab38-43a8-a315-fb4250c09b21, 583e7a6a-3a7a-4420-abd0-c91e346a874d, cbfc4951-4d2c-41cb-89c7-934ac5bd2cbf, 32ab9ef6-2ac4-4d92-a46e-5d5c286af48c, 410d43d4-e698-4747-bbd1-cb62afee258a, 9edcf6f6-660d-470d-b8f0-24f54a335e8f, cd63b4fa-6549-41b4-af8f-576b1f0ef8b3, ff2d47a5-47c8-4fcb-8709-e95b3d0d4340, 8e74290b-4cb6-43a5-9b27-8431643438fd</td></tr><tr><th>steps</th> <td>classification-finserv-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Congratulations!

In this tutorial you have:

* Deployed a single step house price prediction pipeline and sent data to it.
* Create a new Wallaroo connection.
* Assigned the connection to a workspace.
* Retrieved the connection from the workspace.
* Used the data connection to retrieve information from outside of Wallaroo, and use it for an inference.

Great job!
