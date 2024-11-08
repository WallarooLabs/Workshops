# Tutorial Notebook 2: Automation with Wallaroo Connections

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
# run this to load the libraries

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

### Pre-exercise

If needed, log into Wallaroo and go to the workspace, pipeline, and most recent model version from the ones that you created in the previous notebook. Please refer to Notebook 1 to refresh yourself on how to log in and set your working environment to the appropriate workspace.

```python
## blank space to log in 

wl = wallaroo.Client()

# retrieve the previous workspace, model, and pipeline version

workspace_name = "tutorial-workspace-summarization"

workspace = wl.get_workspace(name=workspace_name, create_if_not_exist=True)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

model_name = 'hf-summarizer'

prime_model_version = wl.get_model(model_name)

pipeline_name = 'hf-summarizer'

pipeline = wl.get_pipeline(pipeline_name)

# verify the workspace/pipeline/model

display(wl.get_current_workspace())
display(prime_model_version)
display(pipeline)

```

    {'name': 'tutorial-workspace-summarization', 'id': 7, 'archived': False, 'created_by': 'fca5c4df-37ac-4a78-9602-dd09ca72bc60', 'created_at': '2024-10-29T19:40:06.545232+00:00', 'models': [{'name': 'hf-summarizer', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 10, 29, 19, 51, 19, 477912, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 10, 29, 19, 51, 19, 477912, tzinfo=tzutc())}], 'pipelines': [{'name': 'hf-summarizer', 'create_time': datetime.datetime(2024, 10, 29, 19, 53, 41, 301319, tzinfo=tzutc()), 'definition': '[]'}]}

<table>
        <tr>
          <td>Name</td>
          <td>hf-summarizer</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>8b7a1615-c6bf-47bc-b947-ce4a183cd1be</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>model-auto-conversion_hugging-face_complex-pipelines_hf-summarisation-bart-large-samsun.zip</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>ee71d066a83708e7ca4a3c07caf33fdc528bb000039b6ca2ef77fa2428dc6268</td>
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
          <td>2024-29-Oct 19:53:40</td>
        </tr>
        <tr>
          <td>Workspace id</td>
          <td>7</td>
        </tr>
        <tr>
          <td>Workspace name</td>
          <td>tutorial-workspace-summarization</td>
        </tr>
      </table>

<table><tr><th>name</th> <td>hf-summarizer</td></tr><tr><th>created</th> <td>2024-10-29 19:53:41.301319+00:00</td></tr><tr><th>last_updated</th> <td>2024-10-29 19:54:27.834673+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>workspace_id</th> <td>7</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-summarization</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>74db0fd3-f1b6-429c-a082-04e2aedcc4e6, e216b612-9b34-44c7-9a02-6812b2b8838d</td></tr><tr><th>steps</th> <td>hf-summarizer</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Deploy the Pipeline with the Model Version Step

As per the other tutorials:

1. Clear the pipeline of all steps.
1. Add the model version as a pipeline step.
1. Deploy the pipeline with the following deployment configuration:

```python
deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
```

```python
# run this to set the pipeline deployment

deployment_config = wallaroo.DeploymentConfigBuilder() \
    .cpus(0.25).memory('1Gi') \
    .sidekick_cpus(prime_model_version, 1) \
    .sidekick_memory(prime_model_version, "4Gi") \
    .build()
```

```python
## blank space to deploy the model

pipeline.clear()
pipeline.add_model_step(prime_model_version)

pipeline.deploy(deployment_config=deployment_config)
```

```python
## blank space to check the deployment status

pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.28.2.10',
       'name': 'engine-7bb85686b5-x7vml',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'hf-summarizer',
          'status': 'Running',
          'version': 'e57078b2-0190-4f55-8a6c-4cfdba2c63a6'}]},
       'model_statuses': {'models': [{'name': 'hf-summarizer',
          'sha': 'ee71d066a83708e7ca4a3c07caf33fdc528bb000039b6ca2ef77fa2428dc6268',
          'status': 'Running',
          'version': '8b7a1615-c6bf-47bc-b947-ce4a183cd1be'}]}}],
     'engine_lbs': [{'ip': '10.28.2.9',
       'name': 'engine-lb-6676794678-qx6zj',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.28.2.8',
       'name': 'engine-sidekick-hf-summarizer-5-7fb7c55f75-pkc2k',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

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

# set the connection information for other steps
# suffix is used to create a unique data connection
suffix=''
summary_connection_input_name = f'summary-sample-connection{suffix}'
summary_connection_input_type = "HTTP"
summary_connection_input_argument = { 
    "url": "https://raw.githubusercontent.com/WallarooLabs/Tutorials/20230927-foundations-v2/LLM/Summarization/data/test_summarization.df.json"
    }

wl.create_connection(summary_connection_input_name, summary_connection_input_type, summary_connection_input_argument)
```

```python
## blank space to set the connection information for other steps
# suffix is used to create a unique data connection
suffix=''
summary_connection_input_name = f'summary-sample-connection{suffix}'
summary_connection_input_type = "HTTP"
summary_connection_input_argument = { 
    "url": "https://raw.githubusercontent.com/WallarooLabs/Tutorials/20230927-foundations-v2/LLM/Summarization/data/test_summarization.df.json"
    }

wl.create_connection(summary_connection_input_name, summary_connection_input_type, summary_connection_input_argument)
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>summary-sample-connection</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2024-10-29T20:33:12.209391+00:00</td>
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
## blank space to list the connections here

wl.list_connections()
```

<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th><th>linked workspaces</th></tr><tr><td>summary-sample-connection</td><td>HTTP</td><td>*****</td><td>2024-10-29T20:33:12.209391+00:00</td><td>[]</td></tr></table>

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
## blank space to get the connection by name

this_connection = wl.get_connection(summary_connection_input_name)
this_connection
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>summary-sample-connection</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2024-10-29T20:33:12.209391+00:00</td>
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
## blank space to add connections to the workspace

workspace.add_connection(summary_connection_input_name)
workspace.list_connections()
```

<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th><th>linked workspaces</th></tr><tr><td>summary-sample-connection</td><td>HTTP</td><td>*****</td><td>2024-10-29T20:33:12.209391+00:00</td><td>['tutorial-workspace-summarization']</td></tr></table>

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
## blank space to list connection code

forecast_connection = workspace.list_connections()[-1]
display(forecast_connection)
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>summary-sample-connection</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2024-10-29T20:33:12.209391+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>['tutorial-workspace-summarization']</td>
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
## blank space to perform inference with connection

display(forecast_connection.details()['url'])

import requests

response = requests.get(
                    forecast_connection.details()['url']
                )

# display(response.json())

df = pd.DataFrame(response.json())
display(df)

multiple_result = pipeline.infer(df, timeout=60)
display(multiple_result)
```

    'https://raw.githubusercontent.com/WallarooLabs/Tutorials/20230927-foundations-v2/LLM/Summarization/data/test_summarization.df.json'

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>inputs</th>
      <th>return_text</th>
      <th>return_tensors</th>
      <th>clean_up_tokenization_spaces</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LinkedIn (/lɪŋktˈɪn/) is a business and employment-focused social media platform that works through websites and mobile apps. It launched on May 5, 2003. It is now owned by Microsoft. The platform is primarily used for professional networking and career development, and allows jobseekers to post their CVs and employers to post jobs. From 2015 most of the company's revenue came from selling access to information about its members to recruiters and sales professionals. Since December 2016, it has been a wholly owned subsidiary of Microsoft. As of March 2023, LinkedIn has more than 900 million registered members from over 200 countries and territories. LinkedIn allows members (both workers and employers) to create profiles and connect with each other in an online social network which may represent real-world professional relationships. Members can invite anyone (whether an existing member or not) to become a connection. LinkedIn can also be used to organize offline events, join groups, write articles, publish job postings, post photos and videos, and more</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.clean_up_tokenization_spaces</th>
      <th>in.inputs</th>
      <th>in.return_tensors</th>
      <th>in.return_text</th>
      <th>out.summary_text</th>
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-10-29 20:33:19.512</td>
      <td>False</td>
      <td>LinkedIn (/lɪŋktˈɪn/) is a business and employment-focused social media platform that works through websites and mobile apps. It launched on May 5, 2003. It is now owned by Microsoft. The platform is primarily used for professional networking and career development, and allows jobseekers to post their CVs and employers to post jobs. From 2015 most of the company's revenue came from selling access to information about its members to recruiters and sales professionals. Since December 2016, it has been a wholly owned subsidiary of Microsoft. As of March 2023, LinkedIn has more than 900 million registered members from over 200 countries and territories. LinkedIn allows members (both workers and employers) to create profiles and connect with each other in an online social network which may represent real-world professional relationships. Members can invite anyone (whether an existing member or not) to become a connection. LinkedIn can also be used to organize offline events, join groups, write articles, publish job postings, post photos and videos, and more</td>
      <td>False</td>
      <td>True</td>
      <td>LinkedIn is a business and employment-focused social media platform that works through websites and mobile apps. It launched on May 5, 2003. LinkedIn allows members (both workers and employers) to create profiles and connect with each other in an online social network which may represent real-world professional relationships.</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

## Cleaning up.

Now that the tutorial is complete, don't forget to undeploy your pipeline to free up the resources.

```python
## blank space to undeploy the pipeline

pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s .................................... ok

<table><tr><th>name</th> <td>hf-summarizer</td></tr><tr><th>created</th> <td>2024-10-29 19:53:41.301319+00:00</td></tr><tr><th>last_updated</th> <td>2024-10-29 20:31:07.050434+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>workspace_id</th> <td>7</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-summarization</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>e57078b2-0190-4f55-8a6c-4cfdba2c63a6, 74db0fd3-f1b6-429c-a082-04e2aedcc4e6, e216b612-9b34-44c7-9a02-6812b2b8838d</td></tr><tr><th>steps</th> <td>hf-summarizer</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Congratulations!

In this tutorial you have:

* Deployed a single step house price prediction pipeline and sent data to it.
* Create a new Wallaroo connection.
* Assigned the connection to a workspace.
* Retrieved the connection from the workspace.
* Used the data connection to retrieve information from outside of Wallaroo, and use it for an inference.

Great job!
