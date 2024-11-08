# Tutorial Notebook 4: Automation with ML Workload Orchestrations

Wallaroo provides Data Connections and ML Workload Orchestrations to provide organizations with a method of creating and managing automated tasks that can either be run on demand or a regular schedule.

## Prerequisites

* A Wallaroo instance version 2023.2.1 or greater.

## References

* [Wallaroo SDK Essentials Guide: Model Uploads and Registrations: Python Models](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-python/)
* [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/)
* [Wallaroo SDK Essentials Guide: ML Workload Orchestration](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-ml-workload-orchestration/)

## Orchestrations, Taks, and Tasks Runs

We've details how Wallaroo Connections work.  Now we'll use Orchestrations, Tasks, and Task Runs.

| Item | Description |
|---|---|
| Orchestration | ML Workload orchestration allows data scientists and ML Engineers to automate and scale production ML workflows in Wallaroo to ensure a tight feedback loop and continuous tuning of models from training to production. Wallaroo platform users (data scientists or ML Engineers) have the ability to deploy, automate and scale recurring batch production ML workloads that can ingest data from predefined data sources to run inferences in Wallaroo, chain pipelines, and send inference results to predefined destinations to analyze model insights and assess business outcomes. |
| Task | An implementation of an Orchestration.  Tasks can be either `Run Once`:  They run once and upon completion, stop. `Run Scheduled`: The task runs whenever a specific `cron` like schedule is reached.  Scheduled tasks will run until the `kill` command is issued. |
| Task Run | The execusion of a task.  For `Run Once` tasks, there will be only one `Run Task`.  A `Run Scheduled` tasks will have multiple tasks, one for every time the schedule parameter is met.  Task Runs have their own log files that can be examined to track progress and results. |

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

<table><tr><th>name</th> <td>houseprice-estimator</td></tr><tr><th>created</th> <td>2023-09-11 21:40:29.871251+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-13 16:54:08.320167+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>aa8b92a2-a2b9-4197-9fbd-63ad84e48e52, 4d3230a4-32be-4f47-90b5-d8d3b90cc2a9, 6f788da6-17ee-4df6-b120-354737212559, 7ae39ec3-1486-4bd7-9fa5-6874ee79f245, 11dd78db-9461-4ec0-9003-29538d4c242d, 8ca7a6bf-6529-434f-a3f6-bbc3bcc255d9</td></tr><tr><th>steps</th> <td>house-price-prime</td></tr></table>

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

<table><tr><th>name</th> <td>houseprice-estimator</td></tr><tr><th>created</th> <td>2023-09-11 21:40:29.871251+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-13 17:32:43.282786+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>9d80efd3-ba4b-409e-88e9-e0605db336e3, aa8b92a2-a2b9-4197-9fbd-63ad84e48e52, 4d3230a4-32be-4f47-90b5-d8d3b90cc2a9, 6f788da6-17ee-4df6-b120-354737212559, 7ae39ec3-1486-4bd7-9fa5-6874ee79f245, 11dd78db-9461-4ec0-9003-29538d4c242d, 8ca7a6bf-6529-434f-a3f6-bbc3bcc255d9</td></tr><tr><th>steps</th> <td>house-price-prime</td></tr></table>

### Sample Inference

Verify the pipeline is deployed properly with a sample inference with the file `./data/test_data.df.json`.

```python
# sample inference from previous code here

pipeline.infer_from_file('../data/test_data.df.json')
```

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
      <td>2023-09-13 17:32:48.775</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[659806.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-09-13 17:32:48.775</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[732883.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-09-13 17:32:48.775</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[419508.84]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-09-13 17:32:48.775</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[634028.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-09-13 17:32:48.775</td>
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
      <td>2023-09-13 17:32:48.775</td>
      <td>[4.0, 2.25, 2620.0, 98881.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1820.0, 800.0, 47.4662, -122.453, 1728.0, 95832.0, 63.0, 0.0, 0.0]</td>
      <td>[436151.13]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3996</th>
      <td>2023-09-13 17:32:48.775</td>
      <td>[3.0, 2.5, 2244.0, 4079.0, 2.0, 0.0, 0.0, 3.0, 7.0, 2244.0, 0.0, 47.2606, -122.254, 2077.0, 4078.0, 3.0, 0.0, 0.0]</td>
      <td>[284810.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3997</th>
      <td>2023-09-13 17:32:48.775</td>
      <td>[3.0, 1.75, 1490.0, 5000.0, 1.0, 0.0, 1.0, 3.0, 8.0, 1250.0, 240.0, 47.5257, -122.392, 1980.0, 5000.0, 61.0, 0.0, 0.0]</td>
      <td>[575571.44]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3998</th>
      <td>2023-09-13 17:32:48.775</td>
      <td>[4.0, 2.5, 2740.0, 5700.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2740.0, 0.0, 47.3535, -122.026, 3010.0, 5281.0, 8.0, 0.0, 0.0]</td>
      <td>[432262.38]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3999</th>
      <td>2023-09-13 17:32:48.775</td>
      <td>[5.0, 2.5, 2240.0, 7770.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1340.0, 900.0, 47.7198, -122.171, 1820.0, 7770.0, 36.0, 0.0, 0.0]</td>
      <td>[445873.13]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>4000 rows × 4 columns</p>

## Sample Orchestration

The orchestration that will automate this process is `./orchestration/real-estate-orchestration.zip`.  The files used are stored in the directory `/orchestration/real-estate-orchestration`, created with the command:

`zip -r real-estate-orchestration.zip real-estate-orchestration/*`.

This contains the following:

* `requirements.txt`:  The Python requirements file to specify the following libraries used.  For this example, that will be empty since we will be using the 
* `main.py`: The entry file that uses a deployed pipeline and performs an inference request against it visible from its log files.
* `data/`: Inference data sources.

The `main.py` script performs a workspace and pipeline retrieval, then an inference against the inference input file.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError
import pandas as pd

wl = wallaroo.Client()

# get the arguments
arguments = wl.task_args()

if "workspace_name" in arguments:
    workspace_name = arguments['workspace_name']
else:
    workspace_name="forecast-model-tutorial"

if "pipeline_name" in arguments:
    pipeline_name = arguments['pipeline_name']
else:
    pipeline_name="bikedaypipe"

def get_workspace(name):
    workspace = None
    for ws in wl.list_workspaces():
        if ws.name() == name:
            workspace= ws
    return workspace

def get_pipeline(pipeline_name, workspace):
    plist = workspace.pipelines()
    pipeline = [p for p in plist if p.name() == pipeline_name]
    if len(pipeline) <= 0:
        raise KeyError(f"Pipeline {pipeline_name} not found in this workspace")
        return None
    return pipeline[0]

# pull a single datum from a data frame 
# and convert it to the format the model expects
def get_singleton(df, i):
    singleton = df.iloc[i,:].to_numpy().tolist()
    sdict = {'tensor': [singleton]}
    return pd.DataFrame.from_dict(sdict)

print(f"Workspace: {workspace_name}")
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)
print(workspace)

# the pipeline is assumed to be deployed
print(f"Pipeline: {pipeline_name}")
pipeline = get_pipeline(pipeline_name, workspace)
print(pipeline)

print(pipeline.status())

inference_result = pipeline.infer_file_file('./data/test_data.df.json')
print(inference_result)

```

A few things to go over here.  You'll notice this is almost the exact procedures we've been following so far:  we get a workspace and pipeline, pull data from a CSV file, and perform an inference off the data.

This script assumes that the pipeline has already been deployed, and also includes this part:

`arguments = wl.task_args()`

This allows us to pass arguments into a Task created from an Orchestration, so we can specify a different workspace, pipeline, or any other arguments we construct.  This allows orchestrations to be very flexible.

Also, notice that it refers to a specific file:

`inference_result = pipeline.infer_file_file('./data/test_data.df.json')`

In the `forecast-orchestration` directory is the `data` directory with our sample CSV file.  Orchestrations can include additional artifacts.  We could have used a Wallaroo Connection instead, and we encourage you to try that if you want.

## Upload Orchestration

Orchestrations are uploaded with the Wallaroo client `upload_orchestration(path)` method with the following parameters.

| Parameter | Type | Description |
| --- | --- | ---|
| **path** | string (Required) | The path to the ZIP file to be uploaded. |

Once uploaded, the deployment will be prepared and any requirements will be downloaded and installed.  A typical orchestration upload looks like this:

```python
my_orchestration = wl.upload_orchestration(path-to-zip-file)
```

### Upload Orchestration Exercise

Try uploading our orchestration from `./forecast-orchestration/forecast-orchestration.zip` - or make your own and upload it.

Once uploaded, you can check the status with the `status()`.  If using the orchestration example above, that would be `my_orchestration.status()`  This is handy to make into a loop to check the status until is shows `ready`.

Here's an example of uploading the Orchestration file, then a loop that will keep checking the status every 5 seconds until it returns `ready`.

```python
orchestration = wl.upload_orchestration(name="my real example", path="./orchestration/real-estate-orchestration.zip")

while orchestration.status() != 'ready':
    print(orchestration.status())
    time.sleep(5)
```

```python
orchestration = wl.upload_orchestration(name="real estate 08", path="../orchestration/real-estate-orchestration.zip")

while orchestration.status() != 'ready':
    print(orchestration.status())
    time.sleep(5)
```

    pending_packaging
    pending_packaging
    packaging
    packaging
    packaging
    packaging
    packaging
    packaging
    packaging

## List Orchestrations

Orchestrations are listed with the Wallaroo Client `list_orchestrations()` method.  Orchestrations can be retrieved to a variable by allocated their position in the array - for example:  `orchestration = wl.list_orchestrations()[0]` would return the first orchestration on the list.

### List Orchestrations Exercise

List all of the orchestrations in your Wallaroo instance.  For example, if your client is saved to `wl`, here's some code that would work.

```python
wl.list_orchestrations()
```

```python
# list orchestration here

wl.list_orchestrations()
```

<table><tr><th>id</th><th>name</th><th>status</th><th>filename</th><th>sha</th><th>created at</th><th>updated at</th></tr><tr><td>41d4db24-b2b7-40df-b88b-3f124d9fc2fd</td><td>real estate 05</td><td>packaging</td><td>real-estate-orchestration.zip</td><td>61464c...fa7ae6</td><td>2023-13-Sep 16:58:47</td><td>2023-13-Sep 16:59:00</td></tr><tr><td>fa97c393-f1e8-4343-a117-f9afef0a1e56</td><td>real estate 06</td><td>packaging</td><td>real-estate-orchestration.zip</td><td>61464c...fa7ae6</td><td>2023-13-Sep 17:10:45</td><td>2023-13-Sep 17:10:57</td></tr><tr><td>cafb577e-e9a0-46cd-bcca-0b2a345a7b1b</td><td>real estate 07</td><td>ready</td><td>real-estate-orchestration.zip</td><td>dbc14b...ed866b</td><td>2023-13-Sep 17:32:50</td><td>2023-13-Sep 17:33:39</td></tr><tr><td>2f606e70-89bf-49b2-9ab0-c9720f1cca0a</td><td>real estate 08</td><td>ready</td><td>real-estate-orchestration.zip</td><td>fcc312...7f6437</td><td>2023-13-Sep 17:36:06</td><td>2023-13-Sep 17:36:54</td></tr></table>

## Retrieve Orchestration from List

The command `wallaroo.client.list_orchestrations()` returns a List of orchestrations.  We can assign any of the orchestrations in the list to a variable, then use that for other commands.

## Retrieve Orchestration from List Exercise

Use the `list_orchestrations` command and store the orchestration we just uploaded.

Here's some sample code to get you started that stores the last orchestration in the list to the variable `orchestration_from_list`.

```python
orchestration_from_list = wl.list_orchestrations()[-1]
```

```python
# retrieve the orchestration from the list

orchestration_from_list = wl.list_orchestrations()[-1]
orchestration_from_list
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>ID</td><td>2f606e70-89bf-49b2-9ab0-c9720f1cca0a</td>
  </tr>
  <tr>
    <td>Name</td><td>real estate 08</td>
  </tr>
  <tr>
    <td>File Name</td><td>real-estate-orchestration.zip</td>
  </tr>
  <tr>
    <td>SHA</td><td>fcc3128857650ffa52b53b4463e0508181c2a036390607377b436b49317f6437</td>
  </tr>
  <tr>
    <td>Status</td><td>ready</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-13-Sep 17:36:06</td>
  </tr>
  <tr>
    <td>Updated At</td><td>2023-13-Sep 17:36:54</td>
  </tr>
</table>

## Create Run Once Task from Orchestration

The orchestration is now ready to be implemented as a Wallaroo Task.  We'll just run it once as an example.  This specific Orchestration that creates the Task assumes that the pipeline is deployed, and accepts the arguments:

* workspace_name
* pipeline_name

Tasks are either Run Once, or Run Scheduled.  We create a new task from the Orchestration with either `run_once(task_name, json_args, timeout)` or with `run_scheduled(name, timeout,schedule,json_args)`.  The schedule is based on the Kubernetes cron scheduler.  For example:

```python
schedule={'42 * * * *'}
```

Runs every 42 minutes and contains the answer to life, the universe, and everything.

Creating a scheduled task might be:

```python
task_scheduled = orchestration.run_scheduled(name="schedule example", 
                                             timeout=600, 
                                             schedule=schedule, 
                                             json_args={"workspace_name": workspace_name, 
                                                        "pipeline_name": pipeline_name})
```

### Create Run Once Task from Orchestration Exercise

Using the uploaded orchestration, create a Run Once task using your workspace and pipeline names as the `json_args`.  Here's an example using the variables set above.

```python
task = orchestration.run_once(name="real estate task", 
                              json_args={"workspace_name":workspace_name,
                                         "pipeline_name":pipeline_name}
                              )
```

```python
# create your task here

task = orchestration.run_once(name="real estate task", 
                              json_args={"workspace_name":workspace_name,
                                         "pipeline_name":pipeline_name}
                              )
```

## Monitor Task Run with Task Status

The Task is the **schedule** to execute the instructions within the orchestration.  The actual execution of the task is the **task run**.  A Run Once task will create one Task Run, while a Run Scheduled task will generate a new Task Run each time the schedule pattern is set.

The status task is viewed with the task `status()` command, where it is either `pending` (no tasks runs are generated yet), or `started` (a task run has been started).

## Monitor Task Run with Task Status Example

We'll monitor the run first with it's status with the `Task.status()` command.

Get the status of the task, and once it is `started` proceed to the next step.  Try doing it as a `while` loop if you feel confident.  Here's some sample code where the task was saved to the variable `task`.

```python
task.status
```

Or as a loop pausing ever 5 seconds until the task status is `started`.

```python
while task.status() != "started":
    display(task.status())
    time.sleep(5)
```

```python
while task.status() != "started":
    display(task.status())
    time.sleep(5)
```

    'pending'

## List Tasks

The Wallaroo client `list_tasks` method returns a list of tasks, and shows the the last task run status.

### List Tasks Exercise

List the tasks in your Wallaroo instance.  For example, if your Wallaroo client is stored as `wl`, this would show your tasks.

```python
wl.list_tasks()
```

```python
# empty space to list tasks

wl.list_tasks()
```

<table><tr><th>id</th><th>name</th><th>last run status</th><th>type</th><th>active</th><th>schedule</th><th>created at</th><th>updated at</th></tr><tr><td>8f003301-bb5f-47b9-854f-cb58452a3ad8</td><td>real estate task</td><td>running</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-13-Sep 17:37:00</td><td>2023-13-Sep 17:37:05</td></tr><tr><td>3a385e94-7532-40ef-bab8-b4ae5ff59f48</td><td>real estate task</td><td>failure</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-13-Sep 17:33:58</td><td>2023-13-Sep 17:34:08</td></tr></table>

## Display Task Run Results

The Task Run is the implementation of the task - the actual running of the script and it's results.  Tasks that are Run Once will only have one Task Run, while a Task set to Run Scheduled will have a Task Run for each time the task is executed.  Each Task Run has its own set of logs and results that are monitored through the Task Run `logs()` method.

First, get the Task Run - this is the actual execution of a Task.  The Task is the **scheduled** run of an Orchestration.  The Task Run is the **implementation** of a scheduled Task.  A Run Once Task while generate one Task Run, while a Scheduled Task generated a new Task Run every time the schedule pattern is met until the Task is killed.

We retrieve the task runs with the Task `last_runs()` method, and assign a single Task Run to a variable by selecting it with the list with `last_runs()[index]`.  If you only have one Task Run from a Task, then you can just set the `index` to 0.

### Display Task Run Results Exercise

Retrieve the task run for our generated task, then start checking the logs for our task run.  It may take longer than 30 seconds to launch the task, so be prepared to the command multiple times until is it displayed.  Store the task into a variable for later use.

Here's a code sample where the task was saved to the variable `task`.

```python
task_run = task.last_runs()[0]
task_run
```

```python
task_run = task.last_runs()[0]
task_run
```

<table>
  <tr><th>Field</th><th>Value</th></tr>
  <tr><td>Task</td><td>8f003301-bb5f-47b9-854f-cb58452a3ad8</td></tr>
  <tr><td>Pod ID</td><td>903379c3-1ec7-43d2-9b70-1c2cbff296b3</td></tr>
  <tr><td>Status</td><td>running</td></tr>
  <tr><td>Created At</td><td>2023-13-Sep 17:37:03</td></tr>
  <tr><td>Updated At</td><td>2023-13-Sep 17:37:03</td></tr>
</table>

The Task Run Status is checked with the `_status` method.  This lets you know if there was a failure or if it ran successfully.  If it didn't, you can still get the task run logs to find out why.

```python
task_run._status
```

    'running'

## Retrieve Task Run Logs

The Task Run logs are retrieved with the Wallaroo task runs `log()`, and shows the outputs of the results.  This is why it's useful to have `print` commands in your code to track what it's doing.

### Retrieve Task Run Logs Exercise

Take the task run and display the logs.  It may take a few minutes for the logs to show up, so you may need to refresh the code below a few times.  Here's a quick example of some code.

```python
task_run.logs()
```

```python
task_run.logs()
```

<pre><code>2023-13-Sep 17:37:08 Workspace: tutorial-workspace-john-05
2023-13-Sep 17:37:08 {'name': 'tutorial-workspace-john-05', 'id': 261, 'archived': False, 'created_by': 'd1704c38-2016-4b1d-9407-85e7e6875e6d', 'created_at': '2023-09-11T21:37:02.871776+00:00', 'models': [{'name': 'house-price-prime', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 9, 11, 21, 40, 27, 608874, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 9, 11, 21, 40, 27, 608874, tzinfo=tzutc())}], 'pipelines': [{'name': 'houseprice-estimator', 'create_time': datetime.datetime(2023, 9, 11, 21, 40, 29, 871251, tzinfo=tzutc()), 'definition': '[]'}]}
2023-13-Sep 17:37:08 Pipeline: houseprice-estimator
2023-13-Sep 17:37:08 {'status': 'Running', 'details': [], 'engines': [{'ip': '10.244.5.202', 'name': 'engine-6b947cf755-gdz8g', 'status': 'Running', 'reason': None, 'details': [], 'pipeline_statuses': {'pipelines': [{'id': 'houseprice-estimator', 'status': 'Running'}]}, 'model_statuses': {'models': [{'name': 'house-price-prime', 'version': 'f42b66d3-ed47-4571-910c-5e7184b2b4ad', 'sha': '31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c', 'status': 'Running'}]}}], 'engine_lbs': [{'ip': '10.244.5.203', 'name': 'engine-lb-584f54c899-25gl5', 'status': 'Running', 'reason': None, 'details': []}], 'sidekicks': []}
2023-13-Sep 17:37:08 {'name': 'houseprice-estimator', 'create_time': datetime.datetime(2023, 9, 11, 21, 40, 29, 871251, tzinfo=tzutc()), 'definition': '[]'}
2023-13-Sep 17:37:08                         time  ... check_failures
2023-13-Sep 17:37:08 0    2023-09-13 17:37:08.739  ...              0
2023-13-Sep 17:37:08 1    2023-09-13 17:37:08.739  ...              0
2023-13-Sep 17:37:08 2    2023-09-13 17:37:08.739  ...              0
2023-13-Sep 17:37:08 ...                      ...  ...            ...
2023-13-Sep 17:37:08 4    2023-09-13 17:37:08.739  ...              0
2023-13-Sep 17:37:08 3    2023-09-13 17:37:08.739  ...              0
2023-13-Sep 17:37:08 3996 2023-09-13 17:37:08.739  ...              0
2023-13-Sep 17:37:08 3995 2023-09-13 17:37:08.739  ...              0
2023-13-Sep 17:37:08 3997 2023-09-13 17:37:08.739  ...              0
2023-13-Sep 17:37:08 3998 2023-09-13 17:37:08.739  ...              0
2023-13-Sep 17:37:08 3999 2023-09-13 17:37:08.739  ...              0
2023-13-Sep 17:37:08 [4000 rows x 4 columns]
2023-13-Sep 17:37:08 </code></pre>

You have now walked through setting up a basic assay and running it over historical data.

## Congratulations!
In this tutorial you have
* Deployed a single step house price prediction pipeline and sent data to it.
* Uploaded an ML Orchestration into Wallaroo.
* Created a Run Once Task from the Orchestration.
* Viewed the Task Run's status generated from the Task.
* Viewed the Task Run's logs.

Great job! 

### Cleaning up.

Now that the tutorial is complete, don't forget to undeploy your pipeline to free up the resources.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>houseprice-estimator</td></tr><tr><th>created</th> <td>2023-09-11 21:40:29.871251+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-13 17:32:43.282786+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>9d80efd3-ba4b-409e-88e9-e0605db336e3, aa8b92a2-a2b9-4197-9fbd-63ad84e48e52, 4d3230a4-32be-4f47-90b5-d8d3b90cc2a9, 6f788da6-17ee-4df6-b120-354737212559, 7ae39ec3-1486-4bd7-9fa5-6874ee79f245, 11dd78db-9461-4ec0-9003-29538d4c242d, 8ca7a6bf-6529-434f-a3f6-bbc3bcc255d9</td></tr><tr><th>steps</th> <td>house-price-prime</td></tr></table>

