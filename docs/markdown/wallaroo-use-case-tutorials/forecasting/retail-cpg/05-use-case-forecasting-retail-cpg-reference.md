## Statsmodel Forecast with Wallaroo Features: ML Workload Orchestration

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
```

```python
display(wallaroo.__version__)
```

    '2023.2.1'

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
# used for unique connection names

import string
import random

suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

workspace_name = f'forecast-model-tutorial'

pipeline_name = f'forecast-tutorial-pipeline'
```

### Set the Workspace and Pipeline

The workspace will be either used or created if it does not exist, along with the pipeline.  The models uploaded in the Upload and Deploy tutorial are referenced in this step.

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

```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)

control_model_name = 'forecast-control-model'

bike_day_model = get_model(control_model_name)
```

### Deploy Pipeline

The pipeline is already set witht the model.  For our demo we'll verify that it's deployed.

```python
# Set the deployment to allow for additional engines to run
# Undeploy and clear the pipeline in case it was used in other demonstrations
pipeline.undeploy()
pipeline.clear()

pipeline.add_model_step(bike_day_model)
# pipeline.add_model_step(step)
pipeline.deploy()
```

<table><tr><th>name</th> <td>forecast-tutorial-pipeline</td></tr><tr><th>created</th> <td>2023-08-02 15:50:59.480547+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-03 17:58:19.277566+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>c065b69b-d622-4a7b-93e5-4fcacf62da86, b0a212e3-66b7-4599-9701-f4183045cec6, af0f9c1c-0c28-4aaa-81f5-abb470500960, 980ee03b-694e-47c7-b76b-43b3e927b281, 85af5504-f1e4-4d0d-bd9e-e46891211843, 39b82898-12b6-4a30-ab41-f06cb05c7391, d8edf8c5-07f0-455e-9f34-075b7062f56f, 170402aa-8e83-420e-bee3-51a9fca4a9d9, 14912dd4-5e3a-4314-9e3f-0ea3af3660c1, 3309619d-54b9-4499-8afd-ed7819339b64, 2af1f08c-976c-4d51-9cf6-2cc371788844, 76fbec8d-cebf-40e5-81d5-447170c4a836, c6c10a83-9b6c-449f-a5c3-63b36a3d749b, 436fe308-283f-43b0-a4f0-159c05193d97, eb9e5b9f-41d9-42dc-8e49-13ec4771abad, 4d062242-1477-40fd-bf11-835e6bd62c10, 1f3d774d-7626-4722-b4b8-7dedbaa35803, 12f73035-cf94-4e6c-b2b6-05946ab06aef, b4ec30ef-6724-467e-b42a-d54399198f32, 57e7acf8-b3f0-436b-a236-0b1d6e76ba18, 5697a317-d0e6-402b-9369-7f0e732cc1fa, 5d0cb620-f8ba-4b9d-a81b-0ba333584508, 6b14e208-1319-4bc4-927b-b76a4893d373, 0b44d911-c69e-4030-b481-84e947fe6c70, dc5605d2-bb6a-48d2-b83a-3d77b7e608af, a68819c0-7508-467e-9fc1-60cbf8aaf9e1, b908d302-ce87-4a52-8ef2-b595fac2c67e, 7b94201f-ef5b-4629-ae2f-acf894cb1fcf, dc8bf23f-b598-48c6-bb2d-c5098d264622, 3a8ebc46-6261-4977-8a60-038c99c255d7, 40ab9d3d-ee6c-4f0c-bf38-345385130285, 47792a90-bea8-432a-981f-232bf67288c8, 97b815f3-636b-4424-8be4-3d95bcf32b40, 0d2f2250-9a43-47ce-beef-32371986f798, 46c95b7f-a79e-41ee-8565-578f9c3c20e5, 1ff98a35-3468-4b70-84fc-fe71aed99a75, 73ff8fc2-ca4d-4ea1-887b-0d31190cfe36, f8188956-8b3e-4479-8b15-e8747fe915a6, 33e5cc2c-2bb2-4dc2-8a9e-c058e60f6163, 5d419693-97cc-461b-b72a-a389ab7a001b, 56c78f52-cba5-415c-913a-fee0e1863a90, a109a040-c8f2-46dc-8c0b-373ae10d4fa0, dcaec327-1358-42a7-88de-931602a42a72, debc509f-9481-464b-af7f-5c3138a9cdb4, b0d167aa-cc98-440a-8e85-1ae3f089745a, d9e69c40-c83b-48af-b6b9-caafcb85f08b, 186ffdd2-3a8f-40cc-8362-13cc20bd2f46, 535e6030-ebe5-4c79-b5cd-69b161637a99, c5c0218a-800b-4235-8767-64d18208e68a, 4559d934-33b0-4872-a788-4ef27f554482, 94d3e20b-add7-491c-aedd-4eb094a8aebf, ab4e58bf-3b75-4bf6-b6b3-f703fe61e7af, 3773f5c5-e4c5-4e46-a839-6945af15ca13, 3abf03dd-8eab-4a8d-8432-aa85a30c0eda, 5ec5e8dc-7492-498b-9652-b3733e4c87f7, 1d89287b-4eff-47ec-a7bb-8cedaac1f33f</td></tr><tr><th>steps</th> <td>forecast-control-model</td></tr></table>

### Sample Inference

Verify the pipeline is deployed properly with a sample inference.

```python

sample_count = pd.read_csv('../data/test_data.csv')
inference_df = get_singleton_forecast(sample_count.loc[2:22], 'count')

results = pipeline.infer(inference_df)

display(results)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.count</th>
      <th>out.forecast</th>
      <th>out.weekly_average</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-03 18:00:43.307</td>
      <td>[1349, 1562, 1600, 1606, 1510, 959, 822, 1321, 1263, 1162, 1406, 1421, 1248, 1204, 1000, 683, 1650, 1927, 1543, 981, 986]</td>
      <td>[1278, 1295, 1295, 1295, 1295, 1295, 1295]</td>
      <td>1292.571429</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Forecast Sample Orchestration

The orchestration that will automate this process is `./resources/forecast-orchestration.zip`.  The files used are stored in the directory `forecast-orchestration`, created with the command:

`zip -r forecast-bigquery-connection.zip forecast-orchestration/`.

This contains the following:

* `requirements.txt`:  The Python requirements file to specify the following libraries used.  For this example, that will be empty since we will be using the 
* `main.py`: The entry file that uses a deployed pipeline and performs an inference request against it visible from its log files.
* `data/testdata_dict.json`: An inference input file.

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

def get_pipeline(name):
    try:
        pipeline = wl.pipelines_by_name(name)[0]
    except EntityNotFoundError:
        print(f"Pipeline not found:{name}")
    return pipeline

def get_singleton_forecast(df, field):
    singleton = pd.DataFrame({field: [df[field].values.tolist()]})
    return singleton

print(f"Workspace: {workspace_name}")
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)
print(workspace)

# the pipeline is assumed to be deployed
print(f"Pipeline: {pipeline_name}")
pipeline = get_pipeline(pipeline_name)
print(pipeline)

print(pipeline.status())

sample_count = pd.read_csv('./data/test_data.csv')
inference_df = get_singleton_forecast(sample_count.loc[2:22], 'count')

results = pipeline.infer(inference_df)

print(results)
```

A few things to go over here.  You'll notice this is almost the exact procedures we've been following so far:  we get a workspace and pipeline, pull data from a CSV file, and perform an inference off the data.

This script assumes that the pipeline has already been deployed, and also includes this part:

`arguments = wl.task_args()`

This allows us to pass arguments into a Task created from an Orchestration, so we can specify a different workspace, pipeline, or any other arguments we construct.  This allows orchestrations to be very flexible.

Also, notice that it refers to a specific file:

`sample_count = pd.read_csv('./data/test_data.csv')`

In the `forecast-orchestration` directory is the `data` directory with our sample CSV file.  Orchestrations can include additional artifacts.  We could have used a Wallaroo Connection instead, and we encourage you to try that if you want.

### Upload the Orchestration

Orchestrations are uploaded with the Wallaroo client `upload_orchestration(path)` method with the following parameters.

| Parameter | Type | Description |
| --- | --- | ---|
| **path** | string (Required) | The path to the ZIP file to be uploaded. |

Once uploaded, the deployment will be prepared and any requirements will be downloaded and installed.  A typical orchestration upload looks like this:

```python
my_orchestration = wl.upload_orchestration(path-to-zip-file)
```

Try uploading our orchestration from `./forecast-orchestration/forecast-orchestration.zip` - or make your own and upload it.

Once uploaded, you can check the status with the `status()`.  If using the orchestration example above, that would be `my_orchestration.status()`  This is handy to make into a loop to check the status until is shows `ready`.

```python
orchestration = wl.upload_orchestration(name="forecast example", path="../forecast-orchestration/forecast-orchestration.zip")

while orchestration.status() != 'ready':
    print(orchestration.status())
    time.sleep(5)
```

    pending_packaging
    packaging
    packaging
    packaging
    packaging
    packaging
    packaging
    packaging

### List Orchestrations

Orchestrations are listed with the Wallaroo Client `list_orchestrations()` method.  Orchestrations can be retrieved to a variable by allocated their position in the array - for example:  `orchestration = wl.list_orchestrations()[0]` would return the first orchestration on the list.

```python
# list orchestration here

wl.list_orchestrations()
```

<table><tr><th>id</th><th>name</th><th>status</th><th>filename</th><th>sha</th><th>created at</th><th>updated at</th></tr><tr><td>fc4fd8cb-a108-404b-8ef9-8a1c9e279bb7</td><td>statsmodel-orchestration 5</td><td>ready</td><td>forecast-orchestration.zip</td><td>2c1f30...0f0761</td><td>2023-02-Aug 20:50:47</td><td>2023-02-Aug 20:51:36</td></tr><tr><td>f8cccfd4-5ef3-49f7-a0e0-4bbccc0fc664</td><td>statsmodel-orchestration 6</td><td>ready</td><td>forecast-orchestration.zip</td><td>1b675d...3a4a32</td><td>2023-02-Aug 21:07:51</td><td>2023-02-Aug 21:08:37</td></tr><tr><td>8a476448-06da-43b8-96a6-6f4b492973b0</td><td>statsmodel-orchestration 6</td><td>ready</td><td>forecast-orchestration.zip</td><td>1b675d...3a4a32</td><td>2023-02-Aug 21:13:18</td><td>2023-02-Aug 21:14:02</td></tr><tr><td>db9cdef8-4171-43c2-97ae-2188c7d29b41</td><td>statsmodel-orchestration 6</td><td>ready</td><td>forecast-orchestration.zip</td><td>1b675d...3a4a32</td><td>2023-02-Aug 21:17:33</td><td>2023-02-Aug 21:18:17</td></tr><tr><td>07bfa923-1372-4560-aa5b-c2d1e27f79bf</td><td>forecast example</td><td>ready</td><td>forecast-orchestration.zip</td><td>1a93aa...6f73f5</td><td>2023-03-Aug 18:01:49</td><td>2023-03-Aug 18:02:33</td></tr><tr><td>7f4cbd97-5064-46a4-93a8-1b4f500ad7a8</td><td>forecast example</td><td>ready</td><td>forecast-orchestration.zip</td><td>d38397...fcf803</td><td>2023-03-Aug 18:04:47</td><td>2023-03-Aug 18:05:31</td></tr></table>

```python
# retrieve the orchestration from the list

orchestration_from_list = wl.list_orchestration()[-1]
orchestration_from_list
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>ID</td><td>7f4cbd97-5064-46a4-93a8-1b4f500ad7a8</td>
  </tr>
  <tr>
    <td>Name</td><td>forecast example</td>
  </tr>
  <tr>
    <td>File Name</td><td>forecast-orchestration.zip</td>
  </tr>
  <tr>
    <td>SHA</td><td>d38397d19fa05339a7884cd324208515d3ef2cdc85542af31290c45176fcf803</td>
  </tr>
  <tr>
    <td>Status</td><td>ready</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-03-Aug 18:04:47</td>
  </tr>
  <tr>
    <td>Updated At</td><td>2023-03-Aug 18:05:31</td>
  </tr>
</table>

### Create the Task

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

Using the uploaded orchestration, create a Run Once task using your workspace and pipeline names as the `json_args`.

```python
# create the task here

task = orchestration.run_once(name="forecast single run", 
                              json_args={"workspace_name":workspace_name,
                                         "pipeline_name":pipeline_name}
                              )
```

### Monitor Run with Task Status

We'll monitor the run first with it's status with the `Task.status()` command.

Get the status of the task, and once it is `started` proceed to the next step.  Try doing it as a `while` loop if you feel confident.

```python
while task.status() != "started":
    display(task.status())
    time.sleep(5)
```

    'pending'

### List Tasks

We'll use the Wallaroo client `list_tasks` method to view the tasks currently running or scheduled.

```python
wl.list_tasks()
```

<table><tr><th>id</th><th>name</th><th>last run status</th><th>type</th><th>active</th><th>schedule</th><th>created at</th><th>updated at</th></tr><tr><td>b858db7a-fb70-4bb6-b4bb-49b48cefe504</td><td>forecast single run</td><td>success</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-03-Aug 18:05:36</td><td>2023-03-Aug 18:05:41</td></tr><tr><td>b1ff0ce6-f2cc-4613-91ce-b3a676165c8d</td><td>forecast single run</td><td>failure</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-03-Aug 18:02:38</td><td>2023-03-Aug 18:02:43</td></tr><tr><td>d897ef95-911e-42ee-a874-2a7435b5ca77</td><td>statsmodel single run finale</td><td>success</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-02-Aug 21:18:19</td><td>2023-02-Aug 21:18:30</td></tr><tr><td>f406497a-d8c1-4b20-8fe9-d83c8102da40</td><td>statsmodel single run finale</td><td>success</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-02-Aug 21:08:42</td><td>2023-02-Aug 21:08:48</td></tr><tr><td>7117f780-5fc4-476a-a5d2-0654fdb6271f</td><td>statsmodel single run finale</td><td>failure</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-02-Aug 20:55:17</td><td>2023-02-Aug 20:55:23</td></tr><tr><td>f209c52a-88e2-43e3-a614-b08a35b72a94</td><td>statsmodel single run finale</td><td>failure</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-02-Aug 20:52:24</td><td>2023-02-Aug 20:52:35</td></tr></table>

### Display Task Run Results

The Task Run is the implementation of the task - the actual running of the script and it's results.  Tasks that are Run Once will only have one Task Run, while a Task set to Run Scheduled will have a Task Run for each time the task is executed.  Each Task Run has its own set of logs and results that are monitored through the Task Run `logs()` method.

First, get the Task Run - this is the actual execution of a Task.  The Task is the **scheduled** run of an Orchestration.  The Task Run is the **implementation** of a scheduled Task.  A Run Once Task while generate one Task Run, while a Scheduled Task generated a new Task Run every time the schedule pattern is met until the Task is killed.

We retrieve the task runs with the Task `last_runs()` method, and assign a single Task Run to a variable by selecting it with the list with `last_runs()[index]`.  If you only have one Task Run from a Task, then you can just set the `index` to 0.

Retrieve the task run for our generated task, then start checking the logs for our task run.  It may take longer than 30 seconds to launch the task, so be prepared to run the `.logs()` method again to view the logs.

```python
task
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>ID</td><td>b858db7a-fb70-4bb6-b4bb-49b48cefe504</td>
  </tr>
  <tr>
    <td>Name</td><td>forecast single run</td>
  </tr>
  <tr>
    <td>Last Run Status</td><td>success</td>
  </tr>
  <tr>
    <td>Type</td><td>Temporary Run</td>
  </tr>
  <tr>
    <td>Active</td><td>True</td>
  </tr>
  <tr>
    <td>Schedule</td><td>-</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-03-Aug 18:05:36</td>
  </tr>
  <tr>
    <td>Updated At</td><td>2023-03-Aug 18:05:41</td>
  </tr>
</table>

```python
statsmodel_task_run = task.last_runs()[0]
```

The Task Run Status is checked with the `_status` method.  This lets you know if there was a failure or if it ran successfully.  If it didn't, you can still get the task run logs to find out why.

```python
statsmodel_task_run._status
```

    'success'

### Retrieve Task Run Logs

The Task Run logs are retrieved with the Wallaroo task runs `log()`, and shows the outputs of the results.  This is why it's useful to have `print` commands in your code to track what it's doing.

```python
statsmodel_task_run.logs()
```

<pre><code>2023-03-Aug 18:05:43 Workspace: forecast-model-tutorialjohn
2023-03-Aug 18:05:43 {'name': 'forecast-model-tutorialjohn', 'id': 16, 'archived': False, 'created_by': '0a36fba2-ad42-441b-9a8c-bac8c68d13fa', 'created_at': '2023-08-02T15:50:52.816795+00:00', 'models': [{'name': 'forecast-control-model', 'versions': 3, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 3, 1, 11, 50, 568151, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 2, 15, 50, 54, 223186, tzinfo=tzutc())}, {'name': 'forecast-challenger01-model', 'versions': 3, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 3, 13, 55, 23, 119224, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 2, 15, 50, 55, 208179, tzinfo=tzutc())}, {'name': 'forecast-challenger02-model', 'versions': 3, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 3, 13, 55, 24, 133756, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 2, 15, 50, 56, 291043, tzinfo=tzutc())}], 'pipelines': [{'name': 'forecast-tutorial-pipeline', 'create_time': datetime.datetime(2023, 8, 2, 15, 50, 59, 480547, tzinfo=tzutc()), 'definition': '[]'}]}
2023-03-Aug 18:05:43 Pipeline: forecast-tutorial-pipeline
2023-03-Aug 18:05:43                      time  ... check_failures
2023-03-Aug 18:05:43 {'name': 'forecast-tutorial-pipeline', 'create_time': datetime.datetime(2023, 8, 2, 15, 50, 59, 480547, tzinfo=tzutc()), 'definition': '[]'}
2023-03-Aug 18:05:43 {'status': 'Running', 'details': [], 'engines': [{'ip': '10.244.3.225', 'name': 'engine-5fc486bbf7-wklvf', 'status': 'Running', 'reason': None, 'details': [], 'pipeline_statuses': {'pipelines': [{'id': 'forecast-tutorial-pipeline', 'status': 'Running'}]}, 'model_statuses': {'models': [{'name': 'forecast-control-model', 'version': 'ffca51bd-f9c6-40cf-a36b-c6126ce98dd3', 'sha': 'dcbd11947ae1e51f5c882687a0ec2dbcf60c0b0de8e5156cb6f1d669e0a6d76b', 'status': 'Running'}]}}], 'engine_lbs': [{'ip': '10.244.4.151', 'name': 'engine-lb-584f54c899-rbdhr', 'status': 'Running', 'reason': None, 'details': []}], 'sidekicks': []}
2023-03-Aug 18:05:43 
2023-03-Aug 18:05:43 0 2023-08-03 18:05:43.232  ...              0
2023-03-Aug 18:05:43 [1 rows x 5 columns]</code></pre>

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

    Waiting for undeployment - this will take up to 45s .................................... ok

<table><tr><th>name</th> <td>forecast-tutorial-pipeline</td></tr><tr><th>created</th> <td>2023-08-02 15:50:59.480547+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-02 21:16:55.320303+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>f8188956-8b3e-4479-8b15-e8747fe915a6, 33e5cc2c-2bb2-4dc2-8a9e-c058e60f6163, 5d419693-97cc-461b-b72a-a389ab7a001b, 56c78f52-cba5-415c-913a-fee0e1863a90, a109a040-c8f2-46dc-8c0b-373ae10d4fa0, dcaec327-1358-42a7-88de-931602a42a72, debc509f-9481-464b-af7f-5c3138a9cdb4, b0d167aa-cc98-440a-8e85-1ae3f089745a, d9e69c40-c83b-48af-b6b9-caafcb85f08b, 186ffdd2-3a8f-40cc-8362-13cc20bd2f46, 535e6030-ebe5-4c79-b5cd-69b161637a99, c5c0218a-800b-4235-8767-64d18208e68a, 4559d934-33b0-4872-a788-4ef27f554482, 94d3e20b-add7-491c-aedd-4eb094a8aebf, ab4e58bf-3b75-4bf6-b6b3-f703fe61e7af, 3773f5c5-e4c5-4e46-a839-6945af15ca13, 3abf03dd-8eab-4a8d-8432-aa85a30c0eda, 5ec5e8dc-7492-498b-9652-b3733e4c87f7, 1d89287b-4eff-47ec-a7bb-8cedaac1f33f</td></tr><tr><th>steps</th> <td>forecast-control-model</td></tr></table>

