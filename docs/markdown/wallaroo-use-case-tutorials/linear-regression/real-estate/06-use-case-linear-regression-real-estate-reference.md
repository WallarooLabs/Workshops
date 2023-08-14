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

### Deploy Pipeline

The pipeline is already set witht the model.  For our demo we'll verify that it's deployed.

```python
# Set the deployment to allow for additional engines to run
# Undeploy and clear the pipeline in case it was used in other demonstrations
pipeline.undeploy()
pipeline.clear()

pipeline.add_model_step(control_model)
# pipeline.add_model_step(step)
pipeline.deploy()
```

<table><tr><th>name</th> <td>tutorialpipeline-jch</td></tr><tr><th>created</th> <td>2023-08-03 19:36:31.732163+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-03 19:51:09.492808+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>489fdc01-e8c2-4d72-9640-2f33416d3941, 1a73448b-9012-4258-bb2c-a4d25a1e6f19, d1d0cafe-78a9-4193-84af-cec1b3ed608b, 70438418-4802-4ced-a295-65ef78706fd4</td></tr><tr><th>steps</th> <td>tutorial-model</td></tr></table>

### Sample Inference

Verify the pipeline is deployed properly with a sample inference.

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
      <td>2023-08-03 19:51:27.001</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[704901.9]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Sample Orchestration

The orchestration that will automate this process is `./real-estate-orchestration/real-estate-orchestration.zip`.  The files used are stored in the directory `forecast-orchestration`, created with the command:

`zip -r real-estate-orchestration.zip real-estate-orchestration/`.

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

# get sample inference data
df_from_csv = pd.read_csv('./data/test_data.csv')

singleton = get_singleton(df_from_csv, 0)
display(singleton)

single_result = pipeline.infer(singleton)
display(single_result)

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
orchestration = wl.upload_orchestration(name="house price orchestration example", path="../real-estate-orchestration/real-estate-orchestration.zip")

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
    packaging

### List Orchestrations

Orchestrations are listed with the Wallaroo Client `list_orchestrations()` method.  Orchestrations can be retrieved to a variable by allocated their position in the array - for example:  `orchestration = wl.list_orchestrations()[0]` would return the first orchestration on the list.

```python
# list orchestration here

wl.list_orchestrations()
```

<table><tr><th>id</th><th>name</th><th>status</th><th>filename</th><th>sha</th><th>created at</th><th>updated at</th></tr><tr><td>68fc016f-a013-46a8-95e3-149d46f2ca0c</td><td>house price orchestration example</td><td>ready</td><td>real-estate-orchestration.zip</td><td>2ca71f...8bff63</td><td>2023-03-Aug 19:51:33</td><td>2023-03-Aug 19:52:26</td></tr></table>

```python
# retrieve the orchestration from the list

orchestration_from_list = wl.list_orchestration()[-1]
orchestration_from_list
```

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
# create your task here

# create the task here

task = orchestration.run_once(name="real estate task", 
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

<table><tr><th>id</th><th>name</th><th>last run status</th><th>type</th><th>active</th><th>schedule</th><th>created at</th><th>updated at</th></tr><tr><td>8bec66fb-4f6e-4257-a476-cf39b593db22</td><td>real estate task</td><td>success</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-03-Aug 19:52:51</td><td>2023-03-Aug 19:52:56</td></tr></table>

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
    <td>ID</td><td>8bec66fb-4f6e-4257-a476-cf39b593db22</td>
  </tr>
  <tr>
    <td>Name</td><td>real estate task</td>
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
    <td>Created At</td><td>2023-03-Aug 19:52:51</td>
  </tr>
  <tr>
    <td>Updated At</td><td>2023-03-Aug 19:52:56</td>
  </tr>
</table>

```python
task_run = task.last_runs()[0]
```

The Task Run Status is checked with the `_status` method.  This lets you know if there was a failure or if it ran successfully.  If it didn't, you can still get the task run logs to find out why.

```python
task_run._status
```

    'success'

### Retrieve Task Run Logs

The Task Run logs are retrieved with the Wallaroo task runs `log()`, and shows the outputs of the results.  This is why it's useful to have `print` commands in your code to track what it's doing.

```python
task_run.logs()
```

<pre><code>2023-03-Aug 19:52:58 Workspace: tutorial-workspace-jch
2023-03-Aug 19:52:58 {'name': 'tutorial-workspace-jch', 'id': 19, 'archived': False, 'created_by': '0a36fba2-ad42-441b-9a8c-bac8c68d13fa', 'created_at': '2023-08-03T19:34:42.324336+00:00', 'models': [{'name': 'tutorial-model', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 3, 19, 36, 31, 13200, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 3, 19, 36, 31, 13200, tzinfo=tzutc())}], 'pipelines': [{'name': 'tutorialpipeline-jch', 'create_time': datetime.datetime(2023, 8, 3, 19, 36, 31, 732163, tzinfo=tzutc()), 'definition': '[]'}]}
2023-03-Aug 19:52:58 Pipeline: tutorialpipeline-jch
2023-03-Aug 19:52:58 {'name': 'tutorialpipeline-jch', 'create_time': datetime.datetime(2023, 8, 3, 19, 36, 31, 732163, tzinfo=tzutc()), 'definition': '[]'}
2023-03-Aug 19:52:58 {'status': 'Running', 'details': [], 'engines': [{'ip': '10.244.2.246', 'name': 'engine-6586d95b48-5k6zl', 'status': 'Running', 'reason': None, 'details': [], 'pipeline_statuses': {'pipelines': [{'id': 'tutorialpipeline-jch', 'status': 'Running'}]}, 'model_statuses': {'models': [{'name': 'tutorial-model', 'version': '68b3f094-1b0f-4f6e-940e-4dc1cb1500b2', 'sha': 'ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a', 'status': 'Running'}]}}], 'engine_lbs': [{'ip': '10.244.4.154', 'name': 'engine-lb-584f54c899-k8n25', 'status': 'Running', 'reason': None, 'details': []}], 'sidekicks': []}
2023-03-Aug 19:52:58 0  [4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0,...
2023-03-Aug 19:52:58                                               tensor
2023-03-Aug 19:52:58 0 2023-08-03 19:52:58.828  ...              0
2023-03-Aug 19:52:58                      time  ... check_failures
2023-03-Aug 19:52:58 
2023-03-Aug 19:52:58 [1 rows x 4 columns]</code></pre>

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

<table><tr><th>name</th> <td>tutorialpipeline-jch</td></tr><tr><th>created</th> <td>2023-08-03 19:36:31.732163+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-03 19:51:09.492808+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>489fdc01-e8c2-4d72-9640-2f33416d3941, 1a73448b-9012-4258-bb2c-a4d25a1e6f19, d1d0cafe-78a9-4193-84af-cec1b3ed608b, 70438418-4802-4ced-a295-65ef78706fd4</td></tr><tr><th>steps</th> <td>tutorial-model</td></tr></table>

