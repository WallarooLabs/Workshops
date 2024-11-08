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

<table><tr><th>name</th> <td>rental-forecast</td></tr><tr><th>created</th> <td>2024-10-29 21:00:36.927945+00:00</td></tr><tr><th>last_updated</th> <td>2024-10-30 20:48:14.837079+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>workspace_id</th> <td>8</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-forecast</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>585ee8cd-2f5e-4a1e-bb0d-6c88e6d94d3e, ceff9712-715b-41e6-a124-b174b62a9654, 0250f403-07c6-4b01-83bc-eebdc09bca22, 31b515bb-807f-4d64-b105-fc0ae6a582f2, 614a34e0-6024-4245-9919-1a85b7a1e5d2, 6a593faf-bea3-4f57-b9ec-5c1afe7f93a7, 4dce5be3-926c-419f-9868-3dbea7baf3c1, a601ce07-937c-436a-9735-0ac842173dfb, c0d16da5-5db7-4af1-95e4-cb0c316a4ef3, bd5eb43f-5a2b-493c-a04b-863dccccb55f, 89729096-6581-42b8-9b06-10d580d31e11, b98b86fb-5941-45b6-af5d-c33f80ba7986, aead5518-ffb2-4d18-8898-89575ba90a9f, a2a887c0-a91b-4af7-b579-506c79631fa4, b8ac836a-903b-4327-a4c9-5cc7fb382aa7, 3e18cd2d-c006-497b-a756-5ecc95aa8439, bd3f7d6a-e246-4456-98b9-35b90990b86d</td></tr><tr><th>steps</th> <td>forecast-control-model</td></tr><tr><th>published</th> <td>False</td></tr></table>

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

<table><tr><th>name</th> <td>rental-forecast</td></tr><tr><th>created</th> <td>2024-10-29 21:00:36.927945+00:00</td></tr><tr><th>last_updated</th> <td>2024-10-30 20:55:04.440894+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>workspace_id</th> <td>8</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-forecast</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>beca3565-bb16-41ca-83d6-cb6d9ba3514e, 585ee8cd-2f5e-4a1e-bb0d-6c88e6d94d3e, ceff9712-715b-41e6-a124-b174b62a9654, 0250f403-07c6-4b01-83bc-eebdc09bca22, 31b515bb-807f-4d64-b105-fc0ae6a582f2, 614a34e0-6024-4245-9919-1a85b7a1e5d2, 6a593faf-bea3-4f57-b9ec-5c1afe7f93a7, 4dce5be3-926c-419f-9868-3dbea7baf3c1, a601ce07-937c-436a-9735-0ac842173dfb, c0d16da5-5db7-4af1-95e4-cb0c316a4ef3, bd5eb43f-5a2b-493c-a04b-863dccccb55f, 89729096-6581-42b8-9b06-10d580d31e11, b98b86fb-5941-45b6-af5d-c33f80ba7986, aead5518-ffb2-4d18-8898-89575ba90a9f, a2a887c0-a91b-4af7-b579-506c79631fa4, b8ac836a-903b-4327-a4c9-5cc7fb382aa7, 3e18cd2d-c006-497b-a756-5ecc95aa8439, bd3f7d6a-e246-4456-98b9-35b90990b86d</td></tr><tr><th>steps</th> <td>forecast-control-model</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Sample Inference

Verify the pipeline is deployed properly with a sample inference with the file `./data/test_data.df.json`.

```python
# sample inference from previous code here

single_result = pipeline.infer_from_file('../data/testdata-standard.df.json')
display(single_result)
```

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
      <td>2024-10-30 20:55:33.349</td>
      <td>[1526, 1550, 1708, 1005, 1623, 1712, 1530, 1605, 1538, 1746, 1472, 1589, 1913, 1815, 2115, 2475, 2927, 1635, 1812, 1107, 1450, 1917, 1807, 1461, 1969, 2402, 1446, 1851]</td>
      <td>[1764, 1749, 1743, 1741, 1740, 1740, 1740]</td>
      <td>1745.2858</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

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

def get_pipeline(name):
    try:
        pipeline = wl.pipelines_by_name(name)[0]
    except EntityNotFoundError:
        print(f"Pipeline not found:{name}")
    return pipeline

print(f"Workspace: {workspace_name}")
workspace = wl.get_workspace(name=workspace_name, create_if_not_exist=True)

wl.set_current_workspace(workspace)
print(workspace)

# the pipeline is assumed to be deployed
print(f"Pipeline: {pipeline_name}")
pipeline = wl.build_pipeline(pipeline_name)
print(pipeline)

print(pipeline.status())

single_result = pipeline.infer_from_file('./data/testdata-standard.df.json')

results = pipeline.infer(single_result)

print(results)

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
orchestration = wl.upload_orchestration(name="my real example", path="./forecast-orchestration/forecast-orchestration.zip")

while orchestration.status() != 'ready':
    print(orchestration.status())
    time.sleep(5)
```

```python
orchestration = wl.upload_orchestration(name="forecast sample orchestration", 
                                        path="../forecast-orchestration/forecast-orchestration.zip")

while orchestration.status() != 'ready':
    print(orchestration.status())
    time.sleep(5)
```

    pending_packaging
    pending_packaging
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

<table><tr><th>id</th><th>name</th><th>status</th><th>filename</th><th>sha</th><th>created at</th><th>updated at</th><th>workspace id</th><th>workspace name</th></tr><tr><td>2459b4c4-2c38-4134-b499-ef2f4f375d7c</td><td>forecast sample orchestration</td><td>ready</td><td>forecast-orchestration.zip</td><td>6af413...6ccf88</td><td>2024-30-Oct 20:56:15</td><td>2024-30-Oct 20:57:18</td><td>8</td><td>tutorial-workspace-forecast</td></tr></table>

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
    <td>ID</td><td>2459b4c4-2c38-4134-b499-ef2f4f375d7c</td>
  </tr>
  <tr>
    <td>Name</td><td>forecast sample orchestration</td>
  </tr>
  <tr>
    <td>File Name</td><td>forecast-orchestration.zip</td>
  </tr>
  <tr>
    <td>SHA</td><td>6af4130d4ff8113f5dbe8e62474b2af4ed73900d9b209603e7444017236ccf88</td>
  </tr>
  <tr>
    <td>Status</td><td>ready</td>
  </tr>
  <tr>
    <td>Created At</td><td>2024-30-Oct 20:56:15</td>
  </tr>
  <tr>
    <td>Updated At</td><td>2024-30-Oct 20:57:18</td>
  </tr>
  <tr>
    <td>Workspace ID</td><td>8</td>
  </tr>
  <tr>
    <td>Workspace Name</td><td>tutorial-workspace-forecast</td>
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

task = orchestration.run_once(name="forecast jsh task", 
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

    'pending'

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

<table><tr><th>id</th><th>name</th><th>last run status</th><th>type</th><th>active</th><th>schedule</th><th>created at</th><th>updated at</th><th>workspace id</th><th>workspace name</th></tr><tr><td>3596a22b-7e91-40b8-9445-074af381f4d3</td><td>forecast jsh task</td><td>running</td><td>Temporary Run</td><td>True</td><td>-</td><td>2024-30-Oct 20:59:26</td><td>2024-30-Oct 20:59:41</td><td>8</td><td>tutorial-workspace-forecast</td></tr></table>

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
  <tr><td>Task</td><td>3596a22b-7e91-40b8-9445-074af381f4d3</td></tr>
  <tr><td>Pod ID</td><td>a3eb8af4-9bd7-477e-882b-2f38e7a8e76c</td></tr>
  <tr><td>Status</td><td>success</td></tr>
  <tr><td>Created At</td><td>2024-30-Oct 20:59:39</td></tr>
  <tr><td>Updated At</td><td>2024-30-Oct 20:59:39</td></tr>
</table>

The Task Run Status is checked with the `_status` method.  This lets you know if there was a failure or if it ran successfully.  If it didn't, you can still get the task run logs to find out why.

```python
task_run._status
```

    'success'

## Retrieve Task Run Logs

The Task Run logs are retrieved with the Wallaroo task runs `log()`, and shows the outputs of the results.  This is why it's useful to have `print` commands in your code to track what it's doing.

### Retrieve Task Run Logs Exercise

Take the task run and display the logs.  It may take a few minutes for the logs to show up, so you may need to refresh the code below a few times.  Here's a quick example of some code.

```python
task_run.logs()
```

```python
time.sleep(60)
task_run.logs()
```

<pre><code>2024-30-Oct 20:59:47 Workspace: tutorial-workspace-forecast
2024-30-Oct 20:59:47 {'name': 'tutorial-workspace-forecast', 'id': 8, 'archived': False, 'created_by': 'fca5c4df-37ac-4a78-9602-dd09ca72bc60', 'created_at': '2024-10-29T20:52:00.744998+00:00', 'models': [{'name': 'forecast-control-model', 'versions': 3, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 10, 29, 21, 35, 59, 4303, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 10, 29, 20, 54, 24, 314662, tzinfo=tzutc())}, {'name': 'forecast-alternate01-model', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 10, 30, 19, 56, 17, 519779, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 10, 30, 19, 56, 17, 519779, tzinfo=tzutc())}, {'name': 'forecast-alternate02-model', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 10, 30, 19, 56, 43, 83456, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 10, 30, 19, 56, 43, 83456, tzinfo=tzutc())}], 'pipelines': [{'name': 'rental-forecast', 'create_time': datetime.datetime(2024, 10, 29, 21, 0, 36, 927945, tzinfo=tzutc()), 'definition': '[]'}]}
2024-30-Oct 20:59:47 Pipeline: rental-forecast
2024-30-Oct 20:59:47 {'name': 'rental-forecast', 'create_time': datetime.datetime(2024, 10, 29, 21, 0, 36, 927945, tzinfo=tzutc()), 'definition': '[]'}
2024-30-Oct 20:59:47 {'status': 'Running', 'details': [], 'engines': [{'ip': '10.28.3.19', 'name': 'engine-7d8ffd7bd-tvh7g', 'status': 'Running', 'reason': None, 'details': [], 'pipeline_statuses': {'pipelines': [{'id': 'rental-forecast', 'status': 'Running', 'version': 'beca3565-bb16-41ca-83d6-cb6d9ba3514e'}]}, 'model_statuses': {'models': [{'name': 'forecast-control-model', 'sha': '80b51818171dc1e64e61c3050a0815a68b4d14b1b37e1e18dac9e4719e074eb1', 'status': 'Running', 'version': '4c9a1678-cba3-4db9-97a5-883ce89a9a24'}]}}], 'engine_lbs': [{'ip': '10.28.3.20', 'name': 'engine-lb-6676794678-rzq6t', 'status': 'Running', 'reason': None, 'details': []}], 'sidekicks': [{'ip': '10.28.3.18', 'name': 'engine-sidekick-forecast-control-model-8-5bc98df79b-zlb9q', 'status': 'Running', 'reason': None, 'details': [], 'statuses': '\n'}]}
2024-30-Oct 20:59:47                      time  ... anomaly.count
2024-30-Oct 20:59:47 
2024-30-Oct 20:59:47 0 2024-10-30 20:59:47.257  ...             0
2024-30-Oct 20:59:47 [1 rows x 5 columns]</code></pre>

### Cleaning up.

Now that the tutorial is complete, don't forget to undeploy your pipeline to free up the resources.

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s .................................... ok

<table><tr><th>name</th> <td>rental-forecast</td></tr><tr><th>created</th> <td>2024-10-29 21:00:36.927945+00:00</td></tr><tr><th>last_updated</th> <td>2024-10-30 20:55:04.440894+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>workspace_id</th> <td>8</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-forecast</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>beca3565-bb16-41ca-83d6-cb6d9ba3514e, 585ee8cd-2f5e-4a1e-bb0d-6c88e6d94d3e, ceff9712-715b-41e6-a124-b174b62a9654, 0250f403-07c6-4b01-83bc-eebdc09bca22, 31b515bb-807f-4d64-b105-fc0ae6a582f2, 614a34e0-6024-4245-9919-1a85b7a1e5d2, 6a593faf-bea3-4f57-b9ec-5c1afe7f93a7, 4dce5be3-926c-419f-9868-3dbea7baf3c1, a601ce07-937c-436a-9735-0ac842173dfb, c0d16da5-5db7-4af1-95e4-cb0c316a4ef3, bd5eb43f-5a2b-493c-a04b-863dccccb55f, 89729096-6581-42b8-9b06-10d580d31e11, b98b86fb-5941-45b6-af5d-c33f80ba7986, aead5518-ffb2-4d18-8898-89575ba90a9f, a2a887c0-a91b-4af7-b579-506c79631fa4, b8ac836a-903b-4327-a4c9-5cc7fb382aa7, 3e18cd2d-c006-497b-a756-5ecc95aa8439, bd3f7d6a-e246-4456-98b9-35b90990b86d</td></tr><tr><th>steps</th> <td>forecast-control-model</td></tr><tr><th>published</th> <td>False</td></tr></table>

You have now walked through setting up a basic assay and running it over historical data.

## Congratulations!
In this tutorial you have
* Deployed a single step house price prediction pipeline and sent data to it.
* Uploaded an ML Orchestration into Wallaroo.
* Created a Run Once Task from the Orchestration.
* Viewed the Task Run's status generated from the Task.
* Viewed the Task Run's logs.

Great job! 

