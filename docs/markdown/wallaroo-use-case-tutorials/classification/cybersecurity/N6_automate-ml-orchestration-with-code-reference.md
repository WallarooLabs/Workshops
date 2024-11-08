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

<table><tr><th>name</th> <td>aloha-fraud-detector</td></tr><tr><th>created</th> <td>2024-05-01 16:30:53.995114+00:00</td></tr><tr><th>last_updated</th> <td>2024-05-01 20:03:01.505745+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>8ede9ff5-e42a-404c-b248-fb4c4efd687d, 958397f7-1d3f-472c-8654-21283b8d6d85, ea59aae6-1565-4321-9086-7f8dd2a8e1c2, 41b9260e-21a8-4f16-ad56-c6267d3bae93, 46bdd5a3-fc22-41b7-b1ea-8287c99c241e, 28f5443a-ff67-4f4f-bfc3-f3f95f3c6f83, 435b73f7-76a1-4514-bd41-2cb94d3e78ff, 551242a7-fe4c-4a61-a4c7-e7fcc97509dc, d22eb0d2-9cff-4c5f-a851-10b1a19d8c44, 262909e9-8779-4c56-a994-725ddd0b58c8, 4cdf8e11-1b9c-44ab-a16d-abb054b5e9fe, 6b3529b1-1ff1-454b-8896-460c8c90d667</td></tr><tr><th>steps</th> <td>aloha-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

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

<table><tr><th>name</th> <td>aloha-fraud-detector</td></tr><tr><th>created</th> <td>2024-05-01 16:30:53.995114+00:00</td></tr><tr><th>last_updated</th> <td>2024-05-01 20:27:46.871582+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6a137f96-72dd-448e-9206-b28334b02b8f, 8ede9ff5-e42a-404c-b248-fb4c4efd687d, 958397f7-1d3f-472c-8654-21283b8d6d85, ea59aae6-1565-4321-9086-7f8dd2a8e1c2, 41b9260e-21a8-4f16-ad56-c6267d3bae93, 46bdd5a3-fc22-41b7-b1ea-8287c99c241e, 28f5443a-ff67-4f4f-bfc3-f3f95f3c6f83, 435b73f7-76a1-4514-bd41-2cb94d3e78ff, 551242a7-fe4c-4a61-a4c7-e7fcc97509dc, d22eb0d2-9cff-4c5f-a851-10b1a19d8c44, 262909e9-8779-4c56-a994-725ddd0b58c8, 4cdf8e11-1b9c-44ab-a16d-abb054b5e9fe, 6b3529b1-1ff1-454b-8896-460c8c90d667</td></tr><tr><th>steps</th> <td>aloha-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Sample Inference

Verify the pipeline is deployed properly with a sample inference with the file `./data/test_data.df.json`.

```python
# sample inference from previous code here

pipeline.infer_from_file('../data/data-1k.df.json')
```

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
      <td>2024-05-01 20:28:06.155</td>
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
      <td>2024-05-01 20:28:06.155</td>
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
      <td>2024-05-01 20:28:06.155</td>
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
      <td>2024-05-01 20:28:06.155</td>
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
      <td>2024-05-01 20:28:06.155</td>
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
      <td>2024-05-01 20:28:06.155</td>
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
      <td>2024-05-01 20:28:06.155</td>
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
      <td>2024-05-01 20:28:06.155</td>
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
      <td>2024-05-01 20:28:06.155</td>
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
      <td>2024-05-01 20:28:06.155</td>
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
    workspace_name="tutorial-workspace-john"

if "pipeline_name" in arguments:
    pipeline_name = arguments['pipeline_name']
else:
    pipeline_name="aloha-prime"

print(f"Workspace: {workspace_name}")
workspace = wl.get_workspace(workspace_name)

wl.set_current_workspace(workspace)
print(workspace)

# the pipeline is assumed to be deployed
print(f"Pipeline: {pipeline_name}")
pipeline = wl.get_pipeline(pipeline_name, workspace)
print(pipeline)

print(pipeline.status())

inference_result = pipeline.infer_from_file('./data/data-1k.df.json')
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
orchestration = wl.upload_orchestration(name="cybersecurity-john-sample", 
                                        path="../orchestration/cybersecurity-orchestration.zip")

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

<table><tr><th>id</th><th>name</th><th>status</th><th>filename</th><th>sha</th><th>created at</th><th>updated at</th></tr><tr><td>b347e2b8-2029-4918-b10e-88130f8b4da7</td><td>cybersecurity jch</td><td>ready</td><td>cybersecurity-orchestration.zip</td><td>9631c1...dd42aa</td><td>2024-01-May 20:32:14</td><td>2024-01-May 20:33:05</td></tr><tr><td>d3165b5f-02b6-49b2-a481-bc19b45e736b</td><td>cybersecurity-john</td><td>ready</td><td>cybersecurity-orchestration.zip</td><td>9631c1...dd42aa</td><td>2024-01-May 20:37:39</td><td>2024-01-May 20:38:28</td></tr><tr><td>54d61445-1001-4747-b1bf-3d1f333ff725</td><td>cybersecurity-john-sample</td><td>ready</td><td>cybersecurity-orchestration.zip</td><td>f90818...8772a5</td><td>2024-01-May 20:42:36</td><td>2024-01-May 20:43:25</td></tr></table>

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
    <td>ID</td><td>54d61445-1001-4747-b1bf-3d1f333ff725</td>
  </tr>
  <tr>
    <td>Name</td><td>cybersecurity-john-sample</td>
  </tr>
  <tr>
    <td>File Name</td><td>cybersecurity-orchestration.zip</td>
  </tr>
  <tr>
    <td>SHA</td><td>f9081889579ff91a51d4f1d3ec34bfa8633b6deadd65c7828970151a258772a5</td>
  </tr>
  <tr>
    <td>Status</td><td>ready</td>
  </tr>
  <tr>
    <td>Created At</td><td>2024-01-May 20:42:36</td>
  </tr>
  <tr>
    <td>Updated At</td><td>2024-01-May 20:43:25</td>
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
                                         "pipeline_name":pipeline_name
                                         }
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

<table><tr><th>id</th><th>name</th><th>last run status</th><th>type</th><th>active</th><th>schedule</th><th>created at</th><th>updated at</th></tr><tr><td>fea6316c-77a0-4a63-abdf-1c872d55353a</td><td>real estate task</td><td>running</td><td>Temporary Run</td><td>True</td><td>-</td><td>2024-01-May 20:43:31</td><td>2024-01-May 20:43:37</td></tr><tr><td>012f55ec-11af-4ae5-b980-63298be2eaf7</td><td>real estate task</td><td>failure</td><td>Temporary Run</td><td>True</td><td>-</td><td>2024-01-May 20:38:33</td><td>2024-01-May 20:38:39</td></tr><tr><td>c9ca8499-0cc2-4546-bebe-e99a0de3b1e5</td><td>real estate task</td><td>failure</td><td>Temporary Run</td><td>True</td><td>-</td><td>2024-01-May 20:33:49</td><td>2024-01-May 20:33:55</td></tr></table>

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
  <tr><td>Task</td><td>fea6316c-77a0-4a63-abdf-1c872d55353a</td></tr>
  <tr><td>Pod ID</td><td>107dd2f8-7e3c-4cbe-b072-3ed0d95a9222</td></tr>
  <tr><td>Status</td><td>success</td></tr>
  <tr><td>Created At</td><td>2024-01-May 20:43:34</td></tr>
  <tr><td>Updated At</td><td>2024-01-May 20:43:34</td></tr>
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
task_run.logs()
```

<pre><code>2024-01-May 20:43:44 Workspace: tutorial-workspace-john-cybersecurity
2024-01-May 20:43:44 {'name': 'tutorial-workspace-john-cybersecurity', 'id': 14, 'archived': False, 'created_by': '76b893ff-5c30-4f01-bd9e-9579a20fc4ea', 'created_at': '2024-05-01T16:30:01.177583+00:00', 'models': [{'name': 'aloha-prime', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 5, 1, 16, 30, 43, 651533, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 5, 1, 16, 30, 43, 651533, tzinfo=tzutc())}, {'name': 'aloha-challenger', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 5, 1, 16, 38, 56, 600586, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 5, 1, 16, 38, 56, 600586, tzinfo=tzutc())}], 'pipelines': [{'name': 'aloha-fraud-detector', 'create_time': datetime.datetime(2024, 5, 1, 16, 30, 53, 995114, tzinfo=tzutc()), 'definition': '[]'}]}
2024-01-May 20:43:44 Pipeline: aloha-fraud-detector
2024-01-May 20:43:44 {'name': 'aloha-fraud-detector', 'create_time': datetime.datetime(2024, 5, 1, 16, 30, 53, 995114, tzinfo=tzutc()), 'definition': '[]'}
2024-01-May 20:43:44 {'status': 'Running', 'details': [], 'engines': [{'ip': '10.28.3.118', 'name': 'engine-65c5dfd799-9sm4n', 'status': 'Running', 'reason': None, 'details': [], 'pipeline_statuses': {'pipelines': [{'id': 'aloha-fraud-detector', 'status': 'Running', 'version': '6a137f96-72dd-448e-9206-b28334b02b8f'}]}, 'model_statuses': {'models': [{'name': 'aloha-prime', 'sha': 'fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520', 'status': 'Running', 'version': 'c719bc50-f83f-4c79-b4af-f66395a8da04'}]}}], 'engine_lbs': [{'ip': '10.28.2.121', 'name': 'engine-lb-d7cc8fc9c-q9dzh', 'status': 'Running', 'reason': None, 'details': []}], 'sidekicks': []}
2024-01-May 20:43:44                        time  ... anomaly.count
2024-01-May 20:43:44 0   2024-05-01 20:43:41.349  ...             0
2024-01-May 20:43:44 1   2024-05-01 20:43:41.349  ...             0
2024-01-May 20:43:44 2   2024-05-01 20:43:41.349  ...             0
2024-01-May 20:43:44 3   2024-05-01 20:43:41.349  ...             0
2024-01-May 20:43:44 4   2024-05-01 20:43:41.349  ...             0
2024-01-May 20:43:44 ..                      ...  ...           ...
2024-01-May 20:43:44 995 2024-05-01 20:43:41.349  ...             0
2024-01-May 20:43:44 996 2024-05-01 20:43:41.349  ...             0
2024-01-May 20:43:44 997 2024-05-01 20:43:41.349  ...             0
2024-01-May 20:43:44 998 2024-05-01 20:43:41.349  ...             0
2024-01-May 20:43:44 999 2024-05-01 20:43:41.349  ...             0
2024-01-May 20:43:44 
2024-01-May 20:43:44 [1000 rows x 18 columns]</code></pre>

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

<table><tr><th>name</th> <td>aloha-fraud-detector</td></tr><tr><th>created</th> <td>2024-05-01 16:30:53.995114+00:00</td></tr><tr><th>last_updated</th> <td>2024-05-01 20:27:46.871582+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6a137f96-72dd-448e-9206-b28334b02b8f, 8ede9ff5-e42a-404c-b248-fb4c4efd687d, 958397f7-1d3f-472c-8654-21283b8d6d85, ea59aae6-1565-4321-9086-7f8dd2a8e1c2, 41b9260e-21a8-4f16-ad56-c6267d3bae93, 46bdd5a3-fc22-41b7-b1ea-8287c99c241e, 28f5443a-ff67-4f4f-bfc3-f3f95f3c6f83, 435b73f7-76a1-4514-bd41-2cb94d3e78ff, 551242a7-fe4c-4a61-a4c7-e7fcc97509dc, d22eb0d2-9cff-4c5f-a851-10b1a19d8c44, 262909e9-8779-4c56-a994-725ddd0b58c8, 4cdf8e11-1b9c-44ab-a16d-abb054b5e9fe, 6b3529b1-1ff1-454b-8896-460c8c90d667</td></tr><tr><th>steps</th> <td>aloha-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

