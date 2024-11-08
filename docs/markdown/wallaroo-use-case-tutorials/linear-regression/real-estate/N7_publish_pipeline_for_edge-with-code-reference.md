# Tutorial Notebook 7: Deploy Pipeline to Edge Devices

For this tutorial, we will take a Wallaroo pipeline and publish it to an Open Container (OCI) Registry.  The registry details are stored in the Wallaroo instance as the Edge Registry.  

In this set of exercises, you will:

1. Use a pre-trained model and deploy it to Wallaroo.
1. Perform sample inferences.
1. Publish the pipeline to the Edge Registry.
1. See the steps to deploy the published pipeline to an Edge device and perform inferences through it.

Deployment to the Edge allows data scientists to work in Wallaroo to test their models in Wallaroo, then once satisfied with the results publish those pipelines.  DevOps engineers then take those published pipeline details from the Edge registry and deploy them into Docker and Kubernetes environments.

This tutorial will demonstrate the following concepts:

* [Wallaroo Workspaces](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-workspace/):  Workspaces are environments were users upload models, create pipelines and other artifacts.  The workspace should be considered the fundamental area where work is done.  Workspaces are shared with other users to give them access to the same models, pipelines, etc.
* [Wallaroo Model Upload and Registration](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/): ML Models are uploaded to Wallaroo through the SDK or the MLOps API to a **workspace**.  ML models include default runtimes (ONNX, Python Step, and TensorFlow) that are run directly through the Wallaroo engine, and containerized runtimes (Hugging Face, PyTorch, etc) that are run through in a container through the Wallaroo engine.
* [Wallaroo Pipelines](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/): Pipelines are used to deploy models for inferencing.  Each model is a **pipeline step** in a pipelines, where the inputs of the previous step are fed into the next.  Pipeline steps can be ML models, Python scripts, or Arbitrary Python (these contain necessary models and artifacts for running a model).
* [Pipeline Edge Publication](https://docs.wallaroo.ai/20230300/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-publication/): How to publish a Wallaroo pipeline to an OCI registry, then deploy that pipeline into other environments.

For this tutorial, we will be providing pre-trained models in ONNX format, and have connected a sample Edge Registry to our Wallaroo instance.

For more Wallaroo procedures, see the [Wallaroo Documentation site](https://docs.wallaroo.ai).

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

    {'name': 'tutorial-workspace-john-05', 'id': 17, 'archived': False, 'created_by': 'aa707604-ec80-495a-a9a1-87774c8086d5', 'created_at': '2023-09-13T17:14:52.999726+00:00', 'models': [{'name': 'house-price-prime', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 9, 13, 17, 19, 20, 87935, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 9, 13, 17, 19, 20, 87935, tzinfo=tzutc())}], 'pipelines': [{'name': 'houseprice-estimator', 'create_time': datetime.datetime(2023, 9, 13, 17, 19, 35, 321223, tzinfo=tzutc()), 'definition': '[]'}]}

<table>
        <tr>
          <td>Name</td>
          <td>house-price-prime</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>e49c245b-be7b-48a6-a6b8-93ac4c1fe69a</td>
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
          <td>2023-13-Sep 17:19:20</td>
        </tr>
      </table>

<table><tr><th>name</th> <td>houseprice-estimator</td></tr><tr><th>created</th> <td>2023-09-13 17:19:35.321223+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-13 17:25:07.156747+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>e09b3751-cbbf-4dac-9415-0a4eb469211e, 3b9fbaf9-030a-49d4-b58b-ed20440a8ddf, 7909607d-af4f-4c55-87ae-6413e3b851da, 8d81e28b-a881-4f3f-8a56-09871bf24c84, 81d22f24-4161-441f-ae66-027f2852cc1f, d77fc33f-d830-4e43-9f60-53ee07fb153e</td></tr><tr><th>steps</th> <td>house-price-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

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

<table><tr><th>name</th> <td>houseprice-estimator</td></tr><tr><th>created</th> <td>2023-09-13 17:19:35.321223+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-13 17:54:45.173347+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>c586f765-b67a-402c-9d84-712ad0d1dc2d, e09b3751-cbbf-4dac-9415-0a4eb469211e, 3b9fbaf9-030a-49d4-b58b-ed20440a8ddf, 7909607d-af4f-4c55-87ae-6413e3b851da, 8d81e28b-a881-4f3f-8a56-09871bf24c84, 81d22f24-4161-441f-ae66-027f2852cc1f, d77fc33f-d830-4e43-9f60-53ee07fb153e</td></tr><tr><th>steps</th> <td>house-price-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

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
      <td>2023-09-13 17:54:50.949</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[659806.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-09-13 17:54:50.949</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[732883.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-09-13 17:54:50.949</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[419508.84]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-09-13 17:54:50.949</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[634028.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-09-13 17:54:50.949</td>
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
      <td>2023-09-13 17:54:50.949</td>
      <td>[4.0, 2.25, 2620.0, 98881.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1820.0, 800.0, 47.4662, -122.453, 1728.0, 95832.0, 63.0, 0.0, 0.0]</td>
      <td>[436151.13]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3996</th>
      <td>2023-09-13 17:54:50.949</td>
      <td>[3.0, 2.5, 2244.0, 4079.0, 2.0, 0.0, 0.0, 3.0, 7.0, 2244.0, 0.0, 47.2606, -122.254, 2077.0, 4078.0, 3.0, 0.0, 0.0]</td>
      <td>[284810.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3997</th>
      <td>2023-09-13 17:54:50.949</td>
      <td>[3.0, 1.75, 1490.0, 5000.0, 1.0, 0.0, 1.0, 3.0, 8.0, 1250.0, 240.0, 47.5257, -122.392, 1980.0, 5000.0, 61.0, 0.0, 0.0]</td>
      <td>[575571.44]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3998</th>
      <td>2023-09-13 17:54:50.949</td>
      <td>[4.0, 2.5, 2740.0, 5700.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2740.0, 0.0, 47.3535, -122.026, 3010.0, 5281.0, 8.0, 0.0, 0.0]</td>
      <td>[432262.38]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3999</th>
      <td>2023-09-13 17:54:50.949</td>
      <td>[5.0, 2.5, 2240.0, 7770.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1340.0, 900.0, 47.7198, -122.171, 1820.0, 7770.0, 36.0, 0.0, 0.0]</td>
      <td>[445873.13]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>4000 rows × 4 columns</p>

## Undeploying Your Pipeline

You should always undeploy your pipelines when you are done with them, or don't need them for a while. This releases the resources that the pipeline is using for other processes to use. You can always redeploy the pipeline when you need it again. As a reminder, here are the commands to deploy and undeploy a pipeline:

```python

# "turn off" the pipeline and releaase its resources
my_pipeline.undeploy()
```

```python
# blank space to undeploy the pipeline
pipeline.undeploy()
```

<table><tr><th>name</th> <td>houseprice-estimator</td></tr><tr><th>created</th> <td>2023-09-13 17:19:35.321223+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-13 17:54:45.173347+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>c586f765-b67a-402c-9d84-712ad0d1dc2d, e09b3751-cbbf-4dac-9415-0a4eb469211e, 3b9fbaf9-030a-49d4-b58b-ed20440a8ddf, 7909607d-af4f-4c55-87ae-6413e3b851da, 8d81e28b-a881-4f3f-8a56-09871bf24c84, 81d22f24-4161-441f-ae66-027f2852cc1f, d77fc33f-d830-4e43-9f60-53ee07fb153e</td></tr><tr><th>steps</th> <td>house-price-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Publish the Pipeline for Edge Deployment

It worked! For a demo, we'll take working once as "tested". So now that we've tested our pipeline, we are ready to publish it for edge deployment.

Publishing it means assembling all of the configuration files and model assets and pushing them to an Open Container Initiative (OCI) repository set in the Wallaroo instance as the Edge Registry service.  DevOps engineers then retrieve that image and deploy it through Docker, Kubernetes, or similar deployments.

See [Edge Deployment Registry Guide](https://staging.docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-edge-deployment/) for details on adding an OCI Registry Service to Wallaroo as the Edge Deployment Registry.

This is done through the SDK command `wallaroo.pipeline.publish(deployment_config)` which has the following parameters and returns.

#### Publish a Pipeline Parameters

The `publish` method takes the following parameters.  The containerized pipeline will be pushed to the Edge registry service with the model, pipeline configurations, and other artifacts needed to deploy the pipeline.

| Parameter | Type | Description |
|---|---|---|
| `deployment_config` | `wallaroo.deployment_config.DeploymentConfig` (*Optional*) | Sets the pipeline deployment configuration.  For example:    For more information on pipeline deployment configuration, see the [Wallaroo SDK Essentials Guide: Pipeline Deployment Configuration](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-deployment-config/).

#### Publish a Pipeline Returns

| Field | Type | Description |
|---|---|---|
| id | integer | Numerical Wallaroo id of the published pipeline. |
| pipeline version id | integer | Numerical Wallaroo id of the pipeline version published. |
| status | string | The status of the pipeline publication.  Values include:  <ul><li>PendingPublish: The pipeline publication is about to be uploaded or is in the process of being uploaded.</li><li>Published:  The pipeline is published and ready for use.</li></ul> |
| Engine URL | string | The URL of the published pipeline engine in the edge registry. |
| Pipeline URL | string | The URL of the published pipeline in the edge registry. |
| Helm Chart URL | string | The URL of the helm chart for the published pipeline in the edge registry. |
| Helm Chart Reference | string | The help chart reference. |
| Helm Chart Version | string | The version of the Helm Chart of the published pipeline.  This is also used as the Docker tag. |
| Engine Config | `wallaroo.deployment_config.DeploymentConfig` | The pipeline configuration included with the published pipeline. |
| Created At | DateTime | When the published pipeline was created. |
| Updated At | DateTime | When the published pipeline was updated. |

### Publish the Pipeline for Edge Deployment Exercise

We will now publish the pipeline to our Edge Deployment Registry with the `pipeline.publish(deployment_config)` command.  `deployment_config` is an optional field that specifies the pipeline deployment.  This can be overridden by the DevOps engineer during deployment.

In this example, assuming that the pipeline was saved to the variable `my_pipeline`, we would publish it to the Edge Registry already stored in the Wallaroo instance and store the pipeline publish to the variable `my_pub` with the following command:

```python
my_pub=pipeline.publish(deploy_config)
# display the publish
my_pub
```

```python
## blank space to publish the pipeline

my_pub=pipeline.publish(deploy_config)
# display the publish
my_pub

```

    Waiting for pipeline publish... It may take up to 600 sec.
    Pipeline is Publishing...Published.

<table>
    <tr><td>ID</td><td>20</td></tr>
    <tr><td>Pipeline Version</td><td>be3749d3-9f84-44bf-8354-bfcda725d6da</td></tr>
    <tr><td>Status</td><td>Published</td></tr>
    <tr><td>Engine URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.3.0-3798'>ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.3.0-3798</a></td></tr>
    <tr><td>Pipeline URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/pipelines/houseprice-estimator:be3749d3-9f84-44bf-8354-bfcda725d6da'>ghcr.io/wallaroolabs/doc-samples/pipelines/houseprice-estimator:be3749d3-9f84-44bf-8354-bfcda725d6da</a></td></tr>
    <tr><td>Helm Chart URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/charts/houseprice-estimator'>ghcr.io/wallaroolabs/doc-samples/charts/houseprice-estimator</a></td></tr>
    <tr><td>Helm Chart Reference</td><td>ghcr.io/wallaroolabs/doc-samples/charts@sha256:ae42f60664b4c20cb45710f6c75e09f88e04287ff72c61a06667c6795c1e567b</td></tr>
    <tr><td>Helm Chart Version</td><td>0.0.1-be3749d3-9f84-44bf-8354-bfcda725d6da</td></tr>
    <tr><td>Engine Config</td><td>{'engine': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 1.0, 'memory': '512Mi'}}}, 'engineAux': {'images': {}}, 'enginelb': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 1.0, 'memory': '512Mi'}}}}</td></tr>
    <tr><td>User Images</td><td>[]</td></tr>
    <tr><td>Created By</td><td>john.hummel@wallaroo.ai</td></tr>
    <tr><td>Created At</td><td>2023-09-13 17:55:37.576379+00:00</td></tr>
    <tr><td>Updated At</td><td>2023-09-13 17:55:37.576379+00:00</td></tr>
</table>

## List Published Pipelines

The method `wallaroo.client.list_pipelines()` shows a list of all pipelines in the Wallaroo instance, and includes the `published` field that indicates whether the pipeline was published to the registry (`True`), or has not yet been published (`False`).

### List Published Pipelines Exercise

List all pipelines and see which ones are published or not.  For example, if your client was saved to the variable `wl`, then the following will list the pipelines and display which ones are published.

```python
wl.list_pipelines()
```

```python
# list the pipelines and view which are published

wl.list_pipelines()
```

<table><tr><th>name</th><th>created</th><th>last_updated</th><th>deployed</th><th>tags</th><th>versions</th><th>steps</th><th>published</th></tr><tr><td>houseprice-estimator</td><td>2023-13-Sep 17:19:35</td><td>2023-13-Sep 17:55:35</td><td>False</td><td></td><td>be3749d3-9f84-44bf-8354-bfcda725d6da, c586f765-b67a-402c-9d84-712ad0d1dc2d, e09b3751-cbbf-4dac-9415-0a4eb469211e, 3b9fbaf9-030a-49d4-b58b-ed20440a8ddf, 7909607d-af4f-4c55-87ae-6413e3b851da, 8d81e28b-a881-4f3f-8a56-09871bf24c84, 81d22f24-4161-441f-ae66-027f2852cc1f, d77fc33f-d830-4e43-9f60-53ee07fb153e</td><td>house-price-prime</td><td>True</td></tr><tr><td>houseprice-estimator</td><td>2023-13-Sep 17:16:11</td><td>2023-13-Sep 17:16:14</td><td>False</td><td></td><td>59bbbcbf-5704-4cdf-9619-7251582fc6e5, 03c05d03-20aa-45d5-9830-c0c8f5ba3447</td><td>resnet-50</td><td>False</td></tr><tr><td>housing-pipe</td><td>2023-12-Sep 17:35:52</td><td>2023-12-Sep 17:40:44</td><td>False</td><td></td><td>05d941bb-6547-4608-be5d-4515388d205c, d957ce8d-9d70-477e-bc03-d58b70cd047a, ba8a411e-9318-4ba5-95f5-22c22be8c064, ab42a8de-3551-4551-bc36-9a71d323f81c</td><td>preprocess</td><td>False</td></tr><tr><td>python-step-demo-pipeline</td><td>2023-12-Sep 16:39:26</td><td>2023-12-Sep 16:39:48</td><td>False</td><td></td><td>eacb9246-3416-471c-a9e0-c70867db2144, 1a940a29-ff1c-4ca3-9920-1cedf3416777, 2d4683bb-669e-4a00-b562-582629500fa1</td><td>house-price-sample</td><td>False</td></tr><tr><td>houseprice-estimator</td><td>2023-11-Sep 18:54:03</td><td>2023-11-Sep 21:27:00</td><td>False</td><td></td><td>342f4605-9467-460e-866f-1b74e6e863d1, 47a5fb9f-2456-4132-abea-88a3147c4446, 8df7f3a0-4531-4a28-a757-51d8064b3ead, 4dbd5738-2847-4623-be31-12216ee17b37, b076d650-3c58-462c-acf7-cf04896012e3, 4bfcf67e-6887-4294-b54b-b0fb4369f801, 9d57c76e-3c3c-45f2-967e-4d24a8705de3, 92f2b4f3-494b-4d69-895f-9e767ac1869d, 6de0fcdc-5457-49d2-8f3d-58faa9024f29, ff68e2c7-77cd-4d17-b2df-0a2dd719c1cf, 88d4f659-4b5b-4e44-b0cc-134a6f5efa81, 4d4bdc85-c53e-4fe2-a3fa-1d0467cc9dee, 6d06d1e6-1351-44cd-a123-d47dc7aade25, cdeb5c90-2139-43b3-b1f3-69d3718d6fea, c85b49cc-2b89-4695-800b-ab1c15ce436e, f024be99-ea10-455f-a65a-28050dca7bbe, 7923eb8f-cc16-4b85-9233-a2f206e930a2, 6fbd6236-c642-4f07-8a47-17c6569f57f6, 7bdf459e-72db-4ea9-81bc-97f184caa403, 91bd39e1-acd6-4e96-85a9-8eaabebe7da0</td><td>house-price-prime</td><td>False</td></tr><tr><td>demandcurvepipeline</td><td>2023-11-Sep 18:28:08</td><td>2023-11-Sep 18:28:14</td><td>False</td><td></td><td>cf1e4357-c98c-490d-822f-8c252a43ff99, 6e7452b9-1464-488b-9a28-85ff765f18d6</td><td>curve-preprocess</td><td>False</td></tr><tr><td>edge-cv-retail</td><td>2023-08-Sep 19:09:06</td><td>2023-08-Sep 19:19:11</td><td>False</td><td></td><td>28986e69-0fb2-429d-8a36-08c0609d40cf, be34623f-b651-4331-b0d6-bebb058437ae, d575f835-4f71-4fd3-a7c0-d01cfd624368, c4bb5321-7578-491b-a14b-42209450aee8, 77367cd0-16ea-459b-b27c-407c7f05b542, e31fdf85-e2ac-4264-8175-22e61efceb40</td><td>mobilenet</td><td>True</td></tr><tr><td>edgebiolabspipeline</td><td>2023-08-Sep 18:50:52</td><td>2023-08-Sep 19:02:06</td><td>False</td><td></td><td>b411c8aa-8368-484f-a615-d2a2bce634ca, 0321956b-098c-47eb-8f4c-3bd90e443f2d, 88bcc7a5-d618-4c72-90e5-651f1e252db9, d2425979-98ac-468c-83db-dd4f542e7217, 59a163a3-e9c7-4213-88bf-d732bcae7dbd, d608aa0f-4961-496f-b57c-ce02299b4e39, bf426f14-e3c4-4450-81d4-e833026505a9, b56f68b5-ae1d-49dc-b640-c964b10b117f, 951e4096-b3d7-426d-9eee-2b763d4a0558, e36b6b52-5652-440e-b856-89e42782b62f, 46ad2f62-987f-459d-8283-f495300647fa, 9c283c94-0ecd-4160-998e-b462d03008e1, 78b125a9-5311-46e7-adc8-5794f9ca29f0</td><td>edgebiolabsmodel</td><td>True</td></tr><tr><td>edge-cv-demo</td><td>2023-08-Sep 18:25:24</td><td>2023-08-Sep 18:37:14</td><td>False</td><td></td><td>69a912fb-47da-4049-98d5-aa024e7d66b2, 482fc033-00a6-42e7-b359-90611b76f74d, 32805f9a-40eb-4366-b444-635ab466ef76, b412ff15-c87b-46ea-8d96-48868b7867f0, aaf2c947-af26-4b0e-9819-f8aca5657017, 7ad0a22c-6472-4390-8f33-a8b3eccc7877, c73bbf20-8fe3-4714-be5e-e35773fe4779, fc431a83-22dc-43db-8610-cde3095af584</td><td>resnet-50</td><td>True</td></tr><tr><td>edge-pipeline</td><td>2023-08-Sep 17:30:28</td><td>2023-08-Sep 18:21:00</td><td>False</td><td></td><td>2d8f9e1d-dc65-4e90-a5ce-ee619162d8cd, 1ea2d089-1127-464d-a980-e087d1f052e2</td><td>ccfraud</td><td>True</td></tr><tr><td>edge-pipeline</td><td>2023-08-Sep 17:24:44</td><td>2023-12-Sep 16:17:59</td><td>False</td><td></td><td>072f59a2-177c-44a6-8304-f1896e91b051, 55bb733e-cf49-4dea-87b0-79d561c3891e, 3d217006-de8f-435d-a50e-f498cc489ad8, 66ed071a-1b3c-4a90-b9a2-91891387c81b, 02679ba1-a2d0-47bb-83cb-4ccdcd340b23, 873582f4-4b39-4a69-a2b9-536a0e29927c, 079cf5a1-7e95-4cb7-ae40-381b538371db</td><td>ccfraud</td><td>True</td></tr><tr><td>edge-pipeline-classification-cybersecurity</td><td>2023-08-Sep 15:40:28</td><td>2023-08-Sep 15:45:22</td><td>False</td><td></td><td>60222730-4fb5-4179-b8bf-fa53762fecd1, 86040216-0bbb-4715-b08f-da461857c515, 34204277-bdbd-4ae2-9ce9-86dabe4be5f5, 729ccaa2-41b5-4c8f-89f4-fe1e98f2b303, 216bb86b-f6e8-498f-b8a5-020347355715</td><td>aloha</td><td>True</td></tr><tr><td>edge-pipeline</td><td>2023-08-Sep 15:36:03</td><td>2023-08-Sep 15:36:03</td><td>False</td><td></td><td>83b49e9e-f43d-4459-bb2a-7fa144352307, 73a5d31f-75f5-42c4-9a9d-3ee524113b6c</td><td>aloha</td><td>False</td></tr><tr><td>vgg16-clustering-pipeline</td><td>2023-08-Sep 14:52:44</td><td>2023-08-Sep 14:56:09</td><td>False</td><td></td><td>50d6586a-0661-4f26-802d-c71da2ceea2e, d94e44b3-7ff6-4138-8b76-be1795cb6690, 8d2a8143-2255-408a-bd09-e3008a5bde0b</td><td>vgg16-clustering</td><td>True</td></tr></table>

## List Publishes from a Pipeline

All publishes created from a pipeline are displayed with the `wallaroo.pipeline.publishes` method.  The `pipeline_version_id` is used to know what version of the pipeline was used in that specific publish.  This allows for pipelines to be updated over time, and newer versions to be sent and tracked to the Edge Deployment Registry service.

### List Publishes Parameters

N/A

### List Publishes Returns

A List of the following fields:

| Field | Type | Description |
|---|---|---|
| id | integer | Numerical Wallaroo id of the published pipeline. |
| pipeline_version_id | integer | Numerical Wallaroo id of the pipeline version published. |
| engine_url | string | The URL of the published pipeline engine in the edge registry. |
| pipeline_url | string | The URL of the published pipeline in the edge registry. |
| created_by | string | The email address of the user that published the pipeline.
| Created At | DateTime | When the published pipeline was created. |
| Updated At | DateTime | When the published pipeline was updated. |

### List Publishes from a Pipeline Exercise

List all of the publishes from our pipeline.  For example, if our pipeline is `my_pipeline`, then we would list all publishes from the pipeline with the following:

```python
my_pipeline.publishes()
```

```python
pipeline.publishes()
```

<table><tr><th>id</th><th>pipeline_version_name</th><th>engine_url</th><th>pipeline_url</th><th>created_by</th><th>created_at</th><th>updated_at</th></tr><tr><td>20</td><td>be3749d3-9f84-44bf-8354-bfcda725d6da</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.3.0-3798'>ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.3.0-3798</a></td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/pipelines/houseprice-estimator:be3749d3-9f84-44bf-8354-bfcda725d6da'>ghcr.io/wallaroolabs/doc-samples/pipelines/houseprice-estimator:be3749d3-9f84-44bf-8354-bfcda725d6da</a></td><td>john.hummel@wallaroo.ai</td><td>2023-13-Sep 17:55:37</td><td>2023-13-Sep 17:55:37</td></tr></table>

## Congratulations!

You have now 

* Created a workspace and set it as the current workspace.
* Uploaded an ONNX model.
* Created a Wallaroo pipeline, and set the most recent version of the uploaded model as a pipeline step.
* Successfully send data to your pipeline for inference through the SDK and through an API call.

## DevOps - Pipeline Edge Deployment

Once a pipeline is deployed to the Edge Registry service, it can be deployed in environments such as Docker, Kubernetes, or similar container running services by a DevOps engineer.

### Docker Deployment

First, the DevOps engineer must authenticate to the same OCI Registry service used for the Wallaroo Edge Deployment registry.

For more details, check with the documentation on your artifact service.  The following are provided for the three major cloud services:

* [Set up authentication for Docker](https://cloud.google.com/artifact-registry/docs/docker/authentication)
* [Authenticate with an Azure container registry](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-authentication?tabs=azure-cli)
* [Authenticating Amazon ECR Repositories for Docker CLI with Credential Helper](https://aws.amazon.com/blogs/compute/authenticating-amazon-ecr-repositories-for-docker-cli-with-credential-helper/)

For the deployment, the engine URL is specified with the following environmental variables:

```bash
{published engine url}
-e DEBUG=true -e OCI_REGISTRY={your registry server} \
-e CONFIG_CPUS=4 \ # optional number of CPUs to use
-e OCI_USERNAME={registry username} \
-e OCI_PASSWORD={registry token here} \
-e PIPELINE_URL={published pipeline url}
```

#### Docker Deployment Example

Using our sample environment, here's sample deployment using Docker with a computer vision ML model, the same used in the [Wallaroo Use Case Tutorials Computer Vision: Retail](https://docs.wallaroo.ai/wallaroo-use-case-tutorials/wallaroo-use-case-computer-vision/use-case-computer-vision-retail/) tutorials.

```bash
docker run -p 8080:8080 \
    -e DEBUG=true -e OCI_REGISTRY={your registry server} \
    -e CONFIG_CPUS=4 \
    -e OCI_USERNAME=oauth2accesstoken \
    -e OCI_PASSWORD={registry token here} \
    -e PIPELINE_URL={your registry server}/pipelines/edge-cv-retail:bf70eaf7-8c11-4b46-b751-916a43b1a555 \
    {your registry server}/engine:v2023.3.0-main-3707
```

### Docker Compose Deployment

For users who prefer to use `docker compose`, the following sample `compose.yaml` file is used to launch the Wallaroo Edge pipeline.  This is the same used in the [Wallaroo Use Case Tutorials Computer Vision: Retail](https://docs.wallaroo.ai/wallaroo-use-case-tutorials/wallaroo-use-case-computer-vision/use-case-computer-vision-retail/) tutorials.

```yml
services:
  engine:
    image: {Your Engine URL}
    ports:
      - 8080:8080
    environment:
      PIPELINE_URL: {Your Pipeline URL}
      OCI_REGISTRY: {Your Edge Registry URL}
      OCI_USERNAME:  {Your Registry Username}
      OCI_PASSWORD: {Your Token or Password}
      CONFIG_CPUS: 4
```

For example:

```yml
services:
  engine:
    image: sample-registry.com/engine:v2023.3.0-main-3707
    ports:
      - 8080:8080
    environment:
      PIPELINE_URL: sample-registry.com/pipelines/edge-cv-retail:bf70eaf7-8c11-4b46-b751-916a43b1a555
      OCI_REGISTRY: sample-registry.com
      OCI_USERNAME:  _json_key_base64
      OCI_PASSWORD: abc123
      CONFIG_CPUS: 4
```

#### Docker Compose Deployment Example

The deployment and undeployment is then just a simple `docker compose up` and `docker compose down`.  The following shows an example of deploying the Wallaroo edge pipeline using `docker compose`.

```bash
docker compose up
[+] Running 1/1
 ✔ Container cv_data-engine-1  Recreated                                                                                                                                                                 0.5s
Attaching to cv_data-engine-1
cv_data-engine-1  | Wallaroo Engine - Standalone mode
cv_data-engine-1  | Login Succeeded
cv_data-engine-1  | Fetching manifest and config for pipeline: sample-registry.com/pipelines/edge-cv-retail:bf70eaf7-8c11-4b46-b751-916a43b1a555
cv_data-engine-1  | Fetching model layers
cv_data-engine-1  | digest: sha256:c6c8869645962e7711132a7e17aced2ac0f60dcdc2c7faa79b2de73847a87984
cv_data-engine-1  |   filename: c6c8869645962e7711132a7e17aced2ac0f60dcdc2c7faa79b2de73847a87984
cv_data-engine-1  |   name: resnet-50
cv_data-engine-1  |   type: model
cv_data-engine-1  |   runtime: onnx
cv_data-engine-1  |   version: 693e19b5-0dc7-4afb-9922-e3f7feefe66d
cv_data-engine-1  |
cv_data-engine-1  | Fetched
cv_data-engine-1  | Starting engine
cv_data-engine-1  | Looking for preexisting `yaml` files in //modelconfigs
cv_data-engine-1  | Looking for preexisting `yaml` files in //pipelines
```

### Helm Deployment

Published pipelines can be deployed through the use of helm charts.

Helm deployments take up to two steps - the first step is in retrieving the required `values.yaml` and making updates to override.

1. Pull the helm charts from the published pipeline.  The two fields are the Helm Chart URL and the Helm Chart version to specify the OCI .    This typically takes the format of:

  ```bash
  helm pull oci://{published.helm_chart_url} --version {published.helm_chart_version}
  ```

1. Extract the `tgz` file and copy the `values.yaml` and copy the values used to edit engine allocations, etc.  The following are **required** for the deployment to run:

  ```yml
  ociRegistry:
    registry: {your registry service}
    username:  {registry username here}
    password: {registry token here}
  ```

  Store this into another file, suc as `local-values.yaml`.

1. Create the namespace to deploy the pipeline to.  For example, the namespace `wallaroo-edge-pipeline` would be:

  ```bash
  kubectl create -n wallaroo-edge-pipeline
  ```

1. Deploy the `helm` installation with `helm install` through one of the following options:
    1. Specify the `tgz` file that was downloaded and the local values file.  For example:

        ```bash
        helm install --namespace {namespace} --values {local values file} {helm install name} {tgz path}
        ```

    1. Specify the expended directory from the downloaded `tgz` file.

        ```bash
        helm install --namespace {namespace} --values {local values file} {helm install name} {helm directory path}
        ```

    1. Specify the Helm Pipeline Helm Chart and the Pipeline Helm Version.

        ```bash
        helm install --namespace {namespace} --values {local values file} {helm install name} oci://{published.helm_chart_url} --version {published.helm_chart_version}
        ```

1. Once deployed, the DevOps engineer will have to forward the appropriate ports to the `svc/engine-svc` service in the specific pipeline.  For example, using `kubectl port-forward` to the namespace `ccfraud` that would be:

    ```bash
    kubectl port-forward svc/engine-svc -n ccfraud01 8080 --address 0.0.0.0`
    ```

The following code segment generates a `docker run` template based on the previously published pipeline, assuming our publish was listed as `my_pub`.

```python
docker_deploy = f'''
docker run -p 8080:8080 \\
    -e DEBUG=true -e OCI_REGISTRY=$REGISTRYURL \\
    -e CONFIG_CPUS=4 \\
    -e OCI_USERNAME=$REGISTRYUSERNAME \\
    -e OCI_PASSWORD=$REGISTRYPASSWORD \\
    -e PIPELINE_URL={my_pub.pipeline_url} \\
    {my_pub.engine_url}
'''

print(docker_deploy)
```

    
    docker run -p 8080:8080 \
        -e DEBUG=true -e OCI_REGISTRY=$REGISTRYURL \
        -e CONFIG_CPUS=4 \
        -e OCI_USERNAME=$REGISTRYUSERNAME \
        -e OCI_PASSWORD=$REGISTRYPASSWORD \
        -e PIPELINE_URL=ghcr.io/wallaroolabs/doc-samples/pipelines/houseprice-estimator:be3749d3-9f84-44bf-8354-bfcda725d6da \
        ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.3.0-3798
    

### Docker Compose Deployment Exercise

Use the `docker compose up` command on your own `compose.yaml` using the sample above, replacing the `OCI_USERNAME` and `OCI_PASSWORD` with the values provided by your instructor.

## Edge Deployed Pipeline API Endpoints

Once deployed, we can check the pipelines and models available.  We'll use a `curl` command, but any HTTP based request will work the same way.

### Pipelines Endpoints

The endpoint `/pipelines` returns:

* **id** (*String*):  The name of the pipeline.
* **status** (*String*):  The status as either `Running`, or `Error` if there are any issues.

For this example, the deployment is made on a machine called `testboy.local`.  Replace this URL with the URL of you edge deployment.

#### Pipelines Endpoints Exercise

Use the following `curl` command to view the pipeline data.  For example, if the pipeline was deployed on `localhost`, then the command would be:

```bash
!curl locahost:8080/pipelines
```

```python
# blank space to run the command - replace testboy.local with the host

!curl testboy.local:8080/pipelines
```

    {"pipelines":[{"id":"houseprice-estimator","status":"Running"}]}

### Models Endpoints

The endpoint `/models` returns a List of models with the following fields:

* **name** (*String*): The model name.
* **sha** (*String*): The sha hash value of the ML model.
* **status** (*String*):  The status of either Running or Error if there are any issues.
* **version** (*String*):  The model version.  This matches the version designation used by Wallaroo to track model versions in UUID format.

#### Models Endpoints Exercise

Use the following `curl` command to view the models data.  For example, if the pipeline was deployed on `localhost`, then the command would be:

```bash
!curl locahost:8080/models
```

```python
# blank space to run the command - replace testboy.local with the host

!curl testboy.local:8080/models
```

    {"models":[{"name":"house-price-prime","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c","status":"Running","version":"e49c245b-be7b-48a6-a6b8-93ac4c1fe69a"}]}

### Edge Deployed Inference

The inference endpoint takes the following pattern:

* `/pipelines/{pipeline-name}`:  The `pipeline-name` is the same as returned from the [`/pipelines`](#list-pipelines) endpoint as `id`.

Wallaroo inference endpoint URLs accept the following data inputs through the `Content-Type` header:

* `Content-Type: application/vnd.apache.arrow.file`: For Apache Arrow tables.
* `Content-Type: application/json; format=pandas-records`: For pandas DataFrame in record format.

It returns a `application/json; format=pandas-records` - the same pandas record we've been working with.

### Edge Deployed Inference Exercise

Perform an inference on the deployed pipeline using `curl`.  This command will look like this:

```bash
!curl -X POST localhost:8080/pipelines/{YOUR PIPELINE NAME} -H "Content-Type: application/json; format=pandas-records" --data @../data/singleton.df.json
```

```python
!curl -X POST testboy.local:8080/pipelines/houseprice-estimator -H "Content-Type: application/json; format=pandas-records" --data @../data/singleton.df.json
```

    [{"check_failures":[],"elapsed":[120996489,42784927],"model_name":"house-price-prime","model_version":"e49c245b-be7b-48a6-a6b8-93ac4c1fe69a","original_data":null,"outputs":[{"Float":{"data":[2176827.0],"dim":[1,1],"v":1}}],"pipeline_name":"houseprice-estimator","shadow_data":{},"time":1694628519169}]
