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

<table><tr><th>name</th> <td>rental-forecast</td></tr><tr><th>created</th> <td>2024-10-29 21:00:36.927945+00:00</td></tr><tr><th>last_updated</th> <td>2024-10-30 20:55:04.440894+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>workspace_id</th> <td>8</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-forecast</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>beca3565-bb16-41ca-83d6-cb6d9ba3514e, 585ee8cd-2f5e-4a1e-bb0d-6c88e6d94d3e, ceff9712-715b-41e6-a124-b174b62a9654, 0250f403-07c6-4b01-83bc-eebdc09bca22, 31b515bb-807f-4d64-b105-fc0ae6a582f2, 614a34e0-6024-4245-9919-1a85b7a1e5d2, 6a593faf-bea3-4f57-b9ec-5c1afe7f93a7, 4dce5be3-926c-419f-9868-3dbea7baf3c1, a601ce07-937c-436a-9735-0ac842173dfb, c0d16da5-5db7-4af1-95e4-cb0c316a4ef3, bd5eb43f-5a2b-493c-a04b-863dccccb55f, 89729096-6581-42b8-9b06-10d580d31e11, b98b86fb-5941-45b6-af5d-c33f80ba7986, aead5518-ffb2-4d18-8898-89575ba90a9f, a2a887c0-a91b-4af7-b579-506c79631fa4, b8ac836a-903b-4327-a4c9-5cc7fb382aa7, 3e18cd2d-c006-497b-a756-5ecc95aa8439, bd3f7d6a-e246-4456-98b9-35b90990b86d</td></tr><tr><th>steps</th> <td>forecast-control-model</td></tr><tr><th>published</th> <td>False</td></tr></table>

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

<table><tr><th>name</th> <td>rental-forecast</td></tr><tr><th>created</th> <td>2024-10-29 21:00:36.927945+00:00</td></tr><tr><th>last_updated</th> <td>2024-10-30 21:06:28.542567+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>workspace_id</th> <td>8</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-forecast</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>7936ad28-8cc4-44e9-8002-119a773ef9c6, beca3565-bb16-41ca-83d6-cb6d9ba3514e, 585ee8cd-2f5e-4a1e-bb0d-6c88e6d94d3e, ceff9712-715b-41e6-a124-b174b62a9654, 0250f403-07c6-4b01-83bc-eebdc09bca22, 31b515bb-807f-4d64-b105-fc0ae6a582f2, 614a34e0-6024-4245-9919-1a85b7a1e5d2, 6a593faf-bea3-4f57-b9ec-5c1afe7f93a7, 4dce5be3-926c-419f-9868-3dbea7baf3c1, a601ce07-937c-436a-9735-0ac842173dfb, c0d16da5-5db7-4af1-95e4-cb0c316a4ef3, bd5eb43f-5a2b-493c-a04b-863dccccb55f, 89729096-6581-42b8-9b06-10d580d31e11, b98b86fb-5941-45b6-af5d-c33f80ba7986, aead5518-ffb2-4d18-8898-89575ba90a9f, a2a887c0-a91b-4af7-b579-506c79631fa4, b8ac836a-903b-4327-a4c9-5cc7fb382aa7, 3e18cd2d-c006-497b-a756-5ecc95aa8439, bd3f7d6a-e246-4456-98b9-35b90990b86d</td></tr><tr><th>steps</th> <td>forecast-control-model</td></tr><tr><th>published</th> <td>False</td></tr></table>

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
      <td>2024-10-30 21:06:52.115</td>
      <td>[1526, 1550, 1708, 1005, 1623, 1712, 1530, 1605, 1538, 1746, 1472, 1589, 1913, 1815, 2115, 2475, 2927, 1635, 1812, 1107, 1450, 1917, 1807, 1461, 1969, 2402, 1446, 1851]</td>
      <td>[1764, 1749, 1743, 1741, 1740, 1740, 1740]</td>
      <td>1745.2858</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

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

    Waiting for undeployment - this will take up to 45s .................................... ok

<table><tr><th>name</th> <td>rental-forecast</td></tr><tr><th>created</th> <td>2024-10-29 21:00:36.927945+00:00</td></tr><tr><th>last_updated</th> <td>2024-10-30 21:06:28.542567+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>workspace_id</th> <td>8</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-forecast</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>7936ad28-8cc4-44e9-8002-119a773ef9c6, beca3565-bb16-41ca-83d6-cb6d9ba3514e, 585ee8cd-2f5e-4a1e-bb0d-6c88e6d94d3e, ceff9712-715b-41e6-a124-b174b62a9654, 0250f403-07c6-4b01-83bc-eebdc09bca22, 31b515bb-807f-4d64-b105-fc0ae6a582f2, 614a34e0-6024-4245-9919-1a85b7a1e5d2, 6a593faf-bea3-4f57-b9ec-5c1afe7f93a7, 4dce5be3-926c-419f-9868-3dbea7baf3c1, a601ce07-937c-436a-9735-0ac842173dfb, c0d16da5-5db7-4af1-95e4-cb0c316a4ef3, bd5eb43f-5a2b-493c-a04b-863dccccb55f, 89729096-6581-42b8-9b06-10d580d31e11, b98b86fb-5941-45b6-af5d-c33f80ba7986, aead5518-ffb2-4d18-8898-89575ba90a9f, a2a887c0-a91b-4af7-b579-506c79631fa4, b8ac836a-903b-4327-a4c9-5cc7fb382aa7, 3e18cd2d-c006-497b-a756-5ecc95aa8439, bd3f7d6a-e246-4456-98b9-35b90990b86d</td></tr><tr><th>steps</th> <td>forecast-control-model</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Publish the Pipeline for Edge Deployment

It worked! For a demo, we'll take working once as "tested". So now that we've tested our pipeline, we are ready to publish it for edge deployment.

Publishing it means assembling all of the configuration files and model assets and pushing them to an Open Container Initiative (OCI) repository set in the Wallaroo instance as the Edge Registry service.  DevOps engineers then retrieve that image and deploy it through Docker, Kubernetes, or similar deployments.

See [Edge Deployment Registry Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-edge-deployment/) for details on adding an OCI Registry Service to Wallaroo as the Edge Deployment Registry.

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
    Pipeline is publishing.......... Published.

          <table>
              <tr><td>ID</td><td>3</td></tr>
              <tr><td>Pipeline Name</td><td>rental-forecast</td></tr>
              <tr><td>Pipeline Version</td><td>027b3811-c07e-4807-a201-caa0b22e9fb7</td></tr>
              <tr><td>Status</td><td>Published</td></tr>
              <tr><td>Engine URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/fitzroy-mini:v2024.2.0-5761'>ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/fitzroy-mini:v2024.2.0-5761</a></td></tr>
              <tr><td>Pipeline URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/pipelines/rental-forecast:027b3811-c07e-4807-a201-caa0b22e9fb7'>ghcr.io/wallaroolabs/doc-samples/pipelines/rental-forecast:027b3811-c07e-4807-a201-caa0b22e9fb7</a></td></tr>
              <tr><td>Helm Chart URL</td><td>oci://<a href='https://ghcr.io/wallaroolabs/doc-samples/charts/rental-forecast'>ghcr.io/wallaroolabs/doc-samples/charts/rental-forecast</a></td></tr>
              <tr><td>Helm Chart Reference</td><td>ghcr.io/wallaroolabs/doc-samples/charts@sha256:f7fe0119e2a239922aaadb1ab7639220802f1f5ff7d0b6c85b5819937d5a1844</td></tr>
              <tr><td>Helm Chart Version</td><td>0.0.1-027b3811-c07e-4807-a201-caa0b22e9fb7</td></tr>
              <tr><td>Engine Config</td><td>{'engine': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 1.0, 'memory': '512Mi'}, 'accel': 'none', 'arch': 'x86', 'gpu': False}}, 'engineAux': {'autoscale': {'type': 'none'}, 'images': {}}}</td></tr>
              <tr><td>User Images</td><td>[]</td></tr>
              <tr><td>Created By</td><td>john.hansarick@wallaroo.ai</td></tr>
              <tr><td>Created At</td><td>2024-10-30 21:07:35.680932+00:00</td></tr>
              <tr><td>Updated At</td><td>2024-10-30 21:07:35.680932+00:00</td></tr>
              <tr><td>Replaces</td><td></td></tr>
              <tr>
                  <td>Docker Run Command</td>
                  <td>
                      <table><tr><td>
<pre style="text-align: left">docker run \
    -p $EDGE_PORT:8080 \
    -e OCI_USERNAME=$OCI_USERNAME \
    -e OCI_PASSWORD=$OCI_PASSWORD \
    -e PIPELINE_URL=ghcr.io/wallaroolabs/doc-samples/pipelines/rental-forecast:027b3811-c07e-4807-a201-caa0b22e9fb7 \
    -e CONFIG_CPUS=1 ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/fitzroy-mini:v2024.2.0-5761</pre></td></tr></table>
                      <br />
                      <i>
                          Note: Please set the <code>EDGE_PORT</code>, <code>OCI_USERNAME</code>, and <code>OCI_PASSWORD</code> environment variables.
                      </i>
                  </td>
              </tr>
              <tr>
                  <td>Helm Install Command</td>
                  <td>
                      <table><tr><td>
<pre style="text-align: left">helm install --atomic $HELM_INSTALL_NAME \
    oci://ghcr.io/wallaroolabs/doc-samples/charts/rental-forecast \
    --namespace $HELM_INSTALL_NAMESPACE \
    --version 0.0.1-027b3811-c07e-4807-a201-caa0b22e9fb7 \
    --set ociRegistry.username=$OCI_USERNAME \
    --set ociRegistry.password=$OCI_PASSWORD</pre></td></tr></table>
                      <br />
                      <i>
                          Note: Please set the <code>HELM_INSTALL_NAME</code>, <code>HELM_INSTALL_NAMESPACE</code>,
                          <code>OCI_USERNAME</code>, and <code>OCI_PASSWORD</code> environment variables.
                      </i>
                  </td>
              </tr>

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

<table><tr><th>name</th><th>created</th><th>last_updated</th><th>deployed</th><th>workspace_id</th><th>workspace_name</th><th>arch</th><th>accel</th><th>tags</th><th>versions</th><th>steps</th><th>published</th></tr><tr><td>imdb-reviewer</td><td>2024-29-Oct 17:06:42</td><td>2024-29-Oct 19:01:37</td><td>False</td><td>6</td><td>tutorial-workspace-john-sentiment-analysis</td><td>x86</td><td>none</td><td></td><td>8b0682d5-5601-4bb5-b34f-2e8822e14e45, 78eed0bf-af64-430c-8183-069d66e91e54, 430b0a14-9b70-4c4f-964e-264d906149ee, ddfc04b1-a0a3-4524-a456-b71921de84ba, 31a95d69-2133-4c98-96bb-8069fd45abc8, 0e8ef023-1300-46b2-a341-5885ae131995, db04b467-d79d-4af5-aadd-2dae014aa7ca, 00e32b0f-bfce-4895-8a6c-9716c06245ed, 2542cba0-9ff5-46d8-9204-9e9b5327199e, 8cec1024-a9cf-4b19-b6b6-df506f92de23</td><td>embedder</td><td>True</td></tr><tr><td>hf-summarizer</td><td>2024-29-Oct 19:53:41</td><td>2024-29-Oct 20:37:39</td><td>False</td><td>7</td><td>tutorial-workspace-summarization</td><td>x86</td><td>none</td><td></td><td>bc152b34-793b-4e7e-8a81-8e4345a62bdc, e7d816a1-b458-4f91-b845-4e3b53418154, e57078b2-0190-4f55-8a6c-4cfdba2c63a6, 74db0fd3-f1b6-429c-a082-04e2aedcc4e6, e216b612-9b34-44c7-9a02-6812b2b8838d</td><td>hf-summarizer</td><td>True</td></tr><tr><td>ccfraud-detector</td><td>2024-30-Oct 21:03:49</td><td>2024-30-Oct 21:06:43</td><td>True</td><td>9</td><td>tutorial-finserv-jch</td><td>x86</td><td>none</td><td></td><td>53b85633-3718-40f3-bb4f-833f298ce479, 7e2ceb8b-8bfb-4484-9ceb-be373ddad059, 556952a1-771a-47fa-997e-c0852d830bac, 5d1a453b-38d2-4619-a2b0-4adfec824345, ba96ee7a-e26d-4ce6-8b00-b1bee131592c</td><td>classification-finserv-prime</td><td>False</td></tr><tr><td>rental-forecast</td><td>2024-29-Oct 21:00:36</td><td>2024-30-Oct 21:07:35</td><td>False</td><td>8</td><td>tutorial-workspace-forecast</td><td>x86</td><td>none</td><td></td><td>027b3811-c07e-4807-a201-caa0b22e9fb7, 7936ad28-8cc4-44e9-8002-119a773ef9c6, beca3565-bb16-41ca-83d6-cb6d9ba3514e, 585ee8cd-2f5e-4a1e-bb0d-6c88e6d94d3e, ceff9712-715b-41e6-a124-b174b62a9654, 0250f403-07c6-4b01-83bc-eebdc09bca22, 31b515bb-807f-4d64-b105-fc0ae6a582f2, 614a34e0-6024-4245-9919-1a85b7a1e5d2, 6a593faf-bea3-4f57-b9ec-5c1afe7f93a7, 4dce5be3-926c-419f-9868-3dbea7baf3c1, a601ce07-937c-436a-9735-0ac842173dfb, c0d16da5-5db7-4af1-95e4-cb0c316a4ef3, bd5eb43f-5a2b-493c-a04b-863dccccb55f, 89729096-6581-42b8-9b06-10d580d31e11, b98b86fb-5941-45b6-af5d-c33f80ba7986, aead5518-ffb2-4d18-8898-89575ba90a9f, a2a887c0-a91b-4af7-b579-506c79631fa4, b8ac836a-903b-4327-a4c9-5cc7fb382aa7, 3e18cd2d-c006-497b-a756-5ecc95aa8439, bd3f7d6a-e246-4456-98b9-35b90990b86d</td><td>forecast-control-model</td><td>True</td></tr></table>

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

<table><tr><th>id</th><th>pipeline_version_name</th><th>engine_url</th><th>pipeline_url</th><th>created_by</th><th>created_at</th><th>updated_at</th></tr><tr><td>3</td><td>027b3811-c07e-4807-a201-caa0b22e9fb7</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/fitzroy-mini:v2024.2.0-5761'>ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/fitzroy-mini:v2024.2.0-5761</a></td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/pipelines/rental-forecast:027b3811-c07e-4807-a201-caa0b22e9fb7'>ghcr.io/wallaroolabs/doc-samples/pipelines/rental-forecast:027b3811-c07e-4807-a201-caa0b22e9fb7</a></td><td>john.hansarick@wallaroo.ai</td><td>2024-30-Oct 21:07:35</td><td>2024-30-Oct 21:07:35</td></tr></table>

## Congratulations!

You have now 

* Created a workspace and set it as the current workspace.
* Uploaded an ONNX model.
* Created a Wallaroo pipeline, and set the most recent version of the uploaded model as a pipeline step.
* Successfully send data to your pipeline for inference through the SDK and through an API call.

## DevOps - Pipeline Edge Deployment

Once a pipeline is deployed to the Edge Registry service, it can be deployed in environments such as Docker, Kubernetes, or similar container running services by a DevOps engineer.  Docker Run and Helm Install templates are provided as part of the pipeline publish display from the Wallaroo SDK.

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

### Edge Deployed Inference

The inference endpoint takes the following pattern:

* `/infer`

Wallaroo inference endpoint URLs accept the following data inputs through the `Content-Type` header:

* `Content-Type: application/vnd.apache.arrow.file`: For Apache Arrow tables.
* `Content-Type: application/json; format=pandas-records`: For pandas DataFrame in record format.

It returns a `application/json; format=pandas-records` - the same pandas record we've been working with.

### Edge Deployed Inference Exercise

Perform an inference on the deployed pipeline using `curl`.  This command will look like this:

```bash
!curl -X POST localhost:8080/infer -H "Content-Type: application/json; format=pandas-records" --data @../data/singleton.df.json
```

```python
!curl -X POST testboy.local:8080/pipelines/rental-forecast \
    -H "Content-Type: application/json; format=pandas-records" \
     -H "Accept:application/json; format=pandas-records" \
    --data @../data/testdata-standard.df.json
```
