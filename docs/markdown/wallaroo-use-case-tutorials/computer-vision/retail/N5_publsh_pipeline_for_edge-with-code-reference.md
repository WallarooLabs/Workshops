# Tutorial Notebook 4: Deploy Pipeline to Edge Devices

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
# run this to preload needed libraries 

import wallaroo
from wallaroo.object import EntityNotFoundError
from wallaroo.framework import Framework
from IPython.display import display
from IPython.display import Image
import pandas as pd
import json
import datetime
import time
import cv2
import matplotlib.pyplot as plt
import string
import random
import pyarrow as pa
import sys
import asyncio
pd.set_option('display.max_colwidth', None)

import sys
 
# setting path - only needed when running this from the `with-code` folder.
sys.path.append('../')

import utils
```

### Pre-exercise

If needed, log into Wallaroo and go to the workspace, pipeline, and most recent model version from the ones that you created in the previous notebook. Please refer to Notebook 1 to refresh yourself on how to log in and set your working environment to the appropriate workspace.

```python
## blank space to log in 

wl = wallaroo.Client()

# retrieve the previous workspace, model, and pipeline version

workspace_name = 'tutorial-workspace-john-cv'

workspace = wl.get_workspace(workspace_name)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

# optionally, examine your current workspace
wl.get_current_workspace()

model_name = 'mobilenet'

prime_model_version = wl.get_model(model_name)

module_post_process_model = wl.get_model("cv-post-process-drift-detection")

pipeline_name = 'cv-retail'

pipeline = wl.get_pipeline(pipeline_name)

# display the workspace, pipeline and model version
display(workspace)
display(pipeline)
display(prime_model_version)
display(module_post_process_model)
```

    {'name': 'tutorial-workspace-john-cv', 'id': 13, 'archived': False, 'created_by': 'john.hansarick@wallaroo.ai', 'created_at': '2024-11-04T21:08:24.55981+00:00', 'models': [{'name': 'cv-pixel-intensity', 'versions': 4, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 11, 6, 16, 8, 33, 942644, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 11, 5, 20, 38, 55, 258098, tzinfo=tzutc())}, {'name': 'mobilenet', 'versions': 2, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 11, 6, 17, 10, 14, 651037, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 11, 4, 21, 9, 40, 313224, tzinfo=tzutc())}, {'name': 'cv-post-process-drift-detection', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 11, 6, 17, 10, 16, 758351, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 11, 6, 17, 10, 16, 758351, tzinfo=tzutc())}, {'name': 'resnet50', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 11, 6, 17, 17, 30, 467944, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 11, 6, 17, 17, 30, 467944, tzinfo=tzutc())}], 'pipelines': [{'name': 'cv-retail', 'create_time': datetime.datetime(2024, 11, 4, 21, 10, 5, 287786, tzinfo=tzutc()), 'definition': '[]'}, {'name': 'cv-retail-observe', 'create_time': datetime.datetime(2024, 11, 5, 20, 35, 25, 831787, tzinfo=tzutc()), 'definition': '[]'}, {'name': 'cv-assays-demo', 'create_time': datetime.datetime(2024, 11, 5, 21, 52, 48, 545484, tzinfo=tzutc()), 'definition': '[]'}]}

<table><tr><th>name</th> <td>cv-retail</td></tr><tr><th>created</th> <td>2024-11-04 21:10:05.287786+00:00</td></tr><tr><th>last_updated</th> <td>2024-11-06 17:39:05.484688+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>workspace_id</th> <td>13</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-john-cv</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a0254129-7e5f-4a5a-8194-f175f9ed03f8, 176561de-a406-404b-aa5f-ce496f0c627d, 32a8ee27-39b8-43ad-86a0-69ac4ed8f6ed, ef9d0e34-cee3-4bd7-892c-219130abc978, bb43f0c5-f533-4ebd-9a6f-8ed59c633a9d, b5d76855-a136-4c9c-95e7-20b0b6f9bcd3, d35262a1-4958-42f7-81b0-e4fefec68f39, c4e1078b-26fa-4d11-a37a-266d9820235b, 1e8ec6fa-c8f3-4118-968b-0133cfc18a97, f410b97b-c1dd-4e23-99d9-e63410e100d6, 73d361a7-0b5f-4614-b513-61c141366e84, 4b41e45e-a917-4b48-9786-5e84d189afdd, 44ff0494-e30a-4a93-b5e3-4ce90b1b2368, 42c8d366-583d-44f4-ac4b-513103b5902c, d8be019e-2b7c-4c52-9e41-101b20ab0c2a, dd5e2f8a-e436-4b35-b2fb-189f6059dacc, 5a0772da-cde1-4fea-afc4-16313fcaa229, 7686f3ea-3781-4a95-aa4c-e99e46b9c47c, 4df9df54-6d12-4577-a970-f544128d0575</td></tr><tr><th>steps</th> <td>mobilenet</td></tr><tr><th>published</th> <td>False</td></tr></table>

<table>
        <tr>
          <td>Name</td>
          <td>mobilenet</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>d15d8b9d-9d98-4aa7-8545-ac915862146e</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>mobilenet.pt.onnx</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>9044c970ee061cc47e0c77e20b05e884be37f2a20aa9c0c3ce1993dbd486a830</td>
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
          <td>2024-06-Nov 17:10:14</td>
        </tr>
        <tr>
          <td>Workspace id</td>
          <td>13</td>
        </tr>
        <tr>
          <td>Workspace name</td>
          <td>tutorial-workspace-john-cv</td>
        </tr>
      </table>

<table>
        <tr>
          <td>Name</td>
          <td>cv-post-process-drift-detection</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>a335c538-bccf-40b9-b9a4-9296f03e6eb1</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>post-process-drift-detection.zip</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>eefc55277b091dd90c45704ff51bbd68dbc0f0f7e686930c5409a606659cefcc</td>
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
          <td>2024-06-Nov 17:10:37</td>
        </tr>
        <tr>
          <td>Workspace id</td>
          <td>13</td>
        </tr>
        <tr>
          <td>Workspace name</td>
          <td>tutorial-workspace-john-cv</td>
        </tr>
      </table>

## Deploy the Pipeline with the Model Version Step

As per the other tutorials:

1. Clear the pipeline of all steps.
1. Add the model version as a pipeline step.
1. Deploy the pipeline with the following deployment configuration:

```python
deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(1).memory("1Gi").build()
```

```python
# run this to set the  deployment configuration

deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(1).memory("1Gi").build()
```

```python
## blank space to deploy the pipeline

pipeline.clear()
pipeline.add_model_step(prime_model_version)
pipeline.add_model_step(module_post_process_model)

pipeline.deploy(deployment_config=deploy_config)
```

```python
## blank space to check pipeline status

pipeline.status()
```

    {'status': 'Starting',
     'details': [],
     'engines': [{'ip': '10.28.2.47',
       'name': 'engine-67757844c7-2gxbt',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'cv-retail',
          'status': 'Running',
          'version': '017138df-04f9-40b4-bfcc-900efb3bde2c'}]},
       'model_statuses': {'models': [{'name': 'mobilenet',
          'sha': '9044c970ee061cc47e0c77e20b05e884be37f2a20aa9c0c3ce1993dbd486a830',
          'status': 'Running',
          'version': 'd15d8b9d-9d98-4aa7-8545-ac915862146e'},
         {'name': 'cv-post-process-drift-detection',
          'sha': 'eefc55277b091dd90c45704ff51bbd68dbc0f0f7e686930c5409a606659cefcc',
          'status': 'Running',
          'version': 'a335c538-bccf-40b9-b9a4-9296f03e6eb1'}]}}],
     'engine_lbs': [{'ip': '10.28.2.48',
       'name': 'engine-lb-6676794678-zxbcc',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': None,
       'name': 'engine-sidekick-cv-post-process-drift-detection-26-6557db4p4nb2',
       'status': 'Pending',
       'reason': None,
       'details': ['containers with unready status: [engine-sidekick-cv-post-process-drift-detection-26]',
        'containers with unready status: [engine-sidekick-cv-post-process-drift-detection-26]'],
       'statuses': None}]}

### Sample Inference

Verify the pipeline is deployed properly with a sample inference with the file `./data/test_table.arrow`.

```python
# run this to show sample image

image = cv2.imread('../data/images/example/dairy_bottles.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12,8))
plt.grid(False)
plt.imshow(image)
plt.show()
```

    
{{<figure src="/images/2024.2/wallaroo-use-case-tutorials/computer-vision/retail/N5_publsh_pipeline_for_edge-with-code-reference_files/N5_publsh_pipeline_for_edge-with-code-reference_10_0.png" width="800" label="png">}}
    

```python
# run the following to convert the image to a dataframe

width, height = 640, 480
dfImage, resizedImage = utils.loadImageAndConvertToDataframe('../data/images/example/dairy_bottles.png', width, height)
```

```python
## blank space to run the sample inference

import time
startTime = time.time()
infResults = pipeline.infer(dfImage, timeout=600)
endTime = time.time()
infResults['out.avg_confidence']
```

    0    0.289506
    Name: out.avg_confidence, dtype: float64

## Undeploying Your Pipeline

You should always undeploy your pipelines when you are done with them, or don't need them for a while. This releases the resources that the pipeline is using for other processes to use. You can always redeploy the pipeline when you need it again. As a reminder, here are the commands to deploy and undeploy a pipeline:

```python

# "turn off" the pipeline and releaase its resources
my_pipeline.undeploy()
```

```python
## blank space to undeploy the pipeline

pipeline.undeploy()
```

<table><tr><th>name</th> <td>cv-retail</td></tr><tr><th>created</th> <td>2024-11-04 21:10:05.287786+00:00</td></tr><tr><th>last_updated</th> <td>2024-11-06 19:59:11.684240+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>workspace_id</th> <td>13</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-john-cv</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>017138df-04f9-40b4-bfcc-900efb3bde2c, a0254129-7e5f-4a5a-8194-f175f9ed03f8, 176561de-a406-404b-aa5f-ce496f0c627d, 32a8ee27-39b8-43ad-86a0-69ac4ed8f6ed, ef9d0e34-cee3-4bd7-892c-219130abc978, bb43f0c5-f533-4ebd-9a6f-8ed59c633a9d, b5d76855-a136-4c9c-95e7-20b0b6f9bcd3, d35262a1-4958-42f7-81b0-e4fefec68f39, c4e1078b-26fa-4d11-a37a-266d9820235b, 1e8ec6fa-c8f3-4118-968b-0133cfc18a97, f410b97b-c1dd-4e23-99d9-e63410e100d6, 73d361a7-0b5f-4614-b513-61c141366e84, 4b41e45e-a917-4b48-9786-5e84d189afdd, 44ff0494-e30a-4a93-b5e3-4ce90b1b2368, 42c8d366-583d-44f4-ac4b-513103b5902c, d8be019e-2b7c-4c52-9e41-101b20ab0c2a, dd5e2f8a-e436-4b35-b2fb-189f6059dacc, 5a0772da-cde1-4fea-afc4-16313fcaa229, 7686f3ea-3781-4a95-aa4c-e99e46b9c47c, 4df9df54-6d12-4577-a970-f544128d0575</td></tr><tr><th>steps</th> <td>mobilenet</td></tr><tr><th>published</th> <td>False</td></tr></table>

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
    Pipeline is publishing....... Published.

          <table>
              <tr><td>ID</td><td>9</td></tr>
              <tr><td>Pipeline Name</td><td>cv-retail</td></tr>
              <tr><td>Pipeline Version</td><td>bcda5046-decc-44ce-89d3-143c2054fe68</td></tr>
              <tr><td>Status</td><td>Published</td></tr>
              <tr><td>Engine URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/fitzroy-mini:v2024.2.0-5761'>ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/fitzroy-mini:v2024.2.0-5761</a></td></tr>
              <tr><td>Pipeline URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/pipelines/cv-retail:bcda5046-decc-44ce-89d3-143c2054fe68'>ghcr.io/wallaroolabs/doc-samples/pipelines/cv-retail:bcda5046-decc-44ce-89d3-143c2054fe68</a></td></tr>
              <tr><td>Helm Chart URL</td><td>oci://<a href='https://ghcr.io/wallaroolabs/doc-samples/charts/cv-retail'>ghcr.io/wallaroolabs/doc-samples/charts/cv-retail</a></td></tr>
              <tr><td>Helm Chart Reference</td><td>ghcr.io/wallaroolabs/doc-samples/charts@sha256:986c931e66c76dace563fcf64e9cc8f0b3997fe34c8c3ff58b8ad59022ff6317</td></tr>
              <tr><td>Helm Chart Version</td><td>0.0.1-bcda5046-decc-44ce-89d3-143c2054fe68</td></tr>
              <tr><td>Engine Config</td><td>{'engine': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 1.0, 'memory': '512Mi'}, 'accel': 'none', 'arch': 'x86', 'gpu': False}}, 'engineAux': {'autoscale': {'type': 'none'}, 'images': {}}}</td></tr>
              <tr><td>User Images</td><td>[]</td></tr>
              <tr><td>Created By</td><td>john.hansarick@wallaroo.ai</td></tr>
              <tr><td>Created At</td><td>2024-11-06 20:03:13.404008+00:00</td></tr>
              <tr><td>Updated At</td><td>2024-11-06 20:03:13.404008+00:00</td></tr>
              <tr><td>Replaces</td><td></td></tr>
              <tr>
                  <td>Docker Run Command</td>
                  <td>
                      <table><tr><td>
<pre style="text-align: left">docker run \
    -p $EDGE_PORT:8080 \
    -e OCI_USERNAME=$OCI_USERNAME \
    -e OCI_PASSWORD=$OCI_PASSWORD \
    -e PIPELINE_URL=ghcr.io/wallaroolabs/doc-samples/pipelines/cv-retail:bcda5046-decc-44ce-89d3-143c2054fe68 \
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
    oci://ghcr.io/wallaroolabs/doc-samples/charts/cv-retail \
    --namespace $HELM_INSTALL_NAMESPACE \
    --version 0.0.1-bcda5046-decc-44ce-89d3-143c2054fe68 \
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
## blank space to list the pipelines and view which are published

wl.list_pipelines()
```

<table><tr><th>name</th><th>created</th><th>last_updated</th><th>deployed</th><th>workspace_id</th><th>workspace_name</th><th>arch</th><th>accel</th><th>tags</th><th>versions</th><th>steps</th><th>published</th></tr><tr><td>ccfraud-detector</td><td>2024-31-Oct 18:23:11</td><td>2024-31-Oct 18:37:35</td><td>False</td><td>10</td><td>tutorial-finserv-john</td><td>x86</td><td>none</td><td></td><td>5c75bcaa-1d0d-437a-a425-533172a93112, 93a5709d-76f2-413d-b818-c6cd50767d71, 57659060-ca41-492c-811b-9ae2b368066a, cb236f0f-946a-4cc8-b52e-f1777af7b111, 72e0dbf8-d574-4888-8002-636079575534, ca121c30-8cfa-4420-8f71-bf47148d6ca3, a76ee1ea-bdee-4196-9c25-99af29abb597</td><td>classification-finserv-prime</td><td>True</td></tr><tr><td>imdb-reviewer</td><td>2024-29-Oct 17:06:42</td><td>2024-29-Oct 19:01:37</td><td>False</td><td>6</td><td>tutorial-workspace-john-sentiment-analysis</td><td>x86</td><td>none</td><td></td><td>8b0682d5-5601-4bb5-b34f-2e8822e14e45, 78eed0bf-af64-430c-8183-069d66e91e54, 430b0a14-9b70-4c4f-964e-264d906149ee, ddfc04b1-a0a3-4524-a456-b71921de84ba, 31a95d69-2133-4c98-96bb-8069fd45abc8, 0e8ef023-1300-46b2-a341-5885ae131995, db04b467-d79d-4af5-aadd-2dae014aa7ca, 00e32b0f-bfce-4895-8a6c-9716c06245ed, 2542cba0-9ff5-46d8-9204-9e9b5327199e, 8cec1024-a9cf-4b19-b6b6-df506f92de23</td><td>embedder</td><td>True</td></tr><tr><td>imdb-reviewer</td><td>2024-01-Nov 17:41:20</td><td>2024-01-Nov 18:27:38</td><td>False</td><td>11</td><td>tutorial-workspace-sentiment-analysis</td><td>x86</td><td>none</td><td></td><td>58f9cd8e-81d8-4fa9-8910-1124acbd626d, a84f07f6-2ad4-43c4-9ec5-c6318ea6550a, ea045efc-1cfb-4296-bece-f620cf479ed8, dc0fd017-f43f-4f00-821d-6e1a11ee88e2, a0d82d3f-14a3-4897-90b1-e4ca3e56d23c, 4024dae6-8e62-4720-8feb-c35c7224e1a7, f6287d00-b934-4ad8-9e92-cd20d544e6a5, d8d77a9d-5baa-4979-9e59-d746ce94dde2, 6a333530-d25a-4f42-9c74-97b5a96a150a</td><td>embedder</td><td>True</td></tr><tr><td>hf-summarizer</td><td>2024-29-Oct 19:53:41</td><td>2024-29-Oct 20:37:39</td><td>False</td><td>7</td><td>tutorial-workspace-summarization</td><td>x86</td><td>none</td><td></td><td>bc152b34-793b-4e7e-8a81-8e4345a62bdc, e7d816a1-b458-4f91-b845-4e3b53418154, e57078b2-0190-4f55-8a6c-4cfdba2c63a6, 74db0fd3-f1b6-429c-a082-04e2aedcc4e6, e216b612-9b34-44c7-9a02-6812b2b8838d</td><td>hf-summarizer</td><td>True</td></tr><tr><td>cv-retail</td><td>2024-04-Nov 21:10:05</td><td>2024-06-Nov 20:03:11</td><td>False</td><td>13</td><td>tutorial-workspace-john-cv</td><td>x86</td><td>none</td><td></td><td>bcda5046-decc-44ce-89d3-143c2054fe68, 017138df-04f9-40b4-bfcc-900efb3bde2c, a0254129-7e5f-4a5a-8194-f175f9ed03f8, 176561de-a406-404b-aa5f-ce496f0c627d, 32a8ee27-39b8-43ad-86a0-69ac4ed8f6ed, ef9d0e34-cee3-4bd7-892c-219130abc978, bb43f0c5-f533-4ebd-9a6f-8ed59c633a9d, b5d76855-a136-4c9c-95e7-20b0b6f9bcd3, d35262a1-4958-42f7-81b0-e4fefec68f39, c4e1078b-26fa-4d11-a37a-266d9820235b, 1e8ec6fa-c8f3-4118-968b-0133cfc18a97, f410b97b-c1dd-4e23-99d9-e63410e100d6, 73d361a7-0b5f-4614-b513-61c141366e84, 4b41e45e-a917-4b48-9786-5e84d189afdd, 44ff0494-e30a-4a93-b5e3-4ce90b1b2368, 42c8d366-583d-44f4-ac4b-513103b5902c, d8be019e-2b7c-4c52-9e41-101b20ab0c2a, dd5e2f8a-e436-4b35-b2fb-189f6059dacc, 5a0772da-cde1-4fea-afc4-16313fcaa229, 7686f3ea-3781-4a95-aa4c-e99e46b9c47c, 4df9df54-6d12-4577-a970-f544128d0575</td><td>mobilenet</td><td>True</td></tr><tr><td>yolov8n-tutorial</td><td>2024-04-Nov 16:29:35</td><td>2024-04-Nov 16:53:39</td><td>False</td><td>12</td><td>tutorial-workspace-cv-yolo-john</td><td>x86</td><td>none</td><td></td><td>bf8021b0-08e8-4870-963b-704957c8a887, 0d136492-7fb6-411d-ae36-81099cf75f79, 9151d2ff-4cee-449a-ab01-eb35ae086a8a, 17064dc6-92e2-49c6-9bb5-19ad183b148d, 98851747-c689-4696-b3fc-040de773cc3a, 2e95dd38-566e-483e-8625-3fac1f69eef8, 1f2d7d9d-c627-4043-a462-fc0445e643a6, 685008b0-d557-4e6d-a17b-e1dd13767d40, ebeccd79-e6c3-4ccb-9d01-69b95c179e6d</td><td>yolov8n-tutorial</td><td>True</td></tr><tr><td>cv-retail-observe</td><td>2024-05-Nov 20:35:25</td><td>2024-05-Nov 20:47:41</td><td>False</td><td>13</td><td>tutorial-workspace-john-cv</td><td>x86</td><td>none</td><td></td><td>9f39c71e-244e-49c3-8327-1fb64825e621, 78d1711b-455a-4ecd-a024-c324391340e3, 8e86b43d-b538-4358-b8fa-1a11f63cd82b</td><td>cv-pixel-intensity</td><td>False</td></tr><tr><td>retail-inv-tracker-edge-obs</td><td>2024-05-Nov 21:39:16</td><td>2024-05-Nov 21:39:16</td><td>False</td><td>14</td><td>cv-retail-edge</td><td>x86</td><td>none</td><td></td><td>c858205a-b635-4b08-929c-406cf8766a24, 68e35ec8-6fad-4280-8652-e61d6a585a28</td><td>resnet-with-intensity</td><td>False</td></tr><tr><td>rental-forecast</td><td>2024-29-Oct 21:00:36</td><td>2024-30-Oct 21:07:35</td><td>False</td><td>8</td><td>tutorial-workspace-forecast</td><td>x86</td><td>none</td><td></td><td>027b3811-c07e-4807-a201-caa0b22e9fb7, 7936ad28-8cc4-44e9-8002-119a773ef9c6, beca3565-bb16-41ca-83d6-cb6d9ba3514e, 585ee8cd-2f5e-4a1e-bb0d-6c88e6d94d3e, ceff9712-715b-41e6-a124-b174b62a9654, 0250f403-07c6-4b01-83bc-eebdc09bca22, 31b515bb-807f-4d64-b105-fc0ae6a582f2, 614a34e0-6024-4245-9919-1a85b7a1e5d2, 6a593faf-bea3-4f57-b9ec-5c1afe7f93a7, 4dce5be3-926c-419f-9868-3dbea7baf3c1, a601ce07-937c-436a-9735-0ac842173dfb, c0d16da5-5db7-4af1-95e4-cb0c316a4ef3, bd5eb43f-5a2b-493c-a04b-863dccccb55f, 89729096-6581-42b8-9b06-10d580d31e11, b98b86fb-5941-45b6-af5d-c33f80ba7986, aead5518-ffb2-4d18-8898-89575ba90a9f, a2a887c0-a91b-4af7-b579-506c79631fa4, b8ac836a-903b-4327-a4c9-5cc7fb382aa7, 3e18cd2d-c006-497b-a756-5ecc95aa8439, bd3f7d6a-e246-4456-98b9-35b90990b86d</td><td>forecast-control-model</td><td>True</td></tr><tr><td>ccfraud-detector</td><td>2024-30-Oct 21:03:49</td><td>2024-30-Oct 21:11:18</td><td>False</td><td>9</td><td>tutorial-finserv-jch</td><td>x86</td><td>none</td><td></td><td>f0d85564-7370-4e23-9a7a-d71b7c76bd71, fce1feba-954d-4118-8e48-bdabf404502b, 53b85633-3718-40f3-bb4f-833f298ce479, 7e2ceb8b-8bfb-4484-9ceb-be373ddad059, 556952a1-771a-47fa-997e-c0852d830bac, 5d1a453b-38d2-4619-a2b0-4adfec824345, ba96ee7a-e26d-4ce6-8b00-b1bee131592c</td><td>classification-finserv-prime</td><td>False</td></tr><tr><td>cv-assays-demo</td><td>2024-05-Nov 21:52:48</td><td>2024-06-Nov 16:08:59</td><td>True</td><td>13</td><td>tutorial-workspace-john-cv</td><td>x86</td><td>none</td><td></td><td>8a2f1f10-2f0c-465e-b00e-8e49122aeab1, ae57bdad-4fc1-4fae-a372-6d6ef3ba6799, 4bbfd131-1dac-4769-9922-1c8b03b90cb0, dbbeb481-087f-4bfa-8ebb-87772fb5624f, 37b3ff06-8961-4146-95c5-d78422674307, c780686b-6513-4d55-9625-b9e909ea0c22, 33141597-883b-4ecb-9f67-69aa444b3567, 7a1b6d81-07ca-4a9e-8e88-d49eed847b5e</td><td>cv-pixel-intensity</td><td>False</td></tr></table>

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
## blank space to show the pipeline publishes

pipeline.publishes()
```

<table><tr><th>id</th><th>pipeline_version_name</th><th>engine_url</th><th>pipeline_url</th><th>created_by</th><th>created_at</th><th>updated_at</th></tr><tr><td>9</td><td>bcda5046-decc-44ce-89d3-143c2054fe68</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/fitzroy-mini:v2024.2.0-5761'>ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/fitzroy-mini:v2024.2.0-5761</a></td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/pipelines/cv-retail:bcda5046-decc-44ce-89d3-143c2054fe68'>ghcr.io/wallaroolabs/doc-samples/pipelines/cv-retail:bcda5046-decc-44ce-89d3-143c2054fe68</a></td><td>john.hansarick@wallaroo.ai</td><td>2024-06-Nov 20:03:13</td><td>2024-06-Nov 20:03:13</td></tr></table>

## Congratulations!

You have now 

* Created a workspace and set it as the current workspace.
* Uploaded an ONNX model.
* Created a Wallaroo pipeline, and set the most recent version of the uploaded model as a pipeline step.
* Successfully send data to your pipeline for inference through the SDK and through an API call.

## DevOps - Pipeline Edge Deployment

Once a pipeline is deployed to the Edge Registry service, it can be deployed in environments such as Docker, Kubernetes, or similar container running services by a DevOps engineer.  Docker Run and Helm Install templates are provided as part of the pipeline publish display from the Wallaroo SDK.

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
## blank space to  run the command - replace testboy.local with the host

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
## blank space to  run the command - replace testboy.local with the host

!curl testboy.local:8080/models
```

### Edge Deployed Inference

The inference endpoint takes the following pattern:

* `/pipelines/infer`.

Wallaroo inference endpoint URLs accept the following data inputs through the `Content-Type` header:

* `Content-Type: application/vnd.apache.arrow.file`: For Apache Arrow tables.
* `Content-Type: application/json; format=pandas-records`: For pandas DataFrame in record format.

It returns a `application/json; format=pandas-records` - the same pandas record we've been working with.

### Edge Deployed Inference Exercise

Perform an inference on the deployed pipeline using `curl`.  This command will look like this:

```bash
!curl -X POST testboy.local:8080/infer \
    -H "Content-Type:Content-Type: application/json; format=pandas-records" \
    --data @'../data/dfimage.df.json' > edge_example.df.json
```

```python
## blank space to show edge inference

!curl -X POST testboy.local:8080/infer \
    -H "Content-Type:Content-Type: application/json; format=pandas-records" \
    --data @'../data/dfimage.df.json' > edge_example.df.json

```
