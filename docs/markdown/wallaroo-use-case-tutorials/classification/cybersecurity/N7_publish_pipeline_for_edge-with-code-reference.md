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

<table><tr><th>name</th> <td>aloha-fraud-detector</td></tr><tr><th>created</th> <td>2024-05-01 16:30:53.995114+00:00</td></tr><tr><th>last_updated</th> <td>2024-05-01 20:27:46.871582+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6a137f96-72dd-448e-9206-b28334b02b8f, 8ede9ff5-e42a-404c-b248-fb4c4efd687d, 958397f7-1d3f-472c-8654-21283b8d6d85, ea59aae6-1565-4321-9086-7f8dd2a8e1c2, 41b9260e-21a8-4f16-ad56-c6267d3bae93, 46bdd5a3-fc22-41b7-b1ea-8287c99c241e, 28f5443a-ff67-4f4f-bfc3-f3f95f3c6f83, 435b73f7-76a1-4514-bd41-2cb94d3e78ff, 551242a7-fe4c-4a61-a4c7-e7fcc97509dc, d22eb0d2-9cff-4c5f-a851-10b1a19d8c44, 262909e9-8779-4c56-a994-725ddd0b58c8, 4cdf8e11-1b9c-44ab-a16d-abb054b5e9fe, 6b3529b1-1ff1-454b-8896-460c8c90d667</td></tr><tr><th>steps</th> <td>aloha-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

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

<table><tr><th>name</th> <td>aloha-fraud-detector</td></tr><tr><th>created</th> <td>2024-05-01 16:30:53.995114+00:00</td></tr><tr><th>last_updated</th> <td>2024-05-01 20:48:49.049405+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>66d70a35-21e2-4d16-868e-370d3cd4cc75, 6a137f96-72dd-448e-9206-b28334b02b8f, 8ede9ff5-e42a-404c-b248-fb4c4efd687d, 958397f7-1d3f-472c-8654-21283b8d6d85, ea59aae6-1565-4321-9086-7f8dd2a8e1c2, 41b9260e-21a8-4f16-ad56-c6267d3bae93, 46bdd5a3-fc22-41b7-b1ea-8287c99c241e, 28f5443a-ff67-4f4f-bfc3-f3f95f3c6f83, 435b73f7-76a1-4514-bd41-2cb94d3e78ff, 551242a7-fe4c-4a61-a4c7-e7fcc97509dc, d22eb0d2-9cff-4c5f-a851-10b1a19d8c44, 262909e9-8779-4c56-a994-725ddd0b58c8, 4cdf8e11-1b9c-44ab-a16d-abb054b5e9fe, 6b3529b1-1ff1-454b-8896-460c8c90d667</td></tr><tr><th>steps</th> <td>aloha-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

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
      <td>2024-05-01 20:49:10.647</td>
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
      <td>2024-05-01 20:49:10.647</td>
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
      <td>2024-05-01 20:49:10.647</td>
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
      <td>2024-05-01 20:49:10.647</td>
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
      <td>2024-05-01 20:49:10.647</td>
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
      <td>2024-05-01 20:49:10.647</td>
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
      <td>2024-05-01 20:49:10.647</td>
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
      <td>2024-05-01 20:49:10.647</td>
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
      <td>2024-05-01 20:49:10.647</td>
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
      <td>2024-05-01 20:49:10.647</td>
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

<table><tr><th>name</th> <td>aloha-fraud-detector</td></tr><tr><th>created</th> <td>2024-05-01 16:30:53.995114+00:00</td></tr><tr><th>last_updated</th> <td>2024-05-01 20:48:49.049405+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>66d70a35-21e2-4d16-868e-370d3cd4cc75, 6a137f96-72dd-448e-9206-b28334b02b8f, 8ede9ff5-e42a-404c-b248-fb4c4efd687d, 958397f7-1d3f-472c-8654-21283b8d6d85, ea59aae6-1565-4321-9086-7f8dd2a8e1c2, 41b9260e-21a8-4f16-ad56-c6267d3bae93, 46bdd5a3-fc22-41b7-b1ea-8287c99c241e, 28f5443a-ff67-4f4f-bfc3-f3f95f3c6f83, 435b73f7-76a1-4514-bd41-2cb94d3e78ff, 551242a7-fe4c-4a61-a4c7-e7fcc97509dc, d22eb0d2-9cff-4c5f-a851-10b1a19d8c44, 262909e9-8779-4c56-a994-725ddd0b58c8, 4cdf8e11-1b9c-44ab-a16d-abb054b5e9fe, 6b3529b1-1ff1-454b-8896-460c8c90d667</td></tr><tr><th>steps</th> <td>aloha-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

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
    Pipeline is publishing....... Published.

          <table>
              <tr><td>ID</td><td>19</td></tr>
              <tr><td>Pipeline Name</td><td>aloha-fraud-detector</td></tr>
              <tr><td>Pipeline Version</td><td>aba25c79-b700-4eb9-8d4a-db8ba0b5a0ca</td></tr>
              <tr><td>Status</td><td>Published</td></tr>
              <tr><td>Engine URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/fitzroy-mini:v2024.1.0-main-4963'>ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/fitzroy-mini:v2024.1.0-main-4963</a></td></tr>
              <tr><td>Pipeline URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/pipelines/aloha-fraud-detector:aba25c79-b700-4eb9-8d4a-db8ba0b5a0ca'>ghcr.io/wallaroolabs/doc-samples/pipelines/aloha-fraud-detector:aba25c79-b700-4eb9-8d4a-db8ba0b5a0ca</a></td></tr>
              <tr><td>Helm Chart URL</td><td>oci://<a href='https://ghcr.io/wallaroolabs/doc-samples/charts/aloha-fraud-detector'>ghcr.io/wallaroolabs/doc-samples/charts/aloha-fraud-detector</a></td></tr>
              <tr><td>Helm Chart Reference</td><td>ghcr.io/wallaroolabs/doc-samples/charts@sha256:b83a489d3bdfc90744f003f274cee63c66b4483f63d96dba6ac0f6bec8125171</td></tr>
              <tr><td>Helm Chart Version</td><td>0.0.1-aba25c79-b700-4eb9-8d4a-db8ba0b5a0ca</td></tr>
              <tr><td>Engine Config</td><td>{'engine': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 1.0, 'memory': '512Mi'}, 'accel': 'none', 'arch': 'x86', 'gpu': False}}, 'engineAux': {'autoscale': {'type': 'none'}, 'images': {}}}</td></tr>
              <tr><td>User Images</td><td>[]</td></tr>
              <tr><td>Created By</td><td>john.hummel@wallaroo.ai</td></tr>
              <tr><td>Created At</td><td>2024-05-01 20:49:53.680954+00:00</td></tr>
              <tr><td>Updated At</td><td>2024-05-01 20:49:53.680954+00:00</td></tr>
              <tr><td>Replaces</td><td></td></tr>
              <tr>
                  <td>Docker Run Command</td>
                  <td>
                      <table><tr><td>
<pre style="text-align: left">docker run \
    -p $EDGE_PORT:8080 \
    -e OCI_USERNAME=$OCI_USERNAME \
    -e OCI_PASSWORD=$OCI_PASSWORD \
    -e PIPELINE_URL=ghcr.io/wallaroolabs/doc-samples/pipelines/aloha-fraud-detector:aba25c79-b700-4eb9-8d4a-db8ba0b5a0ca \
    -e CONFIG_CPUS=1 ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/fitzroy-mini:v2024.1.0-main-4963</pre></td></tr></table>
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
    oci://ghcr.io/wallaroolabs/doc-samples/charts/aloha-fraud-detector \
    --namespace $HELM_INSTALL_NAMESPACE \
    --version 0.0.1-aba25c79-b700-4eb9-8d4a-db8ba0b5a0ca \
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

<table><tr><th>name</th><th>created</th><th>last_updated</th><th>deployed</th><th>arch</th><th>tags</th><th>versions</th><th>steps</th><th>published</th></tr><tr><td>aloha-fraud-detector</td><td>2024-01-May 16:30:53</td><td>2024-01-May 20:49:52</td><td>False</td><td>x86</td><td>none</td><td></td><td>aba25c79-b700-4eb9-8d4a-db8ba0b5a0ca, 66d70a35-21e2-4d16-868e-370d3cd4cc75, 6a137f96-72dd-448e-9206-b28334b02b8f, 8ede9ff5-e42a-404c-b248-fb4c4efd687d, 958397f7-1d3f-472c-8654-21283b8d6d85, ea59aae6-1565-4321-9086-7f8dd2a8e1c2, 41b9260e-21a8-4f16-ad56-c6267d3bae93, 46bdd5a3-fc22-41b7-b1ea-8287c99c241e, 28f5443a-ff67-4f4f-bfc3-f3f95f3c6f83, 435b73f7-76a1-4514-bd41-2cb94d3e78ff, 551242a7-fe4c-4a61-a4c7-e7fcc97509dc, d22eb0d2-9cff-4c5f-a851-10b1a19d8c44, 262909e9-8779-4c56-a994-725ddd0b58c8, 4cdf8e11-1b9c-44ab-a16d-abb054b5e9fe, 6b3529b1-1ff1-454b-8896-460c8c90d667</td><td>aloha-prime</td><td>True</td></tr><tr><td>model-pipe-3125</td><td>2024-29-Apr 18:35:53</td><td>2024-29-Apr 18:36:07</td><td>False</td><td>x86</td><td>none</td><td></td><td>31ef22ee-ff12-4ae7-b5ac-aa29dd134de4, 77be9ec8-5eb2-4e07-8154-70e2e7588ac2</td><td>ols-model-3125</td><td>False</td></tr><tr><td>edge-low-connection-demonstration</td><td>2024-23-Apr 17:01:22</td><td>2024-23-Apr 17:02:18</td><td>True</td><td>x86</td><td>none</td><td></td><td>226508c6-7205-4847-afe3-3d0f62c73096, b1cecb46-1777-4acf-93dd-c23880651517, fc666068-ede1-4020-b3c3-2e78d675dfbc</td><td>rf-house-price-estimator</td><td>True</td></tr><tr><td>edge-low-connection-demonstration</td><td>2024-23-Apr 16:10:34</td><td>2024-23-Apr 16:54:18</td><td>(unknown)</td><td>None</td><td>None</td><td></td><td>93efec2a-1fe0-4608-b76c-54fbfa6c0ebb, f9fa077e-0f42-48bb-ab41-ab746353af56, 00780a96-db2e-4f8e-b9ed-4d3d9b798054, 92d2062a-30c0-4123-985b-a6c687b727d3, 9989c65a-c662-4c0e-b25d-43bc48787633</td><td></td><td>True</td></tr><tr><td>houseprice-estimator</td><td>2024-22-Apr 20:34:49</td><td>2024-24-Apr 16:27:56</td><td>False</td><td>x86</td><td>none</td><td></td><td>42cb257d-3d21-4b71-9645-74919b4c862f, 6912f5f1-6fbe-4795-8cec-033d97291a18, eb94fe18-363a-498d-bd36-943289f0f74d, 4d4fdba1-dc5d-4c08-9375-e08ce8738730, b68ae767-dd6b-4f09-a196-a96d2cdaefb1, 3eedd163-bd54-4bde-943e-3f8b32a1f47c, 957ea8c5-d5c0-4629-82fb-fde09cf06958, e6fc1ebc-1a8f-48d2-9709-41c6820d6158, d0517194-4275-4684-860b-b35d64efba0a, 95cf17b2-19b1-40b6-82a5-d50b7f78094d, fad60d26-5f2f-467b-82ab-b97e60cfae2a, 71d8ae65-6b5e-422e-9d91-90fed507f74a, d80ef023-a703-4822-910d-b84f8d20174d, b99c6d8c-1d57-49ce-9690-07fa8f6988db, fbdf00f2-104e-41c6-9df7-86127bd2322e, 1ba316a5-6523-47f1-9e44-ec4c30e5710a</td><td>house-price-prime</td><td>True</td></tr><tr><td>edge-pipeline</td><td>2024-22-Apr 18:26:06</td><td>2024-22-Apr 18:38:22</td><td>True</td><td>x86</td><td>none</td><td></td><td>5e5d2584-a767-4145-84ee-c9c75b1f4e68, 3be0c7d7-709c-4399-8289-ec842f81902b, f7376d38-3793-41b4-8a81-2e2b969b37d0, ae8a787b-ccf7-4ed4-b579-319c983cb179, 22e3077a-86ae-4b5a-9e54-fe8090cdf673, 05c45b4d-527b-4d70-971b-b030940f7664</td><td>ccfraud</td><td>True</td></tr><tr><td>edge-pipeline</td><td>2024-22-Apr 18:08:21</td><td>2024-22-Apr 18:17:01</td><td>False</td><td>x86</td><td>none</td><td></td><td>5a85f1c8-2a56-4887-9b1e-13f44968d390, 98795ff5-2e56-465a-9041-77e426d8fd19, 5f492ec9-feb9-497d-933c-8f1018e29d8d, 2e75cba0-12df-45d2-8386-3f8c3e236b0d, e5964f75-ed49-4115-afa0-a9cca6ee50af, c52a053f-8ff8-47f1-aa13-a2c1711ec773, 068e0163-0ec5-4a5f-9ab8-f1939b743142, 0a81ccbe-84aa-4230-a7d2-ab079b6be605, 8975c2ec-ddfa-4cf5-afa4-538a2b3fddfd, 575ed34b-5435-4cf3-80be-99c0f832d83a, ee5783ac-a620-4e3e-885a-ed92718a00ef</td><td>ccfraud-keras</td><td>True</td></tr><tr><td>edge-low-connection-demo</td><td>2024-22-Apr 17:53:56</td><td>2024-23-Apr 15:58:20</td><td>False</td><td>x86</td><td>none</td><td></td><td>05b3a6b8-bfcb-4561-a2ca-a18c28e2932c, 2bfb13a7-8664-4d1b-b7c0-916376ef9f89, 3b19665f-f78c-4583-b27b-15b4a98c8ab2, ef5488c4-6a46-4b64-84eb-665280dc63f5, e9ae4850-fcdb-44f1-a3aa-523632390aef, cc4a10a2-4866-4ed6-a1d8-8b731e7bf8f8, 758d7e59-38c7-4fe7-b9b6-0b94bb902312, 0488841d-02d4-42e6-8b8f-387a5d42cc9d</td><td>rf-house-price-estimator</td><td>True</td></tr><tr><td>edge-pipeline-classification-cybersecurity</td><td>2024-22-Apr 17:04:23</td><td>2024-22-Apr 17:05:19</td><td>False</td><td>x86</td><td>none</td><td></td><td>21d4aa33-3a9a-4e55-9feb-6689eb822b7a, 93ca84ca-9121-4933-b44c-95539fe5c0ee, 3f585deb-71bd-4a0a-b08b-4970d4e60a77</td><td>aloha</td><td>True</td></tr><tr><td>vgg16-clustering-pipeline</td><td>2024-22-Apr 16:17:08</td><td>2024-22-Apr 16:48:01</td><td>False</td><td>x86</td><td>none</td><td></td><td>7a7509d5-c30b-4f33-82ae-49deaf79dbd1, 29d94f80-3c21-44fb-9e71-a5498c3bce3d, 412b8da5-ad4c-417c-9f6e-ad79d71522a4, 4233c4e7-517a-48e8-807a-b626834f45ec, 4ca1d45a-507d-42e2-8038-d608c543681a, a99f0a28-ad9e-4db3-9eea-113bdd9ca1cd, be19886c-3896-47d5-9935-35592f44ad7c</td><td>vgg16-clustering</td><td>True</td></tr><tr><td>new-edge-inline-replacement</td><td>2024-22-Apr 15:43:04</td><td>2024-22-Apr 15:44:46</td><td>False</td><td>x86</td><td>none</td><td></td><td>455a7840-08c3-43bb-b6a6-7535894e6055, 5210e01e-6d0f-4cdc-92ea-d3499bcc42fc, d61fcf2d-95ad-41e7-9e53-50610d9e0419</td><td>gbr-house-price-estimator</td><td>True</td></tr><tr><td>edge-inline-replacement-demon</td><td>2024-22-Apr 15:27:36</td><td>2024-22-Apr 15:40:50</td><td>False</td><td>x86</td><td>none</td><td></td><td>c6a1e945-7de0-4f2c-addb-4f4746114a86, 0ad2e53e-6c00-4949-8bd4-08ae289430d5, 7c702eca-8acf-45e0-bdcd-bbb48a5102e5, 2ef51c5c-bc58-49b3-9ecf-9aa4bb0a0bae, fbc4bf00-d97f-4be1-a47c-85c788dd90d5</td><td>xgb-house-price-estimator</td><td>True</td></tr></table>

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

<table><tr><th>id</th><th>pipeline_version_name</th><th>engine_url</th><th>pipeline_url</th><th>created_by</th><th>created_at</th><th>updated_at</th></tr><tr><td>19</td><td>aba25c79-b700-4eb9-8d4a-db8ba0b5a0ca</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/fitzroy-mini:v2024.1.0-main-4963'>ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/fitzroy-mini:v2024.1.0-main-4963</a></td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/pipelines/aloha-fraud-detector:aba25c79-b700-4eb9-8d4a-db8ba0b5a0ca'>ghcr.io/wallaroolabs/doc-samples/pipelines/aloha-fraud-detector:aba25c79-b700-4eb9-8d4a-db8ba0b5a0ca</a></td><td>john.hummel@wallaroo.ai</td><td>2024-01-May 20:49:53</td><td>2024-01-May 20:49:53</td></tr></table>

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

* `/pipelines/infer`.

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
!curl -X POST testboy.local:8080/infer -H "Content-Type: application/json; format=pandas-records" --data @../data/singleton.df.json
```

    [{"check_failures":[],"elapsed":[120996489,42784927],"model_name":"house-price-prime","model_version":"e49c245b-be7b-48a6-a6b8-93ac4c1fe69a","original_data":null,"outputs":[{"Float":{"data":[2176827.0],"dim":[1,1],"v":1}}],"pipeline_name":"houseprice-estimator","shadow_data":{},"time":1694628519169}]
