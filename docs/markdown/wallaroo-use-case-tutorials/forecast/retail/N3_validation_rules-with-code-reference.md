# Tutorial Notebook 3: Observability Part 1 - Validation Rules

In the previous notebooks you uploaded the models and artifacts, then deployed the models to production through provisioning workspaces and pipelines. Now you're ready to put your feet up! But to keep your models operational, your work's not done once the model is in production. You must continue to monitor the behavior and performance of the model to insure that the model provides value to the business.

In this notebook, you will learn about adding validation rules to pipelines.

## Preliminaries

In the blocks below we will preload some required libraries.

```python
# preload needed libraries 

import wallaroo
from wallaroo.object import EntityNotFoundError
from wallaroo.framework import Framework

from IPython.display import display

# used to display DataFrame information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)

import json
import datetime
import time

# used for unique connection names

import string
import random
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

<table><tr><th>name</th> <td>rental-forecast</td></tr><tr><th>created</th> <td>2024-10-29 21:00:36.927945+00:00</td></tr><tr><th>last_updated</th> <td>2024-10-30 20:20:43.920831+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>workspace_id</th> <td>8</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-forecast</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a601ce07-937c-436a-9735-0ac842173dfb, c0d16da5-5db7-4af1-95e4-cb0c316a4ef3, bd5eb43f-5a2b-493c-a04b-863dccccb55f, 89729096-6581-42b8-9b06-10d580d31e11, b98b86fb-5941-45b6-af5d-c33f80ba7986, aead5518-ffb2-4d18-8898-89575ba90a9f, a2a887c0-a91b-4af7-b579-506c79631fa4, b8ac836a-903b-4327-a4c9-5cc7fb382aa7, 3e18cd2d-c006-497b-a756-5ecc95aa8439, bd3f7d6a-e246-4456-98b9-35b90990b86d</td></tr><tr><th>steps</th> <td>forecast-control-model</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Deploy the Pipeline

Add the model version as a pipeline step to our pipeline, and deploy the pipeline.  You may want to check the pipeline steps to verify that the right model version is set for the pipeline step.

```python
pipeline.clear()
pipeline.add_model_step(prime_model_version)

deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
pipeline.deploy(deployment_config=deploy_config)
```

    Waiting for deployment - this will take up to 45s .............. ok

<table><tr><th>name</th> <td>rental-forecast</td></tr><tr><th>created</th> <td>2024-10-29 21:00:36.927945+00:00</td></tr><tr><th>last_updated</th> <td>2024-10-30 20:42:37.744603+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>workspace_id</th> <td>8</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-forecast</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>0250f403-07c6-4b01-83bc-eebdc09bca22, 31b515bb-807f-4d64-b105-fc0ae6a582f2, 614a34e0-6024-4245-9919-1a85b7a1e5d2, 6a593faf-bea3-4f57-b9ec-5c1afe7f93a7, 4dce5be3-926c-419f-9868-3dbea7baf3c1, a601ce07-937c-436a-9735-0ac842173dfb, c0d16da5-5db7-4af1-95e4-cb0c316a4ef3, bd5eb43f-5a2b-493c-a04b-863dccccb55f, 89729096-6581-42b8-9b06-10d580d31e11, b98b86fb-5941-45b6-af5d-c33f80ba7986, aead5518-ffb2-4d18-8898-89575ba90a9f, a2a887c0-a91b-4af7-b579-506c79631fa4, b8ac836a-903b-4327-a4c9-5cc7fb382aa7, 3e18cd2d-c006-497b-a756-5ecc95aa8439, bd3f7d6a-e246-4456-98b9-35b90990b86d</td></tr><tr><th>steps</th> <td>forecast-control-model</td></tr><tr><th>published</th> <td>False</td></tr></table>

```python
while pipeline.status()['status'] != 'Running':
    time.sleep(15)
    print("Waiting for deployment.")
pipeline.status()['status']
```

    'Running'

```python
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
      <td>2024-10-30 20:43:01.694</td>
      <td>[1526, 1550, 1708, 1005, 1623, 1712, 1530, 1605, 1538, 1746, 1472, 1589, 1913, 1815, 2115, 2475, 2927, 1635, 1812, 1107, 1450, 1917, 1807, 1461, 1969, 2402, 1446, 1851]</td>
      <td>[1764, 1749, 1743, 1741, 1740, 1740, 1740]</td>
      <td>1745.2858</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

## Model Validation Rules

A simple way to try to keep your model's behavior up to snuff is to make sure that it receives inputs that it expects, and that its output is something that downstream systems can handle. This can entail specifying rules that document what you expect, and either enforcing these rules (by refusing to make a prediction), or at least logging an alert that the expectations described by your validation rules have been violated. As the developer of the model, the data scientist (along with relevant subject matter experts) will often be the person in the best position to specify appropriate validation rules.

In our house price prediction example, suppose you know that house prices in your market are typically in the range $750,000 to $1.5M dollars. Then you might want to set validation rules on your model pipeline to specify that you expect the model's predictions to also be in that range. Then, if the model predicts a value outside that range, the pipeline will log that one of the validation checks has failed; this allows you to investigate that instance further.

Wallaroo provides **validations** to detect anomalous data from inference inputs and outputs.  Validations are added to a Wallaroo pipeline with the `wallaroo.pipeline.add_validations` method.

Adding validations takes the format:

```python
pipeline.add_validations(
    validation_name_01 = polars.col(in|out.{column_name}) EXPRESSION,
    validation_name_02 = polars.col(in|out.{column_name}) EXPRESSION
    ...{additional rules}
)
```

* `validation_name`: The user provided name of the validation.  The names must match Python variable naming requirements.
  * **IMPORTANT NOTE**: Using the name `count` as a validation name **returns a warning**.  Any validation rules named `count` are dropped upon request and an warning returned.
* `polars.col(in|out.{column_name})`: Specifies the **input** or **output** for a specific field aka "column" in an inference result.  Wallaroo inference requests are in the format `in.{field_name}` for **inputs**, and `out.{field_name}` for **outputs**.
  * More than one field can be selected, as long as they follow the rules of the [polars 0.18 Expressions library](https://docs.pola.rs/docs/python/version/0.18/reference/expressions/index.html).
* `EXPRESSION`:  The expression to validate. When the expression returns **True**, that indicates an anomaly detected.

The [`polars` library version 0.18.5](https://docs.pola.rs/docs/python/version/0.18/index.html) is used to create the validation rule.  This is installed by default with the Wallaroo SDK.  This provides a powerful range of comparisons to organizations tracking anomalous data from their ML models.

When validations are added to a pipeline, inference request outputs return the following fields:

| Field | Type | Description |
|---|---|---|
| **anomaly.count** | **Integer** | The total of all validations that returned **True**. |
| **anomaly.{validation name}** | **Bool** | The output of the validation `{validation_name}`. |

When validation returns `True`, **an anomaly is detected**.

For example, adding the validation `fraud` to the following pipeline returns `anomaly.count` of `1` when the validation `fraud` returns `True`.  The validation `fraud` returns `True` when the **output** field **dense_1** at index **0** is greater than 0.9.

```python
sample_pipeline = wallaroo.client.build_pipeline("sample-pipeline")
sample_pipeline.add_model_step(model)

# add the validation
sample_pipeline.add_validations(
    fraud=pl.col("out.dense_1").list.get(0) > 0.9,
    )

# deploy the pipeline
sample_pipeline.deploy()

# sample inference
display(sample_pipeline.infer_from_file("dev_high_fraud.json", data_format='pandas-records'))
```

|&nbsp;|time|in.tensor|out.dense_1|anomaly.count|anomaly.fraud|
|---|---|---|---|---|---|
|0|2024-02-02 16:05:42.152|[1.0678324729, 18.1555563975, -1.6589551058, 5...]|[0.981199]|1|True|

### Detecting Anomalies from Inference Request Results

When an inference request is submitted to a Wallaroo pipeline with validations, the following fields are output:

| Field | Type | Description |
|---|---|---|
| **anomaly.count** | **Integer** | The total of all validations that returned **True**. |
| **anomaly.{validation name}** | **Bool** | The output of each pipeline validation `{validation_name}`. |

For example, adding the validation `fraud` to the following pipeline returns `anomaly.count` of `1` when the validation `fraud` returns `True`.

```python
sample_pipeline = wallaroo.client.build_pipeline("sample-pipeline")
sample_pipeline.add_model_step(model)

# add the validation
sample_pipeline.add_validations(
    fraud=pl.col("out.dense_1").list.get(0) > 0.9,
    )

# deploy the pipeline
sample_pipeline.deploy()

# sample inference
display(sample_pipeline.infer_from_file("dev_high_fraud.json", data_format='pandas-records'))
```

|&nbsp;|time|in.tensor|out.dense_1|anomaly.count|anomaly.fraud|
|---|---|---|---|---|---|
|0|2024-02-02 16:05:42.152|[1.0678324729, 18.1555563975, -1.6589551058, 5...]|[0.981199]|1|True|

### Model Validation Rules Exercise

Add some simple validation rules to the model pipeline that you created in a previous exercise.

* Add an upper bound or a lower bound to the model predictions.
* Try to create predictions that fall both in and out of the specified range.
* Look through the logs to find anomalies.

**HINT 1**: since the purpose of this exercise is try out validation rules, it might be a good idea to take a small data set and make predictions on that data set first, *then* set the validation rules based on those predictions, so that you can see the check failures trigger.

Here's an example:

```python
import polars as pl

sample_pipeline = sample_pipeline.add_validations(
    too_low=pl.col("out.cryptolocker").list.get(0) < 0.4
)
```

```python
from resources import simdb
from resources import util

def get_singleton_forecast(df, field):
    singleton = pd.DataFrame({field: [df[field].values.tolist()]})
    return singleton
```

```python
# blank space to set a validation rule on the pipeline and check if it triggers as expected

import polars as pl

pipeline = pipeline.add_validations(
    high_fraud=pl.col("out.weekly_average") > 1700
)

```

```python
# blank space to set a validation rule on the pipeline and check if it triggers as expected

deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
pipeline.deploy(deployment_config=deploy_config)
```

     ok

<table><tr><th>name</th> <td>rental-forecast</td></tr><tr><th>created</th> <td>2024-10-29 21:00:36.927945+00:00</td></tr><tr><th>last_updated</th> <td>2024-10-30 20:43:23.101933+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>workspace_id</th> <td>8</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-forecast</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>ceff9712-715b-41e6-a124-b174b62a9654, 0250f403-07c6-4b01-83bc-eebdc09bca22, 31b515bb-807f-4d64-b105-fc0ae6a582f2, 614a34e0-6024-4245-9919-1a85b7a1e5d2, 6a593faf-bea3-4f57-b9ec-5c1afe7f93a7, 4dce5be3-926c-419f-9868-3dbea7baf3c1, a601ce07-937c-436a-9735-0ac842173dfb, c0d16da5-5db7-4af1-95e4-cb0c316a4ef3, bd5eb43f-5a2b-493c-a04b-863dccccb55f, 89729096-6581-42b8-9b06-10d580d31e11, b98b86fb-5941-45b6-af5d-c33f80ba7986, aead5518-ffb2-4d18-8898-89575ba90a9f, a2a887c0-a91b-4af7-b579-506c79631fa4, b8ac836a-903b-4327-a4c9-5cc7fb382aa7, 3e18cd2d-c006-497b-a756-5ecc95aa8439, bd3f7d6a-e246-4456-98b9-35b90990b86d</td></tr><tr><th>steps</th> <td>forecast-control-model</td></tr><tr><th>published</th> <td>False</td></tr></table>

```python
sample_count = pd.read_csv('../data/test_data.csv')

inference_df = pd.DataFrame()

for i in range(30):
    inference_df = inference_df.append(get_singleton_forecast(sample_count.loc[i:i+30], 'count'), ignore_index=True)
inference_df

results = pipeline.infer(inference_df)
display(results.loc[results['anomaly.count'] > 0, ['time', 'out.weekly_average']])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.weekly_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18</th>
      <td>2024-10-30 20:45:59.602</td>
      <td>1800.1428</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2024-10-30 20:45:59.602</td>
      <td>1719.2858</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2024-10-30 20:45:59.602</td>
      <td>1794.5714</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2024-10-30 20:45:59.602</td>
      <td>1711.2858</td>
    </tr>
  </tbody>
</table>

## Clean Up

At this point, if you are not continuing on to the next notebook, undeploy your pipeline to give the resources back to the environment.

```python
## blank space to undeploy the pipeline

pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s .................................... ok

<table><tr><th>name</th> <td>rental-forecast</td></tr><tr><th>created</th> <td>2024-10-29 21:00:36.927945+00:00</td></tr><tr><th>last_updated</th> <td>2024-10-30 20:43:23.101933+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>workspace_id</th> <td>8</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-forecast</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>ceff9712-715b-41e6-a124-b174b62a9654, 0250f403-07c6-4b01-83bc-eebdc09bca22, 31b515bb-807f-4d64-b105-fc0ae6a582f2, 614a34e0-6024-4245-9919-1a85b7a1e5d2, 6a593faf-bea3-4f57-b9ec-5c1afe7f93a7, 4dce5be3-926c-419f-9868-3dbea7baf3c1, a601ce07-937c-436a-9735-0ac842173dfb, c0d16da5-5db7-4af1-95e4-cb0c316a4ef3, bd5eb43f-5a2b-493c-a04b-863dccccb55f, 89729096-6581-42b8-9b06-10d580d31e11, b98b86fb-5941-45b6-af5d-c33f80ba7986, aead5518-ffb2-4d18-8898-89575ba90a9f, a2a887c0-a91b-4af7-b579-506c79631fa4, b8ac836a-903b-4327-a4c9-5cc7fb382aa7, 3e18cd2d-c006-497b-a756-5ecc95aa8439, bd3f7d6a-e246-4456-98b9-35b90990b86d</td></tr><tr><th>steps</th> <td>forecast-control-model</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Congratulations!

In this tutorial you have

* Set a validation rule on your house price prediction pipeline.
* Detected model predictions that failed the validation rule.

In the next notebook, you will learn how to monitor the distribution of model outputs for drift away from expected behavior.
