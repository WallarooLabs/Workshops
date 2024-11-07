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

```

## Login to Wallaroo

Retrieve the previous workspace, model versions, and pipelines used in the previous notebook.

```python
## blank space to log in 

wl = wallaroo.Client()

# retrieve the previous workspace, model, and pipeline version

workspace_name = "tutorial-workspace-john-cybersecurity"

workspace = wl.get_workspace(name=workspace_name)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

model_name = 'aloha-prime'

prime_model_version = wl.get_model(model_name)

pipeline_name = 'aloha-fraud-detector'

pipeline = wl.get_pipeline(pipeline_name)

# verify the workspace/pipeline/model

display(wl.get_current_workspace())
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

<table><tr><th>name</th> <td>aloha-fraud-detector</td></tr><tr><th>created</th> <td>2024-05-01 16:30:53.995114+00:00</td></tr><tr><th>last_updated</th> <td>2024-05-01 16:45:19.372610+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>551242a7-fe4c-4a61-a4c7-e7fcc97509dc, d22eb0d2-9cff-4c5f-a851-10b1a19d8c44, 262909e9-8779-4c56-a994-725ddd0b58c8, 4cdf8e11-1b9c-44ab-a16d-abb054b5e9fe, 6b3529b1-1ff1-454b-8896-460c8c90d667</td></tr><tr><th>steps</th> <td>aloha-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Deploy the Pipeline

Add the model version as a pipeline step to our pipeline, and deploy the pipeline.  You may want to check the pipeline steps to verify that the right model version is set for the pipeline step.

Once deployed, perform a sample inference to make sure everythign is running smooth.

```python
## blank space to get your pipeline and run a small batch of data through it to see the range of predictions

pipeline.clear()
pipeline.add_model_step(prime_model_version)

deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
pipeline.deploy(deployment_config=deploy_config)

multiple_result = pipeline.infer_from_file('../data/data-1k.df.json')
display(multiple_result.sort_values(by=["out.cryptolocker"]).loc[:, ['out.cryptolocker']])

```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out.cryptolocker</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0.012099549]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0.025435215]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0.036468606]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0.100503676]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[0.0143616935]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>[0.011983096]</td>
    </tr>
    <tr>
      <th>996</th>
      <td>[0.10195666]</td>
    </tr>
    <tr>
      <th>997</th>
      <td>[0.09424207]</td>
    </tr>
    <tr>
      <th>998</th>
      <td>[0.13612257]</td>
    </tr>
    <tr>
      <th>999</th>
      <td>[0.014839977]</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 1 columns</p>

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
# blank space to set a validation rule on the pipeline and check if it triggers as expected

import polars as pl

pipeline = pipeline.add_validations(
    too_low=pl.col("out.cryptolocker").list.get(0) < 0.1
)

```

```python
deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
pipeline.deploy(deployment_config=deploy_config)

multiple_result = pipeline.infer_from_file('../data/data-1k.df.json')

# show the first to results
display(multiple_result.head(20))

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
      <th>anomaly.too_low</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-05-01 18:56:42.249</td>
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
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-05-01 18:56:42.249</td>
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
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-05-01 18:56:42.249</td>
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
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-05-01 18:56:42.249</td>
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
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-05-01 18:56:42.249</td>
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
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 18, 29, 25, 31, 21, 20, 19, 32, 30, 28, 12, 27, 34, 22, 29, 33]</td>
      <td>[8.510186e-19]</td>
      <td>[1.9487171e-08]</td>
      <td>[0.112345986]</td>
      <td>[4.2180977e-07]</td>
      <td>[3.845866e-25]</td>
      <td>[0.08677925]</td>
      <td>[0.26976353]</td>
      <td>[1.0]</td>
      <td>[0.28034106]</td>
      <td>[0.1772562]</td>
      <td>[1.1434392e-08]</td>
      <td>[0.18945046]</td>
      <td>[0.031674143]</td>
      <td>[1.7841078e-30]</td>
      <td>[1.9434036e-36]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 27, 29, 19, 35, 27, 12, 15, 15, 33, 27, 27, 33, 33]</td>
      <td>[8.3179456e-16]</td>
      <td>[2.1139394e-07]</td>
      <td>[0.10018868]</td>
      <td>[2.164278e-08]</td>
      <td>[3.9452582e-27]</td>
      <td>[0.06368506]</td>
      <td>[0.08174924]</td>
      <td>[0.9999998]</td>
      <td>[0.09102885]</td>
      <td>[0.05443449]</td>
      <td>[2.754149e-09]</td>
      <td>[0.06614307]</td>
      <td>[0.004162426]</td>
      <td>[2.3701194e-05]</td>
      <td>[3.0342645e-32]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 22, 34, 16, 33, 32, 35, 36, 28, 19, 32, 22, 35, 27]</td>
      <td>[2.3569645e-18]</td>
      <td>[5.8529037e-10]</td>
      <td>[0.02562021]</td>
      <td>[1.3246556e-08]</td>
      <td>[9.615627e-30]</td>
      <td>[0.3749346]</td>
      <td>[0.17899293]</td>
      <td>[0.9999982]</td>
      <td>[0.21193951]</td>
      <td>[0.09348915]</td>
      <td>[2.2368522e-10]</td>
      <td>[0.123645365]</td>
      <td>[0.026871203]</td>
      <td>[7.0041276e-36]</td>
      <td>[0.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 23, 22, 17, 15, 25, 32, 31, 16, 28, 16, 22, 19, 31, 12, 23]</td>
      <td>[1.051414e-09]</td>
      <td>[4.6342116e-08]</td>
      <td>[0.039195027]</td>
      <td>[2.8384156e-10]</td>
      <td>[3.1367284e-16]</td>
      <td>[0.11919281]</td>
      <td>[0.1371625]</td>
      <td>[0.9999869]</td>
      <td>[0.18870851]</td>
      <td>[0.08751174]</td>
      <td>[3.7058056e-05]</td>
      <td>[0.09724228]</td>
      <td>[0.012924853]</td>
      <td>[2.5838341e-33]</td>
      <td>[4.099651e-34]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 30, 26, 16, 27, 12, 26, 31, 28, 13]</td>
      <td>[0.00028143774]</td>
      <td>[0.03695946]</td>
      <td>[0.039600268]</td>
      <td>[5.4820653e-06]</td>
      <td>[4.1267756e-10]</td>
      <td>[0.021739561]</td>
      <td>[0.07481046]</td>
      <td>[0.9980819]</td>
      <td>[0.075143255]</td>
      <td>[0.038673826]</td>
      <td>[0.009808683]</td>
      <td>[0.03394704]</td>
      <td>[0.004953161]</td>
      <td>[4.427488e-35]</td>
      <td>[1.1760158e-30]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 25, 30, 27, 16, 27, 20]</td>
      <td>[1.0623e-07]</td>
      <td>[3.44891e-05]</td>
      <td>[0.06656312]</td>
      <td>[9.905916e-07]</td>
      <td>[1.2954507e-13]</td>
      <td>[0.0017106688]</td>
      <td>[0.04309076]</td>
      <td>[0.9993519]</td>
      <td>[0.02822486]</td>
      <td>[0.017890908]</td>
      <td>[1.1848524e-06]</td>
      <td>[0.011960788]</td>
      <td>[0.017825156]</td>
      <td>[2.0426182e-32]</td>
      <td>[1.944474e-26]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 27, 17, 25, 26, 28, 18, 14, 27, 20, 32, 30, 14, 30]</td>
      <td>[2.9365646e-14]</td>
      <td>[7.402574e-07]</td>
      <td>[0.03670107]</td>
      <td>[1.9542877e-06]</td>
      <td>[6.662594e-27]</td>
      <td>[0.13474435]</td>
      <td>[0.13534914]</td>
      <td>[0.99999994]</td>
      <td>[0.16737573]</td>
      <td>[0.08720821]</td>
      <td>[5.128044e-10]</td>
      <td>[0.117759936]</td>
      <td>[0.020653522]</td>
      <td>[1.9017125e-35]</td>
      <td>[2.514033e-34]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 22, 21, 34, 31, 34, 34, 13, 31, 18, 17, 25, 28, 13, 16, 34, 30]</td>
      <td>[8.164666e-14]</td>
      <td>[5.154334e-10]</td>
      <td>[0.017082075]</td>
      <td>[3.2816735e-07]</td>
      <td>[1.37248e-16]</td>
      <td>[0.07431223]</td>
      <td>[0.087764025]</td>
      <td>[0.99998844]</td>
      <td>[0.07380712]</td>
      <td>[0.041434564]</td>
      <td>[5.403246e-07]</td>
      <td>[0.056005254]</td>
      <td>[0.007238474]</td>
      <td>[9.882496e-35]</td>
      <td>[1.7473628e-33]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 20, 23, 13, 28, 34, 28, 14]</td>
      <td>[1.3123198e-09]</td>
      <td>[3.002701e-11]</td>
      <td>[0.047422577]</td>
      <td>[4.668125e-05]</td>
      <td>[2.3931444e-15]</td>
      <td>[0.08620628]</td>
      <td>[0.17082322]</td>
      <td>[0.9962702]</td>
      <td>[0.15127072]</td>
      <td>[0.07085356]</td>
      <td>[3.7702117e-05]</td>
      <td>[0.05850159]</td>
      <td>[0.12628299]</td>
      <td>[3.3260953e-27]</td>
      <td>[2.2508502e-26]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 36, 15, 26, 23, 31, 12, 32, 19, 33, 12, 21, 24, 13]</td>
      <td>[3.4205932e-05]</td>
      <td>[0.0785699]</td>
      <td>[0.027089605]</td>
      <td>[6.866653e-13]</td>
      <td>[7.996386e-10]</td>
      <td>[0.018607156]</td>
      <td>[0.037185796]</td>
      <td>[0.95840836]</td>
      <td>[0.04340277]</td>
      <td>[0.026010705]</td>
      <td>[3.318159e-05]</td>
      <td>[0.025464533]</td>
      <td>[0.0058436417]</td>
      <td>[2.524863e-35]</td>
      <td>[2.8656257e-32]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 35, 16, 12, 22, 13, 19, 13, 31, 24]</td>
      <td>[0.002541592]</td>
      <td>[0.00048265897]</td>
      <td>[0.053151637]</td>
      <td>[3.2713058e-07]</td>
      <td>[1.4396314e-12]</td>
      <td>[0.02697211]</td>
      <td>[0.0425427]</td>
      <td>[0.97252196]</td>
      <td>[0.043152798]</td>
      <td>[0.02238737]</td>
      <td>[0.0012540827]</td>
      <td>[0.023678368]</td>
      <td>[0.0068876254]</td>
      <td>[5.1229324e-17]</td>
      <td>[1.6712352e-19]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 14, 25, 16, 32, 23, 32, 34]</td>
      <td>[1.1404552e-08]</td>
      <td>[0.00028144714]</td>
      <td>[0.2999726]</td>
      <td>[3.1985126e-06]</td>
      <td>[1.5959413e-09]</td>
      <td>[0.0087030465]</td>
      <td>[0.19416207]</td>
      <td>[0.9973451]</td>
      <td>[0.12070679]</td>
      <td>[0.06955199]</td>
      <td>[0.0014974618]</td>
      <td>[0.030090276]</td>
      <td>[0.332936]</td>
      <td>[7.264472e-27]</td>
      <td>[1.2870859e-22]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 21, 18, 32, 13, 17, 25, 19, 14, 22, 19, 12, 36, 31, 17, 12]</td>
      <td>[4.512332e-13]</td>
      <td>[0.00071659556]</td>
      <td>[0.035884183]</td>
      <td>[6.633806e-09]</td>
      <td>[9.596767e-20]</td>
      <td>[0.2649986]</td>
      <td>[0.13681291]</td>
      <td>[0.9999994]</td>
      <td>[0.18176991]</td>
      <td>[0.09768724]</td>
      <td>[6.8252644e-05]</td>
      <td>[0.10130761]</td>
      <td>[0.033053454]</td>
      <td>[3.4586507e-31]</td>
      <td>[2.270425e-34]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 25, 35, 30, 24, 14, 12, 17, 17, 20, 23, 30, 12, 21, 29]</td>
      <td>[7.1145604e-07]</td>
      <td>[3.2433495e-06]</td>
      <td>[0.0059578065]</td>
      <td>[4.2087473e-07]</td>
      <td>[4.4182375e-06]</td>
      <td>[0.13091077]</td>
      <td>[0.037190206]</td>
      <td>[0.9930535]</td>
      <td>[0.046440512]</td>
      <td>[0.02659296]</td>
      <td>[0.0009189554]</td>
      <td>[0.03220919]</td>
      <td>[0.015011885]</td>
      <td>[6.769061e-35]</td>
      <td>[3.0584734e-23]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 36, 30, 14, 33, 22, 29]</td>
      <td>[9.998708e-15]</td>
      <td>[8.658513e-07]</td>
      <td>[0.20481537]</td>
      <td>[8.193097e-05]</td>
      <td>[3.0058433e-17]</td>
      <td>[0.068068124]</td>
      <td>[0.23884542]</td>
      <td>[0.9999949]</td>
      <td>[0.18595874]</td>
      <td>[0.1148858]</td>
      <td>[7.921312e-08]</td>
      <td>[0.087733045]</td>
      <td>[0.18423225]</td>
      <td>[1.4899968e-21]</td>
      <td>[1.7388379e-26]</td>
      <td>0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>

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
      <th>anomaly.too_low</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-05-01 18:56:42.249</td>
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
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-05-01 18:56:42.249</td>
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
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-05-01 18:56:42.249</td>
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
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-05-01 18:56:42.249</td>
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
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 22, 34, 16, 33, 32, 35, 36, 28, 19, 32, 22, 35, 27]</td>
      <td>[2.3569645e-18]</td>
      <td>[5.8529037e-10]</td>
      <td>[0.02562021]</td>
      <td>[1.3246556e-08]</td>
      <td>[9.615627e-30]</td>
      <td>[0.3749346]</td>
      <td>[0.17899293]</td>
      <td>[0.9999982]</td>
      <td>[0.21193951]</td>
      <td>[0.09348915]</td>
      <td>[2.2368522e-10]</td>
      <td>[0.123645365]</td>
      <td>[0.026871203]</td>
      <td>[7.0041276e-36]</td>
      <td>[0.0]</td>
      <td>1</td>
      <td>True</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>985</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 30, 34, 19, 14, 12, 22, 29, 30, 22, 15, 23, 18, 22, 30, 31, 23]</td>
      <td>[1.9509931e-08]</td>
      <td>[3.8211652e-05]</td>
      <td>[0.074470066]</td>
      <td>[1.5295203e-10]</td>
      <td>[4.951934e-13]</td>
      <td>[0.11857636]</td>
      <td>[0.23397556]</td>
      <td>[0.9999181]</td>
      <td>[0.26928952]</td>
      <td>[0.13449039]</td>
      <td>[0.00023073937]</td>
      <td>[0.15711878]</td>
      <td>[0.04463736]</td>
      <td>[0.0]</td>
      <td>[0.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>988</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 14, 32, 35, 25, 20, 31, 18, 26, 26, 35]</td>
      <td>[0.00017462275]</td>
      <td>[2.0345014e-05]</td>
      <td>[0.086056456]</td>
      <td>[5.3362342e-08]</td>
      <td>[2.9563812e-06]</td>
      <td>[0.0585213]</td>
      <td>[0.14665367]</td>
      <td>[0.9945637]</td>
      <td>[0.12845428]</td>
      <td>[0.07318983]</td>
      <td>[0.048846442]</td>
      <td>[0.07254204]</td>
      <td>[0.09940087]</td>
      <td>[2.7488947e-32]</td>
      <td>[1.0424284e-31]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>995</th>
      <td>2024-05-01 18:56:42.249</td>
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
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>997</th>
      <td>2024-05-01 18:56:42.249</td>
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
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>999</th>
      <td>2024-05-01 18:56:42.249</td>
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
      <td>1</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>647 rows × 19 columns</p>

```python
# show only the results that triggered an anomaly
multiple_result.loc[multiple_result['anomaly.count'] > 0, ['time', 'out.cryptolocker']]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.cryptolocker</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0.012099549]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0.025435215]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0.036468606]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0.0143616935]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0.02562021]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>985</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0.074470066]</td>
    </tr>
    <tr>
      <th>988</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0.086056456]</td>
    </tr>
    <tr>
      <th>995</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0.011983096]</td>
    </tr>
    <tr>
      <th>997</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0.09424207]</td>
    </tr>
    <tr>
      <th>999</th>
      <td>2024-05-01 18:56:42.249</td>
      <td>[0.014839977]</td>
    </tr>
  </tbody>
</table>
<p>647 rows × 2 columns</p>

## Clean Up

At this point, if you are not continuing on to the next notebook, undeploy your pipeline to give the resources back to the environment.

```python
## blank space to undeploy the pipeline

pipeline.undeploy()
```

<table><tr><th>name</th> <td>aloha-fraud-detector</td></tr><tr><th>created</th> <td>2024-05-01 16:30:53.995114+00:00</td></tr><tr><th>last_updated</th> <td>2024-05-01 18:56:40.187633+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>28f5443a-ff67-4f4f-bfc3-f3f95f3c6f83, 435b73f7-76a1-4514-bd41-2cb94d3e78ff, 551242a7-fe4c-4a61-a4c7-e7fcc97509dc, d22eb0d2-9cff-4c5f-a851-10b1a19d8c44, 262909e9-8779-4c56-a994-725ddd0b58c8, 4cdf8e11-1b9c-44ab-a16d-abb054b5e9fe, 6b3529b1-1ff1-454b-8896-460c8c90d667</td></tr><tr><th>steps</th> <td>aloha-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Congratulations!

In this tutorial you have

* Set a validation rule on your house price prediction pipeline.
* Detected model predictions that failed the validation rule.

In the next notebook, you will learn how to monitor the distribution of model outputs for drift away from expected behavior.
