# Tutorial Notebook 3: Observability Part 1 - Validation Rules

In the previous notebooks you uploaded the models and artifacts, then deployed the models to production through provisioning workspaces and pipelines. Now you're ready to put your feet up! But to keep your models operational, your work's not done once the model is in production. You must continue to monitor the behavior and performance of the model to insure that the model provides value to the business.

In this notebook, you will learn about adding validation rules to pipelines.

## Preliminaries

In the blocks below we will preload some required libraries.

```python
# run to preload needed libraries 

import wallaroo
from wallaroo.object import EntityNotFoundError
from wallaroo.framework import Framework

from IPython.display import display

# used to display DataFrame information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)

```

## Pre-exercise

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

<table><tr><th>name</th> <td>cv-retail</td></tr><tr><th>created</th> <td>2024-11-04 21:10:05.287786+00:00</td></tr><tr><th>last_updated</th> <td>2024-11-06 17:24:36.733297+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>workspace_id</th> <td>13</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-john-cv</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>1e8ec6fa-c8f3-4118-968b-0133cfc18a97, f410b97b-c1dd-4e23-99d9-e63410e100d6, 73d361a7-0b5f-4614-b513-61c141366e84, 4b41e45e-a917-4b48-9786-5e84d189afdd, 44ff0494-e30a-4a93-b5e3-4ce90b1b2368, 42c8d366-583d-44f4-ac4b-513103b5902c, d8be019e-2b7c-4c52-9e41-101b20ab0c2a, dd5e2f8a-e436-4b35-b2fb-189f6059dacc, 5a0772da-cde1-4fea-afc4-16313fcaa229, 7686f3ea-3781-4a95-aa4c-e99e46b9c47c, 4df9df54-6d12-4577-a970-f544128d0575</td></tr><tr><th>steps</th> <td>resnet50</td></tr><tr><th>published</th> <td>False</td></tr></table>

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

## Deploy the Pipeline

Add the model version as a pipeline step to our pipeline, and deploy the pipeline.  You may want to check the pipeline steps to verify that the right model version is set for the pipeline step.

```python
# run this to set the deployment configuration

deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
```

```python
## blank space to set the model steps and deploy the pipeline

pipeline.undeploy()
pipeline.clear()
pipeline.add_model_step(prime_model_version)
pipeline.add_model_step(module_post_process_model)

pipeline.deploy(deployment_config=deploy_config)
```

<table><tr><th>name</th> <td>cv-retail</td></tr><tr><th>created</th> <td>2024-11-04 21:10:05.287786+00:00</td></tr><tr><th>last_updated</th> <td>2024-11-06 17:38:40.964114+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>workspace_id</th> <td>13</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-john-cv</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>176561de-a406-404b-aa5f-ce496f0c627d, 32a8ee27-39b8-43ad-86a0-69ac4ed8f6ed, ef9d0e34-cee3-4bd7-892c-219130abc978, bb43f0c5-f533-4ebd-9a6f-8ed59c633a9d, b5d76855-a136-4c9c-95e7-20b0b6f9bcd3, d35262a1-4958-42f7-81b0-e4fefec68f39, c4e1078b-26fa-4d11-a37a-266d9820235b, 1e8ec6fa-c8f3-4118-968b-0133cfc18a97, f410b97b-c1dd-4e23-99d9-e63410e100d6, 73d361a7-0b5f-4614-b513-61c141366e84, 4b41e45e-a917-4b48-9786-5e84d189afdd, 44ff0494-e30a-4a93-b5e3-4ce90b1b2368, 42c8d366-583d-44f4-ac4b-513103b5902c, d8be019e-2b7c-4c52-9e41-101b20ab0c2a, dd5e2f8a-e436-4b35-b2fb-189f6059dacc, 5a0772da-cde1-4fea-afc4-16313fcaa229, 7686f3ea-3781-4a95-aa4c-e99e46b9c47c, 4df9df54-6d12-4577-a970-f544128d0575</td></tr><tr><th>steps</th> <td>mobilenet</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Model Validation Rules

A simple way to try to keep your model's behavior up to snuff is to make sure that it receives inputs that it expects, and that its output is something that downstream systems can handle. This can entail specifying rules that document what you expect, and either enforcing these rules (by refusing to make a prediction), or at least logging an alert that the expectations described by your validation rules have been violated. As the developer of the model, the data scientist (along with relevant subject matter experts) will often be the person in the best position to specify appropriate validation rules.

In our house price prediction example, suppose you know that house prices in your market are typically in the range $750,000 to $1.5M dollars. Then you might want to set validation rules on your model pipeline to specify that you expect the model's predictions to also be in that range. Then, if the model predicts a value outside that range, the pipeline will log that one of the validation checks has failed; this allows you to investigate that instance further.

Note that in this specific example, a model prediction outside the specified range may not necessarily be "wrong"; but out-of-range predictions are likely unusual enough that you may want to "sanity-check" the model's behavior in these situations.

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
sample_pipeline.add_model_step(ccfraud_model)

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

* Add an upper bound or a lower bound to the model predictions
* Try to create predictions that fall both in and out of the specified range
* Look through the inference results to check for detected anomalies.

**HINT 1**: since the purpose of this exercise is try out validation rules, it might be a good idea to take a small data set and make predictions on that data set first, *then* set the validation rules based on those predictions, so that you can see the check failures trigger.

Here's an example:

```python
import polars as pl

sample_pipeline = pipeline.add_validations(
    too_low=pl.col("out.avg_confidence") < 0.50
)

pipeline.deploy(deployment_config=deploy_config)

sample_pipeline.steps()
```

For the inference request, isolate only the inferences that trigger the anomaly.  Here's one way using the DataFrame functions:

```python
# sample infer

# run the following to convert the image to a daataframe
import sys
 
# setting path - only needed when running this from the `with-code` folder.
sys.path.append('../')

import utils
width, height = 640, 480
dfImage, resizedImage = utils.loadImageAndConvertToDataframe('../data/images/example/dairy_bottles.png', width, height)

results = pipeline.infer(dfImage)

display(results['out.avg_confidence'])

display(results.loc[results['anomaly.too_low'] == True,['time', 'out.avg_confidence', 'anomaly.too_low', 'anomaly.count']].head(20))
```

```python
## blank space to set the validation

import polars as pl

sample_pipeline = pipeline.add_validations(
    too_low=pl.col("out.avg_confidence") < 0.50
)

pipeline.deploy(deployment_config=deploy_config)

sample_pipeline.steps()
```

    [{'ModelInference': {'models': [{'name': 'mobilenet', 'version': 'd15d8b9d-9d98-4aa7-8545-ac915862146e', 'sha': '9044c970ee061cc47e0c77e20b05e884be37f2a20aa9c0c3ce1993dbd486a830'}]}},
     {'ModelInference': {'models': [{'name': 'cv-post-process-drift-detection', 'version': 'a335c538-bccf-40b9-b9a4-9296f03e6eb1', 'sha': 'eefc55277b091dd90c45704ff51bbd68dbc0f0f7e686930c5409a606659cefcc'}]}},
     {'Check': {'tree': ['{"Alias":[{"BinaryExpr":{"left":{"Column":"out.avg_confidence"},"op":"Lt","right":{"Literal":{"Float64":0.5}}}},"too_low"]}']}}]

```python
## blank space to perform sample infer

# run the following to convert the image to a daataframe
import sys
 
# setting path - only needed when running this from the `with-code` folder.
sys.path.append('../')

import utils
width, height = 640, 480
dfImage, resizedImage = utils.loadImageAndConvertToDataframe('../data/images/example/dairy_bottles.png', width, height)

results = pipeline.infer(dfImage)

display(results['out.avg_confidence'])

display(results.loc[results['anomaly.too_low'] == True,['time', 'out.avg_confidence', 'anomaly.too_low', 'anomaly.count']].head(20))
```

    0    0.289506
    Name: out.avg_confidence, dtype: float64

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.avg_confidence</th>
      <th>anomaly.too_low</th>
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-11-06 17:39:08.934</td>
      <td>0.289506</td>
      <td>True</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

## Clean Up

At this point, if you are not continuing on to the next notebook, undeploy your pipeline to give the resources back to the environment.

```python
## blank space to undeploy the pipeline

pipeline.undeploy()
```

<table><tr><th>name</th> <td>cv-retail</td></tr><tr><th>created</th> <td>2024-11-04 21:10:05.287786+00:00</td></tr><tr><th>last_updated</th> <td>2024-11-06 17:39:05.484688+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>workspace_id</th> <td>13</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-john-cv</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a0254129-7e5f-4a5a-8194-f175f9ed03f8, 176561de-a406-404b-aa5f-ce496f0c627d, 32a8ee27-39b8-43ad-86a0-69ac4ed8f6ed, ef9d0e34-cee3-4bd7-892c-219130abc978, bb43f0c5-f533-4ebd-9a6f-8ed59c633a9d, b5d76855-a136-4c9c-95e7-20b0b6f9bcd3, d35262a1-4958-42f7-81b0-e4fefec68f39, c4e1078b-26fa-4d11-a37a-266d9820235b, 1e8ec6fa-c8f3-4118-968b-0133cfc18a97, f410b97b-c1dd-4e23-99d9-e63410e100d6, 73d361a7-0b5f-4614-b513-61c141366e84, 4b41e45e-a917-4b48-9786-5e84d189afdd, 44ff0494-e30a-4a93-b5e3-4ce90b1b2368, 42c8d366-583d-44f4-ac4b-513103b5902c, d8be019e-2b7c-4c52-9e41-101b20ab0c2a, dd5e2f8a-e436-4b35-b2fb-189f6059dacc, 5a0772da-cde1-4fea-afc4-16313fcaa229, 7686f3ea-3781-4a95-aa4c-e99e46b9c47c, 4df9df54-6d12-4577-a970-f544128d0575</td></tr><tr><th>steps</th> <td>mobilenet</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Congratulations!

In this tutorial you have

* Set a validation rule on your house price prediction pipeline.
* Detected model predictions that failed the validation rule.

In the next notebook, you will learn how to monitor the distribution of model outputs for drift away from expected behavior.
