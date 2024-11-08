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

workspace_name = "tutorial-finserv-john"

workspace = wl.get_workspace(workspace_name)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

# optionally, examine your current workspace
wl.get_current_workspace()

model_name = 'classification-finserv-prime'

prime_model_version = wl.get_model(model_name)

pipeline_name = 'ccfraud-detector'

pipeline = wl.get_pipeline(pipeline_name)

display(workspace)
display(prime_model_version)
display(pipeline)

```

    {'name': 'tutorial-finserv-john', 'id': 11, 'archived': False, 'created_by': '94016008-4e0e-45bc-b2c6-d6f06236b4f5', 'created_at': '2024-09-05T16:18:31.258882+00:00', 'models': [{'name': 'classification-finserv-prime', 'versions': 3, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 9, 5, 16, 48, 18, 348992, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 9, 5, 16, 18, 42, 470017, tzinfo=tzutc())}, {'name': 'ccfraud-xgboost-version', 'versions': 3, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 9, 5, 16, 53, 34, 714217, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 9, 5, 16, 32, 28, 543534, tzinfo=tzutc())}], 'pipelines': [{'name': 'assay-demonstration-tutorial', 'create_time': datetime.datetime(2024, 9, 5, 16, 18, 46, 604658, tzinfo=tzutc()), 'definition': '[]'}, {'name': 'ccfraud-detector', 'create_time': datetime.datetime(2024, 9, 5, 16, 18, 43, 626892, tzinfo=tzutc()), 'definition': '[]'}]}

<table>
        <tr>
          <td>Name</td>
          <td>classification-finserv-prime</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>24761304-0cb3-40e7-9462-d2a454637152</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>keras_ccfraud.onnx</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507</td>
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
          <td>2024-05-Sep 16:48:18</td>
        </tr>
        <tr>
          <td>Workspace id</td>
          <td>11</td>
        </tr>
        <tr>
          <td>Workspace name</td>
          <td>tutorial-finserv-john</td>
        </tr>
      </table>

<table><tr><th>name</th> <td>ccfraud-detector</td></tr><tr><th>created</th> <td>2024-09-05 16:18:43.626892+00:00</td></tr><tr><th>last_updated</th> <td>2024-09-05 16:55:36.887872+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>workspace_id</th> <td>11</td></tr><tr><th>workspace_name</th> <td>tutorial-finserv-john</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>cb026715-9ced-40bb-9108-333b88c9de64, e90c10e3-ab38-43a8-a315-fb4250c09b21, 583e7a6a-3a7a-4420-abd0-c91e346a874d, cbfc4951-4d2c-41cb-89c7-934ac5bd2cbf, 32ab9ef6-2ac4-4d92-a46e-5d5c286af48c, 410d43d4-e698-4747-bbd1-cb62afee258a, 9edcf6f6-660d-470d-b8f0-24f54a335e8f, cd63b4fa-6549-41b4-af8f-576b1f0ef8b3, ff2d47a5-47c8-4fcb-8709-e95b3d0d4340, 8e74290b-4cb6-43a5-9b27-8431643438fd</td></tr><tr><th>steps</th> <td>ccfraud-xgboost-version</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Deploy the Pipeline

Add the model version as a pipeline step to our pipeline, and deploy the pipeline.  You may want to check the pipeline steps to verify that the right model version is set for the pipeline step.

```python
## blank space to get your pipeline and run a small batch of data through it to see the range of predictions

pipeline.clear()
pipeline.add_model_step(prime_model_version)

deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
pipeline.deploy(deployment_config=deploy_config)

```

```python
# sample inference here

multiple_result = pipeline.infer_from_file('../data/cc_data_1k.df.json')
display(multiple_result)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-09-05 17:37:42.984</td>
      <td>[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-09-05 17:37:42.984</td>
      <td>[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-09-05 17:37:42.984</td>
      <td>[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-09-05 17:37:42.984</td>
      <td>[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-09-05 17:37:42.984</td>
      <td>[0.5817662108, 0.097881551, 0.1546819424, 0.4754101949, -0.1978862306, -0.4504344854, 0.0166540447, -0.0256070551, 0.0920561602, -0.2783917153, 0.0593299441, -0.0196585416, -0.4225083157, -0.1217538877, 1.5473094894, 0.2391622864, 0.3553974881, -0.7685165301, -0.7000849355, -0.1190043285, -0.3450517133, -1.1065114108, 0.2523411195, 0.0209441826, 0.2199267436, 0.2540689265, -0.0450225094, 0.1086773898, 0.2547179311]</td>
      <td>[0.0010916889]</td>
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
      <th>996</th>
      <td>2024-09-05 17:37:42.984</td>
      <td>[1.052355506, -0.7602601059, -0.3124601687, -0.5580714587, -0.6198353331, 0.6635428464, -1.2171685083, 0.3144529308, 0.2360632058, 0.878209955, -0.5518803042, -0.2781328417, -0.5675947058, -0.0982688053, 0.1475098349, -0.3097481612, -1.0898892231, 2.804466934, -0.4211447753, -0.7315488305, -0.5311840374, -0.9053830525, 0.5382443229, -0.68327623, -1.1848642272, 0.9872236995, -0.0260721428, -0.1405966468, 0.0759031399]</td>
      <td>[0.00011596084]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>997</th>
      <td>2024-09-05 17:37:42.984</td>
      <td>[-0.8464537996, -0.7608807925, 2.186072883, -0.1614362994, -0.4069378894, 0.734079177, -0.4611705734, 0.4751492626, 1.4952832213, -0.9349105827, -0.7654272171, 0.4362793613, -0.6623354486, -1.5326388376, -1.4311992842, -1.0573215483, 0.9304904478, -1.2836000946, -1.079419331, 0.7138847264, 0.2710369668, 1.1943291742, 0.2527110226, 0.3107779567, 0.4219366694, 2.4854295825, 0.1754876037, -0.2362979978, 0.9979986569]</td>
      <td>[0.0002785325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>998</th>
      <td>2024-09-05 17:37:42.984</td>
      <td>[1.0046377125, 0.0343666504, -1.3512533246, 0.4160460291, 0.5910548281, -0.8187740907, 0.5840864966, -0.447623496, 1.1193896296, -0.1156579903, 0.1298919303, -2.6410683948, 1.1658091033, 2.3607999565, -0.4265055896, -0.4862102299, 0.5102253659, -0.3384745171, -0.4081285365, -0.199414607, 0.0151691668, 0.2644673476, -0.0483547565, 0.9869714364, 0.629627219, 0.8990505678, -0.3731273846, -0.2166148809, 0.6374669208]</td>
      <td>[0.0011070371]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>999</th>
      <td>2024-09-05 17:37:42.984</td>
      <td>[0.4951101913, -0.2499369449, 0.4553345161, 0.9242750451, -0.3643510229, 0.602688482, -0.3785553207, 0.3170957153, 0.7368986387, -0.1195106678, 0.4017042912, 0.7371143425, -1.2229791154, 0.0061993212, -1.3541149574, -0.5839052891, 0.1648461272, -0.1527212037, 0.2456232399, -0.1432012313, -0.0383696111, 0.0865420131, -0.284099885, -0.5027591867, 1.1117147574, -0.5666540195, 0.121220185, 0.0667640208, 0.6583281816]</td>
      <td>[0.0008533001]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>2024-09-05 17:37:42.984</td>
      <td>[0.6118805301, 0.1726081102, 0.4310545502, 0.5032148221, -0.2746663262, -0.464798859, -0.1098384885, -0.0978937224, 0.9820529526, -0.2237381949, 2.3315375168, -1.5852745605, 1.6050692254, 1.9720759474, -0.4217479714, 0.5348796175, 0.0875849983, 0.3280840192, -0.0394716814, -0.1796805095, -0.4955020407, -1.1889449446, 0.246698494, 0.4185131811, 0.3026018698, 0.0812114542, -0.1557850823, 0.0171892918, -0.7236631158]</td>
      <td>[0.0012498498]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1001 rows × 4 columns</p>

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
    too_low=pl.col("out.main").list.get(0) < 0.90
)
```

For the inference request, isolate only the inferences that trigger the anomaly.  Here's one way using the DataFrame functions:

```python
# sample infer

results = pipeline.infer_from_file('../data/data-1k.df.json')

results.loc[results['anomaly.too_low'] == True,['time', 'out.main', 'anomaly.too_low', 'anomaly.count']].head(20)
```

```python
# blank space to set up validation rule

import polars as pl

sample_pipeline = pipeline.add_validations(
    too_low=pl.col("out.dense_1").list.get(0) < 0.90
)

pipeline.deploy(deployment_config=deploy_config)

sample_pipeline.steps()
```

    [{'ModelInference': {'models': [{'name': 'classification-finserv-prime', 'version': '24761304-0cb3-40e7-9462-d2a454637152', 'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507'}]}},
     {'Check': {'tree': ['{"Alias":[{"BinaryExpr":{"left":{"Function":{"input":[{"Column":"out.dense_1"},{"Literal":{"Int32":0}}],"function":{"ListExpr":"Get"},"options":{"collect_groups":"ApplyFlat","fmt_str":"","input_wildcard_expansion":false,"auto_explode":true,"cast_to_supertypes":false,"allow_rename":false,"pass_name_to_apply":false,"changes_length":false,"check_lengths":true,"allow_group_aware":true}}},"op":"Lt","right":{"Literal":{"Float64":0.9}}}},"too_low"]}']}}]

```python
# blank space for inferences

# sample infer

results = pipeline.infer_from_file('../data/cc_data_1k.df.json')

results.loc[results['anomaly.too_low'] == True,['time', 'out.dense_1', 'anomaly.too_low', 'anomaly.count']].head(20)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.dense_1</th>
      <th>anomaly.too_low</th>
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>2024-09-05 17:38:02.858</td>
      <td>[0.0010916889]</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2024-09-05 17:38:02.858</td>
      <td>[0.00047266483]</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2024-09-05 17:38:02.858</td>
      <td>[0.00082170963]</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2024-09-05 17:38:02.858</td>
      <td>[0.0011294782]</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2024-09-05 17:38:02.858</td>
      <td>[0.0018743575]</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2024-09-05 17:38:02.858</td>
      <td>[0.0011520088]</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2024-09-05 17:38:02.858</td>
      <td>[0.0016568303]</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2024-09-05 17:38:02.858</td>
      <td>[0.0010267198]</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2024-09-05 17:38:02.858</td>
      <td>[0.00019043684]</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2024-09-05 17:38:02.858</td>
      <td>[0.00032365322]</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2024-09-05 17:38:02.858</td>
      <td>[0.00062185526]</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2024-09-05 17:38:02.858</td>
      <td>[0.00033929944]</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2024-09-05 17:38:02.858</td>
      <td>[0.00094124675]</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2024-09-05 17:38:02.858</td>
      <td>[0.00040614605]</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2024-09-05 17:38:02.858</td>
      <td>[0.00014156103]</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2024-09-05 17:38:02.858</td>
      <td>[0.0006403029]</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2024-09-05 17:38:02.858</td>
      <td>[0.0008019507]</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2024-09-05 17:38:02.858</td>
      <td>[0.0011220276]</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2024-09-05 17:38:02.858</td>
      <td>[0.0007892251]</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2024-09-05 17:38:02.858</td>
      <td>[0.00040519238]</td>
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

<table><tr><th>name</th> <td>ccfraud-detector</td></tr><tr><th>created</th> <td>2024-09-05 16:18:43.626892+00:00</td></tr><tr><th>last_updated</th> <td>2024-09-05 17:37:49.045951+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>workspace_id</th> <td>11</td></tr><tr><th>workspace_name</th> <td>tutorial-finserv-john</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>93b4dd36-5e02-440d-931c-80198d1ee48a, 252d1e6f-35a5-420a-9590-d08cd76943a7, cb026715-9ced-40bb-9108-333b88c9de64, e90c10e3-ab38-43a8-a315-fb4250c09b21, 583e7a6a-3a7a-4420-abd0-c91e346a874d, cbfc4951-4d2c-41cb-89c7-934ac5bd2cbf, 32ab9ef6-2ac4-4d92-a46e-5d5c286af48c, 410d43d4-e698-4747-bbd1-cb62afee258a, 9edcf6f6-660d-470d-b8f0-24f54a335e8f, cd63b4fa-6549-41b4-af8f-576b1f0ef8b3, ff2d47a5-47c8-4fcb-8709-e95b3d0d4340, 8e74290b-4cb6-43a5-9b27-8431643438fd</td></tr><tr><th>steps</th> <td>classification-finserv-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Congratulations!

In this tutorial you have

* Set a validation rule on your house price prediction pipeline.
* Detected model predictions that failed the validation rule.

In the next notebook, you will learn how to monitor the distribution of model outputs for drift away from expected behavior.
