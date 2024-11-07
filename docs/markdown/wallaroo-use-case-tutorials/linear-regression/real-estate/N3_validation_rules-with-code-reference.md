# Tutorial Notebook 3: Observability Part 1 - Validation Rules

In the previous notebooks you uploaded the models and artifacts, then deployed the models to production through provisioning workspaces and pipelines. Now you're ready to put your feet up! But to keep your models operational, your work's not done once the model is in production. You must continue to monitor the behavior and performance of the model to insure that the model provides value to the business.

In this notebook, you will learn about adding validation rules to pipelines.

## Preliminaries

In the blocks below we will preload some required libraries.

For convenience, the following `helper functions` are defined to retrieve previously created workspaces, models, and pipelines:

* `get_workspace(name, client)`: This takes in the name and the Wallaroo client being used in this session, and returns the workspace matching `name`.  If no workspaces are found matching the name, raises a `KeyError` and returns `None`.
* `get_model_version(model_name, workspace)`: Retrieves the most recent model version from the model matching the `model_name` within the provided `workspace`.  If no model matches that name, raises a `KeyError` and returns `None`.
* `get_pipeline(pipeline_name, workspace)`: Retrieves the most pipeline from the workspace matching the `pipeline_name` within the provided `workspace`.  If no model matches that name, raises a `KeyError` and returns `None`.

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

```python
## convenience functions from the previous notebooks
## these functions assume your connection to wallaroo is called wl

## convenience functions from the previous notebook

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

## Login to Wallaroo

Retrieve the previous workspace, model versions, and pipelines used in the previous notebook.

```python
## blank space to log in 

wl = wallaroo.Client()

# retrieve the previous workspace, model, and pipeline version

workspace_name = "tutorial-workspace-john"

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

## Deploy the Pipeline

Add the model version as a pipeline step to our pipeline, and deploy the pipeline.  You may want to check the pipeline steps to verify that the right model version is set for the pipeline step.

```python
## blank space to get your pipeline and run a small batch of data through it to see the range of predictions

pipeline.clear()
pipeline.add_model_step(prime_model_version)

deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
pipeline.deploy(deployment_config=deploy_config)

multiple_result = pipeline.infer_from_file('../data/test_data.df.json')
display(multiple_result)

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
      <td>2023-09-11 21:09:39.989</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[659806.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-09-11 21:09:39.989</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[732883.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-09-11 21:09:39.989</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[419508.84]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-09-11 21:09:39.989</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[634028.8]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-09-11 21:09:39.989</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[427209.44]</td>
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
      <td>2023-09-11 21:09:39.989</td>
      <td>[4.0, 2.25, 2620.0, 98881.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1820.0, 800.0, 47.4662, -122.453, 1728.0, 95832.0, 63.0, 0.0, 0.0]</td>
      <td>[436151.13]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3996</th>
      <td>2023-09-11 21:09:39.989</td>
      <td>[3.0, 2.5, 2244.0, 4079.0, 2.0, 0.0, 0.0, 3.0, 7.0, 2244.0, 0.0, 47.2606, -122.254, 2077.0, 4078.0, 3.0, 0.0, 0.0]</td>
      <td>[284810.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3997</th>
      <td>2023-09-11 21:09:39.989</td>
      <td>[3.0, 1.75, 1490.0, 5000.0, 1.0, 0.0, 1.0, 3.0, 8.0, 1250.0, 240.0, 47.5257, -122.392, 1980.0, 5000.0, 61.0, 0.0, 0.0]</td>
      <td>[575571.44]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3998</th>
      <td>2023-09-11 21:09:39.989</td>
      <td>[4.0, 2.5, 2740.0, 5700.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2740.0, 0.0, 47.3535, -122.026, 3010.0, 5281.0, 8.0, 0.0, 0.0]</td>
      <td>[432262.38]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3999</th>
      <td>2023-09-11 21:09:39.989</td>
      <td>[5.0, 2.5, 2240.0, 7770.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1340.0, 900.0, 47.7198, -122.171, 1820.0, 7770.0, 36.0, 0.0, 0.0]</td>
      <td>[445873.1]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>4000 rows × 4 columns</p>

## Model Validation Rules

A simple way to try to keep your model's behavior up to snuff is to make sure that it receives inputs that it expects, and that its output is something that downstream systems can handle. This can entail specifying rules that document what you expect, and either enforcing these rules (by refusing to make a prediction), or at least logging an alert that the expectations described by your validation rules have been violated. As the developer of the model, the data scientist (along with relevant subject matter experts) will often be the person in the best position to specify appropriate validation rules.

In our house price prediction example, suppose you know that house prices in your market are typically in the range $750,000 to $1.5M dollars. Then you might want to set validation rules on your model pipeline to specify that you expect the model's predictions to also be in that range. Then, if the model predicts a value outside that range, the pipeline will log that one of the validation checks has failed; this allows you to investigate that instance further.

Note that in this specific example, a model prediction outside the specified range may not necessarily be "wrong"; but out-of-range predictions are likely unusual enough that you may want to "sanity-check" the model's behavior in these situations.

Wallaroo has functionality for specifying simple validation rules on model input and output values.

```python
pipeline.add_validation(<rulename>, <expression>)

```

Here, `<rulename>` is the name of the rule, and `<expression>` is a simple logical expression that the data scientist **expects to be true**.  This means if the expression proves **false**, then a `check_failure` flag is set in the inference results.

To add a validation step to a simple one-step pipeline, you need a handle to the pipeline (here called `pipeline`), and a handle to the model in the pipeline (here called `model`).  Then you can specify an expected prediction range as follows:

* Get the pipeline.
* Depending on the steps, the pipeline can be cleared and the sample model added as a step.
* Add the validation.
* Deploy the pipeline to set the validation as part of its steps

```python
# get the existing pipeline (in your workspace)
pipeline = get_pipeline("pipeline")

# you also need a handle to the model in this single-step pipeline.
# here are two ways to do it:
#
# (1) If you know the name of the model, you can also just use the get_model() convenience function above.
# In this example, the model has been uploaded to wallaroo with the name "mymodel"

model = get_model("mymodel") 

# (2) To get the model without knowing its name (for a single-step pipeline)
model = pipeline.model_configs()[0].model()

# specify the bounds
hi_bnd = 1500000.0 # 1.5M

#
# some examples of validation rules
#

# (1)  validation rule: prediction should be < 1.5 million
pipeline = pipeline.add_validation("less than 1.5m", model.outputs[0][0] < hi_bnd)

# deploy the pipeline to set the steps
pipeline.deploy()

```

When data is passed to the pipeline for inferences, the pipeline will log a check failure whenever one of the validation expressions evaluates to false. Here are some examples of inference results from a pipeline with the validation rule `model.outputs[0][0] < 400000.0`.

{{<figure src="/images/2024.2/wallaroo-use-case-tutorials/linear-regression/real-estate/validation_results.png" width="800" label="Inferences with check failures">}}

You can also find check failures in the logs:

```python
logs = pipeline.logs()
logs.loc[logs['check_failures'] > 0]
```

### Model Validation Rules Exercise

Add some simple validation rules to the model pipeline that you created in a previous exercise.

* Add an upper bound or a lower bound to the model predictions
* Try to create predictions that fall both in and out of the specified range
* Look through the logs to find the check failures.

**HINT 1**: since the purpose of this exercise is try out validation rules, it might be a good idea to take a small data set and make predictions on that data set first, *then* set the validation rules based on those predictions, so that you can see the check failures trigger.

Here's an example:

```python
hi_bnd = 600000.0

pipeline = pipeline.add_validation("less than 600k", prime_model_version.outputs[0][0] < hi_bnd)

pipeline.deploy()

multiple_result = pipeline.infer_from_file('../data/test_data.df.json')

display(multiple_result[multiple_result['check_failures'] > 0])
```

```python
# blank space to set a validation rule on the pipeline and check if it triggers as expected

hi_bnd = 600000.0

pipeline = pipeline.add_validation("less than 600k", prime_model_version.outputs[0][0] < hi_bnd)

pipeline.deploy()

multiple_result = pipeline.infer_from_file('../data/test_data.df.json')

display(multiple_result[multiple_result['check_failures'] > 0])
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
      <td>2023-09-11 21:15:03.782</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[659806.0]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-09-11 21:15:03.782</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[732883.5]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-09-11 21:15:03.782</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[634028.8]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-09-11 21:15:03.782</td>
      <td>[3.0, 2.0, 2140.0, 4923.0, 1.0, 0.0, 0.0, 4.0, 8.0, 1070.0, 1070.0, 47.6902, -122.339, 1470.0, 4923.0, 86.0, 0.0, 0.0]</td>
      <td>[615501.9]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-09-11 21:15:03.782</td>
      <td>[4.0, 3.5, 3590.0, 5334.0, 2.0, 0.0, 2.0, 3.0, 9.0, 3140.0, 450.0, 47.6763, -122.267, 2100.0, 6250.0, 9.0, 0.0, 0.0]</td>
      <td>[1139732.5]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3984</th>
      <td>2023-09-11 21:15:03.782</td>
      <td>[3.0, 3.0, 1850.0, 5000.0, 1.5, 0.0, 0.0, 2.0, 6.0, 1850.0, 0.0, 47.6711, -122.386, 1360.0, 2500.0, 115.0, 0.0, 0.0]</td>
      <td>[676122.1]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3985</th>
      <td>2023-09-11 21:15:03.782</td>
      <td>[3.0, 2.5, 2940.0, 15875.0, 2.0, 0.0, 0.0, 3.0, 10.0, 2940.0, 0.0, 47.5947, -122.016, 2980.0, 15875.0, 21.0, 0.0, 0.0]</td>
      <td>[757490.3]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3988</th>
      <td>2023-09-11 21:15:03.782</td>
      <td>[4.0, 3.25, 2740.0, 7266.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2060.0, 680.0, 47.5346, -122.066, 3030.0, 6546.0, 11.0, 0.0, 0.0]</td>
      <td>[698858.5]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3991</th>
      <td>2023-09-11 21:15:03.782</td>
      <td>[5.0, 4.0, 3210.0, 7200.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2410.0, 800.0, 47.7329, -121.966, 2750.0, 7200.0, 3.0, 0.0, 0.0]</td>
      <td>[636425.25]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3994</th>
      <td>2023-09-11 21:15:03.782</td>
      <td>[4.0, 2.5, 4070.0, 7800.0, 3.0, 0.0, 0.0, 4.0, 8.0, 3390.0, 680.0, 47.6838, -122.327, 2020.0, 6760.0, 12.0, 0.0, 0.0]</td>
      <td>[879390.75]</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1183 rows × 4 columns</p>

## Clean Up

At this point, if you are not continuing on to the next notebook, undeploy your pipeline to give the resources back to the environment.

```python
## blank space to undeploy the pipeline

pipeline.undeploy()
```

<table><tr><th>name</th> <td>houseprice-estimator</td></tr><tr><th>created</th> <td>2023-09-11 18:54:03.604291+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-11 21:15:00.151045+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>47a5fb9f-2456-4132-abea-88a3147c4446, 8df7f3a0-4531-4a28-a757-51d8064b3ead, 4dbd5738-2847-4623-be31-12216ee17b37, b076d650-3c58-462c-acf7-cf04896012e3, 4bfcf67e-6887-4294-b54b-b0fb4369f801, 9d57c76e-3c3c-45f2-967e-4d24a8705de3, 92f2b4f3-494b-4d69-895f-9e767ac1869d, 6de0fcdc-5457-49d2-8f3d-58faa9024f29, ff68e2c7-77cd-4d17-b2df-0a2dd719c1cf, 88d4f659-4b5b-4e44-b0cc-134a6f5efa81, 4d4bdc85-c53e-4fe2-a3fa-1d0467cc9dee, 6d06d1e6-1351-44cd-a123-d47dc7aade25, cdeb5c90-2139-43b3-b1f3-69d3718d6fea, c85b49cc-2b89-4695-800b-ab1c15ce436e, f024be99-ea10-455f-a65a-28050dca7bbe, 7923eb8f-cc16-4b85-9233-a2f206e930a2, 6fbd6236-c642-4f07-8a47-17c6569f57f6, 7bdf459e-72db-4ea9-81bc-97f184caa403, 91bd39e1-acd6-4e96-85a9-8eaabebe7da0</td></tr><tr><th>steps</th> <td>house-price-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Congratulations!

In this tutorial you have

* Set a validation rule on your house price prediction pipeline.
* Detected model predictions that failed the validation rule.

In the next notebook, you will learn how to monitor the distribution of model outputs for drift away from expected behavior.
