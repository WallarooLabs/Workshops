# Tutorial Notebook 3: Observability Part 1 - Validation Rules

In the previous notebooks you uploaded the models and artifacts, then deployed the models to production through provisioning workspaces and pipelines. Now you're ready to put your feet up! But to keep your models operational, your work's not done once the model is in production. You must continue to monitor the behavior and performance of the model to insure that the model provides value to the business.

In this notebook, you will learn about adding validation rules to pipelines.

## Preliminaries

In the blocks below we will preload some required libraries; we will also redefine some of the convenience functions that you saw in the previous notebooks.

After that, you should log into Wallaroo and set your working environment to the workspace that you created for this tutorial. Please refer to Notebook 1 to refresh yourself on how to log in and set your working environment to the appropriate workspace.

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

# return the workspace called <name>, or create it if it does not exist.
# this function assumes your connection to wallaroo is called wl
def get_workspace(name):
    workspace = None
    for ws in wl.list_workspaces():
        if ws.name() == name:
            workspace= ws
    if(workspace == None):
        workspace = wl.create_workspace(name)
    return workspace

# pull a single datum from a data frame 
# and convert it to the format the model expects
def get_singleton(df, i):
    singleton = df.iloc[i,:].to_numpy().tolist()
    sdict = {'tensor': [singleton]}
    return pd.DataFrame.from_dict(sdict)

# pull a batch of data from a data frame
# and convert to the format the model expects
def get_batch(df, first=0, nrows=1):
    last = first + nrows
    batch = df.iloc[first:last, :].to_numpy().tolist()
    return pd.DataFrame.from_dict({'tensor': batch})

# Get the most recent version of a model in the workspace
# Assumes that the most recent version is the first in the list of versions.
# wl.get_current_workspace().models() returns a list of models in the current workspace

def get_model(mname):
    modellist = wl.get_current_workspace().models()
    model = [m.versions()[0] for m in modellist if m.name() == mname]
    if len(model) <= 0:
        raise KeyError(f"model {mname} not found in this workspace")
    return model[0]

# get a pipeline by name in the workspace
def get_pipeline(pname):
    plist = wl.get_current_workspace().pipelines()
    pipeline = [p for p in plist if p.name() == pname]
    if len(pipeline) <= 0:
        raise KeyError(f"pipeline {pname} not found in this workspace")
    return pipeline[0]
```

```python
## blank space to log in and go to correct workspace

wl = wallaroo.Client()

workspace_name = "tutorial-workspace-jch"

workspace = get_workspace(workspace_name)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

# optionally, examine your current workspace
wl.get_current_workspace()

```

    {'name': 'tutorial-workspace-jch', 'id': 9, 'archived': False, 'created_by': 'c3a45eb6-37ff-4020-8d59-7166c3e153d0', 'created_at': '2023-07-20T14:35:12.650973+00:00', 'models': [{'name': 'tutorial-model', 'versions': 3, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 7, 20, 16, 3, 10, 617651, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 7, 20, 14, 43, 43, 886638, tzinfo=tzutc())}, {'name': 'challenger-model', 'versions': 3, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 7, 20, 16, 35, 29, 301552, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 7, 20, 16, 14, 1, 624750, tzinfo=tzutc())}], 'pipelines': [{'name': 'tutorialpipeline-jch', 'create_time': datetime.datetime(2023, 7, 20, 14, 43, 44, 606725, tzinfo=tzutc()), 'definition': '[]'}]}

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

{{<figure src="/images/2023.2.1/wallaroo-use-case-tutorials/linear-regression/real-estate/validation_results.png" width="800" label="Inferences with check failures">}}

You can also find check failures in the logs:

```python
logs = pipeline.logs()
logs.loc[logs['check_failures'] > 0]
```
<hr/>

#### Exercise: Add validation rules to your model pipeline

Add some simple validation rules to the model pipeline that you created in a previous exercise.

* Add an upper bound or a lower bound to the model predictions
* Try to create predictions that fall both in and out of the specified range
* Look through the logs to find the check failures.

**HINT 1**: since the purpose of this exercise is try out validation rules, it might be a good idea to take a small data set and make predictions on that data set first, *then* set the validation rules based on those predictions, so that you can see the check failures trigger.

**Don't forget to undeploy your pipeline after you are done** to free up resources.

```python
## blank space to get your pipeline and run a small batch of data through it to see the range of predictions

pipeline = get_pipeline('tutorialpipeline-jch')

control_model = get_model('tutorial-model')

pipeline.clear()
pipeline.add_model_step(control_model)

pipeline.deploy()

df_from_csv = pd.read_csv('./data/test_data.csv')

multiple_batch = get_batch(df_from_csv, nrows=5)
multiple_result = pipeline.infer(multiple_batch)
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
      <td>2023-07-20 16:42:47.741</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[704901.9]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-07-20 16:42:47.741</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[695994.44]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-07-20 16:42:47.741</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[416164.8]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-07-20 16:42:47.741</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[655277.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-07-20 16:42:47.741</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[426854.66]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
# blank space to set a validation rule on the pipeline and check if it triggers as expected

hi_bnd = 600000.0

pipeline = pipeline.add_validation("less than 1.5m", control_model.outputs[0][0] < hi_bnd)

pipeline.deploy()

multiple_batch = get_batch(df_from_csv, nrows=5)
multiple_result = pipeline.infer(multiple_batch)
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
      <td>2023-07-20 16:44:59.277</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[704901.9]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-07-20 16:44:59.277</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[695994.44]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-07-20 16:44:59.277</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[416164.8]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-07-20 16:44:59.277</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[655277.2]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-07-20 16:44:59.277</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[426854.66]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

## Congratulations!

In this tutorial you have

* Set a validation rule on your house price prediction pipeline.
* Detected model predictions that failed the validation rule.

In the next notebook, you will learn how to monitor the distribution of model outputs for drift away from expected behavior.

### Cleaning up.

At this point, if you are not continuing on to the next notebook, undeploy your pipeline to give the resources back to the environment.

```python
## blank space to undeploy the pipeline

pipeline.undeploy()
```

<table><tr><th>name</th> <td>tutorialpipeline-jch</td></tr><tr><th>created</th> <td>2023-07-20 14:43:44.606725+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-20 16:44:57.005341+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>e6c9d6be-11ee-4dc9-b065-fbb2ae82d262, f8c21055-9732-4a89-9fd7-face19785200, acbbf366-711b-4904-b5d7-2459015f2d70, 3da3ade7-52d5-4fe0-8609-027322769f39, f27623ad-90dc-4161-b826-65084d0f198d, d886d504-d7fd-42b7-9b95-7eadfa6cafbe, 117dac7b-78fe-449d-99c7-6233c100874c, 67f9bfe6-7f03-4087-a6db-46896729c78f, aa7f65e4-0edd-4cb7-8a81-7c5f46c323a7, 4edac46e-22eb-46c0-8ee7-e2cf68181958, ed2b8ae9-eb77-46b5-9985-aa520be6c6b7, 83b4129b-3271-4e5a-aa5c-203b8b95e674, d7c7b6b1-04d5-4d20-a85b-39159cb79ede, 0f5b2df6-6498-4ace-a739-7c585284d6c6, da128b7e-26f5-4b9d-a58e-b244735843bd, 89094eea-ef47-4d4c-bf33-2829b2aeaf64</td></tr><tr><th>steps</th> <td>tutorial-model</td></tr></table>

