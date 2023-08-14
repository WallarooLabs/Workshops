# Workshop Notebook 3: Observability Part 1 - Validation Rules

In the previous notebooks you uploaded the models and artifacts, then deployed the models to production through provisioning workspaces and pipelines. Now you're ready to put your feet up! But to keep your models operational, your work's not done once the model is in production. You must continue to monitor the behavior and performance of the model to insure that the model provides value to the business.

In this notebook, you will learn about adding validation rules to pipelines.

## Preliminaries

In the blocks below we will preload some required libraries; we will also redefine some of the convenience functions that you saw in the previous notebooks.

After that, you should log into Wallaroo and set your working environment to the workspace that you created for this workshop. Please refer to Notebook 1 to refresh yourself on how to log in and set your working environment to the appropriate workspace.

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
    model = [m.versions()[-1] for m in modellist if m.name() == mname]
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

workspace_name = "classification-finserv-jch"

workspace = get_workspace(workspace_name)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

# optionally, examine your current workspace
wl.get_current_workspace()

```

    {'name': 'classification-finserv-jch', 'id': 21, 'archived': False, 'created_by': '0a36fba2-ad42-441b-9a8c-bac8c68d13fa', 'created_at': '2023-08-07T16:26:26.779098+00:00', 'models': [{'name': 'ccfraud-model-keras', 'versions': 2, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 7, 16, 28, 46, 566311, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 7, 16, 26, 36, 806125, tzinfo=tzutc())}, {'name': 'ccfraud-model-xgboost', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 7, 17, 20, 51, 978426, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 7, 17, 20, 51, 978426, tzinfo=tzutc())}], 'pipelines': [{'name': 'finserv-ccfraud', 'create_time': datetime.datetime(2023, 8, 7, 16, 26, 37, 485326, tzinfo=tzutc()), 'definition': '[]'}]}

## Model Validation Rules

A simple way to try to keep your model's behavior up to snuff is to make sure that it receives inputs that it expects, and that its output is something that downstream systems can handle. This can entail specifying rules that document what you expect, and either enforcing these rules (by refusing to make a prediction), or at least logging an alert that the expectations described by your validation rules have been violated. As the developer of the model, the data scientist (along with relevant subject matter experts) will often be the person in the best position to specify appropriate validation rules.

In our financial fraud example, supposed you know that confidences from a set of sample data shouldn't go below 75% likelihood of fraud.  Then you might want to set validation rules on your model pipeline to specify that you expect the model's predictions to also be in that range. That way if the model predicts a value under that range, the pipeline will log that one of the validation checks has failed; this allows you to investigate that instance further.

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

{{<figure src="/images/2023.2.1/wallaroo-use-case-tutorials/classification/finserv/validation_results.png" width="800" label="Inferences with check failures">}}

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

See [Anomaly Testing](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/#anomaly-testing) for more information.

```python
## blank space to get your pipeline and run a small batch of data through it to see the range of predictions

pipeline = get_pipeline('finserv-ccfraud')

ccfraud_keras_model = get_model('ccfraud-model-keras')

pipeline.clear()
pipeline.add_model_step(ccfraud_keras_model)

pipeline.deploy()

df = pd.read_json('../data/cc_data_10k.df.json')

singleton = get_singleton(df, 0)
display(singleton)

single_result = pipeline.infer(singleton)
display(single_result)

```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[[-1.0603297501, 2.3544967095000002, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526000001, 1.9870535692, 0.7005485718000001, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756000001, -0.1466244739, -1.4463212439]]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-07 17:35:16.080</td>
      <td>[[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
# blank space to set a validation rule on the pipeline and check if it triggers as expected

low_bnd = 0.75

pipeline = pipeline.add_validation("less than 75%", ccfraud_keras_model.outputs[0][0] > low_bnd)

pipeline.deploy()

multiple_batch = get_batch(df, nrows=5)
multiple_result = pipeline.infer(multiple_batch)
display(multiple_result)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-07 17:35:27.089</td>
      <td>[[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-08-07 17:35:27.089</td>
      <td>[[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-08-07 17:35:27.089</td>
      <td>[[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-08-07 17:35:27.089</td>
      <td>[[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-08-07 17:35:27.089</td>
      <td>[[0.5817662108, 0.097881551, 0.1546819424, 0.4754101949, -0.1978862306, -0.4504344854, 0.0166540447, -0.0256070551, 0.0920561602, -0.2783917153, 0.0593299441, -0.0196585416, -0.4225083157, -0.1217538877, 1.5473094894, 0.2391622864, 0.3553974881, -0.7685165301, -0.7000849355, -0.1190043285, -0.3450517133, -1.1065114108, 0.2523411195, 0.0209441826, 0.2199267436, 0.2540689265, -0.0450225094, 0.1086773898, 0.2547179311]]</td>
      <td>[0.0010916889]</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

## Congratulations!

In this workshop you have

* Set a validation rule on your financial fraud classification pipeline.
* Detected model predictions that failed the validation rule.

In the next notebook, you will learn how to monitor the distribution of model outputs for drift away from expected behavior.

### Cleaning up.

At this point, if you are not continuing on to the next notebook, undeploy your pipeline to give the resources back to the environment.

```python
## blank space to undeploy the pipeline

pipeline.undeploy()
```

<table><tr><th>name</th> <td>finserv-ccfraud</td></tr><tr><th>created</th> <td>2023-08-07 16:26:37.485326+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-07 17:35:24.149368+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>095ef880-06f0-45d5-bfef-ac4e11d2b06c, f9004ecd-7a8c-4963-bebc-e40d1cde357c, 2513faec-2385-4733-a42d-56a8ae38761a, 10e495af-9b23-4973-913e-4cdebca73461, 782b9edc-d42a-4984-90ca-3bdf35922b87, e80d21a4-21c2-46b8-85d4-6652bbac9506, 8ff48a62-fd9d-43f5-ba3f-11a7f1bcc474, 36faf126-dac5-419d-b0b1-7d7b698b587e, 230d585a-52db-476d-ab28-7b4baed9d023, 192f92e9-9a97-4339-8c1d-f89541ff2cef, 5d2d9c84-13c2-4e35-a41f-ec3c4e8d297b, 41927bef-d8fb-49ee-914e-d106ffc304b3</td></tr><tr><th>steps</th> <td>ccfraud-model-keras</td></tr></table>

