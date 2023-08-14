# Tutorial Notebook 2: Observability Part 1 - Validation Rules

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

workspace_name = f"tutorial-workspace"

workspace = get_workspace(workspace_name)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

# optionally, examine your current workspace
wl.get_current_workspace()

```

    {'name': 'tutorial-workspace-jch', 'id': 19, 'archived': False, 'created_by': '0a36fba2-ad42-441b-9a8c-bac8c68d13fa', 'created_at': '2023-08-03T19:34:42.324336+00:00', 'models': [{'name': 'tutorial-model', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 3, 19, 36, 31, 13200, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 3, 19, 36, 31, 13200, tzinfo=tzutc())}, {'name': 'embedder', 'versions': 2, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 11, 15, 43, 19, 353975, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 11, 15, 34, 48, 164613, tzinfo=tzutc())}, {'name': 'sentiment', 'versions': 2, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 11, 15, 43, 20, 40661, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 11, 15, 34, 48, 913135, tzinfo=tzutc())}], 'pipelines': [{'name': 'tutorialpipeline-jch', 'create_time': datetime.datetime(2023, 8, 3, 19, 36, 31, 732163, tzinfo=tzutc()), 'definition': '[]'}, {'name': 'sentiment-analysis', 'create_time': datetime.datetime(2023, 8, 11, 15, 34, 49, 622995, tzinfo=tzutc()), 'definition': '[]'}]}

## Model Validation Rules

A simple way to try to keep your model's behavior up to snuff is to make sure that it receives inputs that it expects, and that its output is something that downstream systems can handle. This can entail specifying rules that document what you expect, and either enforcing these rules (by refusing to make a prediction), or at least logging an alert that the expectations described by your validation rules have been violated. As the developer of the model, the data scientist (along with relevant subject matter experts) will often be the person in the best position to specify appropriate validation rules.

In our IMDB sentiment model, rankings are typically arount 50-90%.  So any rankings below 50% may indicate either a very negative review, or that some prediction is off with the model and should be investigated further.

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

{{<figure src="/images/2023.2.1/wallaroo-use-case-tutorials/nlp-classification/sentiment-analysis/validation_results.png" width="800" label="Inferences with check failures">}}

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

embedder_name = 'embedder'
sentiment_model_name = 'sentiment'

pipeline_name = 'sentiment-analysis'

pipeline = get_pipeline(pipeline_name)

embedder_model = get_model(embedder_name)
sentiment_model = get_model(sentiment_model_name)

pipeline.clear()
pipeline.add_model_step(embedder_model)
pipeline.add_model_step(sentiment_model)

pipeline.deploy()

df = pd.read_json('../data/test_data_50K.df.json')

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
      <td>[[11.0, 6.0, 1.0, 12.0, 112.0, 13.0, 14.0, 73.0, 14.0, 10.0, 470.0, 5.0, 116.0, 9.0, 207.0, 465.0, 96.0, 15.0, 69.0, 5.0, 231.0, 15.0, 9.0, 91.0, 812.0, 6.0, 28.0, 4.0, 58.0, 511.0, 9654.0, 148.0, 6792.0, 20.0, 1.0, 82.0, 505.0, 1098.0, 30.0, 3.0, 7476.0, 2.0, 2032.0, 96.0, 547.0, 1059.0, 2.0, 148.0, 42.0, 640.0, 4716.0, 8.0, 91.0, 1670.0, 4939.0, 783.0, 41.0, 3.0, 529.0, 449.0, 9.0, 492.0, 85.0, 3050.0, 2.0, 1.0, 357.0, 4.0, 1.0, 174.0, 468.0, 8.0, 84.0, 351.0, 155.0, 155.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]</td>
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
      <td>2023-08-11 16:18:23.435</td>
      <td>[[11.0, 6.0, 1.0, 12.0, 112.0, 13.0, 14.0, 73.0, 14.0, 10.0, 470.0, 5.0, 116.0, 9.0, 207.0, 465.0, 96.0, 15.0, 69.0, 5.0, 231.0, 15.0, 9.0, 91.0, 812.0, 6.0, 28.0, 4.0, 58.0, 511.0, 9654.0, 148.0, 6792.0, 20.0, 1.0, 82.0, 505.0, 1098.0, 30.0, 3.0, 7476.0, 2.0, 2032.0, 96.0, 547.0, 1059.0, 2.0, 148.0, 42.0, 640.0, 4716.0, 8.0, 91.0, 1670.0, 4939.0, 783.0, 41.0, 3.0, 529.0, 449.0, 9.0, 492.0, 85.0, 3050.0, 2.0, 1.0, 357.0, 4.0, 1.0, 174.0, 468.0, 8.0, 84.0, 351.0, 155.0, 155.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]</td>
      <td>[0.8980188]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
# blank space to set a validation rule on the pipeline and check if it triggers as expected

# 50%
hi_bnd = 50.0

pipeline = pipeline.add_validation("less than 50%", sentiment_model.outputs[0][0] > hi_bnd / 100)

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
      <td>2023-08-11 16:18:26.141</td>
      <td>[[11.0, 6.0, 1.0, 12.0, 112.0, 13.0, 14.0, 73.0, 14.0, 10.0, 470.0, 5.0, 116.0, 9.0, 207.0, 465.0, 96.0, 15.0, 69.0, 5.0, 231.0, 15.0, 9.0, 91.0, 812.0, 6.0, 28.0, 4.0, 58.0, 511.0, 9654.0, 148.0, 6792.0, 20.0, 1.0, 82.0, 505.0, 1098.0, 30.0, 3.0, 7476.0, 2.0, 2032.0, 96.0, 547.0, 1059.0, 2.0, 148.0, 42.0, 640.0, 4716.0, 8.0, 91.0, 1670.0, 4939.0, 783.0, 41.0, 3.0, 529.0, 449.0, 9.0, 492.0, 85.0, 3050.0, 2.0, 1.0, 357.0, 4.0, 1.0, 174.0, 468.0, 8.0, 84.0, 351.0, 155.0, 155.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]</td>
      <td>[0.8980188]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-08-11 16:18:26.141</td>
      <td>[[54.0, 548.0, 86.0, 70.0, 1213.0, 24.0, 746.0, 6.0, 11.0, 19.0, 6.0, 3.0, 1898.0, 90.0, 370.0, 113.0, 832.0, 367.0, 154.0, 10.0, 78.0, 21.0, 121.0, 135.0, 4717.0, 5.0, 350.0, 2.0, 1594.0, 122.0, 3.0, 26.0, 6.0, 2315.0, 30.0, 9.0, 22.0, 103.0, 1.0, 2253.0, 20.0, 1.0, 285.0, 2.0, 1.0, 93.0, 26.0, 44.0, 3.0, 367.0, 790.0, 87.0, 184.0, 26.0, 40.0, 9.0, 53.0, 26.0, 1383.0, 109.0, 1.0, 2211.0, 4.0, 688.0, 26.0, 6.0, 3.0, 75.0, 281.0, 26.0, 1784.0, 69.0, 4.0, 157.0, 4311.0, 1720.0, 2124.0, 46.0, 86.0, 44.0, 66.0, 11.0, 19.0, 614.0, 30.0, 540.0, 1927.0, 4588.0, 2.0, 159.0, 555.0, 118.0, 5924.0, 81.0, 264.0, 15.0, 2.0, 688.0, 530.0, 20.0]]</td>
      <td>[0.056596935]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-08-11 16:18:26.141</td>
      <td>[[1.0, 9259.0, 6.0, 8.0, 1.0, 3.0, 62.0, 4.0, 32.0, 4416.0, 34.0, 457.0, 8595.0, 31.0, 1.0, 497.0, 2.0, 8.0, 1.0, 972.0, 2847.0, 2178.0, 24.0, 110.0, 2.0, 1.0, 1918.0, 60.0, 1072.0, 1.0, 129.0, 26.0, 44.0, 410.0, 2353.0, 8.0, 49.0, 2.0, 442.0, 8.0, 1.0, 4287.0, 4.0, 24.0, 24.0, 116.0, 599.0, 5074.0, 2.0, 1135.0, 7093.0, 2602.0, 5120.0, 2.0, 22.0, 25.0, 3.0, 450.0, 8596.0, 16.0, 3036.0, 2.0, 1975.0, 385.0, 16.0, 1.0, 1023.0, 931.0, 4.0, 2137.0, 2.0, 1.0, 3022.0, 4.0, 309.0, 4416.0, 294.0, 32.0, 318.0, 19.0, 15.0, 145.0, 80.0, 807.0, 3264.0, 1.0, 4416.0, 294.0, 5034.0, 15.0, 3023.0, 6.0, 32.0, 5514.0, 4.0, 1.0, 1299.0, 2205.0, 493.0, 1.0]]</td>
      <td>[0.9260802]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-08-11 16:18:26.141</td>
      <td>[[10.0, 25.0, 107.0, 1.0, 343.0, 17.0, 3.0, 168.0, 150.0, 593.0, 100.0, 12.0, 10.0, 103.0, 29.0, 2278.0, 1.0, 83.0, 28.0, 6.0, 63.0, 21.0, 1.0, 115.0, 18.0, 42.0, 1.0, 88.0, 1060.0, 28.0, 204.0, 458.0, 103.0, 1.0, 228.0, 4.0, 6887.0, 4252.0, 297.0, 42.0, 63.0, 84.0, 48.0, 131.0, 490.0, 119.0, 79.0, 1.0, 2278.0, 23.0, 318.0, 8.0, 1.0, 315.0, 299.0, 190.0, 126.0, 576.0, 5.0, 103.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]</td>
      <td>[0.926919]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-08-11 16:18:26.141</td>
      <td>[[10.0, 37.0, 1.0, 49.0, 2.0, 442.0, 982.0, 10.0, 420.0, 1807.0, 8.0, 11.0, 17.0, 125.0, 71.0, 98.0, 17.0, 26.0, 44.0, 123.0, 221.0, 26.0, 283.0, 1.0, 1389.0, 9260.0, 121.0, 9.0, 29.0, 26.0, 628.0, 295.0, 26.0, 284.0, 480.0, 2.0, 3.0, 50.0, 4484.0, 482.0, 1.0, 189.0, 12.0, 9.0, 284.0, 47.0, 23.0, 3108.0, 180.0, 8.0, 1822.0, 2.0, 20.0, 699.0, 71.0, 72.0, 67.0, 101.0, 4.0, 405.0, 69.0, 437.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]</td>
      <td>[0.6618577]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

## Congratulations!

In this tutorial you have

* Set a validation rule on your sentiment model pipeline pipeline.
* Detected model predictions that failed the validation rule.

In the next notebook, you will learn how to monitor the distribution of model outputs for drift away from expected behavior.

### Cleaning up.

At this point, if you are not continuing on to the next notebook, undeploy your pipeline to give the resources back to the environment.

```python
## blank space to undeploy the pipeline

pipeline.undeploy()
```

<table><tr><th>name</th> <td>sentiment-analysis</td></tr><tr><th>created</th> <td>2023-08-11 15:34:49.622995+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-11 16:18:23.859761+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>3665c777-a8ee-409c-a058-40e90aeb60f7, 5bdf82eb-25f0-41c5-8812-06da87b5d2e1, 71fdf82d-9e04-4977-a69b-1e49e9452b5c, 1210dd5d-38db-4561-8721-694ac690063a, f7853459-c4f5-4b71-a3b4-520e3b257245, cf689a7d-e51a-4d58-96fd-a024ebd3ddba, d1a918df-04fe-4b98-8d1e-01f7831aab44, 7668ad9a-c12d-40a5-8370-c681c87e4786, 8028e5fc-81b7-45a0-8347-61e0e17e20c4</td></tr><tr><th>steps</th> <td>sentiment</td></tr></table>

