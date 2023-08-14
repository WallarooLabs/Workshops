# Workshop Notebook 2: Vetting a Model With Production Experiments

So far, we've discussed practices and methods for transitioning an ML model and related artifacts from development to production. However, just the act of pushing a model into production is not the only consideration. In many situations, it's important to vet a model's performance in the real world before fully activating it. Real world vetting can surface issues that may not have arisen during the development stage, when models are only checked using hold-out data.

In this notebook, you will learn about two kinds of production ML model validation methods: A/B testing and Shadow Deployments. A/B tests and other types of experimentation are part of the ML lifecycle. The ability to quickly experiment and test new models in the real world helps data scientists to continually learn, innovate, and improve AI-driven decision processes.

## Preliminaries

In the blocks below we will preload some required libraries; we will also redefine some of the convenience functions that you saw in the previous notebook.

After that, you should log into Wallaroo and set your working environment to the workspace that you created in the previous notebook. 

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
## convenience functions from the previous notebook

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

```

#### Pre-exercise

If needed, log into Wallaroo and go to the workspace that you created in the previous notebook. Please refer to Notebook 1 to refresh yourself on how to log in and set your working environment to the appropriate workspace.

```python
## blank space to log in and go to the appropriate workspace

wl = wallaroo.Client()

workspace_name = "classification-finserv-jch"

workspace = get_workspace(workspace_name)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

# optionally, examine your current workspace
wl.get_current_workspace()

```

    {'name': 'classification-finserv-jch', 'id': 21, 'archived': False, 'created_by': '0a36fba2-ad42-441b-9a8c-bac8c68d13fa', 'created_at': '2023-08-07T16:26:26.779098+00:00', 'models': [{'name': 'ccfraud-model-keras', 'versions': 2, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 7, 16, 28, 46, 566311, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 7, 16, 26, 36, 806125, tzinfo=tzutc())}], 'pipelines': [{'name': 'finserv-ccfraud', 'create_time': datetime.datetime(2023, 8, 7, 16, 26, 37, 485326, tzinfo=tzutc()), 'definition': '[]'}]}

## A/B Testing

An [A/B test](https://en.wikipedia.org/wiki/A/B_testing), also called a controlled experiment or a randomized control trial, is a statistical method of determining which of a set of variants is the best. A/B tests allow organizations and policy-makers to make smarter, data-driven decisions that are less dependent on guesswork.

In the simplest version of an A/B test, subjects are randomly assigned to either the **_control group_** (group A) or the **_treatment group_** (group B). Subjects in the treatment group receive the treatment (such as a new medicine, a special offer, or a new web page design) while the control group proceeds as normal without the treatment. Data is then collected on the outcomes and used to study the effects of the treatment.

In data science, A/B tests are often used to choose between two or more candidate models in production, by measuring which model performs best in the real world. In this formulation, the control is often an existing model that is currently in production, sometimes called the **_champion_**. The treatment is a new model being considered to replace the old one. This new model is sometimes called the **_challenger_**. In our discussion, we'll use the terms *champion* and *challenger*, rather than *control* and *treatment*.

When data is sent to a Wallaroo A/B test pipeline for inference, each datum is randomly sent to either the champion or challenger. After enough data has been sent to collect statistics on all the models in the A/B test pipeline, then those outcomes can be analyzed to determine the difference (if any) in the performance of the champion and challenger. Usually, the purpose of an A/B test is to decide whether or not to replace the champion with the challenger.

Keep in mind that in machine learning, the terms experiments and trials also often refer to the process of finding a training configuration that works best for the problem at hand (this is sometimes called hyperparameter optimization). In this guide, we will use the term experiment to refer to the use of A/B tests to compare the performance of different models in production.
<hr/>

#### Exercise: Create some challenger models and upload them to Wallaroo

Use the house price data from Notebook 1 to create at least one alternate house price prediction model. You can do this by varying the modeling algorithm, the inputs, the feature engineering, or all of the above. 

For the purpose of these exercises, please make sure that the predictions from the new model(s) are in the same units as the (champion) model that you created in Chapter 3. For example, if the champion model predicts log price, then the challenger models should also predict log price. If the champion model predicts price in units of $10,000, then the challenger models should, also.

* **If you prefer to shortcut this step, you can use some of the pretrained model onnx files in the `models` directory**
* Upload your new model(s) to Wallaroo, into your workspace

At the end of this exercise, you should have at least one challenger model to compare to your champion model uploaded to your workspace.

```python
# blank space to train, convert, and upload new model

challenger_model = wl.upload_model('ccfraud-model-xgboost', '../models/xgboost_ccfraud.onnx', framework=Framework.ONNX)
```

There are a number of considerations to designing an A/B test; you can check out the article [*The What, Why, and How of A/B Testing*](https://wallarooai.medium.com/the-what-why-and-how-of-a-b-testing-64471847cd7e) for more details. In these exercises, we will concentrate on the deployment aspects.  You will need a champion model and  at least one challenger model. You also need to decide on a data split: for example 50-50 between the champion and challenger, or a 2:1 ratio between champion and challenger (two-thirds of the data to the champion, one-third to the challenger).

As an example of creating an A/B test deployment, suppose you have a champion model called "champion" that you have been running in a one-step pipeline called "pipeline". You now want to compare it to a challenger model called "challenger". For your A/B test, you will send two-thirds of the data to the champion, and the other third to the challenger. Both models have already been uploaded. 

To help you with the exercises, here some convenience functions to retrieve a models and pipelines that have been previously uploaded to your workspace (in this example, `wl` is your `wallaroo.client()` object). 

```python
# Get the most recent version of a model.
# Assumes that the most recent version is the first in the list of versions.
# wl.get_current_workspace().models() returns a list of models in the current workspace

def get_model(mname, modellist=wl.get_current_workspace().models()):
    model = [m.versions()[-1] for m in modellist if m.name() == mname]
    if len(model) <= 0:
        raise KeyError(f"model {mname} not found in this workspace")
    return model[0]

# get a pipeline by name in the workspace
def get_pipeline(pname, plist = wl.get_current_workspace().pipelines()):
    pipeline = [p for p in plist if p.name() == pname]
    if len(pipeline) <= 0:
        raise KeyError(f"pipeline {pname} not found in this workspace")
    return pipeline[0]
```

```python
# use the space here for retrieving the models and pipeline

pipeline = get_pipeline('finserv-ccfraud')

ccfraud_keras_model = get_model('ccfraud-model-keras')

```

Pipelines may have already been issued with pipeline steps.  Pipeline steps can be removed or replaced with other steps.

The easiest way to clear **all** pipeline steps is with the Pipeline `clear()` method.

To remove one step, use the Pipeline `remove_step(index)` method, where `index` is the step number ordered from zero.  For example, if a pipeline has one step, then `remove_step(0)` would remove that step.

To replace a pipeline step, use the Pipeline `replace_with_model_step(index, model)`, where `index` is the step number ordered from zero, and the `model` is the model to be replacing it with.

Updated pipeline steps are not saved until the pipeline is redeployed with the Pipeline `deploy()` method.

Reference:  [Wallaroo SDK Essentials Guide: Pipeline Management
](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipeline/).

For A/B testing, pipeline steps are **added** or **replace** an existing step.

To **add** a A/B testing step use the Pipeline `add_random_split` method with the following parameters:

| Parameter | Type | Description |
| --- | --- | ---|
| **champion_weight** | Float (Required) | The weight for the champion model. |
| **champion_model** | Wallaroo.Model (Required) | The uploaded champion model. |
| **challenger_weight** | Float (Required) | The weight of the challenger model. |
| **challenger_model** | Wallaroo.Model (Required) | The uploaded challenger model. |
| **hash_key** | String(Optional) | A key used instead of a random number for model selection.  This must be between 0.0 and 1.0. |

Note that multiple challenger models with different weights can be added as the random split step.

In this example, a pipeline will be built with a 2:1 weighted ratio between the champion and a single challenger model.

```python
pipeline.add_random_split([(2, control), (1, challenger)]))
```

To **replace** an existing pipeline step with an A/B testing step use the Pipeline `replace_with_random_split` method.

| Parameter | Type | Description |
| --- | --- | ---|
| **index** | Integer (Required) | The pipeline step being replaced. |
| **champion_weight** | Float (Required) | The weight for the champion model. |
| **champion_model** | Wallaroo.Model (Required) | The uploaded champion model. |
| **challenger_weight** | Float (Required) | The weight of the challenger model. |
| **challenger_model** | Wallaroo.Model (Required) | The uploaded challenger model. |
| **hash_key** | String(Optional) | A key used instead of a random number for model selection.  This must be between 0.0 and 1.0. |

This example replaces the first pipeline step with a 2:1 champion to challenger radio.

```python
pipeline.replace_with_random_split(0,[(2, control), (1, challenger)]))
```

In either case, the random split will randomly send inference data to one model based on the weighted ratio.  As more inferences are performed, the ratio between the champion and challengers will align more and more to the ratio specified.

Reference:  [Wallaroo SDK Essentials Guide: Pipeline Management A/B Testing
](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipeline/#ab-testing).

Then creating an A/B test deployment would look something like this:

First get the models used.

```
# retrieve handles to the most recent versions 
# of the champion and challenger models
champion = get_model("champion")
challenger = get_model("challenger")

```

```python
# blank space to get the model(s)

ccfraud_xgboost_model = get_model('ccfraud-model-xgboost')

```

Second step is to retrieve the pipeline created in the previous Notebook, then redeploy it with the A/B testing split step.

Here's some sample code:

```python
# get an existing single-step pipeline and undeploy it
pipeline = get_pipeline("pipeline")
pipeline.undeploy()

# clear the pipeline and add a random split
pipeline.clear()
pipeline.add_random_split([(2, champion), (1, challenger)])
pipeline.deploy()
```

The above code clears out all the steps of the pipeline and adds a new step with a A/B test deployment, where the incoming data is randomly sent in a 2:1 ratio to the champion and the challenger, respectively.

You can add multiple challengers to an A/B test::

```
pipeline.add_random_split([ (2, champion), (1, challenger01), (1, challenger02) ])
```

This pipeline will distribute data in the ratio 2:1:1 (or half to the champion, a quarter each to the challlengers) to the champion and challenger models, respectively.

You can also create an A/B test deployment from scratch:

```
pipeline = wl.build_pipeline("pipeline")
pipeline.add_random_split([(2, champion), (1, challenger)])
```

<hr/>

#### Exercise: Create an A/B test deployment of your house price models

Use the champion and challenger models that you created in the previous exercises to create an A/B test deployment. You can either create one from scratch, or reconfigure an existing pipeline. 

* Send half the data to the champion, and distribute the rest among the challenger(s).

At the end of this exercise, you should have an A/B test deployment and be ready to compare  multiple models.

```python
# blank space to retrieve pipeline and redeploy with a/b testing step

pipeline.clear()
pipeline.add_random_split([(2, ccfraud_keras_model), (1, ccfraud_xgboost_model)])
pipeline.deploy()

```

<table><tr><th>name</th> <td>finserv-ccfraud</td></tr><tr><th>created</th> <td>2023-08-07 16:26:37.485326+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-07 17:22:20.920743+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>36faf126-dac5-419d-b0b1-7d7b698b587e, 230d585a-52db-476d-ab28-7b4baed9d023, 192f92e9-9a97-4339-8c1d-f89541ff2cef, 5d2d9c84-13c2-4e35-a41f-ec3c4e8d297b, 41927bef-d8fb-49ee-914e-d106ffc304b3</td></tr><tr><th>steps</th> <td>ccfraud-model-keras</td></tr></table>

The pipeline steps are displayed with the Pipeline `steps()` method.  This is used to verify the current **deployed** steps in the pipeline.

* **IMPORTANT NOTE**: Verify that the pipeline is deployed before checking for pipeline steps.  Deploying the pipeline sets the steps into the Wallaroo system - until that happens, the steps only exist in the local system as *potential* steps.

```python
# blank space to get the current pipeline steps

pipeline.steps()
```

    [{'RandomSplit': {'hash_key': None, 'weights': [{'model': {'name': 'ccfraud-model-keras', 'version': '53dbee25-2e64-4acf-8f5b-44feab1de488', 'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507'}, 'weight': 2}, {'model': {'name': 'ccfraud-model-xgboost', 'version': 'e71c9c1d-9777-4967-b32c-a074ea8aa467', 'sha': '054810e3e3ebbdd34438d9c1a08ed6a6680ef10bf97b9223f78ebf38e14b3b52'}, 'weight': 1}]}}]

Please note that for batch inferences, the entire batch will be sent to the same model. So in order to verify that your pipeline is distributing inferences in the proportion you specified, you will need to send your queries one datum at a time.

To help with the next exercise, here is another convenience function you might find useful.

```python
# get the names of the inferring models
# from a dataframe of a/b test results
def get_names(resultframe):
    modelcol = resultframe['out._model_split']
    jsonstrs = [mod[0]  for mod in modelcol]
    return [json.loads(jstr)['name'] for jstr in jsonstrs]
```

Here's an example of how to send a large number of queries one at a time to your pipeline in the SDK

```python
results = []

# get a list of result frames
for i in range(1000):
    query = get_singleton(testdata, i)
    results.append(pipeline.infer(query))

# make one data frame of all results    
allresults = pd.concat(results, ignore_index=True)

# add a column to indicate which model made the inference
allresults['modelname'] = get_names(allresults)

# get the counts of how many inferences were made by each model
allresults.modelname.value_counts()
```  

* **NOTE**:  Performing 1,000 inferences sequentially may take several minutes to complete.  Adjust the range for time as required.

As with the single-step pipeline, the model predictions will be in a column named `out.<outputname>`. In addition, there will be a column named `out._model_split`  that contains information about the model that made a particular prediction. The `get_names()` convenience function above extracts the model name from the `out._model_split` column.

<hr/>

#### Exercise: Send some queries to your A/B test deployment

1. Send a single datum to the A/B test pipeline you created in the previous exercise. You can use the same test data set that you created/downloaded in the previous notebook. Observe what the inference result looks like. If you send the singleton through the pipeline multiple times, you should observe that the model making the inference changes.
2. Send a large number of queries (at least 100) one at a time to the pipeline.
  * Note that approximately half the inferences were made by the champion model.
  * The remaining inferences should be distributed as you specified.

The more queries you send, the closer the distribution should be to what you specified.

If you can align the actual house prices from your test data to the predictions, you can also compare the accuracy of the different models.

**Don't forget to undeploy your pipeline after you are done**, to free up resources.

```python
## blank space to test one inference

##  blank space to create test data, and send some data to your model

df = pd.read_json('../data/cc_data_10k.df.json')

singleton = get_singleton(df, 0)
display(singleton)

single_result = pipeline.infer(singleton)
display(single_result)

display(get_names(single_result))

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
      <th>out._model_split</th>
      <th>out.dense_1</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-07 17:24:57.870</td>
      <td>[[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]]</td>
      <td>[{"name":"ccfraud-model-keras","version":"53dbee25-2e64-4acf-8f5b-44feab1de488","sha":"bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507"}]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

    ['ccfraud-model-keras']

```python
# blank space to send queries to A/B test pipeline and examine the results

results = []

# get a list of result frames
for i in range(20):
    query = get_singleton(df, i)
    results.append(pipeline.infer(query))

# make one data frame of all results    
allresults = pd.concat(results, ignore_index=True)

# add a column to indicate which model made the inference
allresults['modelname'] = get_names(allresults)

# get the counts of how many inferences were made by each model
allresults.modelname.value_counts()
```

    ccfraud-model-keras      12
    ccfraud-model-xgboost     8
    Name: modelname, dtype: int64

## Shadow Deployments

Another way to vet your new model is to set it up in a shadow deployment. With shadow deployments, all the models in the experiment pipeline get all the data, and all inferences are recorded. However, the pipeline returns only one "official" prediction: the one from default, or champion model.

Shadow deployments are useful for "sanity checking" a model before it goes truly live. For example, you might have built a smaller, leaner version of an existing model using knowledge distillation or other model optimization techniques, as discussed [here](https://wallaroo.ai/how-to-accelerate-computer-vision-model-inference/). A shadow deployment of the new model alongside the original model can help ensure that the new model meets desired accuracy and performance requirements before it's put into production.

As an example of creating a shadow deployment, suppose you have a champion model called "champion", that you have been running in a one-step pipeline called "pipeline". You now want to put a challenger model called "challenger" into a shadow deployment with the champion. Both models have already been uploaded. 

Shadow deployments can be **added** as a pipeline step, or **replace** an existing pipeline step.

Shadow deployment steps are added with the `add_shadow_deploy(champion, [model2, model3,...])` method, where the `champion` is the model that the inference results will be returned.  The array of models listed after are the models where inference data is also submitted with their results displayed as as shadow inference results.

Shadow deployment steps replace an existing pipeline step with the  `replace_with_shadow_deploy(index, champion, [model2, model3,...])` method.  The `index` is the step being replaced with pipeline steps starting at 0, and the `champion` is the model that the inference results will be returned.  The array of models listed after are the models where inference data is also submitted with their results displayed as as shadow inference results.

Then creating a shadow deployment from a previously created (and deployed) pipeline could look something like this:

```python
# retrieve handles to the most recent versions 
# of the champion and challenger models
# see the A/B test section for the definition of get_model()
champion = get_model("champion")
challenger = get_model("challenger")

# get the existing pipeline and undeploy it
# see the A/B test section for the definition of get_pipeline()
pipeline = get_pipeline("pipeline")
pipeline.undeploy()

# clear the pipeline and add a shadow deploy step
pipeline.clear()
pipeline.add_shadow_deploy(champion, [challenger])
pipeline.deploy()
```

The above code clears the pipeline and adds a shadow deployment. The pipeline will still only return the inferences from the champion model, but it will also run the challenger model in parallel and log the inferences, so that you can compare what all the models do on the same inputs.

You can add multiple challengers to a shadow deploy:

```python
pipeline.add_shadow_deploy(champion, [challenger01, challenger02])
```

You can also create a shadow deployment from scratch with a new pipeline.  This example just uses two models - one champion, one challenger.

```
newpipeline = wl.build_pipeline("pipeline")
newpipeline.add_shadow_deploy(champion, [challenger])
```

<hr/>

#### Exercise: Create a house price model shadow deployment

Use the champion and challenger models that you created in the previous exercises to create a shadow deployment. You can either create one from scratch, or reconfigure an existing pipeline.

At the end of this exercise, you should have a shadow deployment running multiple models in parallel.

For more information, see [Pipeline Shadow Deployments](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/#pipeline-shadow-deployments)

```python
# blank space to create a shadow deployment

pipeline.clear()

pipeline.add_shadow_deploy(ccfraud_keras_model, [ccfraud_xgboost_model])

pipeline.deploy()
```

<table><tr><th>name</th> <td>finserv-ccfraud</td></tr><tr><th>created</th> <td>2023-08-07 16:26:37.485326+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-07 17:29:08.595377+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>8ff48a62-fd9d-43f5-ba3f-11a7f1bcc474, 36faf126-dac5-419d-b0b1-7d7b698b587e, 230d585a-52db-476d-ab28-7b4baed9d023, 192f92e9-9a97-4339-8c1d-f89541ff2cef, 5d2d9c84-13c2-4e35-a41f-ec3c4e8d297b, 41927bef-d8fb-49ee-914e-d106ffc304b3</td></tr><tr><th>steps</th> <td>ccfraud-model-keras</td></tr></table>

Since a shadow deployment returns multiple predictions for a single datum, its inference result will look a little different from those of an A/B test or a single-step pipelne. The next exercise will show you how to examine all the inferences from all the models.

<hr/>

#### Exercise: Examine shadow deployment inferences

Use the test data that you created in a previous exercise to send a single datum to the shadow deployment that you created in the previous exercise.

* Observe the inference result
* You should see a column called `out.<outputname>`; this is the prediction from the champion model. It is the "official" prediction from the pipeline. If you used the same champion model in the A/B test exercise above, and in the single-step pipeline from the previous notebook, you should see the inference results from all those pipelines was also called `out.<outputname>`.
* You should also see a column called `out_<challengermodel>.<outputname>` (or more than one, if you had multiple challengers). These are the predictions from the challenger models. 

For example, if your champion model is called "champion", your challenger model is called "challenger", and the outputname is "output",
then you should see the "official" prediction `out.output` and the shadow prediction `out_challenger.output`.

**Save the datum and the inference result from this exercise.** You will need it for the next exercise.

```python
# blank space to send an inference and examine the result

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
      <th>out_ccfraud-model-xgboost.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-07 17:29:16.803</td>
      <td>[[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]]</td>
      <td>[0.99300325]</td>
      <td>0</td>
      <td>[1.0094898]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-08-07 17:29:16.803</td>
      <td>[[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]]</td>
      <td>[0.99300325]</td>
      <td>0</td>
      <td>[1.0094898]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-08-07 17:29:16.803</td>
      <td>[[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]]</td>
      <td>[0.99300325]</td>
      <td>0</td>
      <td>[1.0094898]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-08-07 17:29:16.803</td>
      <td>[[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]]</td>
      <td>[0.99300325]</td>
      <td>0</td>
      <td>[1.0094898]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-08-07 17:29:16.803</td>
      <td>[[0.5817662108, 0.097881551, 0.1546819424, 0.4754101949, -0.1978862306, -0.4504344854, 0.0166540447, -0.0256070551, 0.0920561602, -0.2783917153, 0.0593299441, -0.0196585416, -0.4225083157, -0.1217538877, 1.5473094894, 0.2391622864, 0.3553974881, -0.7685165301, -0.7000849355, -0.1190043285, -0.3450517133, -1.1065114108, 0.2523411195, 0.0209441826, 0.2199267436, 0.2540689265, -0.0450225094, 0.1086773898, 0.2547179311]]</td>
      <td>[0.0010916889]</td>
      <td>0</td>
      <td>[-1.9073486e-06]</td>
    </tr>
  </tbody>
</table>

## After the Experiment: Swapping in New Models

You have seen two methods to validate models in production with test (challenger) models. 
The end result of an experiment is a decision about which model becomes the new champion. Let's say that you have been running the shadow deployment that you created in the previous exercise,  and you have decided that you want to replace the model "champion" with the model "challenger". To do this, you will clear all the steps out of the pipeline, and add only "challenger" back in.

```python
# retrieve a handle to the challenger model
# see the A/B test section for the definition of get_model()
challenger = get_model("challenger")

# get the existing pipeline and undeploy it
# see the A/B test section for the definition of get_pipeline()
pipeline = get_pipeline("pipeline")
pipeline.undeploy()

# clear out all the steps and add the champion back in 
pipeline.clear() 
pipeline.add_model_step(challenger).deploy()
```

<hr/> 

#### Exercise: Set a challenger model as the new active model

Pick one of your challenger models as the new champion, and reconfigure your shadow deployment back into a single-step pipeline with the new chosen model.

* Run the test datum from the previous exercise through the reconfigured pipeline.
* Compare the results to the results from the previous exercise.
* Notice that the pipeline predictions are different from the old champion, and consistent with the new one.

At the end of this exercise, you should have a single step pipeline, running a new model.

More information is available through [Replace a Pipeline Step](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/#replace-a-pipeline-step).

```python
# Blank space - remove all steps, then redeploy with new champion model

pipeline.clear()

pipeline.add_model_step(ccfraud_xgboost_model)

pipeline.deploy()

singleton = get_singleton(df, 0)
display(singleton)

display(pipeline.steps())

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

    [{'ModelInference': {'models': [{'name': 'ccfraud-model-xgboost', 'version': 'e71c9c1d-9777-4967-b32c-a074ea8aa467', 'sha': '054810e3e3ebbdd34438d9c1a08ed6a6680ef10bf97b9223f78ebf38e14b3b52'}]}}]

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
      <td>2023-08-07 17:30:17.358</td>
      <td>[[-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526, 1.9870535692, 0.7005485718, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756, -0.1466244739, -1.4463212439]]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

## Congratulations!

You have now 
* successfully trained new challenger models for the house price prediction problem
* compared your models using an A/B test
* compared your models using a shadow deployment
* replaced your old model for a new one in the house price prediction pipeline

In the next notebook, you will learn how to monitor your production pipeline for "anomalous" or out-of-range behavior.

### Cleaning up.

At this point, if you are not continuing on to the next notebook, undeploy your pipeline(s) to give the resources back to the environment.

```python
## blank space to undeploy the pipelines

pipeline.undeploy()
```

<table><tr><th>name</th> <td>finserv-ccfraud</td></tr><tr><th>created</th> <td>2023-08-07 16:26:37.485326+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-07 17:30:15.093178+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>2513faec-2385-4733-a42d-56a8ae38761a, 10e495af-9b23-4973-913e-4cdebca73461, 782b9edc-d42a-4984-90ca-3bdf35922b87, e80d21a4-21c2-46b8-85d4-6652bbac9506, 8ff48a62-fd9d-43f5-ba3f-11a7f1bcc474, 36faf126-dac5-419d-b0b1-7d7b698b587e, 230d585a-52db-476d-ab28-7b4baed9d023, 192f92e9-9a97-4339-8c1d-f89541ff2cef, 5d2d9c84-13c2-4e35-a41f-ec3c4e8d297b, 41927bef-d8fb-49ee-914e-d106ffc304b3</td></tr><tr><th>steps</th> <td>ccfraud-model-keras</td></tr></table>

