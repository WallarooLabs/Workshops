# Tutorial Notebook 2: Vetting a Model With Production Experiments

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

workspace_name = "tutorial-workspace-jch"

workspace = get_workspace(workspace_name)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

# optionally, examine your current workspace
wl.get_current_workspace()
```

    {'name': 'tutorial-workspace-jch', 'id': 9, 'archived': False, 'created_by': 'c3a45eb6-37ff-4020-8d59-7166c3e153d0', 'created_at': '2023-07-20T14:35:12.650973+00:00', 'models': [{'name': 'tutorial-model', 'versions': 3, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 7, 20, 16, 3, 10, 617651, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 7, 20, 14, 43, 43, 886638, tzinfo=tzutc())}, {'name': 'challenger-model', 'versions': 2, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 7, 20, 16, 14, 50, 450715, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 7, 20, 16, 14, 1, 624750, tzinfo=tzutc())}], 'pipelines': [{'name': 'tutorialpipeline-jch', 'create_time': datetime.datetime(2023, 7, 20, 14, 43, 44, 606725, tzinfo=tzutc()), 'definition': '[]'}]}

## A/B Testing

An [A/B test](https://en.wikipedia.org/wiki/A/B_testing), also called a controlled experiment or a randomized control trial, is a statistical method of determining which of a set of variants is the best. A/B tests allow organizations and policy-makers to make smarter, data-driven decisions that are less dependent on guesswork.

In the simplest version of an A/B test, subjects are randomly assigned to either the **_control group_** (group A) or the **_treatment group_** (group B). Subjects in the treatment group receive the treatment (such as a new medicine, a special offer, or a new web page design) while the control group proceeds as normal without the treatment. Data is then collected on the outcomes and used to study the effects of the treatment.

In data science, A/B tests are often used to choose between two or more candidate models in production, by measuring which model performs best in the real world. In this formulation, the control is often an existing model that is currently in production, sometimes called the **_champion_**. The treatment is a new model being considered to replace the old one. This new model is sometimes called the **_challenger_**. In our discussion, we'll use the terms *champion* and *challenger*, rather than *control* and *treatment*.

When data is sent to a Wallaroo A/B test pipeline for inference, each datum is randomly sent to either the champion or challenger. After enough data has been sent to collect statistics on all the models in the A/B test pipeline, then those outcomes can be analyzed to determine the difference (if any) in the performance of the champion and challenger. Usually, the purpose of an A/B test is to decide whether or not to replace the champion with the challenger.

Keep in mind that in machine learning, the terms experiments and trials also often refer to the process of finding a training configuration that works best for the problem at hand (this is sometimes called hyperparameter optimization). In this guide, we will use the term experiment to refer to the use of A/B tests to compare the performance of different models in production.
<hr/>

#### Exercise: Create some house price challenger models and upload them to Wallaroo

Use the house price data from Notebook 1 to create at least one alternate house price prediction model. You can do this by varying the modeling algorithm, the inputs, the feature engineering, or all of the above. 

For the purpose of these exercises, please make sure that the predictions from the new model(s) are in the same units as the (champion) model that you created in Chapter 3. For example, if the champion model predicts log price, then the challenger models should also predict log price. If the champion model predicts price in units of $10,000, then the challenger models should, also.

* **If you prefer to shortcut this step, you can use some of the pretrained model onnx files in the `models` directory**
* Upload your new model(s) to Wallaroo, into your houseprice workspace

At the end of this exercise, you should have at least one challenger model to compare to your champion model uploaded to your workspace.

```python
# blank space to train, convert, and upload new model

```

There are a number of considerations to designing an A/B test; you can check out the article [*The What, Why, and How of A/B Testing*](https://wallarooai.medium.com/the-what-why-and-how-of-a-b-testing-64471847cd7e) for more details. In these exercises, we will concentrate on the deployment aspects.  You will need a champion model and  at least one challenger model. You also need to decide on a data split: for example 50-50 between the champion and challenger, or a 2:1 ratio between champion and challenger (two-thirds of the data to the champion, one-third to the challenger).

As an example of creating an A/B test deployment, suppose you have a champion model called "champion", that you have been running in a one-step pipeline called "pipeline". You now want to compare it to a challenger model called "challenger". For your A/B test, you will send two-thirds of the data to the champion, and the other third to the challenger. Both models have already been uploaded. 

To help you with the exercises, here some convenience functions to retrieve a models and pipelines that have been previously uploaded to your workspace (in this example, `wl` is your `wallaroo.client()` object). 

```python
# Get the most recent version of a model.
# Assumes that the most recent version is the first in the list of versions.
# wl.get_current_workspace().models() returns a list of models in the current workspace

def get_model(mname, modellist=wl.get_current_workspace().models()):
    model = [m.versions()[0] for m in modellist if m.name() == mname]
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

pipeline = get_pipeline('tutorialpipeline-jch')

challenger_model = wl.upload_model('challenger-model', './models/rf_model.onnx', framework=Framework.ONNX)

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

control_model = get_model('tutorial-model')

challenger_model = get_model('challenger-model')
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
pipeline.add_random_split([(2, control_model), (1, challenger_model)])
pipeline.deploy()

```

<table><tr><th>name</th> <td>tutorialpipeline-jch</td></tr><tr><th>created</th> <td>2023-07-20 14:43:44.606725+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-20 16:35:31.296928+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>f27623ad-90dc-4161-b826-65084d0f198d, d886d504-d7fd-42b7-9b95-7eadfa6cafbe, 117dac7b-78fe-449d-99c7-6233c100874c, 67f9bfe6-7f03-4087-a6db-46896729c78f, aa7f65e4-0edd-4cb7-8a81-7c5f46c323a7, 4edac46e-22eb-46c0-8ee7-e2cf68181958, ed2b8ae9-eb77-46b5-9985-aa520be6c6b7, 83b4129b-3271-4e5a-aa5c-203b8b95e674, d7c7b6b1-04d5-4d20-a85b-39159cb79ede, 0f5b2df6-6498-4ace-a739-7c585284d6c6, da128b7e-26f5-4b9d-a58e-b244735843bd, 89094eea-ef47-4d4c-bf33-2829b2aeaf64</td></tr><tr><th>steps</th> <td>tutorial-model</td></tr></table>

The pipeline steps are displayed with the Pipeline `steps()` method.  This is used to verify the current **deployed** steps in the pipeline.

* **IMPORTANT NOTE**: Verify that the pipeline is deployed before checking for pipeline steps.  Deploying the pipeline sets the steps into the Wallaroo system - until that happens, the steps only exist in the local system as *potential* steps.

```python
# blank space to get the current pipeline steps

pipeline.steps()
```

    [{'RandomSplit': {'hash_key': None, 'weights': [{'model': {'name': 'tutorial-model', 'version': '44f9e250-7636-4800-be08-da624b51d057', 'sha': 'ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a'}, 'weight': 2}, {'model': {'name': 'challenger-model', 'version': 'bd69c37d-8e8d-4cfa-8cf7-6f47a411c893', 'sha': 'e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6'}, 'weight': 1}]}}]

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

df_from_csv = pd.read_csv('./data/test_data.csv')

singleton = get_singleton(df_from_csv, 0)
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
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
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
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-07-20 16:35:46.180</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[{"name":"tutorial-model","version":"44f9e250-7636-4800-be08-da624b51d057","sha":"ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a"}]</td>
      <td>[704901.9]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

    ['tutorial-model']

```python
# blank space to send queries to A/B test pipeline and examine the results

results = []

# get a list of result frames
for i in range(20):
    query = get_singleton(df_from_csv, i)
    results.append(pipeline.infer(query))

# make one data frame of all results    
allresults = pd.concat(results, ignore_index=True)

# add a column to indicate which model made the inference
allresults['modelname'] = get_names(allresults)

# get the counts of how many inferences were made by each model
allresults.modelname.value_counts()
```

    tutorial-model      14
    challenger-model     6
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

```python
newpipeline = wl.build_pipeline("pipeline")
newpipeline.add_shadow_deploy(champion, [challenger])
```

<hr/>

#### Exercise: Create a house price model shadow deployment

Use the champion and challenger models that you created in the previous exercises to create a shadow deployment. You can either create one from scratch, or reconfigure an existing pipeline.

At the end of this exercise, you should have a shadow deployment running multiple models in parallel.

```python
# blank space to create a shadow deployment

pipeline.clear()

pipeline.add_shadow_deploy(control_model, [challenger_model])

pipeline.deploy()
```

<table><tr><th>name</th> <td>tutorialpipeline-jch</td></tr><tr><th>created</th> <td>2023-07-20 14:43:44.606725+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-20 16:35:55.285589+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>3da3ade7-52d5-4fe0-8609-027322769f39, f27623ad-90dc-4161-b826-65084d0f198d, d886d504-d7fd-42b7-9b95-7eadfa6cafbe, 117dac7b-78fe-449d-99c7-6233c100874c, 67f9bfe6-7f03-4087-a6db-46896729c78f, aa7f65e4-0edd-4cb7-8a81-7c5f46c323a7, 4edac46e-22eb-46c0-8ee7-e2cf68181958, ed2b8ae9-eb77-46b5-9985-aa520be6c6b7, 83b4129b-3271-4e5a-aa5c-203b8b95e674, d7c7b6b1-04d5-4d20-a85b-39159cb79ede, 0f5b2df6-6498-4ace-a739-7c585284d6c6, da128b7e-26f5-4b9d-a58e-b244735843bd, 89094eea-ef47-4d4c-bf33-2829b2aeaf64</td></tr><tr><th>steps</th> <td>tutorial-model</td></tr></table>

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
      <th>out_challenger-model.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-07-20 16:36:02.097</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[704901.9]</td>
      <td>0</td>
      <td>[718013.7]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-07-20 16:36:02.097</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[695994.44]</td>
      <td>0</td>
      <td>[615094.6]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-07-20 16:36:02.097</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[416164.8]</td>
      <td>0</td>
      <td>[448627.8]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-07-20 16:36:02.097</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[655277.2]</td>
      <td>0</td>
      <td>[758714.3]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-07-20 16:36:02.097</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[426854.66]</td>
      <td>0</td>
      <td>[513264.66]</td>
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

```python
# Blank space - remove all steps, then redeploy with new champion model

pipeline.clear()

pipeline.add_model_step(challenger_model)

pipeline.deploy()

singleton = get_singleton(df_from_csv, 0)
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
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
    </tr>
  </tbody>
</table>

    [{'ModelInference': {'models': [{'name': 'challenger-model', 'version': 'bd69c37d-8e8d-4cfa-8cf7-6f47a411c893', 'sha': 'e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6'}]}}]

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
      <td>2023-07-20 16:36:04.730</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
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

<table><tr><th>name</th> <td>tutorialpipeline-jch</td></tr><tr><th>created</th> <td>2023-07-20 14:43:44.606725+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-20 16:36:02.479007+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>acbbf366-711b-4904-b5d7-2459015f2d70, 3da3ade7-52d5-4fe0-8609-027322769f39, f27623ad-90dc-4161-b826-65084d0f198d, d886d504-d7fd-42b7-9b95-7eadfa6cafbe, 117dac7b-78fe-449d-99c7-6233c100874c, 67f9bfe6-7f03-4087-a6db-46896729c78f, aa7f65e4-0edd-4cb7-8a81-7c5f46c323a7, 4edac46e-22eb-46c0-8ee7-e2cf68181958, ed2b8ae9-eb77-46b5-9985-aa520be6c6b7, 83b4129b-3271-4e5a-aa5c-203b8b95e674, d7c7b6b1-04d5-4d20-a85b-39159cb79ede, 0f5b2df6-6498-4ace-a739-7c585284d6c6, da128b7e-26f5-4b9d-a58e-b244735843bd, 89094eea-ef47-4d4c-bf33-2829b2aeaf64</td></tr><tr><th>steps</th> <td>tutorial-model</td></tr></table>

