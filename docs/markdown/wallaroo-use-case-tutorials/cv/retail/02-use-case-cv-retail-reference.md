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

import pyarrow as pa

import sys
 
# setting path - only needed when running this from the `with-code` folder.
sys.path.append('../')

from CVDemoUtils import CVDemo
cvDemo = CVDemo()
cvDemo.COCO_CLASSES_PATH = "../models/coco_classes.pickle"
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

# Translated a column from a dataframe into a single array
# used for the Statsmodel forecast model

def get_singleton_forecast(df, field):
    singleton = pd.DataFrame({field: [df[field].values.tolist()]})
    return singleton

```

#### Pre-exercise

If needed, log into Wallaroo and go to the workspace that you created in the previous notebook. Please refer to Notebook 1 to refresh yourself on how to log in and set your working environment to the appropriate workspace.

```python
## blank space to log in and go to the appropriate workspace

wl = wallaroo.Client()

import string
import random

workspace_name = f'computer-vision-tutorial'

workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)
```

    {'name': 'computer-vision-tutorialjohn', 'id': 20, 'archived': False, 'created_by': '0a36fba2-ad42-441b-9a8c-bac8c68d13fa', 'created_at': '2023-08-04T19:16:04.283819+00:00', 'models': [{'name': 'resnet', 'versions': 2, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 5, 17, 25, 35, 579362, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 5, 17, 20, 21, 886290, tzinfo=tzutc())}, {'name': 'mobilenet', 'versions': 5, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 5, 18, 2, 2, 812811, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 4, 19, 19, 46, 286247, tzinfo=tzutc())}, {'name': 'cv-post-process-drift-detection', 'versions': 4, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 5, 18, 2, 4, 862817, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 4, 19, 23, 10, 189958, tzinfo=tzutc())}], 'pipelines': [{'name': 'cv-retail-pipeline', 'create_time': datetime.datetime(2023, 8, 4, 19, 23, 11, 179176, tzinfo=tzutc()), 'definition': '[]'}]}

## A/B Testing

An [A/B test](https://en.wikipedia.org/wiki/A/B_testing), also called a controlled experiment or a randomized control trial, is a statistical method of determining which of a set of variants is the best. A/B tests allow organizations and policy-makers to make smarter, data-driven decisions that are less dependent on guesswork.

In the simplest version of an A/B test, subjects are randomly assigned to either the **_control group_** (group A) or the **_treatment group_** (group B). Subjects in the treatment group receive the treatment (such as a new medicine, a special offer, or a new web page design) while the control group proceeds as normal without the treatment. Data is then collected on the outcomes and used to study the effects of the treatment.

In data science, A/B tests are often used to choose between two or more candidate models in production, by measuring which model performs best in the real world. In this formulation, the control is often an existing model that is currently in production, sometimes called the **_champion_**. The treatment is a new model being considered to replace the old one. This new model is sometimes called the **_challenger_**. In our discussion, we'll use the terms *champion* and *challenger*, rather than *control* and *treatment*.

When data is sent to a Wallaroo A/B test pipeline for inference, each datum is randomly sent to either the champion or challenger. After enough data has been sent to collect statistics on all the models in the A/B test pipeline, then those outcomes can be analyzed to determine the difference (if any) in the performance of the champion and challenger. Usually, the purpose of an A/B test is to decide whether or not to replace the champion with the challenger.

Keep in mind that in machine learning, the terms experiments and trials also often refer to the process of finding a training configuration that works best for the problem at hand (this is sometimes called hyperparameter optimization). In this guide, we will use the term experiment to refer to the use of A/B tests to compare the performance of different models in production.
<hr/>

#### Exercise: Create some challenger models and upload them to Wallaroo

Use the computer vision data from Notebook 1 to create at least one alternate computer vision model. You can do this by varying the modeling algorithm, the inputs, the feature engineering, or all of the above. 

For the purpose of these exercises, please make sure that the predictions from the new model(s) are in the same units as the (champion) model that you created in Chapter 3. For example, if the champion model predicts log price, then the challenger models should also predict log price. If the champion model predicts price in units of $10,000, then the challenger models should, also.

* **If you prefer to shortcut this step, you can use some of the pretrained model Python model files in the `models` directory**
  * If the Python models are used, ensure that the proper input and output schemas are set.  See the N1_deploy_a_model notebook for instructions.
* Upload your new model(s) to Wallaroo, into your workspace

At the end of this exercise, you should have at least one challenger model to compare to your champion model uploaded to your workspace.

```python
# blank space to train, convert, and upload new model

resnet_model_name = 'resnet'
resnet_model_path = "../models/frcnn-resnet.pt.onnx"

resnet_model = wl.upload_model(resnet_model_name, 
                               resnet_model_path, 
                               framework=Framework.ONNX).configure('onnx', 
                                                                      batch_config="single"
                                                                      )

```

There are a number of considerations to designing an A/B test; you can check out the article [*The What, Why, and How of A/B Testing*](https://wallarooai.medium.com/the-what-why-and-how-of-a-b-testing-64471847cd7e) for more details. In these exercises, we will concentrate on the deployment aspects.  You will need a champion model and  at least one challenger model. You also need to decide on a data split: for example 50-50 between the champion and challenger, or a 2:1 ratio between champion and challenger (two-thirds of the data to the champion, one-third to the challenger).

As an example of creating an A/B test deployment, suppose you have a champion model called "champion", that you have been running in a one-step pipeline called "pipeline". You now want to compare it to a challenger model called "challenger". For your A/B test, you will send two-thirds of the data to the champion, and the other third to the challenger. Both models have already been uploaded. 

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

mobilenet_model_name = 'mobilenet'

module_post_process_name = "cv-post-process-drift-detection"

mobilenet_model = get_model(mobilenet_model_name)

module_post_process_model = get_model(module_post_process_name)

pipeline_name = 'cv-retail-pipeline'

pipeline = get_pipeline(pipeline_name)
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

Reference:  [Wallaroo SDK Essentials Guide: Pipeline Management A/B Testing](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipeline/#ab-testing).

Then creating an A/B test deployment would look something like this:

First get the models used.

```python
# retrieve handles to the most recent versions 
# of the champion and challenger models
champion = get_model("champion")
challenger = get_model("challenger")

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
# add in the post-processing step as a normal step
pipeline.add_model_step(module_post_process)
pipeline.deploy()
```

The above code clears out all the steps of the pipeline and adds a new step with a A/B test deployment, where the incoming data is randomly sent in a 2:1 ratio to the champion and the challenger, respectively.

You can add multiple challengers to an A/B test::

```python
pipeline.add_random_split([ (2, champion), (1, challenger01), (1, challenger02) ])
```

This pipeline will distribute data in the ratio 2:1:1 (or half to the champion, a quarter each to the challlengers) to the champion and challenger models, respectively.

You can also create an A/B test deployment from scratch:

```python
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

# blank space to get the model(s)
pipeline.undeploy()
pipeline.clear()
pipeline.add_random_split([(1, mobilenet_model), (1, resnet_model)])
pipeline.deploy()

```

<table><tr><th>name</th> <td>cv-retail-pipeline</td></tr><tr><th>created</th> <td>2023-08-04 19:23:11.179176+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-14 14:34:08.984836+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>cfe4f208-f01e-4fe2-a3f4-2da0b42b3547, 5bc400fb-4ea7-4e84-80a7-8c60794c3575, 6500b4d3-4031-4b0c-b219-4af552e3100c, db77cdac-d02f-4f85-a7fa-35833a480eff, fdee0a8f-e540-4c48-bd14-628a3417f24f, 5a59498b-f2f2-4254-9ca4-5c19460bb42a, e6be0eb9-f387-471a-8e6b-4b5d845e8aec, 4e74bc78-3501-4135-a3ff-8e24a9132d0f, a719a198-bb04-462e-bd6f-85644c357e62, 5f280423-8ac1-45b7-b645-27b15e0bd7d4, 9eb22dbc-c035-4ac4-bba9-b7cd3a9f30ba, 5ce99fc6-4463-4ab0-abbe-8b490ce9fc29, 8faa0d21-11ed-4186-8f5d-a586ead7ab00, 305db319-db20-4be8-94a7-ecb3d8bee4d4, 15cc7825-03a1-4794-8a31-744d290db193</td></tr><tr><th>steps</th> <td>mobilenet</td></tr></table>

The pipeline steps are displayed with the Pipeline `steps()` method.  This is used to verify the current **deployed** steps in the pipeline.

* **IMPORTANT NOTE**: Verify that the pipeline is deployed before checking for pipeline steps.  Deploying the pipeline sets the steps into the Wallaroo system - until that happens, the steps only exist in the local system as *potential* steps.

```python
# blank space to get the current pipeline steps
pipeline.steps()
```

    [{'RandomSplit': {'hash_key': None, 'weights': [{'model': {'name': 'mobilenet', 'version': '484fffe8-70fe-44b9-937f-e98838bcc245', 'sha': '9044c970ee061cc47e0c77e20b05e884be37f2a20aa9c0c3ce1993dbd486a830'}, 'weight': 1}, {'model': {'name': 'resnet', 'version': '5aaf7fbc-81aa-40ad-b784-721f203d9532', 'sha': '43326e50af639105c81372346fb9ddf453fea0fe46648b2053c375360d9c1647'}, 'weight': 1}]}}]

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

##  blank space to create test data, and send some data to your model

image = '../data/images/input/example/dairy_bottles.png'

width, height = 640, 480
dfImage, resizedImage = cvDemo.loadImageAndConvertToDataframe(image, 
                                                              width, 
                                                              height
                                                              )

for i in range(10):
    results = pipeline.infer(dfImage, timeout=60)
    # display(results)
    display(get_names(results))

```

    ['resnet']

    ['resnet']

    ['resnet']

    ['mobilenet']

    ['mobilenet']

    ['resnet']

    ['resnet']

    ['resnet']

    ['mobilenet']

    ['mobilenet']

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
# add in the post-processing step as a normal step
pipeline.add_model_step(module_post_process)
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
pipeline.add_shadow_deploy(mobilenet_model, [resnet_model])
pipeline.deploy()

```

<table><tr><th>name</th> <td>cv-retail-pipeline</td></tr><tr><th>created</th> <td>2023-08-04 19:23:11.179176+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-14 14:35:52.266795+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>acc7d8bc-3b43-48ba-b771-d7e97ad3e79d, cfe4f208-f01e-4fe2-a3f4-2da0b42b3547, 5bc400fb-4ea7-4e84-80a7-8c60794c3575, 6500b4d3-4031-4b0c-b219-4af552e3100c, db77cdac-d02f-4f85-a7fa-35833a480eff, fdee0a8f-e540-4c48-bd14-628a3417f24f, 5a59498b-f2f2-4254-9ca4-5c19460bb42a, e6be0eb9-f387-471a-8e6b-4b5d845e8aec, 4e74bc78-3501-4135-a3ff-8e24a9132d0f, a719a198-bb04-462e-bd6f-85644c357e62, 5f280423-8ac1-45b7-b645-27b15e0bd7d4, 9eb22dbc-c035-4ac4-bba9-b7cd3a9f30ba, 5ce99fc6-4463-4ab0-abbe-8b490ce9fc29, 8faa0d21-11ed-4186-8f5d-a586ead7ab00, 305db319-db20-4be8-94a7-ecb3d8bee4d4, 15cc7825-03a1-4794-8a31-744d290db193</td></tr><tr><th>steps</th> <td>mobilenet</td></tr></table>

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

results = pipeline.infer(dfImage, timeout=60)
display(results.filter(regex='time|out.*'))
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.boxes</th>
      <th>out.classes</th>
      <th>out.confidences</th>
      <th>out_resnet.boxes</th>
      <th>out_resnet.classes</th>
      <th>out_resnet.confidences</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-14 14:36:05.202</td>
      <td>[0.0, 210.2901, 85.26463, 479.07495, 72.03781, 197.3227, 151.44223, 468.43225, 211.28015, 184.72838, 277.2192, 420.42746, 143.23904, 203.83005, 216.85547, 448.8881, 13.095016, 41.91339, 640.0, 480.0, 106.51464, 206.14499, 159.54643, 463.96756, 278.0637, 1.5217237, 321.45782, 93.563484, 462.31833, 104.16202, 510.5396, 224.75314, 310.4559, 1.3959818, 352.8513, 94.123825, 528.0485, 268.4225, 636.26715, 475.7666, 220.06293, 0.51385117, 258.31833, 90.18019, 552.87115, 96.30235, 600.7256, 233.53384, 349.24072, 0.27034378, 404.17325, 98.68022, 450.89346, 264.2356, 619.6033, 472.65173, 261.51385, 193.4335, 307.17914, 408.75247, 509.22018, 101.16539, 544.1857, 235.7374, 592.4824, 100.38687, 633.77985, 239.13432, 475.54208, 297.6141, 551.0544, 468.01547, 368.81982, 163.61407, 423.90942, 362.7888, 120.669, 0.0, 175.9362, 81.774574, 72.48429, 0.0, 143.50789, 85.4698, 271.12686, 200.89185, 305.626, 274.59537, 161.80728, 0.0, 213.08308, 85.42828, 162.13324, 0.0, 214.60814, 83.81444, 310.89108, 190.95468, 367.32925, 397.28137, ...]</td>
      <td>[44, 44, 44, 44, 82, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 84, 84, 44, 84, 44, 44, 44, 61, 44, 86, 44, 44]</td>
      <td>[0.98649, 0.90115356, 0.6077846, 0.5922323, 0.53729033, 0.4513168, 0.43728516, 0.43094054, 0.4084834, 0.39185277, 0.35759133, 0.3181266, 0.26451287, 0.23062895, 0.20482065, 0.174621, 0.17313862, 0.15999581, 0.14913696, 0.1366402, 0.13322707, 0.12218643, 0.121301256, 0.11956108, 0.11527827, 0.09616333, 0.08654833, 0.078406945, 0.07234089, 0.062820904, 0.052787986]</td>
      <td>[2.1511102, 193.98323, 76.26535, 475.40292, 610.82245, 98.606316, 639.8868, 232.27054, 544.2867, 98.726524, 581.28845, 230.20497, 454.99344, 113.08567, 484.78464, 210.1282, 502.58884, 331.87665, 551.2269, 476.49182, 538.54254, 292.1205, 587.46545, 468.1288, 578.5417, 99.70756, 617.2247, 233.57082, 548.552, 191.84564, 577.30585, 238.47737, 459.83328, 344.29712, 505.42633, 456.7118, 483.47168, 110.56585, 514.0936, 205.00156, 262.1222, 190.36658, 323.4903, 405.20584, 511.6675, 104.53834, 547.01715, 228.23663, 75.39197, 205.62312, 168.49893, 453.44086, 362.50656, 173.16858, 398.66956, 371.8243, 490.42468, 337.627, 534.1234, 461.0242, 351.3856, 169.14897, 390.7583, 244.06992, 525.19824, 291.73895, 570.5553, 417.6439, 563.4224, 285.3889, 609.30853, 452.25943, 425.57935, 366.24915, 480.63535, 474.54, 154.538, 198.0377, 227.64284, 439.84412, 597.02893, 273.60458, 637.2067, 439.03214, 473.88763, 293.41992, 519.7537, 349.2304, 262.77597, 192.03581, 313.3096, 258.3466, 521.1493, 152.89026, 534.8596, 246.52365, 389.89633, 178.07867, 431.87555, 360.59323, ...]</td>
      <td>[44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 86, 82, 44, 44, 44, 44, 44, 44, 84, 84, 44, 44, 44, 44, 86, 84, 44, 44, 44, 44, 44, 84, 44, 44, 84, 44, 44, 44, 44, 51, 44, 44, 44, 44, 44, 44, 44, 44, 44, 82, 44, 44, 44, 44, 44, 86, 44, 44, 1, 84, 44, 44, 44, 44, 84, 47, 47, 84, 14, 44, 44, 53, 84, 47, 47, 44, 84, 44, 44, 82, 44, 44, 44]</td>
      <td>[0.9965358, 0.9883404, 0.9700247, 0.9696426, 0.96478045, 0.96037567, 0.9542889, 0.9467539, 0.946524, 0.94484967, 0.93611854, 0.91653466, 0.9133634, 0.8874814, 0.84405905, 0.825526, 0.82326967, 0.81740034, 0.7956525, 0.78669065, 0.7731486, 0.75193685, 0.7360918, 0.7009186, 0.6932351, 0.65077204, 0.63243586, 0.57877576, 0.5023476, 0.50163734, 0.44628552, 0.42804396, 0.4253787, 0.39086252, 0.36836442, 0.3473236, 0.32950658, 0.3105372, 0.29076362, 0.28558296, 0.26680034, 0.26302803, 0.25444376, 0.24568668, 0.2353662, 0.23321979, 0.22612995, 0.22483191, 0.22332378, 0.21442991, 0.20122288, 0.19754867, 0.19439234, 0.19083925, 0.1871393, 0.17646024, 0.16628945, 0.16326219, 0.14825206, 0.13694529, 0.12920643, 0.12815322, 0.122357555, 0.121289656, 0.116281696, 0.11498632, 0.111848116, 0.11016138, 0.1095062, 0.1039151, 0.10385688, 0.097573474, 0.09632071, 0.09557622, 0.091599144, 0.09062039, 0.08262358, 0.08223499, 0.07993951, 0.07989185, 0.078758545, 0.078201495, 0.07737936, 0.07690251, 0.07593444, 0.07503418, 0.07482597, 0.068981, 0.06841128, 0.06764157, 0.065750405, 0.064908616, 0.061884128, 0.06010121, 0.0578873, 0.05717648, 0.056616478, 0.056017116, 0.05458274, 0.053669468]</td>
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

pipeline.undeploy()
pipeline.clear()
pipeline.add_model_step(mobilenet_model)
pipeline.add_model_step(module_post_process_model)
pipeline.deploy()

# regular inference
results = pipeline.infer(dfImage, timeout=60)
display(results.loc[:, ['out.avg_conf']])

# swap the model

pipeline.replace_with_model_step(0, resnet_model)
pipeline.deploy()

# gives time for the update to happen - usually milliseconds, sometimes longer.  
# This gives enough time for the database updates to happen
import time
time.sleep(15)

display(pipeline.steps())

# regular inference
results = pipeline.infer(dfImage, timeout=60)
display(results.loc[:, ['out.avg_conf']])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out.avg_conf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0.2895053208114637]</td>
    </tr>
  </tbody>
</table>

    [{'ModelInference': {'models': [{'name': 'resnet', 'version': '5aaf7fbc-81aa-40ad-b784-721f203d9532', 'sha': '43326e50af639105c81372346fb9ddf453fea0fe46648b2053c375360d9c1647'}]}},
     {'ModelInference': {'models': [{'name': 'cv-post-process-drift-detection', 'version': '3ae7dfc2-1b3c-44d3-9957-97c5704e3592', 'sha': 'f60c8ca55c6350d23a4e76d24cc3e5922616090686e88c875fadd6e79c403be5'}]}}]

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out.avg_conf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0.3588037962093945]</td>
    </tr>
  </tbody>
</table>

## Congratulations!

You have now 
* successfully trained new challenger models for the computer vision problem
* compared your models using an A/B test
* compared your models using a shadow deployment
* replaced your old model for a new one in the computer vision pipeline

In the next notebook, you will learn how to monitor your production pipeline for "anomalous" or out-of-range behavior.

### Cleaning up.

At this point, if you are not continuing on to the next notebook, undeploy your pipeline(s) to give the resources back to the environment.

```python
## blank space to undeploy the pipelines
pipeline.undeploy()
```

<table><tr><th>name</th> <td>cv-retail-pipeline</td></tr><tr><th>created</th> <td>2023-08-04 19:23:11.179176+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-14 14:40:11.091157+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>9afba3df-25ea-41eb-b0b5-5b0ab5266ed6, 47a96466-a53e-4a7c-92c5-56f1f5c44e31, eb8d202f-7193-4124-a8a4-d52ca355f1e2, 32ed1c76-07b2-4472-ac4d-929687f2710b, acc7d8bc-3b43-48ba-b771-d7e97ad3e79d, cfe4f208-f01e-4fe2-a3f4-2da0b42b3547, 5bc400fb-4ea7-4e84-80a7-8c60794c3575, 6500b4d3-4031-4b0c-b219-4af552e3100c, db77cdac-d02f-4f85-a7fa-35833a480eff, fdee0a8f-e540-4c48-bd14-628a3417f24f, 5a59498b-f2f2-4254-9ca4-5c19460bb42a, e6be0eb9-f387-471a-8e6b-4b5d845e8aec, 4e74bc78-3501-4135-a3ff-8e24a9132d0f, a719a198-bb04-462e-bd6f-85644c357e62, 5f280423-8ac1-45b7-b645-27b15e0bd7d4, 9eb22dbc-c035-4ac4-bba9-b7cd3a9f30ba, 5ce99fc6-4463-4ab0-abbe-8b490ce9fc29, 8faa0d21-11ed-4186-8f5d-a586ead7ab00, 305db319-db20-4be8-94a7-ecb3d8bee4d4, 15cc7825-03a1-4794-8a31-744d290db193</td></tr><tr><th>steps</th> <td>mobilenet</td></tr></table>

