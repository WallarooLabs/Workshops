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

wallarooPrefix = "doc-test."
wallarooSuffix = "wallarooexample.ai"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}keycloak.{wallarooSuffix}", 
                    auth_type="sso")

import string
import random

suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
suffix='john'

workspace_name = f'forecast-model-tutorial{suffix}'

workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)
```

    Please log into the following URL in a web browser:
    
    	https://doc-test.keycloak.wallarooexample.ai/auth/realms/master/device?user_code=PDKG-QSIF
    
    Login successful!

    {'name': 'forecast-model-tutorialjohn', 'id': 16, 'archived': False, 'created_by': '0a36fba2-ad42-441b-9a8c-bac8c68d13fa', 'created_at': '2023-08-02T15:50:52.816795+00:00', 'models': [{'name': 'forecast-control-model', 'versions': 3, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 3, 1, 11, 50, 568151, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 2, 15, 50, 54, 223186, tzinfo=tzutc())}, {'name': 'forecast-challenger01-model', 'versions': 3, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 3, 13, 55, 23, 119224, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 2, 15, 50, 55, 208179, tzinfo=tzutc())}, {'name': 'forecast-challenger02-model', 'versions': 3, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 3, 13, 55, 24, 133756, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 2, 15, 50, 56, 291043, tzinfo=tzutc())}], 'pipelines': [{'name': 'forecast-tutorial-pipeline', 'create_time': datetime.datetime(2023, 8, 2, 15, 50, 59, 480547, tzinfo=tzutc()), 'definition': '[]'}]}

## A/B Testing

An [A/B test](https://en.wikipedia.org/wiki/A/B_testing), also called a controlled experiment or a randomized control trial, is a statistical method of determining which of a set of variants is the best. A/B tests allow organizations and policy-makers to make smarter, data-driven decisions that are less dependent on guesswork.

In the simplest version of an A/B test, subjects are randomly assigned to either the **_control group_** (group A) or the **_treatment group_** (group B). Subjects in the treatment group receive the treatment (such as a new medicine, a special offer, or a new web page design) while the control group proceeds as normal without the treatment. Data is then collected on the outcomes and used to study the effects of the treatment.

In data science, A/B tests are often used to choose between two or more candidate models in production, by measuring which model performs best in the real world. In this formulation, the control is often an existing model that is currently in production, sometimes called the **_champion_**. The treatment is a new model being considered to replace the old one. This new model is sometimes called the **_challenger_**. In our discussion, we'll use the terms *champion* and *challenger*, rather than *control* and *treatment*.

When data is sent to a Wallaroo A/B test pipeline for inference, each datum is randomly sent to either the champion or challenger. After enough data has been sent to collect statistics on all the models in the A/B test pipeline, then those outcomes can be analyzed to determine the difference (if any) in the performance of the champion and challenger. Usually, the purpose of an A/B test is to decide whether or not to replace the champion with the challenger.

Keep in mind that in machine learning, the terms experiments and trials also often refer to the process of finding a training configuration that works best for the problem at hand (this is sometimes called hyperparameter optimization). In this guide, we will use the term experiment to refer to the use of A/B tests to compare the performance of different models in production.
<hr/>

#### Exercise: Create some forecast models and upload them to Wallaroo

Use the forecast process from Notebook 1 to create at least one alternate forecast. You can do this by varying the modeling algorithm, the inputs, the feature engineering, or all of the above. 

For the purpose of these exercises, please make sure that the predictions from the new model(s) are in the same units as the (champion) model that you created in Chapter 3. For example, if the champion model predicts log price, then the challenger models should also predict log price. If the champion model predicts price in units of $10,000, then the challenger models should, also.

This applies to the Python step if using one similar to the python step provided from Notebook 1 - if the outputs of the new forecast model match the previous one, that post-processing python step can be applied to both models.

* **If you prefer to shortcut this step, you can use some of the pretrained model Python model files in the `models` directory**
  * If the Python models are used, ensure that the proper input and output schemas are set.  See the N1_deploy_a_model notebook for instructions.
* Upload your new model(s) to Wallaroo, into your forecast workspace

At the end of this exercise, you should have at least one challenger model to compare to your champion model uploaded to your workspace.

```python
challenger01_model_name = "forecast-challenger01-model"
challenger01_model_path = "../models/forecast_alternate01.py"

challenger02_model_name = "forecast-challenger02-model"
challenger02_model_path = "../models/forecast_alternate02.py"

# Holding on these for later
input_schema = pa.schema([
    pa.field('count', pa.list_(pa.int64()))
])

output_schema = pa.schema([
    pa.field('forecast', pa.list_(pa.int64())),
    pa.field('weekly_average', pa.list_(pa.float64()))
])

# upload the models

challenger01_model = (wl.upload_model(challenger01_model_name, 
                                 challenger01_model_path, 
                                 framework=Framework.PYTHON)
                                 .configure("python", 
                                 input_schema=input_schema, 
                                 output_schema=output_schema)
                )

challenger02_model = (wl.upload_model(challenger02_model_name, 
                                 challenger02_model_path, 
                                 framework=Framework.PYTHON)
                                 .configure("python", 
                                 input_schema=input_schema, 
                                 output_schema=output_schema)
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
# use the space here for retrieving the pipeline

pipeline_name = 'forecast-tutorial-pipeline'

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

Reference:  [Wallaroo SDK Essentials Guide: Pipeline Management A/B Testing
](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipeline/#ab-testing).

Then creating an A/B test deployment would look something like this:

First get the models used.

```python
# retrieve handles to the most recent versions 
# of the champion and challenger models
champion = get_model("champion")
challenger = get_model("challenger")

```

```python
# blank space to get the model(s)

control_model_name = 'forecast-control-model'

bike_day_model = get_model(control_model_name)
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

```python
pipeline.add_random_split([ (2, champion), (1, challenger01), (1, challenger02) ])
```

This pipeline will distribute data in the ratio 2:1:1 (or half to the champion, a quarter each to the challlengers) to the champion and challenger models, respectively.

You can also create an A/B test deployment from scratch:

```python
pipeline = wl.build_pipeline("pipeline")
pipeline.add_random_split([(2, champion), (1, challenger)])
```

If there is a pre or post process step, those would be added normally.  For example:

```python
pipeline = wl.build_pipeline("pipeline")
pipeline.add_random_split([(2, champion), (1, challenger)])
pipeline.add_model_step(postprocess-python)
```

This assumes that both the champion and challenger return the same data output schema that the post process step is expecting.

<hr/>

#### Exercise: Create an A/B test deployment of your house price models

Use the champion and challenger models that you created in the previous exercises to create an A/B test deployment. You can either create one from scratch, or reconfigure an existing pipeline. 

* Send half the data to the champion, and distribute the rest among the challenger(s).

At the end of this exercise, you should have an A/B test deployment and be ready to compare  multiple models.

```python
# blank space to retrieve pipeline and redeploy with a/b testing step

pipeline.undeploy()
pipeline.clear()
pipeline.add_random_split([(2, bike_day_model), (1, challenger01_model), (1, challenger02_model)])
pipeline.deploy()
```

<table><tr><th>name</th> <td>forecast-tutorial-pipeline</td></tr><tr><th>created</th> <td>2023-08-02 15:50:59.480547+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-14 15:46:31.432410+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>3c5a263b-05de-4305-b121-a9a6b3749dbc, c065b69b-d622-4a7b-93e5-4fcacf62da86, b0a212e3-66b7-4599-9701-f4183045cec6, af0f9c1c-0c28-4aaa-81f5-abb470500960, 980ee03b-694e-47c7-b76b-43b3e927b281, 85af5504-f1e4-4d0d-bd9e-e46891211843, 39b82898-12b6-4a30-ab41-f06cb05c7391, d8edf8c5-07f0-455e-9f34-075b7062f56f, 170402aa-8e83-420e-bee3-51a9fca4a9d9, 14912dd4-5e3a-4314-9e3f-0ea3af3660c1, 3309619d-54b9-4499-8afd-ed7819339b64, 2af1f08c-976c-4d51-9cf6-2cc371788844, 76fbec8d-cebf-40e5-81d5-447170c4a836, c6c10a83-9b6c-449f-a5c3-63b36a3d749b, 436fe308-283f-43b0-a4f0-159c05193d97, eb9e5b9f-41d9-42dc-8e49-13ec4771abad, 4d062242-1477-40fd-bf11-835e6bd62c10, 1f3d774d-7626-4722-b4b8-7dedbaa35803, 12f73035-cf94-4e6c-b2b6-05946ab06aef, b4ec30ef-6724-467e-b42a-d54399198f32, 57e7acf8-b3f0-436b-a236-0b1d6e76ba18, 5697a317-d0e6-402b-9369-7f0e732cc1fa, 5d0cb620-f8ba-4b9d-a81b-0ba333584508, 6b14e208-1319-4bc4-927b-b76a4893d373, 0b44d911-c69e-4030-b481-84e947fe6c70, dc5605d2-bb6a-48d2-b83a-3d77b7e608af, a68819c0-7508-467e-9fc1-60cbf8aaf9e1, b908d302-ce87-4a52-8ef2-b595fac2c67e, 7b94201f-ef5b-4629-ae2f-acf894cb1fcf, dc8bf23f-b598-48c6-bb2d-c5098d264622, 3a8ebc46-6261-4977-8a60-038c99c255d7, 40ab9d3d-ee6c-4f0c-bf38-345385130285, 47792a90-bea8-432a-981f-232bf67288c8, 97b815f3-636b-4424-8be4-3d95bcf32b40, 0d2f2250-9a43-47ce-beef-32371986f798, 46c95b7f-a79e-41ee-8565-578f9c3c20e5, 1ff98a35-3468-4b70-84fc-fe71aed99a75, 73ff8fc2-ca4d-4ea1-887b-0d31190cfe36, f8188956-8b3e-4479-8b15-e8747fe915a6, 33e5cc2c-2bb2-4dc2-8a9e-c058e60f6163, 5d419693-97cc-461b-b72a-a389ab7a001b, 56c78f52-cba5-415c-913a-fee0e1863a90, a109a040-c8f2-46dc-8c0b-373ae10d4fa0, dcaec327-1358-42a7-88de-931602a42a72, debc509f-9481-464b-af7f-5c3138a9cdb4, b0d167aa-cc98-440a-8e85-1ae3f089745a, d9e69c40-c83b-48af-b6b9-caafcb85f08b, 186ffdd2-3a8f-40cc-8362-13cc20bd2f46, 535e6030-ebe5-4c79-b5cd-69b161637a99, c5c0218a-800b-4235-8767-64d18208e68a, 4559d934-33b0-4872-a788-4ef27f554482, 94d3e20b-add7-491c-aedd-4eb094a8aebf, ab4e58bf-3b75-4bf6-b6b3-f703fe61e7af, 3773f5c5-e4c5-4e46-a839-6945af15ca13, 3abf03dd-8eab-4a8d-8432-aa85a30c0eda, 5ec5e8dc-7492-498b-9652-b3733e4c87f7, 1d89287b-4eff-47ec-a7bb-8cedaac1f33f</td></tr><tr><th>steps</th> <td>forecast-challenger01-model</td></tr></table>

The pipeline steps are displayed with the Pipeline `steps()` method.  This is used to verify the current **deployed** steps in the pipeline.

* **IMPORTANT NOTE**: Verify that the pipeline is deployed before checking for pipeline steps.  Deploying the pipeline sets the steps into the Wallaroo system - until that happens, the steps only exist in the local system as *potential* steps.

```python
# blank space to get the current pipeline steps
pipeline.steps()
```

    [{'RandomSplit': {'hash_key': None, 'weights': [{'model': {'name': 'forecast-control-model', 'version': '4c5ade81-ae25-4200-a69e-01e24d15fac5', 'sha': '3cd2acdd1f513f46615be7aa5beac16f09903be851e91f20f6dcdead4a48faa0'}, 'weight': 2}, {'model': {'name': 'forecast-challenger01-model', 'version': 'c99efea6-70b0-4bb0-a5c6-58f50478ca34', 'sha': '5035aca1989226ec1fa16ab325ed2ca7f88de22813d41f1a343f3acbca181dc4'}, 'weight': 1}, {'model': {'name': 'forecast-challenger02-model', 'version': 'fc12d991-9d79-499f-91cb-a7332cb91af6', 'sha': '94473071d321c00670dda36c7e7f953f4ed5fd2f33c2188b3a96dace19ece71d'}, 'weight': 1}]}}]

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
# blank space to send queries to A/B test pipeline and examine the results

sample_count = pd.read_csv('../data/test_data.csv')
inference_df = get_singleton_forecast(sample_count.loc[2:22], 'count')
display(inference_df)

for i in range(10):
    results = pipeline.infer(inference_df)
    display(results.loc[:, ["time", "out.weekly_average"]])
    display(get_names(results))
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[1349, 1562, 1600, 1606, 1510, 959, 822, 1321, 1263, 1162, 1406, 1421, 1248, 1204, 1000, 683, 1650, 1927, 1543, 981, 986]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.weekly_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-14 15:47:55.828</td>
      <td>[1292.5714285714287]</td>
    </tr>
  </tbody>
</table>

    ['forecast-control-model']

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.weekly_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-14 15:47:56.327</td>
      <td>[1292.5714285714287]</td>
    </tr>
  </tbody>
</table>

    ['forecast-control-model']

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.weekly_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-14 15:47:56.793</td>
      <td>[986.0]</td>
    </tr>
  </tbody>
</table>

    ['forecast-challenger01-model']

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.weekly_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-14 15:47:57.249</td>
      <td>[1292.5714285714287]</td>
    </tr>
  </tbody>
</table>

    ['forecast-control-model']

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.weekly_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-14 15:47:57.690</td>
      <td>[1292.5714285714287]</td>
    </tr>
  </tbody>
</table>

    ['forecast-control-model']

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.weekly_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-14 15:47:58.126</td>
      <td>[1292.5714285714287]</td>
    </tr>
  </tbody>
</table>

    ['forecast-control-model']

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.weekly_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-14 15:47:58.582</td>
      <td>[1048.0]</td>
    </tr>
  </tbody>
</table>

    ['forecast-challenger02-model']

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.weekly_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-14 15:47:59.027</td>
      <td>[1292.5714285714287]</td>
    </tr>
  </tbody>
</table>

    ['forecast-control-model']

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.weekly_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-14 15:47:59.487</td>
      <td>[986.0]</td>
    </tr>
  </tbody>
</table>

    ['forecast-challenger01-model']

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.weekly_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-14 15:47:59.938</td>
      <td>[1292.5714285714287]</td>
    </tr>
  </tbody>
</table>

    ['forecast-control-model']

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
pipeline.add_shadow_deploy(bike_day_model, [challenger01_model, challenger02_model])
pipeline.deploy()

```

<table><tr><th>name</th> <td>forecast-tutorial-pipeline</td></tr><tr><th>created</th> <td>2023-08-02 15:50:59.480547+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-14 15:48:29.360779+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>2be416d2-bad8-4210-9fcd-aa1b3f197f64, 3c5a263b-05de-4305-b121-a9a6b3749dbc, c065b69b-d622-4a7b-93e5-4fcacf62da86, b0a212e3-66b7-4599-9701-f4183045cec6, af0f9c1c-0c28-4aaa-81f5-abb470500960, 980ee03b-694e-47c7-b76b-43b3e927b281, 85af5504-f1e4-4d0d-bd9e-e46891211843, 39b82898-12b6-4a30-ab41-f06cb05c7391, d8edf8c5-07f0-455e-9f34-075b7062f56f, 170402aa-8e83-420e-bee3-51a9fca4a9d9, 14912dd4-5e3a-4314-9e3f-0ea3af3660c1, 3309619d-54b9-4499-8afd-ed7819339b64, 2af1f08c-976c-4d51-9cf6-2cc371788844, 76fbec8d-cebf-40e5-81d5-447170c4a836, c6c10a83-9b6c-449f-a5c3-63b36a3d749b, 436fe308-283f-43b0-a4f0-159c05193d97, eb9e5b9f-41d9-42dc-8e49-13ec4771abad, 4d062242-1477-40fd-bf11-835e6bd62c10, 1f3d774d-7626-4722-b4b8-7dedbaa35803, 12f73035-cf94-4e6c-b2b6-05946ab06aef, b4ec30ef-6724-467e-b42a-d54399198f32, 57e7acf8-b3f0-436b-a236-0b1d6e76ba18, 5697a317-d0e6-402b-9369-7f0e732cc1fa, 5d0cb620-f8ba-4b9d-a81b-0ba333584508, 6b14e208-1319-4bc4-927b-b76a4893d373, 0b44d911-c69e-4030-b481-84e947fe6c70, dc5605d2-bb6a-48d2-b83a-3d77b7e608af, a68819c0-7508-467e-9fc1-60cbf8aaf9e1, b908d302-ce87-4a52-8ef2-b595fac2c67e, 7b94201f-ef5b-4629-ae2f-acf894cb1fcf, dc8bf23f-b598-48c6-bb2d-c5098d264622, 3a8ebc46-6261-4977-8a60-038c99c255d7, 40ab9d3d-ee6c-4f0c-bf38-345385130285, 47792a90-bea8-432a-981f-232bf67288c8, 97b815f3-636b-4424-8be4-3d95bcf32b40, 0d2f2250-9a43-47ce-beef-32371986f798, 46c95b7f-a79e-41ee-8565-578f9c3c20e5, 1ff98a35-3468-4b70-84fc-fe71aed99a75, 73ff8fc2-ca4d-4ea1-887b-0d31190cfe36, f8188956-8b3e-4479-8b15-e8747fe915a6, 33e5cc2c-2bb2-4dc2-8a9e-c058e60f6163, 5d419693-97cc-461b-b72a-a389ab7a001b, 56c78f52-cba5-415c-913a-fee0e1863a90, a109a040-c8f2-46dc-8c0b-373ae10d4fa0, dcaec327-1358-42a7-88de-931602a42a72, debc509f-9481-464b-af7f-5c3138a9cdb4, b0d167aa-cc98-440a-8e85-1ae3f089745a, d9e69c40-c83b-48af-b6b9-caafcb85f08b, 186ffdd2-3a8f-40cc-8362-13cc20bd2f46, 535e6030-ebe5-4c79-b5cd-69b161637a99, c5c0218a-800b-4235-8767-64d18208e68a, 4559d934-33b0-4872-a788-4ef27f554482, 94d3e20b-add7-491c-aedd-4eb094a8aebf, ab4e58bf-3b75-4bf6-b6b3-f703fe61e7af, 3773f5c5-e4c5-4e46-a839-6945af15ca13, 3abf03dd-8eab-4a8d-8432-aa85a30c0eda, 5ec5e8dc-7492-498b-9652-b3733e4c87f7, 1d89287b-4eff-47ec-a7bb-8cedaac1f33f</td></tr><tr><th>steps</th> <td>forecast-challenger01-model</td></tr></table>

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

inference_df = get_singleton_forecast(sample_count.loc[4:30], 'count')
display(inference_df)

results = pipeline.infer(inference_df)
display(results.filter(regex='time|out.*?weekly_average'))
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[1600, 1606, 1510, 959, 822, 1321, 1263, 1162, 1406, 1421, 1248, 1204, 1000, 683, 1650, 1927, 1543, 981, 986, 1416, 1985, 506, 431, 1167, 1098, 1096, 1501]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.weekly_average</th>
      <th>out_forecast-challenger01-model.weekly_average</th>
      <th>out_forecast-challenger02-model.weekly_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-14 15:49:25.882</td>
      <td>[1260.4285714285713]</td>
      <td>[1472.7142857142858]</td>
      <td>[1235.0]</td>
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

pipeline.add_model_step(bike_day_model)
pipeline.deploy()
display(pipeline.steps())

inference_df = get_singleton_forecast(sample_count.loc[4:30], 'count')
display(inference_df)

results = pipeline.infer(inference_df)
display(results.filter(regex='time|out.*?weekly_average'))

# swap the model

pipeline.replace_with_model_step(0, challenger02_model)
pipeline.deploy()

# gives time for the update to happen - usually milliseconds, sometimes longer.  This gives enough time for the database updates to happen
import time
time.sleep(15)

display(pipeline.steps())

results = pipeline.infer(inference_df)
display(results.filter(regex='time|out.*?weekly_average'))

```

    [{'ModelInference': {'models': [{'name': 'forecast-control-model', 'version': '4c5ade81-ae25-4200-a69e-01e24d15fac5', 'sha': '3cd2acdd1f513f46615be7aa5beac16f09903be851e91f20f6dcdead4a48faa0'}]}}]

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[1600, 1606, 1510, 959, 822, 1321, 1263, 1162, 1406, 1421, 1248, 1204, 1000, 683, 1650, 1927, 1543, 981, 986, 1416, 1985, 506, 431, 1167, 1098, 1096, 1501]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.weekly_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-14 15:52:23.301</td>
      <td>[1260.4285714285713]</td>
    </tr>
  </tbody>
</table>

    [{'ModelInference': {'models': [{'name': 'forecast-challenger02-model', 'version': 'fc12d991-9d79-499f-91cb-a7332cb91af6', 'sha': '94473071d321c00670dda36c7e7f953f4ed5fd2f33c2188b3a96dace19ece71d'}]}}]

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.weekly_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-14 15:52:43.318</td>
      <td>[1235.0]</td>
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

<table><tr><th>name</th> <td>forecast-tutorial-pipeline</td></tr><tr><th>created</th> <td>2023-08-02 15:50:59.480547+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-14 15:52:23.736145+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>826ffd60-0113-4101-b356-395d8aa713a1, 9897a80f-a81e-4170-9d11-9ad6800f0fa7, f3e2c7a6-7a5a-4f7d-a441-9a6945812735, 2be416d2-bad8-4210-9fcd-aa1b3f197f64, 3c5a263b-05de-4305-b121-a9a6b3749dbc, c065b69b-d622-4a7b-93e5-4fcacf62da86, b0a212e3-66b7-4599-9701-f4183045cec6, af0f9c1c-0c28-4aaa-81f5-abb470500960, 980ee03b-694e-47c7-b76b-43b3e927b281, 85af5504-f1e4-4d0d-bd9e-e46891211843, 39b82898-12b6-4a30-ab41-f06cb05c7391, d8edf8c5-07f0-455e-9f34-075b7062f56f, 170402aa-8e83-420e-bee3-51a9fca4a9d9, 14912dd4-5e3a-4314-9e3f-0ea3af3660c1, 3309619d-54b9-4499-8afd-ed7819339b64, 2af1f08c-976c-4d51-9cf6-2cc371788844, 76fbec8d-cebf-40e5-81d5-447170c4a836, c6c10a83-9b6c-449f-a5c3-63b36a3d749b, 436fe308-283f-43b0-a4f0-159c05193d97, eb9e5b9f-41d9-42dc-8e49-13ec4771abad, 4d062242-1477-40fd-bf11-835e6bd62c10, 1f3d774d-7626-4722-b4b8-7dedbaa35803, 12f73035-cf94-4e6c-b2b6-05946ab06aef, b4ec30ef-6724-467e-b42a-d54399198f32, 57e7acf8-b3f0-436b-a236-0b1d6e76ba18, 5697a317-d0e6-402b-9369-7f0e732cc1fa, 5d0cb620-f8ba-4b9d-a81b-0ba333584508, 6b14e208-1319-4bc4-927b-b76a4893d373, 0b44d911-c69e-4030-b481-84e947fe6c70, dc5605d2-bb6a-48d2-b83a-3d77b7e608af, a68819c0-7508-467e-9fc1-60cbf8aaf9e1, b908d302-ce87-4a52-8ef2-b595fac2c67e, 7b94201f-ef5b-4629-ae2f-acf894cb1fcf, dc8bf23f-b598-48c6-bb2d-c5098d264622, 3a8ebc46-6261-4977-8a60-038c99c255d7, 40ab9d3d-ee6c-4f0c-bf38-345385130285, 47792a90-bea8-432a-981f-232bf67288c8, 97b815f3-636b-4424-8be4-3d95bcf32b40, 0d2f2250-9a43-47ce-beef-32371986f798, 46c95b7f-a79e-41ee-8565-578f9c3c20e5, 1ff98a35-3468-4b70-84fc-fe71aed99a75, 73ff8fc2-ca4d-4ea1-887b-0d31190cfe36, f8188956-8b3e-4479-8b15-e8747fe915a6, 33e5cc2c-2bb2-4dc2-8a9e-c058e60f6163, 5d419693-97cc-461b-b72a-a389ab7a001b, 56c78f52-cba5-415c-913a-fee0e1863a90, a109a040-c8f2-46dc-8c0b-373ae10d4fa0, dcaec327-1358-42a7-88de-931602a42a72, debc509f-9481-464b-af7f-5c3138a9cdb4, b0d167aa-cc98-440a-8e85-1ae3f089745a, d9e69c40-c83b-48af-b6b9-caafcb85f08b, 186ffdd2-3a8f-40cc-8362-13cc20bd2f46, 535e6030-ebe5-4c79-b5cd-69b161637a99, c5c0218a-800b-4235-8767-64d18208e68a, 4559d934-33b0-4872-a788-4ef27f554482, 94d3e20b-add7-491c-aedd-4eb094a8aebf, ab4e58bf-3b75-4bf6-b6b3-f703fe61e7af, 3773f5c5-e4c5-4e46-a839-6945af15ca13, 3abf03dd-8eab-4a8d-8432-aa85a30c0eda, 5ec5e8dc-7492-498b-9652-b3733e4c87f7, 1d89287b-4eff-47ec-a7bb-8cedaac1f33f</td></tr><tr><th>steps</th> <td>forecast-challenger01-model</td></tr></table>

