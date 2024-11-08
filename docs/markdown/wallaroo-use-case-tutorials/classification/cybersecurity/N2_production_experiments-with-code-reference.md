# Tutorial Notebook 2: Vetting a Model With Production Experiments

So far, we've discussed practices and methods for transitioning an ML model and related artifacts from development to production. However, just the act of pushing a model into production is not the only consideration. In many situations, it's important to vet a model's performance in the real world before fully activating it. Real world vetting can surface issues that may not have arisen during the development stage, when models are only checked using hold-out data.

In this notebook, you will learn about two kinds of production ML model validation methods: A/B testing and Shadow Deployments. A/B tests and other types of experimentation are part of the ML lifecycle. The ability to quickly experiment and test new models in the real world helps data scientists to continually learn, innovate, and improve AI-driven decision processes.

## Prerequisites

This notebook assumes that you have completed Notebook 1 "Deploy a Model", and that at this point you have:

* Created and worked with a Workspace.
* Uploaded ML Models and worked with Wallaroo Model Versions.
* Created a Wallaroo Pipeline, added model versions as pipeline steps, and deployed the pipeline.
* Performed sample inferences and undeployed the pipeline.

The same workspace, models, and pipelines will be used for this notebook.

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

#### Pre-exercise

If needed, log into Wallaroo and go to the workspace, pipeline, and most recent model version from the ones that you created in the previous notebook. Please refer to Notebook 1 to refresh yourself on how to log in and set your working environment to the appropriate workspace.

```python
## blank space to log in 

wl = wallaroo.Client()

# retrieve the previous workspace, model, and pipeline version

workspace_name = "tutorial-workspace-john-cybersecurity"

workspace = wl.get_workspace(name=workspace_name, create_if_not_exist=True)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

model_name = 'aloha-prime'

prime_model_version = wl.get_model(model_name)

pipeline_name = 'aloha-fraud-detector'

pipeline = wl.get_pipeline(pipeline_name)

# verify the workspace/pipeline/model

display(wl.get_current_workspace())
display(prime_model_version)
display(pipeline)

```

    {'name': 'tutorial-workspace-john-cybersecurity', 'id': 14, 'archived': False, 'created_by': '76b893ff-5c30-4f01-bd9e-9579a20fc4ea', 'created_at': '2024-05-01T16:30:01.177583+00:00', 'models': [{'name': 'aloha-prime', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 5, 1, 16, 30, 43, 651533, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 5, 1, 16, 30, 43, 651533, tzinfo=tzutc())}], 'pipelines': [{'name': 'aloha-fraud-detector', 'create_time': datetime.datetime(2024, 5, 1, 16, 30, 53, 995114, tzinfo=tzutc()), 'definition': '[]'}]}

<table>
        <tr>
          <td>Name</td>
          <td>aloha-prime</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>c719bc50-f83f-4c79-b4af-f66395a8da04</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>aloha-cnn-lstm.zip</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520</td>
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
          <td>2024-01-May 16:30:43</td>
        </tr>
      </table>

<table><tr><th>name</th> <td>aloha-fraud-detector</td></tr><tr><th>created</th> <td>2024-05-01 16:30:53.995114+00:00</td></tr><tr><th>last_updated</th> <td>2024-05-01 16:31:08.534681+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>4cdf8e11-1b9c-44ab-a16d-abb054b5e9fe, 6b3529b1-1ff1-454b-8896-460c8c90d667</td></tr><tr><th>steps</th> <td>aloha-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Upload Challenger Models

Multiple models can be uploaded to a Wallaroo workspace and deployed in a Pipeline as a `pipeline step`.  These can include pre or post processing Python steps, models that take in different input and output types, or ML models that are of totally different frameworks.

For this module, we will are focuses on different models that have the same input and outputs that are compared to each other to find the "best" model to use.  Before we start, we'll upload another set of models that were pre-trained to provide house prices.  In 'Deploy a Model' we used the file `xgb_model.onnx` as our model.

### Upload Challenger Models Exercise

Upload to the workspace set in the steps above the challenger models `./models/rf_model.onnx` and `./models/gbr_model.onnx`.  Recall this is done with the Wallaroo SDK command `wallaroo.Client.upload_model(name, path, framework, input_schema, output_schema)`.  Because we're working with ONNX models which are part of the native Wallaroo runtimes, we only need to provide a `name`, `path`, and `framework`, and the framework is `wallaroo.framework.Framework.ONNX`.

Upload both `./models/rf_model.onnx` and `./models/gbr_model.onnx` as new models, and return the returned model versions as variables for later use.  Make sure to use **unique** names when uploading the models, or else they will be uploaded as new model versions to an existing model.  Here's an example, assuming that the model `xgb_model.onnx` was uploaded as `house-price-prime`

```python
aloha_challenger = wl.upload_model('aloha-challenger',
                                    '../models/aloha-cnn-lstm-new.zip',
                                    framework=wallaroo.framework.Framework.TENSORFLOW)
display(aloha_challenger)
```

```python
aloha_challenger = wl.upload_model('aloha-challenger',
                                    '../models/aloha-cnn-lstm-new.zip',
                                    framework=wallaroo.framework.Framework.TENSORFLOW)
display(aloha_challenger)
```

<table>
        <tr>
          <td>Name</td>
          <td>aloha-challenger</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>79102468-9ac3-494e-a9f8-8c3968e4eca7</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>aloha-cnn-lstm-new.zip</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>223d26869d24976942f53ccb40b432e8b7c39f9ffcf1f719f3929d7595bceaf3</td>
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
          <td>2024-01-May 16:38:56</td>
        </tr>
      </table>

## A/B Pipeline Steps

An [A/B test](https://en.wikipedia.org/wiki/A/B_testing), also called a controlled experiment or a randomized control trial, is a statistical method of determining which of a set of variants is the best. A/B tests allow organizations and policy-makers to make smarter, data-driven decisions that are less dependent on guesswork.

In the simplest version of an A/B test, subjects are randomly assigned to either the **_control group_** (group A) or the **_treatment group_** (group B). Subjects in the treatment group receive the treatment (such as a new medicine, a special offer, or a new web page design) while the control group proceeds as normal without the treatment. Data is then collected on the outcomes and used to study the effects of the treatment.

In data science, A/B tests are often used to choose between two or more candidate models in production, by measuring which model performs best in the real world. In this formulation, the control is often an existing model that is currently in production, sometimes called the **_champion_**. The treatment is a new model being considered to replace the old one. This new model is sometimes called the **_challenger_**. In our discussion, we'll use the terms *champion* and *challenger*, rather than *control* and *treatment*.

When data is sent to a Wallaroo A/B test pipeline for inference, each datum is randomly sent to either the champion or challenger. After enough data has been sent to collect statistics on all the models in the A/B test pipeline, then those outcomes can be analyzed to determine the difference (if any) in the performance of the champion and challenger. Usually, the purpose of an A/B test is to decide whether or not to replace the champion with the challenger.

Keep in mind that in machine learning, the terms experiments and trials also often refer to the process of finding a training configuration that works best for the problem at hand (this is sometimes called hyperparameter optimization). In this guide, we will use the term experiment to refer to the use of A/B tests to compare the performance of different models in production.

There are a number of considerations to designing an A/B test; you can check out the article [*The What, Why, and How of A/B Testing*](https://wallarooai.medium.com/the-what-why-and-how-of-a-b-testing-64471847cd7e) for more details. In these exercises, we will concentrate on the deployment aspects.  You will need a champion model and  at least one challenger model. You also need to decide on a data split: for example 50-50 between the champion and challenger, or a 2:1 ratio between champion and challenger (two-thirds of the data to the champion, one-third to the challenger).

As an example of creating an A/B test deployment, suppose you have a champion model called "champion", that you have been running in a one-step pipeline called "pipeline". You now want to compare it to a challenger model called "challenger". For your A/B test, you will send two-thirds of the data to the champion, and the other third to the challenger. Both models have already been uploaded.

A/B pipeline steps are created with one of the two following commands:

* `wallaroo.pipeline.add_random_split([(weight1, model1), (weight2, model2)...])`: Create a new A/B Pipeline Step with the provided models and weights.
* `wallaroo.pipeline.replace_with_random_split(index, [(weight1, model1), (weight2, model2)...])`: Replace an existing Pipeline step with an A/B pipeline step at the specified index.

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

Each model receives inputs that are approximately proportional to the weight it is assigned. For example, with two models having weights 1 and 1, each will receive roughly equal amounts of inference inputs. If the weights were changed to 1 and 2, the models would receive roughly 33% and 66% respectively instead.

When choosing the model to use, a random number between 0.0 and 1.0 is generated. The weighted inputs are mapped to that range, and the random input is then used to select the model to use. For example, for the two-models equal-weight case, a random key of 0.4 would route to the first model, 0.6 would route to the second.

Models used for A/B pipeline steps should have the **same** inputs and outputs to accurately compare each other.

Reference:  [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipeline/).

### A/B Pipeline Steps Exercise

Create an A/B pipeline step with the uploaded model versions using the `wallaroo.pipeline.add_random_split` method.  Since we have 3 models, apply a 2:1:1 ratio to the control:challenger1:challenger2 models.

Since this pipeline was used in the previous notebook, use the `wallaroo.pipeline.clear()` method to clear all of the previous steps before adding the new one.  Recall that pipeline steps are not saved from the Notebook to the Wallaroo instance until the `wallaroo.pipeline.deploy(deployment_configuration)` method is applied.

One done, deploy the pipeline with `deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()`.

Here's an example of adding our A/B pipeline step and deploying it.

```python
my_pipeline.clear()
my_pipeline.add_random_split([(2, prime_model_version), (1, house_price_rf_model_version), (1, house_price_gbr_model_version)])
deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
my_pipeline.deploy(deployment_config=deploy_config)
```

```python
pipeline.clear()
pipeline.add_random_split([(2, prime_model_version), (1, aloha_challenger)])

deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
pipeline.deploy(deployment_config=deploy_config)
```

<table><tr><th>name</th> <td>aloha-fraud-detector</td></tr><tr><th>created</th> <td>2024-05-01 16:30:53.995114+00:00</td></tr><tr><th>last_updated</th> <td>2024-05-01 16:39:02.287551+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>262909e9-8779-4c56-a994-725ddd0b58c8, 4cdf8e11-1b9c-44ab-a16d-abb054b5e9fe, 6b3529b1-1ff1-454b-8896-460c8c90d667</td></tr><tr><th>steps</th> <td>aloha-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

The pipeline steps are displayed with the Pipeline `steps()` method.  This is used to verify the current **deployed** steps in the pipeline.

* **IMPORTANT NOTE**: Verify that the pipeline is deployed before checking for pipeline steps.  Deploying the pipeline sets the steps into the Wallaroo system - until that happens, the steps only exist in the local system as *potential* steps.

```python
# blank space to get the current pipeline steps

pipeline.steps()
```

    [{'RandomSplit': {'hash_key': None, 'weights': [{'model': {'name': 'aloha-prime', 'version': 'c719bc50-f83f-4c79-b4af-f66395a8da04', 'sha': 'fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520'}, 'weight': 2}, {'model': {'name': 'aloha-challenger', 'version': '79102468-9ac3-494e-a9f8-8c3968e4eca7', 'sha': '223d26869d24976942f53ccb40b432e8b7c39f9ffcf1f719f3929d7595bceaf3'}, 'weight': 1}]}}]

## A/B Pipeline Inferences

Inferences through pipelines that have A/B steps is the same any other pipeline:  use either `wallaroo.pipeline.infer(pandas record|Apache Arrow)` or `wallaroo.pipeline.infer_from_file(path)`.  The distinction is that inference data will randomly be assigned to one of the models in the A/B pipeline step based on the weighted ratio pattens.

The output is the same as any other inference request, with one difference:  the `out._model_split` field that lists which model version was used for the model split step.  Here's an example:

```python
single_result = pipeline.infer_from_file('./data/singleton.df.json')
display(single_result)
```

|&nbsp;|out._model_split|out.variable|
|0|[{"name":"house-price-prime","version":"69f344e6-2ab5-47e8-82a6-030e328c8d35","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]|[2176827.0]|

Please note that for batch inferences, the entire batch will be sent to the same model. So in order to verify that your pipeline is distributing inferences in the proportion you specified, you will need to send your queries one datum at a time.  This example has 4,000 rows of data submitted in one batch as an inference request.  Note that the same model is listed in the `out._model_split` field.

```python
multiple_result = pipeline.infer_from_file('../data/test_data.df.json')
display(multiple_result.head(10).loc[:, ['out._model_split', 'out.variable']])
```

|&nbsp;|out._model_split|out.variable|
|---|---|---|
|0|[{"name":"house-price-rf-model","version":"616c2306-bf93-417b-9656-37bee6f14379","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]|[718013.7]|
|1|[{"name":"house-price-rf-model","version":"616c2306-bf93-417b-9656-37bee6f14379","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]|[615094.6]|
|2|[{"name":"house-price-rf-model","version":"616c2306-bf93-417b-9656-37bee6f14379","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]|[448627.8]|
|3|[{"name":"house-price-rf-model","version":"616c2306-bf93-417b-9656-37bee6f14379","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]|[758714.3]|
|4|[{"name":"house-price-rf-model","version":"616c2306-bf93-417b-9656-37bee6f14379","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]|[513264.66]|
|5|[{"name":"house-price-rf-model","version":"616c2306-bf93-417b-9656-37bee6f14379","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]|[668287.94]|
|6|[{"name":"house-price-rf-model","version":"616c2306-bf93-417b-9656-37bee6f14379","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]|[1004846.56]|
|7|[{"name":"house-price-rf-model","version":"616c2306-bf93-417b-9656-37bee6f14379","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]|[684577.25]|
|8|[{"name":"house-price-rf-model","version":"616c2306-bf93-417b-9656-37bee6f14379","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]|[727898.25]|
|9|[{"name":"house-price-rf-model","version":"616c2306-bf93-417b-9656-37bee6f14379","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]|[559631.06]|

To help with the next exercise, here is another convenience function you might find useful.  It takes the inference result and returns the model version used in the A/B Testing step.

Run the cell block below before going on to the exercise.

```python
# get the names of the inferring models
# from a dataframe of a/b test results
def get_names(resultframe):
    modelcol = resultframe['out._model_split']
    jsonstrs = [mod[0]  for mod in modelcol]
    return [json.loads(jstr)['model_version']['name'] for jstr in jsonstrs]
```

### A/B Pipeline Inferences Exercise

Perform a set of inferences with the same data, and show the model version used for the A/B Testing step.  Here's an example.

```python
for x in range(10):
    single_result = pipeline.infer_from_file('../data/data-1.df.json')
    display(f"{get_names(single_result)}: {single_result.loc[0, 'out.main']}")
```

```python
for x in range(10):
    single_result = pipeline.infer_from_file('../data/data-1.df.json')
    display(f"{get_names(single_result)}: {single_result.loc[0, 'out.cryptolocker']}")
```

    "['aloha-prime']: [0.012099553]"

    "['aloha-challenger']: [0.012099553]"

    "['aloha-prime']: [0.012099553]"

    "['aloha-prime']: [0.012099553]"

    "['aloha-prime']: [0.012099553]"

    "['aloha-prime']: [0.012099553]"

    "['aloha-challenger']: [0.012099553]"

    "['aloha-prime']: [0.012099553]"

    "['aloha-prime']: [0.012099553]"

    "['aloha-challenger']: [0.012099553]"

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

### Shadow Deployments Exercise

Use the champion and challenger models that you created in the previous exercises to create a shadow deployment. You can either create one from scratch, or reconfigure an existing pipeline.

At the end of this exercise, you should have a shadow deployment running multiple models in parallel.

Here's an example:

```python
pipeline.undeploy()

pipeline.clear()
pipeline.add_shadow_deploy(prime_model_version, 
                           [aloha_challenger]
                        )

deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
pipeline.deploy(deployment_config=deploy_config)
```

```python
# blank space to create a shadow deployment

pipeline.undeploy()

pipeline.clear()
pipeline.add_shadow_deploy(prime_model_version, 
                           [aloha_challenger]
                        )

deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
pipeline.deploy(deployment_config=deploy_config)
```

<table><tr><th>name</th> <td>aloha-fraud-detector</td></tr><tr><th>created</th> <td>2024-05-01 16:30:53.995114+00:00</td></tr><tr><th>last_updated</th> <td>2024-05-01 16:42:05.952258+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>d22eb0d2-9cff-4c5f-a851-10b1a19d8c44, 262909e9-8779-4c56-a994-725ddd0b58c8, 4cdf8e11-1b9c-44ab-a16d-abb054b5e9fe, 6b3529b1-1ff1-454b-8896-460c8c90d667</td></tr><tr><th>steps</th> <td>aloha-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Shadow Deploy Inference

Since a shadow deployment returns multiple predictions for a single datum, its inference result will look a little different from those of an A/B test or a single-step pipeline. The next exercise will show you how to examine all the inferences from all the models.

Model outputs are listed by column based on the model’s outputs. The output data is set by the term out, followed by the name of the model. For the default model, this is out.{variable_name}, while the shadow deployed models are in the format out_{model name}.variable, where {model name} is the name of the shadow deployed model.

Here's an example with the models `ccfraudrf` and `ccfraudxgb`.

```python
sample_data_file = './smoke_test.df.json'
response = pipeline.infer_from_file(sample_data_file)
```

| | time | in.tensor | out.dense_1 | check_failures | out_ccfraudrf.variable | out_ccfraudxgb.variable
|---|---|---|---|---|---|---
|0 | 2023-03-03 17:35:28.859 | [1.0678324729, 0.2177810266, -1.7115145262, 0.682285721, 1.0138553067, -0.4335000013, 0.7395859437, -0.2882839595, -0.447262688, 0.5146124988, 0.3791316964, 0.5190619748, -0.4904593222, 1.1656456469, -0.9776307444, -0.6322198963, -0.6891477694, 0.1783317857, 0.1397992467, -0.3554220649, 0.4394217877, 1.4588397512, -0.3886829615, 0.4353492889, 1.7420053483, -0.4434654615, -0.1515747891, -0.2668451725, -1.4549617756] | [0.0014974177] | 0 | [1.0] | [0.0005066991]

### Shadow Deploy Inference Exercise

Use the test data that from the previous exercise to send a single datum to the shadow deployment that you created in the previous exercise.  View the outputs from each of the shadow deployed models.

Here's an example - adjust based on the name of your models.

```python
single_result = pipeline.infer_from_file('../data/data-1.df.json')
display(single_result.loc[0, ['out.main', 
                              'out_aloha-challenger.main']
                        ])
```

```python
# blank space to send an inference and examine the result

single_result = pipeline.infer_from_file('../data/data-1.df.json')
display(single_result.loc[0, ['out.main', 
                              'out_aloha-challenger.main']
                        ])
```

    out.main                     [0.997564]
    out_aloha-challenger.main    [0.997564]
    Name: 0, dtype: object

## After the Experiment: Swapping in New Models

You have seen two methods to validate models in production with test (challenger) models. 
The end result of an experiment is a decision about which model becomes the new champion. Let's say that you have been running the shadow deployment that you created in the previous exercise,  and you have decided that you want to replace the model "champion" with the model "challenger". To do this, you will clear all the steps out of the pipeline, and add only "challenger" back in.

```python
# undeploy the pipeline and reset it back to the champion model
pipeline.undeploy()
pipeline.clear()

pipeline.add_model_step(prime_model_version)

deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
pipeline.deploy(deployment_config=deploy_config)

single_result = pipeline.infer_from_file('../data/data-1.df.json')
display(single_result.loc[:, ["time", "in.text_input", "out.main"]])
display(pipeline.steps())
```

### After the Experiment: Swapping in New Models Exercise

Pick one of your challenger models as the new champion, and reconfigure your shadow deployment back into a single-step pipeline with the new chosen model.

* Run the test datum from the previous exercise through the reconfigured pipeline.
* Compare the results to the results from the previous exercise.
* Notice that the pipeline predictions are different from the old champion, and consistent with the new one.

At the end of this exercise, you should have a single step pipeline, running a new model.

```python
# Blank space - remove all steps, then redeploy with new champion model
pipeline.undeploy()
pipeline.clear()

pipeline.add_model_step(prime_model_version)

deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
pipeline.deploy(deployment_config=deploy_config)

single_result = pipeline.infer_from_file('../data/data-1.df.json')
display(single_result.loc[:, ["time", "in.text_input", "out.main"]])
display(pipeline.steps())
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.text_input</th>
      <th>out.main</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-05-01 16:45:34.467</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 16, 32, 23, 29, 32, 30, 19, 26, 17]</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>

    [{'ModelInference': {'models': [{'name': 'aloha-prime', 'version': 'c719bc50-f83f-4c79-b4af-f66395a8da04', 'sha': 'fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520'}]}}]

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>aloha-fraud-detector</td></tr><tr><th>created</th> <td>2024-05-01 16:30:53.995114+00:00</td></tr><tr><th>last_updated</th> <td>2024-05-01 16:45:19.372610+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>551242a7-fe4c-4a61-a4c7-e7fcc97509dc, d22eb0d2-9cff-4c5f-a851-10b1a19d8c44, 262909e9-8779-4c56-a994-725ddd0b58c8, 4cdf8e11-1b9c-44ab-a16d-abb054b5e9fe, 6b3529b1-1ff1-454b-8896-460c8c90d667</td></tr><tr><th>steps</th> <td>aloha-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

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

<table><tr><th>name</th> <td>aloha-fraud-detector</td></tr><tr><th>created</th> <td>2023-09-27 16:29:55.160300+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-27 16:35:00.916922+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>79865932-5b89-4b2a-bfb1-cb9ebeb5125f, 4727b985-db36-44f7-a1a3-7f1886bbf894, 07cbfcae-1fa2-4746-b585-55349d230b45, 03824313-6bbb-4ccd-95ea-64340f789b9c, 9ce54998-a667-43b3-8198-b2d95e0d2879, 8a416842-5675-455a-b638-29fe7dbb5ba1</td></tr><tr><th>steps</th> <td>aloha-prime</td></tr></table>

```python

```