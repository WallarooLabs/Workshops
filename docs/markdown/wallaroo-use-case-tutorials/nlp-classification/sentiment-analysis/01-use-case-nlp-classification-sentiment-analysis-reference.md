# Tutorial Notebook 1: Build and Deploy a Model

For this tutorial, let's pretend that you work for a movie promotion company tracking how people feel about a movie, and have developed a ML Model that tracks reviews written in the Internet Movie DataBase (IMDB) to gauge whether a review is positive or negative.

For this tutorial, you will train a ML model that uses data from the the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) with sample data can be downloaded from the [aclIMDB dataset](http://s3.amazonaws.com/text-datasets/aclImdb.zip ) to create a new model.  (Or use the pre-trained one in the `./models` folder).

Before we start, let's load some libraries that we will need for this notebook (note that this may not be a complete list).

* **IMPORTANT NOTE**:  This tutorial is geared towards a Wallaroo 2023.2.1 environment.

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

<hr/>

#### Exercise: Build a model

Sample data can be downloaded from the [aclIMDB dataset](http://s3.amazonaws.com/text-datasets/aclImdb.zip ) to create a new model - or from the `./data/aclimdb` folder.  Some possible other sources in categorizing the original text are available [from this 'Fetching data, training a classifier' example](https://github.com/marcotcr/lime/blob/master/doc/notebooks/Lime%20-%20basic%20usage%2C%20two%20class%20case.ipynb).

At the end of the exercise, you should have a notebook and possibly other artifacts to produce a model for predicting house prices. For the purposes of the exercise, please use a framework that can be converted to ONNX, such as scikit-learn or XGBoost.

For assistance converting a model to ONNX, see the [Wallaroo Model Conversion Tutorials](https://docs.wallaroo.ai/wallaroo-tutorials/wallaroo-tutorials-conversion-tutorials/) for some examples.

**NOTE**

If you prefer to shortcut this step, you can use one of the pre-trained model files in the `models` subdirectory.

```python
## Blank space for training model, if needed

```

## Getting Ready to deploy

Wallaroo natively supports models in the ONNX and Tensorflow frameworks, and other frameworks via containerization. For this exercise, we assume that you have a model that can be converted to the ONNX framework. The first steps to deploying in Wallaroo, then, is to convert your model to ONNX, and to add some extra functions to your processing modules so Wallaroo can call them.
<hr/>

#### Exercise: Convert your Model to ONNX

Take the model that you created in the previous exercises, and convert it to ONNX. If you need help, see the [Wallaroo Conversion Tutorials](https://docs.wallaroo.ai/wallaroo-tutorials/wallaroo-tutorials-conversion-tutorials/), or other [conversion documentation](https://github.com/onnx/tutorials#converting-to-onnx-format). 

At the end of this exercise, you should have your model as a standalone artifact, for example, a file called `model.onnx`.

**NOTE**

If you prefer to shortcut this exercise, you can use one of the pre-converted onnx files in the `models` directory.

```python
# Blank space to load for converting model, if needed

```

## Get ready to work with Wallaroo

Now that you have a model ready to go, you can log into Wallaroo and set up a **workspace** to organize your deployment artifacts. A Wallaroo workspace is place to organize the deployment artifacts for a project, and to collaborate with other team members. For more information, see [the Wallaroo 101](https://docs.wallaroo.ai/wallaroo-101/). 

Logging into Wallaroo via the cluster's integrated JupyterLab is quite straightfoward:

```python
# Login through local Wallaroo instance 
wl = wallaroo.Client()
```
See [the documentation](https://docs.wallaroo.ai/wallaroo-101/#connect-to-the-wallaroo-instance) if you are logging into Wallaroo some other way.

Once you are logged in, you can create a workspace and set it as your working environment. To make the first exercise easier, here is a convenience function to get or create a workspace:

```python
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
```

Then logging in and creating a workspace looks something like this:

```python
# Login through local Wallaroo instance 
wl = wallaroo.Client()
```

Setting up the workspace may resemble this.  Verify that the workspace name is unique across the Wallaroo instance.

```python
# workspace names need to be globally unique, so add a random suffix to insure this
# especially important if the "main" workspace name is potentially a common one

suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
workspace_name = "my-workspace"+suffix

workspace = get_workspace(workspace_name)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

# optionally, examine your current workspace
wl.get_current_workspace()

```

<hr/>

#### Exercise: Log in and create a workspace

Log into wallaroo, and create a workspace for this tutorial. Then set that new workspace to your current workspace.
Make sure you remember the name that you gave the workspace, as you will need it for later notebooks. Set that workspace to be your working environment.

**Notes**
* Workspace names must be globally unique, so don't pick something too common. The "random suffix" trick in the code snippet is one way to try to generate a unique workspace name, if you suspect you are using a common name. 

At the end of the exercise, you should be in a new workspace to do further work.

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

```python
## Blank spot to connect to the workspace

# used to generate a random workspace for testing
suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

workspace_name = f"tutorial-workspace"

workspace = get_workspace(workspace_name)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

# optionally, examine your current workspace
wl.get_current_workspace()
```

    {'name': 'tutorial-workspace-jch', 'id': 19, 'archived': False, 'created_by': '0a36fba2-ad42-441b-9a8c-bac8c68d13fa', 'created_at': '2023-08-03T19:34:42.324336+00:00', 'models': [{'name': 'tutorial-model', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 3, 19, 36, 31, 13200, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 3, 19, 36, 31, 13200, tzinfo=tzutc())}], 'pipelines': [{'name': 'tutorialpipeline-jch', 'create_time': datetime.datetime(2023, 8, 3, 19, 36, 31, 732163, tzinfo=tzutc()), 'definition': '[]'}]}

## Deploy a Simple Single-Step Pipeline

Once your model is in the ONNX format, and you have a workspace to work in, you can easily upload your model to Wallaroo's production platform with just a few lines of code. For example, if you have a model called `model.onnx`, and you wish to upload it to Wallaroo with the name `mymodel`, then upload the model as follows (once you are in the appropriate workspace):

```python

my_model = wl.upload_model("mymodel", "model.onnx", framework=Framework.ONNX).configure()
```

See [Wallaroo SDK Essentials Guide: Model Uploads and Registrations: ONNX](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-onnx/) for full details.

The function `upload_model()` returns a handle to the uploaded model that you will continue to work with in the SDK.

Once the model has been uploaded, you can create a **pipeline** that contains the model. The pipeline is the mechanism that manages deployments. A pipeline contains a series of **steps** - sequential sets of models which take in the data from the preceding step, process it through the model, then return a result. Some pipelines can have just one step, while others may have multiple models with multiple steps or arranged for A/B testing. Deployed pipelines allocate resources and can then process data either through local files or through a **deployment URL**.

So for your model to accept inferences, you must add it to a pipeline. You can create a single step pipeline called `mypipeline` as follows.

```python
# create the pipeline
my_pipeline = wl.build_pipeline("mypipeline").add_model_step(my_model)

# deploy the pipeline
my_pipeline = my_pipeline.deploy()
```

Deploying the pipeline means that resources from the cluster are allocated to the pipeline, and it is ready to accept inferences. You can "turn off" the pipeline with the call `pipeline.undeploy()`, which returns the resources back to the cluster.  This is an important step - leaving pipeline deployed when they're no longer needed takes up resources that may be needed by other pipelines or services.

See [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/) for full details.

### Additional Data Processing Steps

In some cases, additional steps are used to format the data either before reaching the model (pre-process) or after the models output (post-process).

For this sample example, we have two models:  one that prepares the data (aka the embedder) and the actual sentiment model.  The both are ONNX models, and can be deployed as follows:

```python
pipeline.add_model_step(embedder)
pipeline.add_model_step(sentiment_model).configure(runtime="onnx", tensor_fields=["flatten_1"])
```

Note the additional configuration for our sample sentiment model - this is to use the output from the embedder and recognize the tensor fields to look at (aka - `flatten_1`).

Inference data is passed to the first ML model, then the results of that model are passed to the next step.  Any ML model can serve as a Python step in sequence, provided it is trained to input and output the data in the format that the previous or the next model in the sequence provides.

For convenience, pre and post process steps can be Python scripts and deployed to Wallaroo as pipeline model steps to make managing this data processing earlier.

For information on how to set up a pipeline step as a Python script for hosting models or for pre or post processing, see [Wallaroo SDK Essentials Guide: Model Uploads and Registrations: Python Models](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-python/).

**More Hints**

* `workspace = wl.get_current_workspace()` gives you a handle to the current workspace
* then `workspace.models()` will return a list of the models in the workspace
* and `workspace.pipelines()` will return a list of the pipelines in the workspace

<hr/>

#### Exercise: Upload and deploy your model

Upload and deploy the ONNX model that you created in the previous exercise. For simplicity, do any needed pre-processing in the notebook.

At the end of the exercise, you should have a model and a deployed pipeline in your workspace.

```python
## blank space to upload model, and create the pipeline

from wallaroo.framework import Framework

embedder = wl.upload_model('embedder', '../models/embedder.onnx', framework=Framework.ONNX)
sentiment_model = wl.upload_model('sentiment', '../models/sentiment_model.onnx', framework=Framework.ONNX).configure(runtime="onnx", tensor_fields=["flatten_1"])

pipeline = wl.build_pipeline("sentiment-analysis").add_model_step(embedder).add_model_step(sentiment_model)

pipeline.deploy()

```

<table><tr><th>name</th> <td>sentiment-analysis</td></tr><tr><th>created</th> <td>2023-08-11 15:34:49.622995+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-11 15:43:22.426116+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>cf689a7d-e51a-4d58-96fd-a024ebd3ddba, d1a918df-04fe-4b98-8d1e-01f7831aab44, 7668ad9a-c12d-40a5-8370-c681c87e4786, 8028e5fc-81b7-45a0-8347-61e0e17e20c4</td></tr><tr><th>steps</th> <td>sentiment</td></tr></table>

## Sending Data to your Pipeline

ONNX models generally expect their input as an array in a dictionary, keyed by input name. In Wallaroo, the default input name is "tensor". So (outside of Wallaroo), an ONNX model that expected three numeric values as its input would expect input data similar to the below: (**Note: The below examples are only notional, they aren't intended to work with our example models.**)

```python
# one datum
singleton = {'tensor': [[1, 2, 3]] }

# two datums
two_inputs = {'tensor': [[1, 2, 3], [4, 5, 6]] }
```

In the Wallaroo SDK, you can send a pandas DataFrame representation of this dictionary (pandas record format) to the pipeline, via the `pipeline.infer()` method.

```python
import pandas as pd

# one datum (notional example)
sdf = pd.DataFrame(singleton)
sdf
#       tensor
# 0  [1, 2, 3]

# send the datum to a pipeline for inference
# notional example - not houseprice model!
result = my_pipeline.infer(sdf)

# two datums
# Note that the value of 'tensor' must be a list, not a numpy array 
twodf = pd.DataFrame(two_inputs)
twodf
#      tensor
# 0  [1, 2, 3]
# 1  [4, 5, 6]

# send the data to a pipeline for inference
# notional example, not houseprice model!
result = my_pipeline.infer(twodf)
```

To send data to a pipeline via the inference URL (for example, via CURL), you need the JSON representation of these data frames.

```python
#
# notional examples, not houseprice model!
#
sdf.to_json(orient='records')
# '[{"tensor":[1,2,3]}]'

twodf.to_json(orient='records')
# '[{"tensor":[1,2,3]},{"tensor":[4,5,6]}]'
```

If the JSON data is in a file, you can send it to the pipeline from within the SDK via the `pipeline.infer_from_file()` method. 

In either case, a successful inference will return a data frame of inference results. The model inference(s) will be in the `column out.<outputname>`.

For more details, see [Wallaroo SDK Essentials Guide: Inference Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-inferences/).

### Converting inputs

If your input data is in a standard tabular format, then you need to convert to pandas record format to send the data to your pipeline.  See the pandas DataFrame documentation for methods on how to import a CSV file to a DataFrame.

The sample data provided is already tokenized in both Apache Arrow, JSON and pandas record format in the `./data` folder.  If using the pre-built models, try loading the `./data/test_data_50K.df.json` in a pandas DataFrame and proceeding from there.

To help with the following exercises, here are some convenience functions you might find useful for doing this conversion. These functions convert input data in standard tabular format (in a pandas DataFrame) to the pandas record format that the model expects.

```python
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

#### Execute the following code block to see examples of what `get_singleton` and `get_batch` do.

```python
# RUN ME!

print('''TOY data for a model that takes inputs var1, var2, var3.
The dataframe is called df.
Pretend the model is in a Wallaroo pipeline called "toypipeline"''')

df = pd.DataFrame({
    'var1': [1, 3, 5],
    'var2': [33, 88, 45],
    'var3': [6, 20, 5]
})

display(df)

# create a model input from the first row
# this is now in the format that a model would accept
singleton = get_singleton(df, 0)

print('''The command "singleton = get_singleton(df, 0)" converts
the first row of the data frame into the format that Wallaroo pipelines accept.
You could now get a prediction by: "toypipeline.infer(singleton)".
''')
display(singleton)

# create a batch of queries from the entire dataframe
batch = get_batch(df, nrows=2)

print('''The command "batch = get_batch(df, nrows=2)" converts
the the first two rows of the data frame into a batch format that Wallaroo pipelines accept.
You could now get a batch prediction by: "toypipeline.infer(batch)".
''')
display(batch)

```

    TOY data for a model that takes inputs var1, var2, var3.
    The dataframe is called df.
    Pretend the model is in a Wallaroo pipeline called "toypipeline"

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>var1</th>
      <th>var2</th>
      <th>var3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>33</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>88</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>45</td>
      <td>5</td>
    </tr>
  </tbody>
</table>

    The command "singleton = get_singleton(df, 0)" converts
    the first row of the data frame into the format that Wallaroo pipelines accept.
    You could now get a prediction by: "toypipeline.infer(singleton)".
    

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
      <td>[1, 33, 6]</td>
    </tr>
  </tbody>
</table>

    The command "batch = get_batch(df, nrows=2)" converts
    the the first two rows of the data frame into a batch format that Wallaroo pipelines accept.
    You could now get a batch prediction by: "toypipeline.infer(batch)".
    

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
      <td>[1, 33, 6]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[3, 88, 20]</td>
    </tr>
  </tbody>
</table>

<hr/>

#### Exercise: Send data to your pipeline for inference.

Create some test data from the housing data and send it to the pipeline that you deployed in the previous exercise.  

* If you used the pre-provided models, then you can use `test_data.csv` from the `data` directory.  This can be loaded directly into **your** sample pandas DataFrame - check the pandas documentation for a handy function for doing that.  (We mention yours because sometimes people try to use the example code above rather than their own data.)

* Start easy, with just one datum; retrieve the inference results. You can try small batches, as well. Use the above example as a guide.
* Examine the inference results; observe what the model prediction column is called; it should be of the form `out.<outputname>`.

For more hints about the different ways of sending data to the pipeline, and to see an example of the inference result format, see the ["Running Inferences" section of Wallaroo 101](https://docs.wallaroo.ai/wallaroo-101/#running-interfences).

At the end of the exercise, you should have a set of inference results that you got through the Wallaroo pipeline. 

```python
##  blank space to create test data, and send some data to your model

df = pd.read_json('../data/test_data_50K.df.json')

singleton = get_singleton(df, 0)
display(singleton)

single_result = pipeline.infer(singleton)
display(single_result)

multiple_batch = get_batch(df, nrows=5)
multiple_result = pipeline.infer(multiple_batch)
display(multiple_result)

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
      <td>2023-08-11 15:44:16.706</td>
      <td>[[11.0, 6.0, 1.0, 12.0, 112.0, 13.0, 14.0, 73.0, 14.0, 10.0, 470.0, 5.0, 116.0, 9.0, 207.0, 465.0, 96.0, 15.0, 69.0, 5.0, 231.0, 15.0, 9.0, 91.0, 812.0, 6.0, 28.0, 4.0, 58.0, 511.0, 9654.0, 148.0, 6792.0, 20.0, 1.0, 82.0, 505.0, 1098.0, 30.0, 3.0, 7476.0, 2.0, 2032.0, 96.0, 547.0, 1059.0, 2.0, 148.0, 42.0, 640.0, 4716.0, 8.0, 91.0, 1670.0, 4939.0, 783.0, 41.0, 3.0, 529.0, 449.0, 9.0, 492.0, 85.0, 3050.0, 2.0, 1.0, 357.0, 4.0, 1.0, 174.0, 468.0, 8.0, 84.0, 351.0, 155.0, 155.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]</td>
      <td>[0.8980188]</td>
      <td>0</td>
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
      <td>2023-08-11 15:44:17.093</td>
      <td>[[11.0, 6.0, 1.0, 12.0, 112.0, 13.0, 14.0, 73.0, 14.0, 10.0, 470.0, 5.0, 116.0, 9.0, 207.0, 465.0, 96.0, 15.0, 69.0, 5.0, 231.0, 15.0, 9.0, 91.0, 812.0, 6.0, 28.0, 4.0, 58.0, 511.0, 9654.0, 148.0, 6792.0, 20.0, 1.0, 82.0, 505.0, 1098.0, 30.0, 3.0, 7476.0, 2.0, 2032.0, 96.0, 547.0, 1059.0, 2.0, 148.0, 42.0, 640.0, 4716.0, 8.0, 91.0, 1670.0, 4939.0, 783.0, 41.0, 3.0, 529.0, 449.0, 9.0, 492.0, 85.0, 3050.0, 2.0, 1.0, 357.0, 4.0, 1.0, 174.0, 468.0, 8.0, 84.0, 351.0, 155.0, 155.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]</td>
      <td>[0.8980188]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-08-11 15:44:17.093</td>
      <td>[[54.0, 548.0, 86.0, 70.0, 1213.0, 24.0, 746.0, 6.0, 11.0, 19.0, 6.0, 3.0, 1898.0, 90.0, 370.0, 113.0, 832.0, 367.0, 154.0, 10.0, 78.0, 21.0, 121.0, 135.0, 4717.0, 5.0, 350.0, 2.0, 1594.0, 122.0, 3.0, 26.0, 6.0, 2315.0, 30.0, 9.0, 22.0, 103.0, 1.0, 2253.0, 20.0, 1.0, 285.0, 2.0, 1.0, 93.0, 26.0, 44.0, 3.0, 367.0, 790.0, 87.0, 184.0, 26.0, 40.0, 9.0, 53.0, 26.0, 1383.0, 109.0, 1.0, 2211.0, 4.0, 688.0, 26.0, 6.0, 3.0, 75.0, 281.0, 26.0, 1784.0, 69.0, 4.0, 157.0, 4311.0, 1720.0, 2124.0, 46.0, 86.0, 44.0, 66.0, 11.0, 19.0, 614.0, 30.0, 540.0, 1927.0, 4588.0, 2.0, 159.0, 555.0, 118.0, 5924.0, 81.0, 264.0, 15.0, 2.0, 688.0, 530.0, 20.0]]</td>
      <td>[0.056596935]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-08-11 15:44:17.093</td>
      <td>[[1.0, 9259.0, 6.0, 8.0, 1.0, 3.0, 62.0, 4.0, 32.0, 4416.0, 34.0, 457.0, 8595.0, 31.0, 1.0, 497.0, 2.0, 8.0, 1.0, 972.0, 2847.0, 2178.0, 24.0, 110.0, 2.0, 1.0, 1918.0, 60.0, 1072.0, 1.0, 129.0, 26.0, 44.0, 410.0, 2353.0, 8.0, 49.0, 2.0, 442.0, 8.0, 1.0, 4287.0, 4.0, 24.0, 24.0, 116.0, 599.0, 5074.0, 2.0, 1135.0, 7093.0, 2602.0, 5120.0, 2.0, 22.0, 25.0, 3.0, 450.0, 8596.0, 16.0, 3036.0, 2.0, 1975.0, 385.0, 16.0, 1.0, 1023.0, 931.0, 4.0, 2137.0, 2.0, 1.0, 3022.0, 4.0, 309.0, 4416.0, 294.0, 32.0, 318.0, 19.0, 15.0, 145.0, 80.0, 807.0, 3264.0, 1.0, 4416.0, 294.0, 5034.0, 15.0, 3023.0, 6.0, 32.0, 5514.0, 4.0, 1.0, 1299.0, 2205.0, 493.0, 1.0]]</td>
      <td>[0.9260802]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-08-11 15:44:17.093</td>
      <td>[[10.0, 25.0, 107.0, 1.0, 343.0, 17.0, 3.0, 168.0, 150.0, 593.0, 100.0, 12.0, 10.0, 103.0, 29.0, 2278.0, 1.0, 83.0, 28.0, 6.0, 63.0, 21.0, 1.0, 115.0, 18.0, 42.0, 1.0, 88.0, 1060.0, 28.0, 204.0, 458.0, 103.0, 1.0, 228.0, 4.0, 6887.0, 4252.0, 297.0, 42.0, 63.0, 84.0, 48.0, 131.0, 490.0, 119.0, 79.0, 1.0, 2278.0, 23.0, 318.0, 8.0, 1.0, 315.0, 299.0, 190.0, 126.0, 576.0, 5.0, 103.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]</td>
      <td>[0.926919]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-08-11 15:44:17.093</td>
      <td>[[10.0, 37.0, 1.0, 49.0, 2.0, 442.0, 982.0, 10.0, 420.0, 1807.0, 8.0, 11.0, 17.0, 125.0, 71.0, 98.0, 17.0, 26.0, 44.0, 123.0, 221.0, 26.0, 283.0, 1.0, 1389.0, 9260.0, 121.0, 9.0, 29.0, 26.0, 628.0, 295.0, 26.0, 284.0, 480.0, 2.0, 3.0, 50.0, 4484.0, 482.0, 1.0, 189.0, 12.0, 9.0, 284.0, 47.0, 23.0, 3108.0, 180.0, 8.0, 1822.0, 2.0, 20.0, 699.0, 71.0, 72.0, 67.0, 101.0, 4.0, 405.0, 69.0, 437.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]</td>
      <td>[0.6618577]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

## Undeploying Your Pipeline

You should always undeploy your pipelines when you are done with them, or don't need them for a while. This releases the resources that the pipeline is using for other processes to use. You can always redeploy the pipeline when you need it again. As a reminder, here are the commands to deploy and undeploy a pipeline:

```python

# when the pipeline is deployed, it's ready to receive data and infer
pipeline.deploy()

# "turn off" the pipeline and releaase its resources
pipeline.undeploy()

```

If you are continuing on to the next notebook now, you can leave the pipeline deployed to keep working; but if you are taking a break, then you should undeploy.

```python
## blank space to undeploy the pipeline, if needed

pipeline.undeploy()

```

<table><tr><th>name</th> <td>sentiment-analysis</td></tr><tr><th>created</th> <td>2023-08-11 15:34:49.622995+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-11 15:43:22.426116+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>cf689a7d-e51a-4d58-96fd-a024ebd3ddba, d1a918df-04fe-4b98-8d1e-01f7831aab44, 7668ad9a-c12d-40a5-8370-c681c87e4786, 8028e5fc-81b7-45a0-8347-61e0e17e20c4</td></tr><tr><th>steps</th> <td>sentiment</td></tr></table>

## Congratulations!

You have now 

* Successfully trained a model
* Converted your model and uploaded it to Wallaroo
* Created and deployed a simple single-step pipeline
* Successfully send data to your pipeline for inference

In the next notebook, you will look at two different ways to evaluate your model against the real world environment.

