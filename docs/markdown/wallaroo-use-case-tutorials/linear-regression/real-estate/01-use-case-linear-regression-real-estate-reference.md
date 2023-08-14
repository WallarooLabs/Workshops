# Tutorial Notebook 1: Build and Deploy a Model

For this tutorial, let's pretend that you work for a real estate company wants to estimate the likely value of their listings portfolio. You have developed a model to predict the sale price of properties that the company has listed, based on data collected in the company's listings database.

In this set of exercises, you will build a model to predict house sale prices, and deploy it to Wallaroo.

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

Use the house price data `seattle_housing.csv` in the `data` subdirectory to build a model to predict the sales price of homes based on the features in the data set. 

At the end of the exercise, you should have a notebook and possibly other artifacts to produce a model for predicting house prices. For the purposes of the exercise, please use a framework that can be converted to ONNX, such as scikit-learn or XGBoost.

For assistance converting a model to ONNX, see the [Wallaroo Model Conversion Tutorials](https://docs.wallaroo.ai/wallaroo-tutorials/wallaroo-tutorials-conversion-tutorials/) for some examples.

**NOTE**

If you prefer to shortcut this step, you can use one of the pre-trained model pickle files in the `models` subdirectory.

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

suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
workspace_name = f"tutorial-workspace{suffix}"

workspace = get_workspace(workspace_name)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

# optionally, examine your current workspace
wl.get_current_workspace()
```

    {'name': 'tutorial-workspace-jch', 'id': 19, 'archived': False, 'created_by': '0a36fba2-ad42-441b-9a8c-bac8c68d13fa', 'created_at': '2023-08-03T19:34:42.324336+00:00', 'models': [], 'pipelines': []}

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

model = wl.upload_model('tutorial-model', '../models/gbr_model.onnx', framework=Framework.ONNX)

pipeline = wl.build_pipeline("tutorialpipeline-jch").add_model_step(model)

pipeline.deploy()

```

<table><tr><th>name</th> <td>tutorialpipeline-jch</td></tr><tr><th>created</th> <td>2023-08-03 19:36:31.732163+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-03 19:36:32.726371+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>d1d0cafe-78a9-4193-84af-cec1b3ed608b, 70438418-4802-4ced-a295-65ef78706fd4</td></tr><tr><th>steps</th> <td>tutorial-model</td></tr></table>

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

### Converting from tabular format

If your input data is in a standard tabular format (like the `test_data.csv` example data in the `data` directory), then you need to convert to pandas record format to send the data to your pipeline.  See the pandas DataFrame documentation for methods on how to import a CSV file to a DataFrame.

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

df_from_csv = pd.read_csv('../data/test_data.csv')

singleton = get_singleton(df_from_csv, 0)
display(singleton)

single_result = pipeline.infer(singleton)
display(single_result)

multiple_batch = get_batch(df_from_csv, nrows=5)
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
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-03 19:36:50.669</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[704901.9]</td>
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
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-03 19:36:51.057</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[704901.9]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-08-03 19:36:51.057</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[695994.44]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-08-03 19:36:51.057</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[416164.8]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-08-03 19:36:51.057</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[655277.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-08-03 19:36:51.057</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[426854.66]</td>
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

<table><tr><th>name</th> <td>tutorialpipeline-jch</td></tr><tr><th>created</th> <td>2023-08-03 19:36:31.732163+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-03 19:36:32.726371+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>d1d0cafe-78a9-4193-84af-cec1b3ed608b, 70438418-4802-4ced-a295-65ef78706fd4</td></tr><tr><th>steps</th> <td>tutorial-model</td></tr></table>

## Congratulations!

You have now 

* Successfully trained a model
* Converted your model and uploaded it to Wallaroo
* Created and deployed a simple single-step pipeline
* Successfully send data to your pipeline for inference

In the next notebook, you will look at two different ways to evaluate your model against the real world environment.

