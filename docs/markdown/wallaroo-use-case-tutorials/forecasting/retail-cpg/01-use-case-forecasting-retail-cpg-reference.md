# Statsmodel Forecast Tutorial Notebook 1: Build and Deploy a Model

For this tutorial, let's pretend that you work for a bike rental company.  You have developed a model to predict the number of rentals for the week following a given date, based on data collected in the company's rental history database.

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

import pyarrow as pa

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

Wallaroo natively supports models in the ONNX, Python based models, Tensorflow frameworks, and other frameworks via containerization. For this exercise, we assume that you have a model that can be converted to the ONNX framework. The first steps to deploying in Wallaroo, then, is to convert your model to ONNX, and to add some extra functions to your processing modules so Wallaroo can call them.

<hr/>

#### Exercise: Convert your Model to ONNX

Take the model that you created in the previous exercises, and convert it to ONNX supported by Wallaroo. If you need help, see the [Wallaroo ONNX conversion tips](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-onnx/#onnx-conversion-tips).  The model can also be deployed to Wallaroo if is supported by Wallaroo.  See the [Wallaroo SDK Essentials Guide: Model Uploads and Registrations](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/)

At the end of this exercise, you should have your model as a standalone artifact, for example, a file called `model.onnx`.

**NOTE**

If you prefer to shortcut this exercise, you can use one of the Python models in the `models` directory.

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
## Blank spot to log in

wl = wallaroo.Client()
```

```python
## Blank spot to connect to the workspace

import string
import random

suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

workspace_name = f'forecast-model-tutorial'

workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

```

    {'name': 'forecast-model-tutorialjohn', 'id': 16, 'archived': False, 'created_by': '0a36fba2-ad42-441b-9a8c-bac8c68d13fa', 'created_at': '2023-08-02T15:50:52.816795+00:00', 'models': [{'name': 'forecast-control-model', 'versions': 2, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 2, 18, 16, 45, 620061, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 2, 15, 50, 54, 223186, tzinfo=tzutc())}, {'name': 'forecast-challenger01-model', 'versions': 2, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 2, 18, 16, 46, 633644, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 2, 15, 50, 55, 208179, tzinfo=tzutc())}, {'name': 'forecast-challenger02-model', 'versions': 2, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 2, 18, 16, 47, 740983, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 2, 15, 50, 56, 291043, tzinfo=tzutc())}], 'pipelines': [{'name': 'forecast-tutorial-pipeline', 'create_time': datetime.datetime(2023, 8, 2, 15, 50, 59, 480547, tzinfo=tzutc()), 'definition': '[]'}]}

## Deploy a Simple Single-Step Pipeline

Once your model is in the ONNX format, and you have a workspace to work in, you can easily upload your model to Wallaroo's production platform with just a few lines of code. For example, if you have a model called `model.onnx`, and you wish to upload it to Wallaroo with the name `mymodel`, then upload the model as follows (once you are in the appropriate workspace):

```python

my_model = wl.upload_model("mymodel", "model.onnx", framework=Framework.ONNX).configure()
```

See [Wallaroo SDK Essentials Guide: Model Uploads and Registrations: ONNX](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-onnx/) for full details.

The function `upload_model()` returns a handle to the uploaded model that you will continue to work with in the SDK.

### Upload Python Models

If you choose to use one of our sample models, our forecast example uses a Python library ARIMA statsmodel.  The models are all available in the `./models` directory.

To upload a Python model to Wallaroo, the following is required:

* The name of the model:  What to designate the model name once uploaded to Wallaroo.
* Path of the Python script:  This specifies the file location.  For example: `./models/my-python-script.py`
* The input and output schemas:  These inform Wallaroo how inference request data will be formatted, and the shape of the data going out from the model.  The schema is in Apache Arrow schema format, aka `pyarrow.lib.Schema`..  These are required to be in the For example, if the inference inputs are a pandas DataFrame with the following shape:

| &nbsp; | tensor |
|---|---|
| 0 | [15.5, 17.2, 35.724, 0.37894 ]

With the following output:

[
    "forecast": [235, 135, 175],
    "average_forecast": [181.66]
]

In this case, the input schema is represented as:

```python
import pyarrow as py

input_schema = pa.schema([
    pa.field('tensor', pa.list_(pa.float64()))
])

output_schema = pa.schema([
    pa.field('forecast', pa.list_(pa.int64())),
    pa.field('average_forecast', pa.list_(pa.float64()))
])
```

Python models are uploaded to Wallaroo with the `upload_model` method, with the additional `configure` method specifying the `python` framework with the inputs and outputs.  For example:

```python
sample_model = (wl.upload_model(control_model_name, 
                                 control_model_file, 
                                 framework=Framework.PYTHON)
                                 .configure("python", 
                                 input_schema=input_schema, 
                                 output_schema=output_schema)
                )
```

For more information about Python models, see [Wallaroo SDK Essentials Guide: Model Uploads and Registrations: Python Models](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-python/).

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

control_model_name = 'forecast-control-model'
control_model_file = '../models/forecast_standard.py'

# Holding on these for later
input_schema = pa.schema([
    pa.field('count', pa.list_(pa.int64()))
])

output_schema = pa.schema([
    pa.field('forecast', pa.list_(pa.int64())),
    pa.field('weekly_average', pa.list_(pa.float64()))
])

# upload the models

bike_day_model = (wl.upload_model(control_model_name, 
                                 control_model_file, 
                                 framework=Framework.PYTHON)
                                 .configure("python", 
                                 input_schema=input_schema, 
                                 output_schema=output_schema)
                )

# create the pipeline

pipeline_name = 'forecast-tutorial-pipeline'

pipeline = wl.build_pipeline(pipeline_name).clear().add_model_step(bike_day_model).deploy()

```

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

### Inference Input with Forecast Statsmodel

For our particular statsmodel, the input is stored in the file `day.csv`, which tracks bike rentals and conditions from 2011 through 2012.  The sample Python ARIMA Statsmodel used in these demonstrations has the following input and output schemas.

```python
input_schema = pa.schema([
    pa.field('count', pa.list_(pa.int64()))
])

output_schema = pa.schema([
    pa.field('forecast', pa.list_(pa.int64())),
    pa.field('weekly_average', pa.list_(pa.float64()))
])
```

### Converting from tabular format

If your input data is in a standard tabular format (like the `test_data.csv` example data in the `data` directory), then you need to convert to pandas record format to send the data to your pipeline.  See the pandas DataFrame documentation for methods on how to import a CSV file to a DataFrame.

To help with the following exercises, here are some convenience functions you might find useful for doing this conversion. These functions convert input data in standard tabular format (in a pandas DataFrame) to the pandas record format that the model expects.

`get_singleton` assumes that all of the information is on a single row.

`get_singleton_forecast` assumes that the data is composed in a single DataFrame, and then reformats it to be in a single row.  Our example ARIMA forecast model expects data in the input schema detailed above.  So a series of inputs such as:

| &nbsp; | count |
|---|---|
| 0 | 985 |
| 1 | 801 |
| 2 |1349 |
| 3 |1562 |
| 4 |1600 |
| 5 |1606 |
| 6 |1510 |
| 7 |959 |
| 8 |822 |
| 9 |1321 |

Would have to be reformatted as:

| &nbsp; | count |
|---|---|
| 0 | [985, 801, 1349, 1562, 1600, 1606, 1510, 959, 822, 1321] |

```python
# pull a single datum from a data frame 
# and convert it to the format the model expects
def get_singleton(df, i):
    singleton = df.iloc[i,:].to_numpy().tolist()
    sdict = {'tensor': [singleton]}
    return pd.DataFrame.from_dict(sdict)

def get_singleton_forecast(df, field):
    singleton = pd.DataFrame({field: [df[field].values.tolist()]})
    return singleton

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

print('''The command "singleton = get_singleton(df, 0)" converts
the first row of the data frame into the format that Wallaroo pipelines accept.
You could now get a prediction by: "toypipeline.infer(singleton)".
''')

sample_df = pd.DataFrame({"count": [1526, 
                                    1550, 
                                    1708, 
                                    1005, 
                                    1623]
                        })
display(sample_df)

inference_df = get_singleton_forecast(sample_df, 'count')
display(inference_df)
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

    The command "singleton = get_singleton(df, 0)" converts
    the first row of the data frame into the format that Wallaroo pipelines accept.
    You could now get a prediction by: "toypipeline.infer(singleton)".
    

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
      <td>1526</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1550</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1708</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1005</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1623</td>
    </tr>
  </tbody>
</table>

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
      <td>[1526, 1550, 1708, 1005, 1623]</td>
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

sample_count = pd.read_csv('../data/test_data.csv')
# sample_df = sample_count.loc[0:20, ['count']]
inference_df = get_singleton_forecast(sample_count.loc[0:20], 'count')
display(inference_df)

results = pipeline.infer(inference_df)
display(results)

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
      <td>[985, 801, 1349, 1562, 1600, 1606, 1510, 959, 822, 1321, 1263, 1162, 1406, 1421, 1248, 1204, 1000, 683, 1650, 1927, 1543]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.count</th>
      <th>out.forecast</th>
      <th>out.weekly_average</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-03 01:12:30.126</td>
      <td>[985, 801, 1349, 1562, 1600, 1606, 1510, 959, 822, 1321, 1263, 1162, 1406, 1421, 1248, 1204, 1000, 683, 1650, 1927, 1543]</td>
      <td>[1434, 1288, 1296, 1295, 1295, 1295, 1295]</td>
      <td>[1314.0]</td>
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

## Congratulations!

You have now 

* Successfully trained a model
* Converted your model and uploaded it to Wallaroo
* Created and deployed a simple single-step pipeline
* Successfully send data to your pipeline for inference

In the next notebook, you will look at two different ways to evaluate your model against the real world environment.

