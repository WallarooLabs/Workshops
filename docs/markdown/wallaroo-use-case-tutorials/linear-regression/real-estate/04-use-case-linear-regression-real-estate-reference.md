# Tutorial Notebook 4: Observability Part 2 - Drift Detection

In the previous notebook you learned how to add simple validation rules to a pipeline, to monitor whether outputs (or inputs) stray out of some expected range. In this notebook, you will monitor the *distribution* of the pipeline's predictions to see if the model, or the environment that it runs it, has changed.

## Preliminaries

In the blocks below we will preload some required libraries; we will also redefine some of the convenience functions that you saw in the previous notebooks.

After that, you should log into Wallaroo and set your working environment to the workspace that you created for this tutorial. Please refer to Notebook 1 to refresh yourself on how to log in and set your working environment to the appropriate workspace.

```python
# preload needed libraries 

import wallaroo
from wallaroo.object import EntityNotFoundError

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
    model = [m.versions()[0] for m in modellist if m.name() == mname]
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

workspace_name = "tutorial-workspace-jch"

workspace = get_workspace(workspace_name)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

# optionally, examine your current workspace
wl.get_current_workspace()

```

    {'name': 'tutorial-workspace-jch', 'id': 9, 'archived': False, 'created_by': 'c3a45eb6-37ff-4020-8d59-7166c3e153d0', 'created_at': '2023-07-20T14:35:12.650973+00:00', 'models': [{'name': 'tutorial-model', 'versions': 3, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 7, 20, 16, 3, 10, 617651, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 7, 20, 14, 43, 43, 886638, tzinfo=tzutc())}, {'name': 'challenger-model', 'versions': 3, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 7, 20, 16, 35, 29, 301552, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 7, 20, 16, 14, 1, 624750, tzinfo=tzutc())}], 'pipelines': [{'name': 'tutorialpipeline-jch', 'create_time': datetime.datetime(2023, 7, 20, 14, 43, 44, 606725, tzinfo=tzutc()), 'definition': '[]'}]}

## Monitoring for Drift: Shift Happens. 

In machine learning, you use data and known answers to train a model to make predictions for new previously unseen data. You do this with the assumption that the future unseen data will be similar to the data used during training: the future will look somewhat like the past.
But the conditions that existed when a model was created, trained and tested can change over time, due to various factors.

A good model should be robust to some amount of change in the environment; however, if the environment changes too much, your models may no longer be making the correct decisions. This situation is known as concept drift; too much drift can obsolete your models, requiring periodic retraining.

Let's consider the example we've been working on: home sale price prediction. You may notice over time that there has been a change in the mix of properties in the listings portfolio: for example a dramatic increase or decrease in expensive properties (or more precisely, properties that the model thinks are expensive)

Such a change could be due to many factors: a change in interest rates; the appearance or disappearance of major sources of employment; new housing developments opening up in the area. Whatever the cause, detecting such a change quickly is crucial, so that the business can react quickly in the appropriate manner, whether that means simply retraining the model on fresher data, or a pivot in business strategy.

In Wallaroo you can monitor your housing model for signs of drift through the model monitoring and insight capability called Assays. Assays help you track changes in the environment that your model operates within, which can affect the model’s outcome. It does this by tracking the model’s predictions and/or the data coming into the model against an **established baseline**. If the distribution of monitored values in the current observation window differs too much from the baseline distribution, the assay will flag it. The figure below shows an example of a running scheduled assay.

{{<figure src="https://docs.wallaroo.ai/images/current/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_35_0.png" width="800" label="">}}

**Figure:** A daily assay that's been running for a month. The dots represent the difference between the distribution of values in the daily observation window, and the baseline. When that difference exceeds the specified threshold (indicated by a red dot) an alert is set.

This next set of exercises will walk you through setting up an assay to monitor the predictions of your house price model, in order to detect drift.

### NOTE

An assay is a monitoring process that typically runs over an extended, ongoing period of time. For example, one might set up an assay that every day monitors the previous 24 hours' worth of predictions and compares it to a baseline. For the purposes of these exercises, we'll be compressing processes what normally would take hours or days into minutes.

<hr/>

#### Exercise Prep: Create some datasets for demonstrating assays

Because assays are designed to detect changes in distributions, let's try to set up data with different distributions to test with. Take your houseprice data and create two sets: a set with lower prices, and a set with higher prices. You can split however you choose.

The idea is we will pretend that the set of lower priced houses represent the "typical" mix of houses in the housing portfolio at the time you set your baseline; you will introduce the higher priced houses later, to represent an environmental change when more expensive houses suddenly enter the market.

* If you are using the pre-provided models to do these exercises, you can use the provided data sets `lowprice.df.json` and `highprice.df.json`.  This is to establish our baseline as a set of known values, so the higher prices will trigger our assay alerts.

```python
lowprice_data = pd.read_json('lowprice.df.json')
highprice_data = pd.read_json('highprice.df.json')
```

Note that the data in these files are already in the form expected by the models, so you don't need to use the `get_singleton` or `get_batch` convenience functions to infer.

At the end of this exercise, you should have two sets of data to demonstrate assays. In the discussion below, we'll refer to these sets as `lowprice_data` and `highprice_data`.

```python
# blank spot to split or download data

lowprice_data = pd.read_json('./data/lowprice.df.json')
highprice_data = pd.read_json('./data/highprice.df.json')

```

We will use this data to set up some "historical data" in the house price prediction pipeline that you build in the assay exercises.

## Setting up a baseline for the assay

In order to know whether the distribution of your model's predictions have changed, you need a baseline to compare them to. This baseline should represent how you expect the model to behave at the time it was trained. This might be approximated by the distribution of the model's predictions over some "typical" period of time. For example, we might collect the predictions of our model over the first few days after it's been deployed. For these exercises, we'll compress that to a few minutes. Currently, to set up a wallaroo assay the pipeline must have been running for some period of time, and the assumption is that this period of time is "typical", and that the distributions of the inputs and the outputs of the model during this period of time are "typical."

#### Exercise Prep: Run some inferences and set some time stamps

Here, we simulate having a pipeline that's been running for a long enough period of time to set up an assay.

To send enough data through the pipeline to create assays, you execute something like the following code (using the appropriate names for your pipelines and models). Note that this step will take a little while, because of  the `sleep` interval.

You will need the timestamps `baseline_start`, and `baseline_end`, for the next exercises.

```python
# get your pipeline (in this example named "mypipeline")

pipeline = get_pipeline("mypipeline")
pipeline.deploy()

## Run some baseline data
# Where the baseline data will start
baseline_start = datetime.datetime.now()

# the number of samples we'll use for the baseline
nsample = 500

# Wait 30 seconds to set this data apart from the rest
# then send the data in batch
time.sleep(30)

# get a sample
lowprice_data_sample = lowprice_data.sample(nsample, replace=True).reset_index(drop=True)
pipeline.infer(lowprice_data_sample)

# Set the baseline end
baseline_end = datetime.datetime.now()

```

```python
# blank space to get pipeline and set up baseline data

pipeline = get_pipeline('tutorialpipeline-jch')

control_model = get_model('tutorial-model')

pipeline.deploy()

```

<table><tr><th>name</th> <td>tutorialpipeline-jch</td></tr><tr><th>created</th> <td>2023-07-20 14:43:44.606725+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-20 16:58:21.001748+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>fca4b15b-ca41-4865-b7ba-b2ce7b0e0d9d, e6c9d6be-11ee-4dc9-b065-fbb2ae82d262, f8c21055-9732-4a89-9fd7-face19785200, acbbf366-711b-4904-b5d7-2459015f2d70, 3da3ade7-52d5-4fe0-8609-027322769f39, f27623ad-90dc-4161-b826-65084d0f198d, d886d504-d7fd-42b7-9b95-7eadfa6cafbe, 117dac7b-78fe-449d-99c7-6233c100874c, 67f9bfe6-7f03-4087-a6db-46896729c78f, aa7f65e4-0edd-4cb7-8a81-7c5f46c323a7, 4edac46e-22eb-46c0-8ee7-e2cf68181958, ed2b8ae9-eb77-46b5-9985-aa520be6c6b7, 83b4129b-3271-4e5a-aa5c-203b8b95e674, d7c7b6b1-04d5-4d20-a85b-39159cb79ede, 0f5b2df6-6498-4ace-a739-7c585284d6c6, da128b7e-26f5-4b9d-a58e-b244735843bd, 89094eea-ef47-4d4c-bf33-2829b2aeaf64</td></tr><tr><th>steps</th> <td>tutorial-model</td></tr></table>

Before setting up an assay on this pipeline's output, we may want to look at the distribution of the predictions over our selected baseline period. To do that, we'll create an *assay_builder* that specifies the pipeline, the model in the pipeline, and the baseline period.. We'll also specify that we want to look at the output of the model, which in the example code is named `variable`, and would appear as `out.variable` in the logs.

```python
# print out one of the logs to get the name of the output variable
display(pipeline.logs(limit=1))

# get the model name directly off the pipeline (you could just hard code this, if you know the name)

model_name = pipeline.model_configs()[0].model().name()

assay_builder = ( wl.build_assay(assay_name, pipeline, model_name, 
                     baseline_start, baseline_end)
                    .add_iopath("output variable 0") ) # specify that we are looking at the first output of the output variable "variable"
```

where `baseline_start` and `baseline_end` are the beginning and end of the baseline periods as `datetime.datetime` objects. 

You can then examine the distribution of `variable` over the baseline period:

```python
assay_builder.baseline_histogram()
```

<hr/>

#### Exercise: Create an assay builder and set a baseline

Create an assay builder to monitor the output of your house price pipeline. The baseline period should be from `baseline_start` to `baseline_end`. 

* You will need to know the name of your output variable, and the name of the model in the pipeline.

Examine the baseline distribution.

```python
## Blank space to create an assay builder and examine the baseline distribution

display(pipeline.logs(limit=1))
model_name = pipeline.model_configs()[0].model().name()
display(model_name)

import datetime
import time
baseline_start = datetime.datetime.now()
time.sleep(5)

pipeline.infer(lowprice_data)

time.sleep(5)

baseline_end = datetime.datetime.now()

assay_builder = ( wl.build_assay('sample assay', pipeline, model_name, 
                     baseline_start, baseline_end)
                    .add_iopath("output variable 0") )

assay_builder.baseline_histogram()

```

    Warning: There are more logs available. Please set a larger limit or request a file using export_logs.

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
      <td>2023-07-20 16:30:45.959</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

    'tutorial-model'

    
{{<figure src="/images/2023.2.1/wallaroo-use-case-tutorials/linear-regression/real-estate/04-use-case-linear-regression-real-estate-reference_files/04-use-case-linear-regression-real-estate-reference_10_3.png" width="800" label="png">}}
    

An assay should detect if the distribution of model predictions changes from the above distribution over regularly sampled observation windows. This is called *drift*.

To show drift, we'll run more data through the pipeline -- first some data drawn from the same distribution as the baseline (`lowprice_data`). Then, we will gradually introduce more data from a different distribution (`highprice_data`). We should see the difference between the baseline distribution and the distribution in the observation window increase.

To set up the data, you should do something like the below. It will take a while to run, because of all the `sleep` intervals.

You will need the `assay_window_end` for a later exercise.

**IMPORTANT NOTE**:  To generate the data for the assay, this process may take 4-5 minutes.  Because the shortest period of time for an assay window is 1 minute, the intervals of inference data are spaced to fall within that time period.

```python
# Set the start for our assay window period.
assay_window_start = datetime.datetime.now()

# Run a set of house values, spread across a "longer" period of time

# run "typical" data
for x in range(4):
    pipeline.infer(lowprice_data.sample(2*nsample, replace=True).reset_index(drop=True))
    time.sleep(25)
    
# run a mix
for x in range(3):
    pipeline.infer(lowprice_data.sample(nsample, replace=True).reset_index(drop=True))
    pipeline.infer(highprice_data.sample(nsample, replace=True).reset_index(drop=True))
    time.sleep(25)
    
# high price houses dominate the sample
for x in range(3):
    pipeline.infer(highprice_data.sample(2*nsample, replace=True).reset_index(drop=True))
    time.sleep(25)

# End our assay window period
assay_window_end = datetime.datetime.now()
```

<hr/>

#### Exercise Prep: Run some inferences and set some time stamps

Run more data through the pipeline, manifesting a drift, like the example above. It may around 10 minutes depending on how you stagger the inferences.

```python
## Blank space to run more data

assay_window_start = datetime.datetime.now()

# Run a set of house values, spread across a "longer" period of time

nsample = 500

# run "typical" data
for x in range(4):
    pipeline.infer(lowprice_data.sample(2*nsample, replace=True).reset_index(drop=True))
    time.sleep(25)
    
# run a mix
for x in range(3):
    pipeline.infer(lowprice_data.sample(nsample, replace=True).reset_index(drop=True))
    pipeline.infer(highprice_data.sample(nsample, replace=True).reset_index(drop=True))
    time.sleep(25)
    
# high price houses dominate the sample
for x in range(3):
    pipeline.infer(highprice_data.sample(2*nsample, replace=True).reset_index(drop=True))
    time.sleep(25)

# End our assay window period
assay_window_end = datetime.datetime.now()

```

## Defining the Assay Parameters

Now we're finally ready to set up an assay!

### The Observation Window

Once a baseline period has been established, you must define the window of observations that will be compared to the baseline. For instance, you might want to set up an assay that runs *every 12 hours*, collects the *previous 24 hours' predictions* and compares the distribution of predictions within that 24 hour window to the baseline. To set such a comparison up would look like this:

```python
assay_builder.window_builder().add_width(hours=24).add_interval(hours=12)
```

In other words **_width_** is the width of the observation window, and **_interval_** is how often an assay (comparison) is run. The default value of *width* is 24 hours; the default value of *interval* is to set it equal to *width*. The units can be specified in one of: `minutes`, `hours`, `days`, `weeks`.

### The Comparison Threshold
Given an observation window and a baseline distribution, an assay computes the distribution of predictions in the observation window. It then calculates the "difference" (or "distance") between the observed distribution and the baseline distribution. For the assay's default distance metric (which we will use here), a good starting threshold is 0.1. Since a different value may work best for a specific situation, you can try interactive assay runs on historical data to find a good threshold, as we do in these exercises.

To set the assay threshold for the assays to 0.1:

```python
assay_builder.add_alert_threshold(0.1)
```

### Running an Assay on Historical Data

In this exercise, you will build an **interactive assay** over historical data. To do this, you need an end time (`endtime`). 

Depending on the historical history, the window and interval may need adjusting.  If using the previously generated information, an interval window as short as 1 minute may be useful.

Assuming you have an assay builder with the appropriate window parameters and threshold set, you can do an interactive run and look at the results would look like this.

```python
# set the end of the interactive run
assay_builder.add_run_until(endtime)

# set the window

assay_builder.window_builder().add_width(hours=24).add_interval(hours=12)

assay_results = assay_builder.build().interactive_run()
df = assay_results.to_dataframe() # to return the results as a table
assay_results.chart_scores() # to plot the run
```

<hr/>

#### Exercise: Create an interactive assay

Use the assay_builder you created in the previous exercise to set up an interactive assay. 
* The assay should run every minute, on a window that is a minute wide. 
* Set the alert threshold to 0.1.  
* You can use `assay_window_end` (or a later timestamp) as the end of the interactive run.

Examine the assay results. Do you see any drift?

To try other ways of examining the assay results, see the ["Interactive Assay Runs" section of the Model Insights tutorial](https://docs.wallaroo.ai/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights/#interactive-assay-runs).

```python
# blank space for setting assay parameters, creating and examining an interactive assay

# set the end of the interactive run
assay_builder.add_run_until(assay_window_end)

# doing minutes to get our previous values in
assay_builder.window_builder().add_width(minutes=1).add_interval(minutes=1)
assay_builder.add_alert_threshold(0.1)
assay_results = assay_builder.build().interactive_run()
df = assay_results.to_dataframe() # to return the results as a table
assay_results.chart_scores() # to plot the run

```

    
{{<figure src="/images/2023.2.1/wallaroo-use-case-tutorials/linear-regression/real-estate/04-use-case-linear-regression-real-estate-reference_files/04-use-case-linear-regression-real-estate-reference_14_0.png" width="800" label="png">}}
    

### Scheduling an Assay to run on ongoing data

(We won't be doing an exercise here, this is for future reference).

Once you are satisfied with the parameters you have set, you can schedule an assay to run regularly .

```python
# create a fresh assay builder with the correct parameters
assay_builder = ( wl.build_assay(assay_name, pipeline, model_name, 
                     baseline_start, baseline_end)
                    .add_iopath("output variable 0") )

# this assay runs every 24 hours on a 24 hour window
assay_builder.window_builder().add_width(hours=24)
assay_builder.add_alert_threshold(0.1)

# now schedule the assay
assay_id = assay_builder.upload()
```

You can use the assay id later to get the assay results.

You have now walked through setting up a basic assay and running it over historical data.

## Congratulations!
In this tutorial you have
* Deployed a single step house price prediction pipeline and sent data to it.
* Compared two house price prediction models in an A/B test
* Compared two house price prediction models in a shadow deployment.
* Swapped the "winner" of the comparisons into the house price prediction pipeline.
* Set validation rules on the pipeline.
* Set up an assay on the pipeline to monitor for drift in its predictions.

Great job! 

### Cleaning up.

Now that the tutorial is complete, don't forget to undeploy your pipeline to free up the resources.

```python
# blank space to undeploy your pipeline

pipeline.undeploy()

```
