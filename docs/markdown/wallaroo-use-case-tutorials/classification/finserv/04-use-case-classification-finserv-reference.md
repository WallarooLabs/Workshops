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

workspace_name = "classification-finserv-jch"

workspace = get_workspace(workspace_name)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

# optionally, examine your current workspace
wl.get_current_workspace()

```

    {'name': 'classification-finserv-jch', 'id': 21, 'archived': False, 'created_by': '0a36fba2-ad42-441b-9a8c-bac8c68d13fa', 'created_at': '2023-08-07T16:26:26.779098+00:00', 'models': [{'name': 'ccfraud-model-keras', 'versions': 2, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 7, 16, 28, 46, 566311, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 7, 16, 26, 36, 806125, tzinfo=tzutc())}, {'name': 'ccfraud-model-xgboost', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 7, 17, 20, 51, 978426, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 7, 17, 20, 51, 978426, tzinfo=tzutc())}], 'pipelines': [{'name': 'finserv-ccfraud', 'create_time': datetime.datetime(2023, 8, 7, 16, 26, 37, 485326, tzinfo=tzutc()), 'definition': '[]'}]}

## Monitoring for Drift: Shift Happens. 

In machine learning, you use data and known answers to train a model to make predictions for new previously unseen data. You do this with the assumption that the future unseen data will be similar to the data used during training: the future will look somewhat like the past.
But the conditions that existed when a model was created, trained and tested can change over time, due to various factors.

A good model should be robust to some amount of change in the environment; however, if the environment changes too much, your models may no longer be making the correct decisions. This situation is known as concept drift; too much drift can obsolete your models, requiring periodic retraining.

Let's consider the example we've been working on: credit card fraud prediction.  You may start to notice a large number of transactions that that aren't coming across as fraudulent compared to historical baselines.

Such a change could be due to many factors: a change in interest rates; the appearance or disappearance of major sources of employment; changes in travel patterns. Whatever the cause, detecting such a change quickly is crucial, so that the business can react quickly in the appropriate manner, whether that means simply retraining the model on fresher data, or a pivot in business strategy.

In Wallaroo you can monitor your housing model for signs of drift through the model monitoring and insight capability called Assays. Assays help you track changes in the environment that your model operates within, which can affect the model’s outcome. It does this by tracking the model’s predictions and/or the data coming into the model against an **established baseline**. If the distribution of monitored values in the current observation window differs too much from the baseline distribution, the assay will flag it. The figure below shows an example of a running scheduled assay.

{{<figure src="https://docs.wallaroo.ai/images/current/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_35_0.png" width="800" label="">}}

**Figure:** A daily assay that's been running for a month. The dots represent the difference between the distribution of values in the daily observation window, and the baseline. When that difference exceeds the specified threshold (indicated by a red dot) an alert is set.

This next set of exercises will walk you through setting up an assay to monitor the predictions of your house price model, in order to detect drift.

### NOTE

An assay is a monitoring process that typically runs over an extended, ongoing period of time. For example, one might set up an assay that every day monitors the previous 24 hours' worth of predictions and compares it to a baseline. For the purposes of these exercises, we'll be compressing processes what normally would take hours or days into minutes.

<hr/>

#### Exercise Prep: Create some datasets for demonstrating assays

Because assays are designed to detect changes in distributions, let's try to set up data with different distributions to test with. Take your financial transaction data and create two sets: a set with lower prices, and a set with higher prices. You can split however you choose.

The idea is we will pretend that the set of transactions less likely to trigger fraud warnings versus a set that is known to have a high likelihood of being evaluated as fraudulent.

* If you are using the pre-provided models to do these exercises, you can use the provided data sets `cc_data_low_confidence.df.json` and `cc_data_high_confidence.df.json`.  This is to establish our baseline as a set of known values, so the higher prices will trigger our assay alerts.

```python
low_fraud_data = pd.read_json('../data/cc_data_low_confidence.df.json')
high_fraud_data = pd.read_json('../data/cc_data_high_confidence.df.json')
```

Note that the data in these files are already in the form expected by the models, so you don't need to use the `get_singleton` or `get_batch` convenience functions to infer.

At the end of this exercise, you should have two sets of data to demonstrate assays. In the discussion below, we'll refer to these sets as `low_fraud_data` and `high_fraud_data`.

See [Wallaroo SDK Essentials Guide: Assays Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-assays/) for more details.

```python
# let's start with this - get the 10k of inferences.
# then split them into those < 0.25, and those > 0.75

```

```python
# blank spot to split or download data

low_fraud_data = pd.read_json('../data/cc_data_low_confidence.df.json')
high_fraud_data = pd.read_json('../data/cc_data_high_confidence.df.json')

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

pipeline = get_pipeline('finserv-ccfraud')

ccfraud_keras_model = get_model('ccfraud-model-keras')

pipeline.deploy()

```

<table><tr><th>name</th> <td>finserv-ccfraud</td></tr><tr><th>created</th> <td>2023-08-07 16:26:37.485326+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-07 17:55:48.136182+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>af6809a9-088f-47e5-88a4-2005599fde7a, 095ef880-06f0-45d5-bfef-ac4e11d2b06c, f9004ecd-7a8c-4963-bebc-e40d1cde357c, 2513faec-2385-4733-a42d-56a8ae38761a, 10e495af-9b23-4973-913e-4cdebca73461, 782b9edc-d42a-4984-90ca-3bdf35922b87, e80d21a4-21c2-46b8-85d4-6652bbac9506, 8ff48a62-fd9d-43f5-ba3f-11a7f1bcc474, 36faf126-dac5-419d-b0b1-7d7b698b587e, 230d585a-52db-476d-ab28-7b4baed9d023, 192f92e9-9a97-4339-8c1d-f89541ff2cef, 5d2d9c84-13c2-4e35-a41f-ec3c4e8d297b, 41927bef-d8fb-49ee-914e-d106ffc304b3</td></tr><tr><th>steps</th> <td>ccfraud-model-keras</td></tr></table>

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

pipeline.infer(low_fraud_data)

time.sleep(5)

baseline_end = datetime.datetime.now()

assay_builder = ( wl.build_assay('sample finserv assay', pipeline, model_name, 
                     baseline_start, baseline_end)
                    .add_iopath("output dense_1 0") )

assay_builder.baseline_histogram()

```

    Warning: There are more logs available. Please set a larger limit or request a file using export_logs.

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
      <td>2023-08-07 17:11:21.172</td>
      <td>[[0.5817662108, 0.097881551, 0.1546819424, 0.4754101949, -0.1978862306, -0.4504344854, 0.0166540447, -0.0256070551, 0.0920561602, -0.2783917153, 0.0593299441, -0.0196585416, -0.4225083157, -0.1217538877, 1.5473094894, 0.2391622864, 0.3553974881, -0.7685165301, -0.7000849355, -0.1190043285, -0.3450517133, -1.1065114108, 0.2523411195, 0.0209441826, 0.2199267436, 0.2540689265, -0.0450225094, 0.1086773898, 0.2547179311]]</td>
      <td>[0.0010916889]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

    'ccfraud-model-keras'

    
{{<figure src="/images/2023.2.1/wallaroo-use-case-tutorials/classification/finserv/04-use-case-classification-finserv-reference_files/04-use-case-classification-finserv-reference_11_3.png" width="800" label="png">}}
    

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
    pipeline.infer(low_fraud_data.sample(2*nsample, replace=True).reset_index(drop=True))
    time.sleep(25)
    
# run a mix
for x in range(3):
    pipeline.infer(low_fraud_data.sample(nsample, replace=True).reset_index(drop=True))
    pipeline.infer(high_fraud_data.sample(nsample, replace=True).reset_index(drop=True))
    time.sleep(25)
    
# high price houses dominate the sample
for x in range(3):
    pipeline.infer(high_fraud_data.sample(2*nsample, replace=True).reset_index(drop=True))
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
    pipeline.infer(df.sample(2*nsample, replace=True).reset_index(drop=True))
    time.sleep(25)
    
# run a mix
for x in range(3):
    pipeline.infer(low_fraud_data.sample(nsample, replace=True).reset_index(drop=True))
    pipeline.infer(high_fraud_data.sample(nsample, replace=True).reset_index(drop=True))
    time.sleep(25)
    
# high price houses dominate the sample
for x in range(3):
    pipeline.infer(high_fraud_data.sample(2*nsample, replace=True).reset_index(drop=True))
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

    
{{<figure src="/images/2023.2.1/wallaroo-use-case-tutorials/classification/finserv/04-use-case-classification-finserv-reference_files/04-use-case-classification-finserv-reference_15_0.png" width="800" label="png">}}
    

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
In this tutorial you have:

* Deployed a single step financial transaction classification pipeline and sent data to it.
* Compared two financial transaction classification models in an A/B test
* Compared two financial transaction classification models in a shadow deployment.
* Swapped the "winner" of the comparisons into the financial transaction classification pipeline.
* Set validation rules on the pipeline.
* Set up an assay on the pipeline to monitor for drift in its predictions.

Great job! 

### Cleaning up.

Now that the tutorial is complete, don't forget to undeploy your pipeline to free up the resources.

```python
# blank space to undeploy your pipeline

pipeline.undeploy()

```

<table><tr><th>name</th> <td>finserv-ccfraud</td></tr><tr><th>created</th> <td>2023-08-07 16:26:37.485326+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-07 18:11:00.015882+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>405d7b12-309e-44a1-968c-744f765b42a2, a665ba59-bc76-4a87-a836-4dd65da8b115, af6809a9-088f-47e5-88a4-2005599fde7a, 095ef880-06f0-45d5-bfef-ac4e11d2b06c, f9004ecd-7a8c-4963-bebc-e40d1cde357c, 2513faec-2385-4733-a42d-56a8ae38761a, 10e495af-9b23-4973-913e-4cdebca73461, 782b9edc-d42a-4984-90ca-3bdf35922b87, e80d21a4-21c2-46b8-85d4-6652bbac9506, 8ff48a62-fd9d-43f5-ba3f-11a7f1bcc474, 36faf126-dac5-419d-b0b1-7d7b698b587e, 230d585a-52db-476d-ab28-7b4baed9d023, 192f92e9-9a97-4339-8c1d-f89541ff2cef, 5d2d9c84-13c2-4e35-a41f-ec3c4e8d297b, 41927bef-d8fb-49ee-914e-d106ffc304b3</td></tr><tr><th>steps</th> <td>ccfraud-model-keras</td></tr></table>

