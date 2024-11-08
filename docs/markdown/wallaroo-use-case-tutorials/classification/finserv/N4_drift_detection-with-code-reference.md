# Tutorial Notebook 4: Observability Part 2 - Drift Detection

In the previous notebook you learned how to add simple validation rules to a pipeline, to monitor whether outputs (or inputs) stray out of some expected range. In this notebook, you will monitor the *distribution* of the pipeline's predictions to see if the model, or the environment that it runs it, has changed.

## Preliminaries

In the blocks below we will preload some required libraries.

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

## Login to Wallaroo

Retrieve the previous workspace, model versions, and pipelines used in the previous notebook.

```python
## blank space to log in 

wl = wallaroo.Client()
# retrieve the previous workspace, model, and pipeline version

workspace_name = "tutorial-finserv-john"

workspace = wl.get_workspace(workspace_name)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

# optionally, examine your current workspace
wl.get_current_workspace()

model_name = 'classification-finserv-prime'

prime_model_version = wl.get_model(model_name)

pipeline_name = 'ccfraud-detector'

pipeline = wl.get_pipeline(pipeline_name)

display(workspace)
display(prime_model_version)
display(pipeline)

```

    {'name': 'tutorial-finserv-john', 'id': 11, 'archived': False, 'created_by': '94016008-4e0e-45bc-b2c6-d6f06236b4f5', 'created_at': '2024-09-05T16:18:31.258882+00:00', 'models': [{'name': 'classification-finserv-prime', 'versions': 3, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 9, 5, 16, 48, 18, 348992, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 9, 5, 16, 18, 42, 470017, tzinfo=tzutc())}, {'name': 'ccfraud-xgboost-version', 'versions': 3, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 9, 5, 16, 53, 34, 714217, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 9, 5, 16, 32, 28, 543534, tzinfo=tzutc())}], 'pipelines': [{'name': 'assay-demonstration-tutorial', 'create_time': datetime.datetime(2024, 9, 5, 16, 18, 46, 604658, tzinfo=tzutc()), 'definition': '[]'}, {'name': 'ccfraud-detector', 'create_time': datetime.datetime(2024, 9, 5, 16, 18, 43, 626892, tzinfo=tzutc()), 'definition': '[]'}]}

<table>
        <tr>
          <td>Name</td>
          <td>classification-finserv-prime</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>24761304-0cb3-40e7-9462-d2a454637152</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>keras_ccfraud.onnx</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507</td>
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
          <td>2024-05-Sep 16:48:18</td>
        </tr>
        <tr>
          <td>Workspace id</td>
          <td>11</td>
        </tr>
        <tr>
          <td>Workspace name</td>
          <td>tutorial-finserv-john</td>
        </tr>
      </table>

<table><tr><th>name</th> <td>ccfraud-detector</td></tr><tr><th>created</th> <td>2024-09-05 16:18:43.626892+00:00</td></tr><tr><th>last_updated</th> <td>2024-09-05 17:37:49.045951+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>workspace_id</th> <td>11</td></tr><tr><th>workspace_name</th> <td>tutorial-finserv-john</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>93b4dd36-5e02-440d-931c-80198d1ee48a, 252d1e6f-35a5-420a-9590-d08cd76943a7, cb026715-9ced-40bb-9108-333b88c9de64, e90c10e3-ab38-43a8-a315-fb4250c09b21, 583e7a6a-3a7a-4420-abd0-c91e346a874d, cbfc4951-4d2c-41cb-89c7-934ac5bd2cbf, 32ab9ef6-2ac4-4d92-a46e-5d5c286af48c, 410d43d4-e698-4747-bbd1-cb62afee258a, 9edcf6f6-660d-470d-b8f0-24f54a335e8f, cd63b4fa-6549-41b4-af8f-576b1f0ef8b3, ff2d47a5-47c8-4fcb-8709-e95b3d0d4340, 8e74290b-4cb6-43a5-9b27-8431643438fd</td></tr><tr><th>steps</th> <td>classification-finserv-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

Set up the pipeline with the single model step as was done in notebook 1, then deploy it.

```python
# deploy pipeline here

pipeline.clear()
pipeline.add_model_step(prime_model_version)

deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
pipeline.deploy(deployment_config=deploy_config)
```

<table><tr><th>name</th> <td>ccfraud-detector</td></tr><tr><th>created</th> <td>2024-09-05 16:18:43.626892+00:00</td></tr><tr><th>last_updated</th> <td>2024-09-05 17:41:14.081888+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>workspace_id</th> <td>11</td></tr><tr><th>workspace_name</th> <td>tutorial-finserv-john</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>9f9fc75d-9297-4de5-9962-1cd46e5006df, 93b4dd36-5e02-440d-931c-80198d1ee48a, 252d1e6f-35a5-420a-9590-d08cd76943a7, cb026715-9ced-40bb-9108-333b88c9de64, e90c10e3-ab38-43a8-a315-fb4250c09b21, 583e7a6a-3a7a-4420-abd0-c91e346a874d, cbfc4951-4d2c-41cb-89c7-934ac5bd2cbf, 32ab9ef6-2ac4-4d92-a46e-5d5c286af48c, 410d43d4-e698-4747-bbd1-cb62afee258a, 9edcf6f6-660d-470d-b8f0-24f54a335e8f, cd63b4fa-6549-41b4-af8f-576b1f0ef8b3, ff2d47a5-47c8-4fcb-8709-e95b3d0d4340, 8e74290b-4cb6-43a5-9b27-8431643438fd</td></tr><tr><th>steps</th> <td>classification-finserv-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

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

### Assay DataSet Exercise

Because assays are designed to detect changes in distributions, let's try to set up data with different distributions to test with. Take your houseprice data and create two sets: a set with lower prices, and a set with higher prices. You can split however you choose.

The idea is we will pretend that the set of lower priced houses represent the "typical" mix of houses in the housing portfolio at the time you set your baseline; you will introduce the higher priced houses later, to represent an environmental change when more expensive houses suddenly enter the market.

* If you are using the pre-provided models to do these exercises, you can use the provided data sets `lowprice.df.json` and `highprice.df.json`.  This is to establish our baseline as a set of known values, so the higher prices will trigger our assay alerts.

```python
lowprice_data = pd.read_json('./data/lowprice.df.json')
highprice_data = pd.read_json('./data/highprice.df.json')
```

Note that the data in these files are already in the form expected by the models, so you don't need to use the `get_singleton` or `get_batch` convenience functions to infer.

At the end of this exercise, you should have two sets of data to demonstrate assays. In the discussion below, we'll refer to these sets as `lowprice_data` and `highprice_data`.

```python
# blank spot to split or download data

lowprice_data = pd.read_json('../data/cc_data_high_confidence.df.json')
highprice_data = pd.read_json('../data/cc_data_low_confidence.df.json')

```

We will use this data to set up some "historical data" in the house price prediction pipeline that you build in the assay exercises.

## Set Assay Baseline

In order to know whether the distribution of your model's predictions have changed, you need a baseline to compare them to. This baseline should represent how you expect the model to behave at the time it was trained. This might be approximated by the distribution of the model's predictions over some "typical" period of time. For example, we might collect the predictions of our model over the first few days after it's been deployed. For these exercises, we'll compress that to a few minutes. Currently, to set up a wallaroo assay the pipeline must have been running for some period of time, and the assumption is that this period of time is "typical", and that the distributions of the inputs and the outputs of the model during this period of time are "typical."

### Set Assay Baseline Data Exercise

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

## Create Assay Builder and Set Baseline

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

### Create Assay Builder and Set Baseline Exercise

Create an assay builder to monitor the output of your house price pipeline. The baseline period should be from `baseline_start` to `baseline_end`. 

* You will need to know the name of your output variable, and the name of the model in the pipeline.

Here's an example.

```python
## Blank space to create an assay builder and examine the baseline distribution

import datetime
import time
baseline_start = datetime.datetime.now()
time.sleep(5)

pipeline.infer(lowprice_data)

time.sleep(5)

baseline_end = datetime.datetime.now()

assay_builder = wl.build_assay(assay_name="finserv sample assay john", 
                                          pipeline=pipeline, 
                                          iopath="output dense_1 0",
                                          baseline_start=baseline_start, 
                                          baseline_end=baseline_end)

assay_builder.baseline_histogram()
```

```python
## Blank space to create an assay builder and examine the baseline distribution

import datetime
import time
baseline_start = datetime.datetime.now()
time.sleep(5)

pipeline.infer(lowprice_data)

time.sleep(5)

baseline_end = datetime.datetime.now()

assay_builder = wl.build_assay(assay_name="finserv sample assay john", 
                                          pipeline=pipeline, 
                                          iopath="output dense_1 0",
                                          baseline_start=baseline_start, 
                                          baseline_end=baseline_end)

assay_builder.baseline_histogram()
```

    
{{<figure src="/images/2024.2/wallaroo-use-case-tutorials/classification/finserv/N4_drift_detection-with-code-reference_files/N4_drift_detection-with-code-reference_11_0.png" width="800" label="png">}}
    

## Assay Windows

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

## Assay Windows Exercise

Run more data through the pipeline, manifesting a drift, like the example above. It may around 10 minutes depending on how you stagger the inferences.

Here's an example of some code to use:

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
    
# run atypical dominate the sample
for x in range(3):
    pipeline.infer(highprice_data.sample(2*nsample, replace=True).reset_index(drop=True))
    time.sleep(25)

# run "typical" data
for x in range(4):
    pipeline.infer(lowprice_data.sample(2*nsample, replace=True).reset_index(drop=True))
    time.sleep(25)

# End our assay window period
assay_window_end = datetime.datetime.now()

```

## Define Assay Parameters

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

### Define Assay Parameters Exercise

Use the assay_builder you created in the previous exercise to set up an interactive assay. 
* The assay should run every minute, on a window that is a minute wide. 
* Set the alert threshold to 0.1.  
* You can use `assay_window_end` (or a later timestamp) as the end of the interactive run.

Examine the assay results. Do you see any drift?

To try other ways of examining the assay results, see the ["Interactive Assay Runs" section of the Model Insights tutorial](https://docs.wallaroo.ai/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights/#interactive-assay-runs).

Here's some code to use.

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

```python
# blank space for setting assay parameters, creating and examining an interactive assay

# set the end of the interactive run
assay_builder.add_run_until(assay_window_end)

# doing minutes to get our previous values in
assay_builder.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
assay_builder.add_alert_threshold(0.1)
assay_results = assay_builder.build().interactive_run()
df = assay_results.to_dataframe() # to return the results as a table
assay_results.chart_scores() # to plot the run

```

    
{{<figure src="/images/2024.2/wallaroo-use-case-tutorials/classification/finserv/N4_drift_detection-with-code-reference_files/N4_drift_detection-with-code-reference_15_0.png" width="800" label="png">}}
    

## Schedule an Assay for Ongoing Data

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

## Cleaning up.

Now that the tutorial is complete, don't forget to undeploy your pipeline to free up the resources.

```python
# blank space to undeploy your pipeline

pipeline.undeploy()

```

<table><tr><th>name</th> <td>ccfraud-detector</td></tr><tr><th>created</th> <td>2024-09-05 16:18:43.626892+00:00</td></tr><tr><th>last_updated</th> <td>2024-09-05 17:41:14.081888+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>workspace_id</th> <td>11</td></tr><tr><th>workspace_name</th> <td>tutorial-finserv-john</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>9f9fc75d-9297-4de5-9962-1cd46e5006df, 93b4dd36-5e02-440d-931c-80198d1ee48a, 252d1e6f-35a5-420a-9590-d08cd76943a7, cb026715-9ced-40bb-9108-333b88c9de64, e90c10e3-ab38-43a8-a315-fb4250c09b21, 583e7a6a-3a7a-4420-abd0-c91e346a874d, cbfc4951-4d2c-41cb-89c7-934ac5bd2cbf, 32ab9ef6-2ac4-4d92-a46e-5d5c286af48c, 410d43d4-e698-4747-bbd1-cb62afee258a, 9edcf6f6-660d-470d-b8f0-24f54a335e8f, cd63b4fa-6549-41b4-af8f-576b1f0ef8b3, ff2d47a5-47c8-4fcb-8709-e95b3d0d4340, 8e74290b-4cb6-43a5-9b27-8431643438fd</td></tr><tr><th>steps</th> <td>classification-finserv-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

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
