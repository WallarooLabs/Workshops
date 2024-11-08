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
wl = wallaroo.Client()
```

```python
# retrieve the previous workspace, model, and pipeline version

workspace_name = "tutorial-workspace-john-cybersecurity"

workspace = wl.get_workspace(name=workspace_name)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

# optionally, examine your current workspace
wl.get_current_workspace()

model_name = 'aloha-prime'

prime_model_version = wl.get_model(model_name)

pipeline_name = 'aloha-fraud-detector'

pipeline = wl.get_pipeline(pipeline_name)

display(workspace)
display(prime_model_version)
display(pipeline)

```

    {'name': 'tutorial-workspace-john-cybersecurity', 'id': 14, 'archived': False, 'created_by': '76b893ff-5c30-4f01-bd9e-9579a20fc4ea', 'created_at': '2024-05-01T16:30:01.177583+00:00', 'models': [{'name': 'aloha-prime', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 5, 1, 16, 30, 43, 651533, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 5, 1, 16, 30, 43, 651533, tzinfo=tzutc())}, {'name': 'aloha-challenger', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 5, 1, 16, 38, 56, 600586, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 5, 1, 16, 38, 56, 600586, tzinfo=tzutc())}], 'pipelines': [{'name': 'aloha-fraud-detector', 'create_time': datetime.datetime(2024, 5, 1, 16, 30, 53, 995114, tzinfo=tzutc()), 'definition': '[]'}]}

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

<table><tr><th>name</th> <td>aloha-fraud-detector</td></tr><tr><th>created</th> <td>2024-05-01 16:30:53.995114+00:00</td></tr><tr><th>last_updated</th> <td>2024-05-01 19:20:24.682678+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>ea59aae6-1565-4321-9086-7f8dd2a8e1c2, 41b9260e-21a8-4f16-ad56-c6267d3bae93, 46bdd5a3-fc22-41b7-b1ea-8287c99c241e, 28f5443a-ff67-4f4f-bfc3-f3f95f3c6f83, 435b73f7-76a1-4514-bd41-2cb94d3e78ff, 551242a7-fe4c-4a61-a4c7-e7fcc97509dc, d22eb0d2-9cff-4c5f-a851-10b1a19d8c44, 262909e9-8779-4c56-a994-725ddd0b58c8, 4cdf8e11-1b9c-44ab-a16d-abb054b5e9fe, 6b3529b1-1ff1-454b-8896-460c8c90d667</td></tr><tr><th>steps</th> <td>aloha-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

Set up the pipeline with the single model step as was done in notebook 1, then deploy it.

## Monitoring for Drift: Shift Happens. 

In machine learning, you use data and known answers to train a model to make predictions for new previously unseen data. You do this with the assumption that the future unseen data will be similar to the data used during training: the future will look somewhat like the past.
But the conditions that existed when a model was created, trained and tested can change over time, due to various factors.

A good model should be robust to some amount of change in the environment; however, if the environment changes too much, your models may no longer be making the correct decisions. This situation is known as concept drift; too much drift can obsolete your models, requiring periodic retraining.

Let's consider the example we've been working on: home sale price prediction. You may notice over time that there has been a change in the mix of properties in the listings portfolio: for example a dramatic increase or decrease in expensive properties (or more precisely, properties that the model thinks are expensive)

Such a change could be due to many factors: a change in interest rates; the appearance or disappearance of major sources of employment; new housing developments opening up in the area. Whatever the cause, detecting such a change quickly is crucial, so that the business can react quickly in the appropriate manner, whether that means simply retraining the model on fresher data, or a pivot in business strategy.

In Wallaroo you can monitor your housing model for signs of drift through the model monitoring and insight capability called Assays. Assays help you track changes in the environment that your model operates within, which can affect the model’s outcome. It does this by tracking the model’s predictions and/or the data coming into the model against an **established baseline**. If the distribution of monitored values in the current observation window differs too much from the baseline distribution, the assay will flag it. The figure below shows an example of a running scheduled assay.

{{<figure src="../images/wallaroo-model-insights-reference_35_0.png" width="800" label="">}}

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

We will use this data to set up some "historical data" in the house price prediction pipeline that you build in the assay exercises.

## Generate Sample Data

Before creating the assays, we will retrieve some sample data.  The following files are provided by the instructor, and at this point inferences are run in the same workspace and pipeline that was retrieved.

We'll retrieve the following variables for use in our assay generation:

* `small_results_baseline`:  Used to create the baseline from the numpy values from sample inferences.
* `assay_baseline_start`: When to start the baseline from the inference history.
* `assay_baseline_end`: When to end the baseline from the inference history.
* `assay_window_start`: When to start the assay window period for assay samples.

```python
# not needed for 2023.2
import numpy

baseline_numpy = numpy.load('./small_results_baseline.npy')
baseline_numpy
```

    array([1.38411885e-02, 6.98903700e-02, 1.72902760e-01, 1.01371060e-01,
           1.18515120e-02, 3.29993070e-02, 6.35678300e-02, 2.04431410e-02,
           5.54976280e-02, 3.33689700e-02, 1.63933440e-01, 5.01484130e-02,
           9.13416500e-02, 1.93939230e-02, 1.30328190e-02, 1.59948200e-01,
           1.50059490e-02, 1.80134130e-01, 3.50623620e-03, 1.64611250e-02,
           4.29595300e-02, 1.50780880e-02, 1.43549200e-01, 4.56962830e-03,
           4.97932730e-02, 1.64257660e-01, 7.37149900e-02, 1.10372186e-01,
           1.68252210e-02, 8.52306000e-02, 3.29359550e-02, 5.03419380e-02,
           1.25163240e-01, 2.78586650e-02, 6.16877850e-03, 3.07482650e-02,
           2.27204800e-02, 6.45729100e-02, 7.57607000e-02, 7.08777550e-03,
           2.97230650e-02, 3.59103600e-02, 2.30605320e-02, 3.27331540e-02,
           1.86745990e-01, 1.29913540e-02, 2.54665330e-02, 2.75201000e-02,
           7.10707060e-03, 1.50722850e-01, 9.01008800e-02, 8.99820400e-02,
           7.66398000e-04, 5.31407200e-02, 2.59052660e-02, 1.35844560e-01,
           7.88279400e-03, 5.74785550e-02, 8.55128600e-02, 5.78838800e-03,
           2.18456070e-02, 5.83127900e-02, 2.48722880e-02, 1.82142520e-02,
           1.09350510e-01, 1.35561020e-02, 2.56826020e-02, 8.15486300e-02,
           1.07193470e-01, 5.25884630e-02, 2.97664270e-03, 8.46283100e-03,
           1.77558360e-01, 1.15131140e-02, 5.15763100e-02, 1.02688690e-02,
           5.88536800e-02, 9.13388360e-02, 1.10681920e-02, 2.77703160e-02,
           8.27124600e-03, 1.56994830e-01, 2.44069840e-02, 1.74694250e-02,
           2.52948650e-03, 9.49505400e-03, 3.74419600e-02, 4.55613100e-02,
           1.00706150e-02, 1.94774100e-02, 8.64558800e-03, 1.46801350e-01,
           6.16716670e-02, 3.59286930e-02, 1.74182860e-01, 3.98783400e-02,
           8.39603400e-02, 1.86577480e-01, 3.10761230e-03, 2.68125720e-02,
           3.47353400e-02, 8.27692450e-02, 4.44800260e-02, 2.22439900e-02,
           1.17242046e-01, 2.17619300e-03, 2.33667050e-02, 9.22291800e-04,
           1.30676000e-03, 1.74263750e-02, 8.23086900e-03, 3.31561830e-03,
           1.40119060e-01, 3.40707700e-02, 1.34876130e-02, 1.53026670e-01,
           9.71889200e-02, 3.00213380e-02, 1.41497640e-01, 4.11387160e-03,
           3.07076800e-02, 1.87213120e-01, 1.64854140e-02, 4.16542250e-02,
           5.99254400e-02, 1.58361840e-01, 3.89646200e-02, 1.49028840e-02,
           4.76679650e-02, 2.93277000e-02, 4.38593500e-02, 1.97974700e-03,
           1.70390160e-02, 5.22396640e-02, 1.10602520e-02, 5.39108300e-03,
           3.91753170e-02, 8.27682000e-02, 9.94242450e-03, 1.46346750e-01,
           8.59485400e-02, 9.37004300e-03, 8.70854700e-02, 4.70293760e-02,
           6.21358450e-02, 5.19265720e-02, 1.32552570e-02, 1.45563920e-02,
           1.25547020e-01, 1.62168720e-02, 1.29899140e-01, 5.90517200e-02,
           3.54841130e-02, 7.22223000e-02, 1.50737540e-01, 5.79327500e-03,
           1.25484630e-01, 9.66135700e-02, 5.91832250e-02, 1.44371690e-02,
           1.46466660e-01, 6.93768200e-03, 6.80450200e-02, 6.41044500e-03,
           4.56792560e-02, 3.10506040e-02, 1.20614630e-01, 5.36099820e-02,
           1.52909660e-03, 2.07377340e-02, 4.58794970e-03, 4.73935700e-03,
           1.76518340e-01, 9.28278500e-03, 3.82321900e-02, 4.40731920e-03,
           5.01983540e-02, 1.05953574e-01, 2.72795390e-02, 4.92523500e-03,
           1.44624720e-02, 8.99027850e-03, 1.86023160e-01, 1.02547220e-02,
           5.44555440e-03, 1.37948830e-02, 5.22396640e-02, 5.94832470e-03,
           1.20436795e-01, 1.42605490e-02, 4.72300540e-02, 8.29395300e-02,
           7.83373600e-03, 2.65637820e-02, 7.86642100e-03, 7.26907200e-03,
           3.16277930e-02, 6.01660900e-03, 7.76602000e-03, 9.64844000e-02,
           4.96464040e-03, 3.26802700e-03, 8.09673600e-02, 5.59866060e-03,
           1.54808040e-02, 1.45334890e-02, 9.02044600e-03, 1.32668100e-01,
           2.63142470e-02, 4.80901040e-02, 2.15990520e-02, 2.39113660e-02,
           1.39197770e-02, 9.58271300e-02, 6.62867200e-02, 3.84855800e-02,
           1.08147375e-01, 2.00037820e-03, 2.25819180e-02, 1.91687330e-01,
           1.74825820e-02, 1.83839470e-01, 6.33939900e-03, 3.38803870e-02,
           1.58901410e-01, 3.48141400e-03, 9.66674400e-02, 5.17018100e-03,
           7.85653300e-03, 2.36946880e-02, 3.39801100e-02, 3.62642100e-02,
           3.84961070e-02, 4.90643040e-03, 1.59728740e-03, 2.35600790e-02,
           2.21027550e-02, 1.02316390e-02, 8.18040100e-03, 2.31594610e-02,
           1.23900205e-01, 1.84543120e-03, 4.69775270e-02, 6.39349700e-02,
           1.01578660e-02, 9.28876850e-02, 2.30481060e-03, 5.61872720e-02,
           8.70758950e-03, 4.70962560e-02, 1.18207335e-01, 9.12116840e-02,
           1.75370350e-02, 6.32652300e-03, 2.89774760e-02, 1.16218135e-01,
           1.86825600e-02, 5.02174100e-03, 1.16046160e-01, 1.37497130e-01,
           4.10481300e-03, 1.66790410e-02, 4.09364750e-03, 4.55811100e-02,
           1.94109320e-02, 4.71559130e-02, 9.25871360e-02, 2.85627100e-02,
           1.38995500e-01, 1.85749950e-01, 7.89931300e-02, 3.32346220e-02,
           1.17254730e-02, 3.19854360e-02, 1.47787870e-01, 1.85417570e-02,
           1.01933580e-02, 3.39885760e-02, 1.11831340e-01, 1.08295410e-03,
           1.43812500e-01, 1.95146400e-01, 2.47166960e-02, 1.14282250e-01,
           2.29466990e-02, 1.41030460e-01, 9.14114860e-02, 8.80805550e-02,
           7.73094300e-03, 1.76356500e-01, 1.04407705e-01, 1.81357900e-02,
           1.12586450e-02, 1.94578220e-01, 2.84287900e-02, 2.16221810e-02,
           3.69168630e-02, 5.78838800e-03, 6.94113100e-02, 1.14240160e-01,
           3.32645550e-02, 5.59242370e-02, 1.44821050e-01, 7.05835900e-02,
           3.56951730e-02, 8.61230400e-02, 1.66330620e-01, 1.31818800e-01,
           1.85409670e-02, 4.88343830e-03, 2.08539240e-02, 2.73829980e-03,
           6.26096000e-02, 1.53563490e-02, 4.71109560e-02, 5.58597970e-03,
           1.33812340e-01, 1.80868190e-02, 6.72935900e-02, 1.00287960e-01,
           3.82798900e-02, 6.79309700e-03, 6.14267330e-02, 3.11686170e-02,
           1.19035440e-01, 7.07995800e-02, 4.31257340e-02, 1.53811920e-02,
           5.80739800e-02, 5.21259870e-02, 8.71385450e-03, 1.11940150e-01,
           1.16627075e-01, 1.11630490e-01, 1.11695080e-01, 7.45863800e-05,
           3.69427260e-03, 6.19374030e-02, 1.37062800e-02, 9.59035700e-04,
           5.05744220e-02, 8.51463800e-02, 9.98307400e-02, 9.25409200e-03,
           4.05967350e-02, 6.32675500e-02, 6.53976050e-02, 6.16523060e-03,
           1.15123490e-02, 1.42405820e-02, 1.06156826e-01, 3.53979950e-02,
           1.71599180e-01, 1.24943240e-02, 2.50083990e-02, 2.42647620e-02,
           6.63617250e-02, 4.58676500e-03, 1.51891440e-01, 1.54267520e-02,
           1.36224690e-02, 1.42364620e-01, 1.66309100e-02, 3.41719130e-02,
           5.48537930e-02, 9.35613400e-03, 2.86029580e-02, 6.51544900e-03,
           6.36362660e-02, 1.17464720e-01, 4.57605720e-02, 3.78032540e-03,
           1.98746250e-02, 6.44662200e-02, 7.63675500e-02, 1.43812500e-01,
           3.39511100e-02, 1.51270720e-01, 2.13156320e-02, 1.42593690e-02,
           5.05189520e-02, 1.33675690e-02, 1.57063030e-02, 2.97409900e-02,
           7.39630500e-02, 1.52623190e-02, 4.87300570e-02, 2.08271000e-03,
           4.08358400e-03, 4.05166860e-03, 6.70003800e-03, 1.17731570e-01,
           2.12120190e-02, 5.51563800e-02, 6.24075230e-02, 1.39067210e-02,
           5.65910640e-02, 2.59743020e-02, 7.90192200e-02, 3.95232630e-02,
           3.70051260e-02, 9.54685600e-03, 3.08228060e-02, 1.77661910e-01,
           1.36665450e-01, 9.70385600e-02, 4.00257000e-02, 7.79701630e-03,
           1.40410330e-01, 4.43394500e-02, 5.88602130e-02, 5.49641700e-03,
           6.60768240e-03, 3.82705700e-03, 3.17071600e-02, 1.69518010e-02,
           2.29757430e-02, 4.07151470e-02, 7.14481800e-02, 1.46181140e-01,
           3.90153500e-02, 4.34549600e-03, 5.52976800e-03, 1.30328190e-02,
           1.07495030e-02, 2.31704250e-02, 5.75540100e-03, 5.52719720e-02,
           1.30861410e-02, 6.69540700e-02, 4.79600840e-02, 3.29008030e-02,
           2.11389330e-02, 2.23866400e-02, 6.13834500e-05, 1.15114376e-01,
           9.96227200e-02, 3.17312500e-02, 1.23500920e-02, 4.17530350e-02,
           6.21513300e-02, 1.39439380e-02, 1.32357245e-02, 3.43569100e-02,
           3.77594260e-03, 1.39851150e-01, 2.93206440e-02, 7.36732860e-02,
           8.25936700e-03, 7.51185040e-02, 1.60548720e-02, 8.11753050e-02,
           5.29028900e-02, 3.11605390e-02, 8.67323500e-03, 1.58319680e-02,
           4.16296900e-03, 2.86411900e-02, 1.33027030e-02, 5.34343980e-02,
           1.27165710e-02, 3.63482530e-02, 1.26039830e-02, 9.25015960e-02,
           4.01699730e-03, 7.81137300e-02, 1.77701410e-01, 4.73120920e-02,
           9.91035100e-02, 2.80917210e-02, 4.03374950e-03, 1.61084440e-02,
           1.62963120e-01, 1.23026505e-01, 2.00015230e-02, 1.38884040e-02,
           9.45581900e-03, 1.63274310e-02, 9.15808400e-02, 1.92165020e-02,
           1.07956840e-02, 3.10963280e-02, 2.36851500e-02, 1.48576450e-01,
           7.22223000e-02, 4.50184900e-03, 1.42279750e-02, 1.94049210e-03,
           1.63249390e-01, 7.09095200e-02, 1.09045880e-01, 4.73470460e-02,
           1.31590250e-01, 4.57807930e-02, 1.38165750e-02, 8.64240800e-02,
           1.42364620e-01, 1.53426050e-02, 1.00781200e-01, 4.37252600e-03])

```python
# read the assay baseline start datetime

with open('./assay_baseline_start', 'r') as file:
    assay_baseline_start = datetime.datetime.strptime(file.read(), "%d-%b-%Y (%H:%M:%S.%f)")

# read the assay baseline end datetime

with open('./assay_baseline_end', 'r') as file:
    assay_baseline_end = datetime.datetime.strptime(file.read(), "%d-%b-%Y (%H:%M:%S.%f)")

# read the assay window start datetime

with open('./assay_window_start', 'r') as file:
    assay_window_start = datetime.datetime.strptime(file.read(), "%d-%b-%Y (%H:%M:%S.%f)")

display(assay_baseline_start)
display(assay_window_start)
display(assay_baseline_end)

```

    datetime.datetime(2024, 5, 1, 13, 20, 32, 870761)

    datetime.datetime(2024, 5, 1, 13, 27, 1, 25108)

    datetime.datetime(2024, 5, 1, 13, 21, 34, 718837)

## Model Insights via the Wallaroo Dashboard SDK

Assays generated through the Wallaroo SDK can be previewed, configured, and uploaded to the Wallaroo Ops instance.  The following is a condensed version of this process.  For full details see the [Wallaroo SDK Essentials Guide: Assays Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-assays/) guide.

Model drift detection with assays using the Wallaroo SDK follows this general process.

* Define the Baseline: From either historical inference data for a specific model in a pipeline, or from a pre-determine array of data, a **baseline** is formed.
* Assay Preview:  Once the baseline is formed, we **preview the assay** and configure the different options until we have the the best method of detecting environment or model drift.
* Create Assay:  With the previews and configuration complete, we **upload** the assay.  The assay will perform an analysis on a regular scheduled based on the configuration.
* Get Assay Results:  Retrieve the analyses and use them to detect model drift and possible sources.
* Pause/Resume Assay:  Pause or restart an assay as needed.

### Define the Baseline

Assay baselines are defined with the [`wallaroo.client.build_assay`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/client/#Client.build_assay) method. Through this process we define the baseline from either a range of dates or pre-generated values.

`wallaroo.client.build_assay` take the following parameters:

| Parameter | Type | Description |
|---|---|---|
| **assay_name** | *String* (*Required*) - required | The name of the assay.  Assay names **must** be unique across the Wallaroo instance. |
| **pipeline** | *wallaroo.pipeline.Pipeline* (*Required*) | The pipeline the assay is monitoring. |
| **model_name** | *String* (*Required*)  | The name of the model to monitor.
| **iopath** | *String* (*Required*) | The input/output data for the model being tracked in the format `input/output field index`.  Only one value is tracked for any assay.  For example, to track the **output** of the model's field `house_value` at index `0`, the `iopath` is `'output house_value 0`. |
| **baseline_start** | *datetime.datetime* (*Optional*) | The start time for the inferences to use as the baseline.  **Must be included with `baseline_end`.  Cannot be included with `baseline_data`**. |
| **baseline_end** | *datetime.datetime* (*Optional*) | The end time of the baseline window. the baseline. Windows start immediately after the baseline window and are run at regular intervals continuously until the assay is deactivated or deleted.  **Must be included with `baseline_start`.  Cannot be included with `baseline_data`.**. |
| **baseline_data** | *numpy.array* (*Optional*) | The baseline data in numpy array format.  **Cannot be included with either `baseline_start` or `baseline_data`**. |

Baselines are created in one of two ways:

* **Date Range**:  The `baseline_start` and `baseline_end` retrieves the inference requests and results for the pipeline from the start and end period.  This data is summarized and used to create the baseline.
* **Numpy Values**:  The `baseline_data` sets the baseline from a provided numpy array.

#### Define the Baseline Example

This example shows two methods of defining the baseline for an assay:

* `"assays from date baseline"`: This assay uses historical inference requests to define the baseline.  This assay is saved to the variable `assay_builder_from_dates`.
* `"assays from numpy"`:  This assay uses a pre-generated numpy array to define the baseline.  This assay is saved to the variable `assay_builder_from_numpy`.

In both cases, the following parameters are used:

| Parameter | Value |
|---|---|
| **assay_name** | `"assays from date baseline"` and `"assays from numpy"` |
| **pipeline** | `mainpipeline`:  A pipeline with a ML model that predicts house prices.  The output field for this model is `variable`. |
| **model_name** | `"houseprice-predictor"` - the model name set during model upload. |
| **iopath** | These assays monitor the model's **output** field **variable** at index 0.  From this, the `iopath` setting is `"output variable 0"`.  |

The difference between the two assays' parameters determines how the baseline is generated.

* `"assays from date baseline"`: Uses the `baseline_start` and `baseline_end` to set the time period of inference requests and results to gather data from.
* `"assays from numpy"`:  Uses a pre-generated numpy array as for the baseline data.

For each of our assays, we will set the time period of inference data to compare against the baseline data.

Use the sample code below to create an assay off the pipeline and model.

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = ( wl.build_assay('sample assay from baseline_date', 
                                            pipeline, 
                                            model_name, 
                                            baseline_start=assay_baseline_start, 
                                            baseline_end=assay_baseline_end)
                                            .add_iopath("output variable 0")
                           )

assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
assay_config_from_dates = assay_builder_from_dates.build()
assay_results_from_dates = assay_config_from_dates.interactive_run()
```

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0

display(assay_baseline_start)
display(assay_baseline_end)

assay_baseline_from_dates = wl.build_assay(assay_name="assays from date baseline jch", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# create the baseline from the dates
assay_baseline_run_from_dates = assay_baseline_from_dates.build().interactive_baseline_run()
```

    datetime.datetime(2024, 5, 1, 13, 20, 32, 870761)

    datetime.datetime(2024, 5, 1, 13, 21, 34, 718837)

#### Baseline DataFrame

#### Baseline DataFrame

The method [`wallaroo.assay_config.AssayBuilder.baseline_dataframe`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#AssayBuilder.baseline_dataframe) returns a DataFrame of the assay baseline generated from the provided parameters.  This includes:

* `metadata`:  The inference metadata with the model information, inference time, and other related factors.
* `in` data:  Each input field assigned with the label `in.{input field name}`.
* `out` data:  Each output field assigned with the label `out.{output field name}`

Note that for assays generated from numpy values, there is only the `out` data based on the supplied baseline data.

In the following example, the baseline DataFrame is retrieved.  

In the following example, the baseline DataFrame is retrieved.

```python
display(assay_builder_from_dates.baseline_dataframe())
```

```python
display(assay_baseline_from_dates.baseline_dataframe())
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>metadata</th>
      <th>input_text_input_0</th>
      <th>input_text_input_1</th>
      <th>input_text_input_2</th>
      <th>input_text_input_3</th>
      <th>input_text_input_4</th>
      <th>input_text_input_5</th>
      <th>input_text_input_6</th>
      <th>input_text_input_7</th>
      <th>...</th>
      <th>output_kraken_0</th>
      <th>output_locky_0</th>
      <th>output_main_0</th>
      <th>output_matsnu_0</th>
      <th>output_pykspa_0</th>
      <th>output_qakbot_0</th>
      <th>output_ramdo_0</th>
      <th>output_ramnit_0</th>
      <th>output_simda_0</th>
      <th>output_suppobox_0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1714591293408</td>
      <td>{'last_model': '{"model_name":"aloha-prime","model_sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}', 'pipeline_version': 'ea59aae6-1565-4321-9086-7f8dd2a8e1c2', 'elapsed': [4914898, 1037310284], 'dropped': [], 'partition': 'engine-5bc4457cb4-smgvg'}</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1.822409e-01</td>
      <td>3.872248e-02</td>
      <td>0.999998</td>
      <td>6.326576e-02</td>
      <td>0.028688</td>
      <td>8.994720e-08</td>
      <td>0.063389</td>
      <td>0.000680</td>
      <td>1.315745e-29</td>
      <td>1.566795e-29</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1714591293408</td>
      <td>{'last_model': '{"model_name":"aloha-prime","model_sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}', 'pipeline_version': 'ea59aae6-1565-4321-9086-7f8dd2a8e1c2', 'elapsed': [4914898, 1037310284], 'dropped': [], 'partition': 'engine-5bc4457cb4-smgvg'}</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6.881179e-02</td>
      <td>1.606125e-01</td>
      <td>0.998849</td>
      <td>1.527669e-01</td>
      <td>0.097369</td>
      <td>1.268106e-01</td>
      <td>0.070885</td>
      <td>0.045734</td>
      <td>4.409813e-32</td>
      <td>3.995916e-38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1714591293408</td>
      <td>{'last_model': '{"model_name":"aloha-prime","model_sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}', 'pipeline_version': 'ea59aae6-1565-4321-9086-7f8dd2a8e1c2', 'elapsed': [4914898, 1037310284], 'dropped': [], 'partition': 'engine-5bc4457cb4-smgvg'}</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1.200413e-02</td>
      <td>7.000831e-02</td>
      <td>0.999806</td>
      <td>5.570423e-02</td>
      <td>0.040504</td>
      <td>3.660773e-05</td>
      <td>0.023749</td>
      <td>0.067193</td>
      <td>8.728547e-01</td>
      <td>3.854356e-17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1714591293408</td>
      <td>{'last_model': '{"model_name":"aloha-prime","model_sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}', 'pipeline_version': 'ea59aae6-1565-4321-9086-7f8dd2a8e1c2', 'elapsed': [4914898, 1037310284], 'dropped': [], 'partition': 'engine-5bc4457cb4-smgvg'}</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7.545263e-03</td>
      <td>5.215001e-02</td>
      <td>0.999956</td>
      <td>4.482307e-02</td>
      <td>0.028627</td>
      <td>3.603033e-03</td>
      <td>0.033871</td>
      <td>0.082471</td>
      <td>9.670970e-01</td>
      <td>1.193230e-19</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1714591293408</td>
      <td>{'last_model': '{"model_name":"aloha-prime","model_sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}', 'pipeline_version': 'ea59aae6-1565-4321-9086-7f8dd2a8e1c2', 'elapsed': [4914898, 1037310284], 'dropped': [], 'partition': 'engine-5bc4457cb4-smgvg'}</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1.017413e-12</td>
      <td>4.426621e-15</td>
      <td>1.000000</td>
      <td>7.573464e-15</td>
      <td>0.244629</td>
      <td>1.245128e-07</td>
      <td>0.512951</td>
      <td>0.002041</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>1714591293408</td>
      <td>{'last_model': '{"model_name":"aloha-prime","model_sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}', 'pipeline_version': 'ea59aae6-1565-4321-9086-7f8dd2a8e1c2', 'elapsed': [4914898, 1037310284], 'dropped': [], 'partition': 'engine-5bc4457cb4-smgvg'}</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6.701718e-02</td>
      <td>1.607626e-01</td>
      <td>0.999751</td>
      <td>1.390635e-01</td>
      <td>0.081234</td>
      <td>9.239194e-06</td>
      <td>0.055272</td>
      <td>0.108671</td>
      <td>1.053473e-24</td>
      <td>2.099079e-27</td>
    </tr>
    <tr>
      <th>496</th>
      <td>1714591293408</td>
      <td>{'last_model': '{"model_name":"aloha-prime","model_sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}', 'pipeline_version': 'ea59aae6-1565-4321-9086-7f8dd2a8e1c2', 'elapsed': [4914898, 1037310284], 'dropped': [], 'partition': 'engine-5bc4457cb4-smgvg'}</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1.238461e-01</td>
      <td>2.807522e-01</td>
      <td>0.999985</td>
      <td>2.732534e-01</td>
      <td>0.154991</td>
      <td>1.483513e-05</td>
      <td>0.126482</td>
      <td>0.067603</td>
      <td>3.376616e-28</td>
      <td>1.382165e-36</td>
    </tr>
    <tr>
      <th>497</th>
      <td>1714591293408</td>
      <td>{'last_model': '{"model_name":"aloha-prime","model_sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}', 'pipeline_version': 'ea59aae6-1565-4321-9086-7f8dd2a8e1c2', 'elapsed': [4914898, 1037310284], 'dropped': [], 'partition': 'engine-5bc4457cb4-smgvg'}</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4.386075e-02</td>
      <td>1.863457e-02</td>
      <td>0.999992</td>
      <td>2.382919e-02</td>
      <td>0.017675</td>
      <td>9.459920e-07</td>
      <td>0.032047</td>
      <td>0.000450</td>
      <td>9.982694e-01</td>
      <td>1.251219e-21</td>
    </tr>
    <tr>
      <th>498</th>
      <td>1714591293408</td>
      <td>{'last_model': '{"model_name":"aloha-prime","model_sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}', 'pipeline_version': 'ea59aae6-1565-4321-9086-7f8dd2a8e1c2', 'elapsed': [4914898, 1037310284], 'dropped': [], 'partition': 'engine-5bc4457cb4-smgvg'}</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3.643817e-02</td>
      <td>1.621823e-01</td>
      <td>0.999975</td>
      <td>1.465069e-01</td>
      <td>0.080386</td>
      <td>3.081988e-10</td>
      <td>0.075103</td>
      <td>0.149607</td>
      <td>7.931008e-33</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>499</th>
      <td>1714591293408</td>
      <td>{'last_model': '{"model_name":"aloha-prime","model_sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}', 'pipeline_version': 'ea59aae6-1565-4321-9086-7f8dd2a8e1c2', 'elapsed': [4914898, 1037310284], 'dropped': [], 'partition': 'engine-5bc4457cb4-smgvg'}</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1.858206e-12</td>
      <td>1.136425e-15</td>
      <td>0.999999</td>
      <td>2.814451e-15</td>
      <td>0.146142</td>
      <td>9.087309e-11</td>
      <td>0.511674</td>
      <td>0.000147</td>
      <td>6.198446e-38</td>
      <td>0.000000e+00</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 67 columns</p>

#### Baseline Stats

The method `wallaroo.assay.AssayAnalysis.baseline_stats()` returns a `pandas.core.frame.DataFrame` of the baseline stats.

The baseline stats for each assay are displayed in the examples below.

```python
assay_baseline_run_from_dates.baseline_stats()
```

```python
assay_baseline_run_from_dates.baseline_stats()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Baseline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>500</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000061</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.195146</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.051899</td>
    </tr>
    <tr>
      <th>median</th>
      <td>0.032968</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.049993</td>
    </tr>
    <tr>
      <th>start</th>
      <td>2024-05-01T19:20:32.870761+00:00</td>
    </tr>
    <tr>
      <th>end</th>
      <td>2024-05-01T19:21:34.718761+00:00</td>
    </tr>
  </tbody>
</table>

#### Baseline Bins

The method `wallaroo.assay.AssayAnalysis.baseline_bins` a simple dataframe to with the edge/bin data for a baseline.

```python
assay_results_from_dates[0].baseline_bins()
```

```python
assay_baseline_run_from_dates.baseline_bins()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b_edges</th>
      <th>b_edge_names</th>
      <th>b_aggregated_values</th>
      <th>b_aggregation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000061</td>
      <td>left_outlier</td>
      <td>0.0</td>
      <td>Aggregation.DENSITY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.009547</td>
      <td>q_20</td>
      <td>0.2</td>
      <td>Aggregation.DENSITY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.022947</td>
      <td>q_40</td>
      <td>0.2</td>
      <td>Aggregation.DENSITY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.047111</td>
      <td>q_60</td>
      <td>0.2</td>
      <td>Aggregation.DENSITY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.092888</td>
      <td>q_80</td>
      <td>0.2</td>
      <td>Aggregation.DENSITY</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.195146</td>
      <td>q_100</td>
      <td>0.2</td>
      <td>Aggregation.DENSITY</td>
    </tr>
    <tr>
      <th>6</th>
      <td>inf</td>
      <td>right_outlier</td>
      <td>0.0</td>
      <td>Aggregation.DENSITY</td>
    </tr>
  </tbody>
</table>

#### Baseline Histogram Chart

The baseline chart is displayed with `wallaroo.assay.AssayAnalysis.chart()`, which returns a chart with:

* **baseline mean**:  The mean value of the baseline values.
* **baseline median**: The median value of the baseline values.
* **bin_mode**: The binning mode.  See [Binning Mode](#binning-mode)
* **aggregation**:  The aggregation type.  See [Aggregation Options](#aggregation-options)
* **metric**:  The assay's metric type.  See [Score Metric](#score-metric)
* **weighted**:  Whether the binning mode is weighted.  See [Binning Mode](#binning-mode)

```python
assay_builder_from_dates.baseline_histogram()
```

```python
assay_baseline_from_dates.baseline_histogram()
```

    
{{<figure src="/images/2024.2/wallaroo-use-case-tutorials/classification/cybersecurity/N4_drift_detection-with-code-reference_files/N4_drift_detection-with-code-reference_20_0.png" width="800" label="png">}}
    

### Assay Preview

Now that the baseline is defined, we look at different configuration options and view how the assay baseline and results changes.  Once we determine what gives us the best method of determining model drift, we can create the assay.

#### Analysis List Chart Scores

Analysis List scores show the assay scores for each assay result interval in one chart.  Values that are outside of the alert threshold are colored red, while scores within the alert threshold are green.

Assay chart scores are displayed with the method [`wallaroo.assay.AssayAnalysisList.chart_scores(title: Optional[str] = None)`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay/#AssayAnalysisList.chart_scores), with ability to display an optional title with the chart.

The following example shows retrieving the assay results and displaying the chart scores.  From our example, we have two windows - the first should be green, and the second is red showing that values were outside the alert threshold.

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# Preview the assay analyses
assay_results.chart_scores()
```

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=pipeline, 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# Preview the assay analyses
assay_results.chart_scores()
```

    
{{<figure src="/images/2024.2/wallaroo-use-case-tutorials/classification/cybersecurity/N4_drift_detection-with-code-reference_files/N4_drift_detection-with-code-reference_23_0.png" width="800" label="png">}}
    

#### Analysis Chart

The method `wallaroo.assay.AssayAnalysis.chart()` displays a comparison between the baseline and an interval of inference data.

This is compared to the [Chart Scores](#analysis-list-chart-scores), which is a **list** of all of the inference data split into intervals, while the **Analysis Chart** shows the breakdown of one set of inference data against the baseline.

Score from the [Analysis List Chart Scores](#analysis-list-chart-scores) and each element from the [Analysis List DataFrame](#analysis-list-dataframe) generates 

The following fields are included.

| Field | Type | Description |
|---|---|---|
| **baseline mean** | **Float** | The mean of the baseline values. |
| **window mean** | **Float** | The mean of the window values. |
| **baseline median** | **Float** | The median of the baseline values. |
| **window median** | **Float** | The median of the window values. |
| **bin_mode** | **String** | The binning mode used for the assay. |
| **aggregation** | **String** | The aggregation mode used for the assay. |
| **metric** | **String** | The metric mode used for the assay. |
| **weighted** | **Bool** | Whether the bins were manually weighted. |
| **score** | **Float** | The score from the assay window. |
| **scores** | **List(Float)** | The score from each assay window bin. |
| **index** | **Integer/None** | The window index.  Interactive assay runs are `None`. |

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# display one of the analysis from the total results
display(assay_results[0].chart())
display(assay_results[1].chart())
```

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# display one of the analysis from the total results
display(assay_results[0].chart())
```

    baseline mean = 0.05189906566111312
    window mean = 0.26292607237398624
    baseline median = 0.03296763077378273
    window median = 0.2469722330570221
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 8.868483247476384
    scores = [0.0, 0.7193314935522176, 0.7193314935522176, 0.7193314935522176, 0.7193314935522176, 0.7193314935522176, 5.2718257797152965]
    index = None

    
{{<figure src="/images/2024.2/wallaroo-use-case-tutorials/classification/cybersecurity/N4_drift_detection-with-code-reference_files/N4_drift_detection-with-code-reference_25_1.png" width="800" label="png">}}
    

    None

#### Analysis List DataFrame

`wallaroo.assay.AssayAnalysisList.to_dataframe()` returns a DataFrame showing the assay results for each window aka individual analysis.  This DataFrame contains the following fields:

| Field | Type | Description |
|---|---|---|
| **assay_id** | *Integer/None* | The assay id.  Only provided from uploaded and executed assays. |
| **name** | *String/None* | The name of the assay.  Only provided from uploaded and executed assays. |
| **iopath** | *String/None* | The iopath of the assay.  Only provided from uploaded and executed assays. |
| **score** | *Float* | The assay score. |
| **start** | *DateTime* | The DateTime start of the assay window.
| **min** | *Float* | The minimum value in the assay window.
| **max**  | *Float* | The maximum value in the assay window.
| **mean** | *Float* | The mean value in the assay window.
| **median** | *Float* | The median value in the assay window.
| **std** | *Float* | The standard deviation value in the assay window.
| **warning_threshold** | *Float/None* | The warning threshold of the assay window.
| **alert_threshold** | *Float/None* | The alert threshold of the assay window.
| **status** | *String* | The assay window status.  Values are:  <ul><li>`OK`: The score is within accepted thresholds.</li><li>`Warning`: The score has triggered the `warning_threshold` if exists, but not the `alert_threshold`.</li><li>`Alert`: The score has triggered the the `alert_threshold`.</li></ul> |

For this example, the assay analysis list DataFrame is listed.  

From this tutorial, we should have 2 windows of dta to look at, each one minute apart.  The first window should show `status: OK`, with the second window with the very large house prices will show `status: alert`

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# display the dataframe from the analyses
assay_results.to_dataframe()
```

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# display the dataframe from the analyses
assay_results.to_dataframe()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>assay_id</th>
      <th>assay_name</th>
      <th>iopath</th>
      <th>pipeline_id</th>
      <th>pipeline_name</th>
      <th>score</th>
      <th>start</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>median</th>
      <th>std</th>
      <th>warning_threshold</th>
      <th>alert_threshold</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>None</td>
      <td>assays from date baseline</td>
      <td></td>
      <td>None</td>
      <td></td>
      <td>8.868483</td>
      <td>2024-05-01T19:28:01.025108+00:00</td>
      <td>0.200003</td>
      <td>0.567695</td>
      <td>0.262926</td>
      <td>0.246972</td>
      <td>0.056658</td>
      <td>None</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
  </tbody>
</table>

#### Analysis List Full DataFrame

`wallaroo.assay.AssayAnalysisList.to_full_dataframe()` returns a DataFrame showing all values, including the inputs and outputs from the assay results for each window aka individual analysis.  This DataFrame contains the following fields:

	pipeline_id	warning_threshold	bin_index	created_at

| Field | Type | Description |
|---|---|---|
| **window_start** | *DateTime* | The date and time when the window period began. |
| **analyzed_at** | *DateTime* | The date and time when the assay analysis was performed. |
| **elapsed_millis** | *Integer* | How long the analysis took to perform in milliseconds. |
| **baseline_summary_count** | *Integer* | The number of data elements from the baseline. |
| **baseline_summary_min** | *Float* | The minimum value from the baseline summary. |
| **baseline_summary_max** | *Float* | The maximum value from the baseline summary. |
| **baseline_summary_mean** | *Float* | The mean value of the baseline summary. |
| **baseline_summary_median** | *Float* | The median value of the baseline summary. |
| **baseline_summary_std** | *Float* | The standard deviation value of the baseline summary. |
| **baseline_summary_edges_{0...n}** | *Float* | The baseline summary edges for each baseline edge from 0 to number of edges. |
| **summarizer_type** | *String* | The type of summarizer used for the baseline.  See [`wallaroo.assay_config` for other summarizer types](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/). |
| **summarizer_bin_weights** | *List / None* | If baseline bin weights were provided, the list of those weights.  Otherwise, `None`. |
| **summarizer_provided_edges** | *List / None* | If baseline bin edges were provided, the list of those edges.  Otherwise, `None`. |
| **status** | *String* | The assay window status.  Values are:  <ul><li>`OK`: The score is within accepted thresholds.</li><li>`Warning`: The score has triggered the `warning_threshold` if exists, but not the `alert_threshold`.</li><li>`Alert`: The score has triggered the the `alert_threshold`.</li></ul> |
| **id** | *Integer/None* | The id for the window aka analysis.  Only provided from uploaded and executed assays. |
| **assay_id** | *Integer/None* | The assay id.  Only provided from uploaded and executed assays. |
| **pipeline_id** | *Integer/None* | The pipeline id.  Only provided from uploaded and executed assays. |
| **warning_threshold** | *Float* | The warning threshold set for the assay. |
| **warning_threshold** | *Float* | The warning threshold set for the assay.
| **bin_index** | *Integer/None* | The bin index for the window aka analysis.|
| **created_at** | *Datetime/None* | The date and time the window aka analysis was generated.  Only provided from uploaded and executed assays. |

For this example, full DataFrame from an assay preview is generated.

From this tutorial, we should have 2 windows of dta to look at, each one minute apart.  The first window should show `status: OK`, with the second window with the very large house prices will show `status: alert`

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = ( wl.build_assay('sample assay from baseline_date', 
                                            pipeline, 
                                            model_name, 
                                            baseline_start=assay_baseline_start, 
                                            baseline_end=assay_baseline_end)
                                            .add_iopath("output cryptolocker 0")
                           )

assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
assay_config_from_dates = assay_builder_from_dates.build()
assay_results_from_dates = assay_config_from_dates.interactive_run()

assay_results_from_dates.to_full_dataframe()
```

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# display the full dataframe from the analyses
assay_results.to_full_dataframe()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>window_start</th>
      <th>analyzed_at</th>
      <th>elapsed_millis</th>
      <th>baseline_summary_count</th>
      <th>baseline_summary_min</th>
      <th>baseline_summary_max</th>
      <th>baseline_summary_mean</th>
      <th>baseline_summary_median</th>
      <th>baseline_summary_std</th>
      <th>baseline_summary_edges_0</th>
      <th>...</th>
      <th>summarizer_type</th>
      <th>summarizer_bin_weights</th>
      <th>summarizer_provided_edges</th>
      <th>status</th>
      <th>id</th>
      <th>assay_id</th>
      <th>pipeline_id</th>
      <th>warning_threshold</th>
      <th>bin_index</th>
      <th>created_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-05-01T19:28:01.025108+00:00</td>
      <td>2024-05-01T19:41:46.223804+00:00</td>
      <td>112</td>
      <td>500</td>
      <td>0.000061</td>
      <td>0.195146</td>
      <td>0.051899</td>
      <td>0.032968</td>
      <td>0.049993</td>
      <td>0.000061</td>
      <td>...</td>
      <td>UnivariateContinuous</td>
      <td>None</td>
      <td>None</td>
      <td>Alert</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 86 columns</p>

#### Analysis Compare Basic Stats

The method `wallaroo.assay.AssayAnalysis.compare_basic_stats` returns a DataFrame comparing one set of inference data against the baseline.

This is compared to the [Analysis List DataFrame](#analysis-list-dataframe), which is a **list** of all of the inference data split into intervals, while the **Analysis Compare Basic Stats** shows the breakdown of one set of inference data against the baseline.

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# display one analysis against the baseline
assay_results[0].compare_basic_stats()
```

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# display one analysis against the baseline
assay_results[0].compare_basic_stats()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Baseline</th>
      <th>Window</th>
      <th>diff</th>
      <th>pct_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>500.0</td>
      <td>1000.0</td>
      <td>500.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000061</td>
      <td>0.200003</td>
      <td>0.199942</td>
      <td>325725.732613</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.195146</td>
      <td>0.567695</td>
      <td>0.372549</td>
      <td>190.907497</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.051899</td>
      <td>0.262926</td>
      <td>0.211027</td>
      <td>406.610416</td>
    </tr>
    <tr>
      <th>median</th>
      <td>0.032968</td>
      <td>0.246972</td>
      <td>0.214005</td>
      <td>649.135523</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.049993</td>
      <td>0.056658</td>
      <td>0.006664</td>
      <td>13.329998</td>
    </tr>
    <tr>
      <th>start</th>
      <td>2024-05-01T19:20:32.870761+00:00</td>
      <td>2024-05-01T19:28:01.025108+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>end</th>
      <td>2024-05-01T19:21:34.718761+00:00</td>
      <td>2024-05-01T19:29:01.025108+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>

#### Configure Assays

Before creating the assay, **configure** the assay and continue to preview it until the best method for detecting drift is set.  The following options are available.

##### Inference Interval and Inference Width

The inference interval aka window interval sets how often to run the assay analysis.  This is set from the [`wallaroo.assay_config.AssayBuilder.window_builder.add_interval`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#WindowBuilder.add_interval) method to collect data expressed in time units:  "hours=24", "minutes=1", etc.

For example, with an interval of 1 minute, the assay collects data every minute.  Within an hour, 60 intervals of data is collected.

We can adjust the interval and see how the assays change based on how **frequently** they are run.

The width sets the time period from the [`wallaroo.assay_config.AssayBuilder.window_builder.add_width`](/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#WindowBuilder.add_width) method to collect data expressed in time units:  "hours=24", "minutes=1", etc.

For example, an interval of 1 minute and a width of 1 minute collects 1 minutes worth of data every minute.  An interval of 1 minute with a width of 5 minutes collects 5 minute of inference data every minute.

By default, the interval and width is **24 hours**.

Use the code block below and adjust the interval and width to 5 minutes.

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# show the analyses chart
assay_results.chart_scores()
```

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# show the analyses chart
assay_results.chart_scores()
```

    
{{<figure src="/images/2024.2/wallaroo-use-case-tutorials/classification/cybersecurity/N4_drift_detection-with-code-reference_files/N4_drift_detection-with-code-reference_34_0.png" width="800" label="png">}}
    

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=5).add_interval(minutes=5).add_start(assay_window_start)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# show the analyses chart
assay_results.chart_scores()
```

    
{{<figure src="/images/2024.2/wallaroo-use-case-tutorials/classification/cybersecurity/N4_drift_detection-with-code-reference_files/N4_drift_detection-with-code-reference_35_0.png" width="800" label="png">}}
    

##### Add Run Until and Add Inference Start

For previewing assays, setting [`wallaroo.assay_config.AssayBuilder.add_run_until`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#AssayBuilder.add_run_until) sets the **end date and time** for collecting inference data.  When an assay is uploaded, this setting is no longer valid - assays run at the [Inference Interval](#inference-interval-and-inference-width) until the [assay is paused](#pause-and-resume-assay).

Setting the [`wallaroo.assay_config.WindowBuilder.add_start`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#WindowBuilder.add_start) sets the **start date and time** to collect inference data.   When an assay is uploaded, this setting **is included**, and assay results will be displayed starting from that start date at the [Inference Interval](#inference-interval-and-inference-width) until the [assay is paused](#pause-and-resume-assay).  By default, `add_start` begins 24 hours after the assay is uploaded unless set in the assay configuration manually.

For the following example, the `add_run_until` setting is set to `datetime.datetime.now()` to collect all inference data from `assay_window_start` up until now, and the second example limits that example to only two minutes of data.

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results, minus 2 minutes for the period to start gathering analyses
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start+datetime.timedelta(seconds=-120))

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# show the analyses chart
assay_results.chart_scores()
```

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results, minus 2 minutes for the period to start gathering analyses
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start+datetime.timedelta(seconds=-120))

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# show the analyses chart
assay_results.chart_scores()
```

    
{{<figure src="/images/2024.2/wallaroo-use-case-tutorials/classification/cybersecurity/N4_drift_detection-with-code-reference_files/N4_drift_detection-with-code-reference_37_0.png" width="800" label="png">}}
    

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results plus 2 minutes 
assay_baseline_from_dates.add_run_until(assay_window_start+datetime.timedelta(seconds=120))

# Set the interval and window to one minute each
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# show the analyses chart
assay_results.chart_scores()
```

    
{{<figure src="/images/2024.2/wallaroo-use-case-tutorials/classification/cybersecurity/N4_drift_detection-with-code-reference_files/N4_drift_detection-with-code-reference_38_0.png" width="800" label="png">}}
    

##### Score Metric

The `score` is a distance between the baseline and the analysis window.  The larger the score, the greater the difference between the baseline and the analysis window.  The following methods are provided determining the score:

* `PSI` (*Default*) - Population Stability Index (PSI).
* `MAXDIFF`: Maximum difference between corresponding bins.
* `SUMDIFF`: Mum of differences between corresponding bins.

The metric type used is updated with the [`wallaroo.assay_config.AssayBuilder.add_metric(metric: wallaroo.assay_config.Metric)`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#UnivariateContinousSummarizerBuilder.add_metric) method.

The following three charts use each of the metrics.  Note how the scores change based on the score type used.

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# set metric PSI mode
assay_baseline.summarizer_builder.add_metric(wallaroo.assay_config.Metric.PSI)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# display one analysis from the results
assay_results[0].chart()
```

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# set metric PSI mode
assay_baseline.summarizer_builder.add_metric(wallaroo.assay_config.Metric.PSI)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# display one analysis from the results
assay_results[0].chart()
```

    baseline mean = 0.05189906566111312
    window mean = 0.26292607237398624
    baseline median = 0.03296763077378273
    window median = 0.2469722330570221
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 8.868483247476384
    scores = [0.0, 0.7193314935522176, 0.7193314935522176, 0.7193314935522176, 0.7193314935522176, 0.7193314935522176, 5.2718257797152965]
    index = None

    
{{<figure src="/images/2024.2/wallaroo-use-case-tutorials/classification/cybersecurity/N4_drift_detection-with-code-reference_files/N4_drift_detection-with-code-reference_40_1.png" width="800" label="png">}}
    

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# set metric MAXDIFF mode
assay_baseline.summarizer_builder.add_metric(wallaroo.assay_config.Metric.MAXDIFF)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# display one analysis from the results
assay_results[0].chart()
```

    baseline mean = 514227.3124375
    window mean = 507478.46859375
    baseline median = 450623.609375
    window median = 445049.296875
    bin_mode = Quantile
    aggregation = Density
    metric = MaxDiff
    weighted = False
    score = 0.034
    scores = [0.002, 0.021999999999999992, 0.034, 0.025000000000000026, 0.021999999999999992, 0.01100000000000001, 0.0]
    index = 2

    
{{<figure src="/images/2024.2/wallaroo-use-case-tutorials/classification/cybersecurity/N4_drift_detection-with-code-reference_files/N4_drift_detection-with-code-reference_41_1.png" width="800" label="png">}}
    

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# set metric SUMDIFF mode
assay_baseline.summarizer_builder.add_metric(wallaroo.assay_config.Metric.SUMDIFF)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# display one analysis from the results
assay_results[0].chart()
```

    baseline mean = 0.05189906566111312
    window mean = 0.26292607237398624
    baseline median = 0.03296763077378273
    window median = 0.2469722330570221
    bin_mode = Quantile
    aggregation = Density
    metric = SumDiff
    weighted = False
    score = 1.0
    scores = [0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0]
    index = None

    
{{<figure src="/images/2024.2/wallaroo-use-case-tutorials/classification/cybersecurity/N4_drift_detection-with-code-reference_files/N4_drift_detection-with-code-reference_42_1.png" width="800" label="png">}}
    

##### Alert Threshold

Assay alert thresholds are modified with the [`wallaroo.assay_config.AssayBuilder.add_alert_threshold(alert_threshold: float)`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/) method.  By default alert thresholds are `0.1`.

The following example updates the alert threshold to `0.5`.

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# set the alert threshold
assay_baseline.add_alert_threshold(0.5)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# show the analyses with the alert threshold
assay_results.to_dataframe()
```

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# set the alert threshold
assay_baseline.add_alert_threshold(0.5)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# show the analyses with the alert threshold
assay_results.to_dataframe()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>assay_id</th>
      <th>assay_name</th>
      <th>iopath</th>
      <th>pipeline_id</th>
      <th>pipeline_name</th>
      <th>score</th>
      <th>start</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>median</th>
      <th>std</th>
      <th>warning_threshold</th>
      <th>alert_threshold</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>None</td>
      <td>assays from date baseline</td>
      <td></td>
      <td>None</td>
      <td></td>
      <td>8.868483</td>
      <td>2024-05-01T19:28:01.025108+00:00</td>
      <td>0.200003</td>
      <td>0.567695</td>
      <td>0.262926</td>
      <td>0.246972</td>
      <td>0.056658</td>
      <td>None</td>
      <td>0.5</td>
      <td>Alert</td>
    </tr>
  </tbody>
</table>

##### Number of Bins

Number of bins sets how the baseline data is partitioned.  The total number of bins includes the set number plus the left_outlier and the right_outlier, so the total number of bins will be the total set + 2.

The number of bins is set with the [`wallaroo.assay_config.UnivariateContinousSummarizerBuilder.add_num_bins(num_bins: int)`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#UnivariateContinousSummarizerBuilder.add_num_bins) method.

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# update number of bins here
assay_baseline.summarizer_builder.add_num_bins(10)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# show one analysis with the updated bins
assay_results[0].chart()
```

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# update number of bins here
assay_baseline.summarizer_builder.add_num_bins(10)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# show one analysis with the updated bins
assay_results[0].chart()
```

    baseline mean = 0.05189906566111312
    window mean = 0.26292607237398624
    baseline median = 0.03296763077378273
    window median = 0.2469722330570221
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 8.117771439591587
    scores = [0.0, 0.28459456598762917, 0.28459456598762917, 0.28459456598762917, 0.28459456598762917, 0.28459456598762917, 0.28459456598762917, 0.28459456598762917, 0.28459456598762917, 0.28459456598762917, 0.28459456598762917, 5.2718257797152965]
    index = None

    /Users/johnhansarick/.virtualenvs/wallaroosdk2024.1/lib/python3.8/site-packages/wallaroo/assay.py:422: UserWarning: FixedFormatter should only be used together with FixedLocator
      ax.set_xticklabels(labels=edge_names, rotation=45)

    
{{<figure src="/images/2024.2/wallaroo-use-case-tutorials/classification/cybersecurity/N4_drift_detection-with-code-reference_files/N4_drift_detection-with-code-reference_46_2.png" width="800" label="png">}}
    

##### Binning Mode

Binning Mode defines how the bins are separated.  Binning modes are modified through the `wallaroo.assay_config.UnivariateContinousSummarizerBuilder.add_bin_mode(bin_mode: bin_mode: wallaroo.assay_config.BinMode, edges: Optional[List[float]] = None)`.

Available `bin_mode` values from `wallaroo.assay_config.Binmode` are the following:

* `QUANTILE` (*Default*): Based on percentages. If `num_bins` is 5 then quintiles so bins are created at the 20%, 40%, 60%, 80% and 100% points.
* `EQUAL`: Evenly spaced bins where each bin is set with the formula `min - max / num_bins`
* `PROVIDED`: The user provides the edge points for the bins.

If `PROVIDED` is supplied, then a List of float values must be provided for the `edges` parameter that matches the number of bins.

The following examples are used to show how each of the binning modes effects the bins.

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# update binning mode here
assay_baseline.summarizer_builder.add_bin_mode(wallaroo.assay_config.BinMode.QUANTILE)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# show one analysis with the updated bins
assay_results[0].chart()
```

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# update binning mode here
assay_baseline.summarizer_builder.add_bin_mode(wallaroo.assay_config.BinMode.QUANTILE)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# show one analysis with the updated bins
assay_results[0].chart()
```

    baseline mean = 0.05189906566111312
    window mean = 0.26292607237398624
    baseline median = 0.03296763077378273
    window median = 0.2469722330570221
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 8.868483247476384
    scores = [0.0, 0.7193314935522176, 0.7193314935522176, 0.7193314935522176, 0.7193314935522176, 0.7193314935522176, 5.2718257797152965]
    index = None

    
{{<figure src="/images/2024.2/wallaroo-use-case-tutorials/classification/cybersecurity/N4_drift_detection-with-code-reference_files/N4_drift_detection-with-code-reference_48_1.png" width="800" label="png">}}
    

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# update binning mode here
assay_baseline.summarizer_builder.add_bin_mode(wallaroo.assay_config.BinMode.EQUAL)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# show one analysis with the updated bins
assay_results[0].chart()
```

    baseline mean = 0.05189906566111312
    window mean = 0.26292607237398624
    baseline median = 0.03296763077378273
    window median = 0.2469722330570221
    bin_mode = Equal
    aggregation = Density
    metric = PSI
    weighted = False
    score = 9.22754688491299
    scores = [0.0, 2.595943040380882, 0.6637417412224403, 0.34900090293262137, 0.23043766008937197, 0.11659776057237851, 5.2718257797152965]
    index = None

    
{{<figure src="/images/2024.2/wallaroo-use-case-tutorials/classification/cybersecurity/N4_drift_detection-with-code-reference_files/N4_drift_detection-with-code-reference_49_1.png" width="800" label="png">}}
    

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

edges = [200000.0, 400000.0, 600000.0, 800000.0, 1500000.0, 2000000.0]

# update binning mode here
assay_baseline.summarizer_builder.add_bin_mode(wallaroo.assay_config.BinMode.PROVIDED, edges)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# show one analysis with the updated bins
assay_results[0].chart()
```

    baseline mean = 0.05189906566111312
    window mean = 0.26292607237398624
    baseline median = 0.03296763077378273
    window median = 0.2469722330570221
    bin_mode = Provided
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.0
    scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    index = None

    
{{<figure src="/images/2024.2/wallaroo-use-case-tutorials/classification/cybersecurity/N4_drift_detection-with-code-reference_files/N4_drift_detection-with-code-reference_50_1.png" width="800" label="png">}}
    

##### Aggregation Options

Assay aggregation options are modified with the [`wallaroo.assay_config.AssayBuilder.add_aggregation(aggregation: wallaroo.assay_config.Aggregation)`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#UnivariateContinousSummarizerBuilder.add_aggregation) method.  The following options are provided:

* `Aggregation.DENSITY` (*Default*): Count the number/percentage of values that fall in each bin. 
* `Aggregation.CUMULATIVE`: Empirical Cumulative Density Function style, which keeps a cumulative count of the values/percentages that fall in each bin.

The following example demonstrate the different results between the two.

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

#Aggregation.DENSITY - the default
assay_baseline.summarizer_builder.add_aggregation(wallaroo.assay_config.Aggregation.DENSITY)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# show one analysis with the updated bins
assay_results[0].chart()
```

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

#Aggregation.DENSITY - the default
assay_baseline.summarizer_builder.add_aggregation(wallaroo.assay_config.Aggregation.DENSITY)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# show one analysis with the updated bins
assay_results[0].chart()
```

    baseline mean = 0.05189906566111312
    window mean = 0.26292607237398624
    baseline median = 0.03296763077378273
    window median = 0.2469722330570221
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 8.868483247476384
    scores = [0.0, 0.7193314935522176, 0.7193314935522176, 0.7193314935522176, 0.7193314935522176, 0.7193314935522176, 5.2718257797152965]
    index = None

    
{{<figure src="/images/2024.2/wallaroo-use-case-tutorials/classification/cybersecurity/N4_drift_detection-with-code-reference_files/N4_drift_detection-with-code-reference_52_1.png" width="800" label="png">}}
    

```python
# Create the assay baseline
assay_baseline = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the assay parameters

# The end date to gather inference results
assay_baseline.add_run_until(datetime.datetime.now())

# Set the interval and window to one minute each, set the start date for gathering inference results
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

#Aggregation.CUMULATIVE - the default
assay_baseline.summarizer_builder.add_aggregation(wallaroo.assay_config.Aggregation.CUMULATIVE)

# build the assay configuration
assay_config = assay_baseline.build()

# perform an interactive run and collect inference data
assay_results = assay_config.interactive_run()

# show one analysis with the updated bins
assay_results[0].chart()
```

    baseline mean = 0.05189906566111312
    window mean = 0.26292607237398624
    baseline median = 0.03296763077378273
    window median = 0.2469722330570221
    bin_mode = Quantile
    aggregation = Cumulative
    metric = PSI
    weighted = False
    score = 3.0
    scores = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.0]
    index = None

    
{{<figure src="/images/2024.2/wallaroo-use-case-tutorials/classification/cybersecurity/N4_drift_detection-with-code-reference_files/N4_drift_detection-with-code-reference_53_1.png" width="800" label="png">}}
    

### Create Assay

With the assay previewed and configuration options determined, we officially create it by uploading it to the Wallaroo instance.

Once it is uploaded, the assay runs an analysis based on the window width, interval, and the other settings configured.

Assays are uploaded with the `wallaroo.assay_config.upload()` method. This uploads the assay into the Wallaroo database with the configurations applied and returns the assay id. Note that assay names **must be unique across the Wallaroo instance**; attempting to upload an assay with the same name as an existing one will return an error.

`wallaroo.assay_config.upload()` returns the assay id for the assay.

Typically we would just call `wallaroo.assay_config.upload()` after configuring the assay.  For the example below, we will perform the complete configuration in one window to show all of the configuration steps at once before creating the assay.

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_baseline = wl.build_assay(assay_name="assays from date baseline tutorial", 
                                          pipeline=mainpipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# set the width, interval, and assay start date and time
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# add other options
assay_baseline.summarizer_builder.add_aggregation(wallaroo.assay_config.Aggregation.CUMULATIVE)
assay_baseline.summarizer_builder.add_metric(wallaroo.assay_config.Metric.MAXDIFF)
assay_baseline.add_alert_threshold(0.5)

assay_id = assay_baseline.upload()

# wait 65 seconds for the first analysis run performed
time.sleep(65)

```

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_baseline = wl.build_assay(assay_name="assays from date baseline cybersecurity", 
                                          pipeline=pipeline, 
                                          iopath="output cryptolocker 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# set the width, interval, and assay start date and time
assay_baseline.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# add other options
assay_baseline.summarizer_builder.add_aggregation(wallaroo.assay_config.Aggregation.CUMULATIVE)
assay_baseline.summarizer_builder.add_metric(wallaroo.assay_config.Metric.MAXDIFF)
assay_baseline.add_alert_threshold(0.5)

assay_id = assay_baseline.upload()

# wait 65 seconds for the first analysis run performed
time.sleep(65)

```

### Get Assay Info

Assay information is retrieved with the `wallaroo.client.get_assay_info()` which takes the following parameters.

| Parameter | Type | Description |
|---|---|---|
| **assay_id** | *Integer* (*Required*) | The numerical id of the assay. |

This returns the following:

| Parameter | Type | Description |
|---|---|---|
| *id* | *Integer* | The numerical id of the assay. |
| *name* | *String* | The name of the assay. |
| *active* | *Boolean* | `True`: The assay is active and generates analyses based on its configuration. `False`: The assay is disabled and will not generate new analyses. |
| *pipeline_name* | *String* | The name of the pipeline the assay references. |
| *last_run* | *DateTime* | The date and time the assay last ran. |
| *next_run* | *DateTime* | THe date and time the assay analysis will next run. |
| *alert_threshold* | *Float* | The alert threshold setting for the assay. |
| *baseline* | *Dict* | The baseline and settings as set from the assay configuration. |
| *iopath* | *String* | The `iopath` setting for the assay. |
| *metric* | *String* | The `metric` setting for the assay. |
| *num_bins* | *Integer* | The number of bins for the assay. |
| *bin_weights* | *List*/None | The bin weights used if any. |
| *bin_mode* | *String* | The binning mode used. |

```python
display(wl.get_assay_info(assay_id))
```

```python
display(wl.get_assay_info(assay_id))
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>active</th>
      <th>status</th>
      <th>pipeline_name</th>
      <th>last_run</th>
      <th>next_run</th>
      <th>alert_threshold</th>
      <th>baseline</th>
      <th>iopath</th>
      <th>metric</th>
      <th>num_bins</th>
      <th>bin_weights</th>
      <th>bin_mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>assays from date baseline cybersecurity</td>
      <td>True</td>
      <td>created</td>
      <td>aloha-fraud-detector</td>
      <td>None</td>
      <td>2024-05-01T19:46:45.46681+00:00</td>
      <td>0.5</td>
      <td>Start:2024-05-01T19:20:32.870761+00:00, End:2024-05-01T19:21:34.718761+00:00</td>
      <td>output cryptolocker 0</td>
      <td>MaxDiff</td>
      <td>5</td>
      <td>None</td>
      <td>Quantile</td>
    </tr>
  </tbody>
</table>

### Get Assay Results

Once an assay is created the assay runs an analysis based on the window width, interval, and the other settings configured.

Assay results are retrieved with the `wallaroo.client.get_assay_results` method which takes the following parameters:

| Parameter | Type | Description |
|---|---|---|
| **assay_id** | *Integer* (*Required*) | The numerical id of the assay. |
| **start** | *Datetime.Datetime* (*Required*) | The start date and time of historical data from the pipeline to start analyses from. |
| **end** | *Datetime.Datetime* (*Required*) | The end date and time of historical data from the pipeline to limit analyses to. |

* **IMPORTANT NOTE**:  This process requires that additional historical data is generated from the time the assay is created to when the results are available. To add additional inference data, use the [Assay Test Data](#assay-test-data) section above.

```python
assay_results = wl.get_assay_results(assay_id=assay_id,
                     start=assay_window_start,
                     end=datetime.datetime.now())

assay_results.chart_scores()
```

We'll run some additional inferences to create new inference result data.

```python
deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
pipeline.deploy(deployment_config=deploy_config)
```

<table><tr><th>name</th> <td>aloha-fraud-detector</td></tr><tr><th>created</th> <td>2024-05-01 16:30:53.995114+00:00</td></tr><tr><th>last_updated</th> <td>2024-05-01 19:51:36.073881+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>958397f7-1d3f-472c-8654-21283b8d6d85, ea59aae6-1565-4321-9086-7f8dd2a8e1c2, 41b9260e-21a8-4f16-ad56-c6267d3bae93, 46bdd5a3-fc22-41b7-b1ea-8287c99c241e, 28f5443a-ff67-4f4f-bfc3-f3f95f3c6f83, 435b73f7-76a1-4514-bd41-2cb94d3e78ff, 551242a7-fe4c-4a61-a4c7-e7fcc97509dc, d22eb0d2-9cff-4c5f-a851-10b1a19d8c44, 262909e9-8779-4c56-a994-725ddd0b58c8, 4cdf8e11-1b9c-44ab-a16d-abb054b5e9fe, 6b3529b1-1ff1-454b-8896-460c8c90d667</td></tr><tr><th>steps</th> <td>aloha-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

```python
input_ranges = pd.read_json('../data/data-1k.df.json')
baseline_size = 500

# These inputs will be random samples of small priced houses.  Around 30,000 is a good number
normal_inputs = input_ranges.sample(baseline_size, replace=True).reset_index(drop=True)

normal_results = pipeline.infer(normal_inputs)
# Wait 60 seconds to set this data apart from the rest
time.sleep(60)
```

```python
assay_results = wl.get_assay_results(assay_id=assay_id,
                     start=assay_window_start,
                     end=datetime.datetime.now())

assay_results.chart_scores()
```

    
{{<figure src="/images/2024.2/wallaroo-use-case-tutorials/classification/cybersecurity/N4_drift_detection-with-code-reference_files/N4_drift_detection-with-code-reference_62_0.png" width="800" label="png">}}
    

### List Assays

A list of assays is retrieved with the `wallaroo.client.list_assays()` method.

This returns a Object as a List with the following fields.

| Parameter | Type | Description |
|---|---|---|
| *Assay Id* | *Integer* | The numerical id of the assay. |
| *Assay Name* | *String* | The name of the assay. |
| *Active* | *Boolean* | `True`: The assay is active and generates analyses based on its configuration. `False`: The assay is disabled and will not generate new analyses. |
| *Status* | *Dict* | The status of the assay including the 
| *Warning Threshold* | *Float*/*None* | The warning threshold if set. |
| *Alert Threshold* | *Float* | The alert threshold for the assay. |
| *Pipeline Id* | *Integer* | The numerical id of the pipeline the assay references. |
| *Pipeline Name* | *String* | The name of the pipeline the assay references. |

```python
wl.list_assays()
```

```python
wl.list_assays()
```

<table><tr><th>Assay ID</th><th>Assay Name</th><th>Active</th><th>Status</th><th>Warning Threshold</th><th>Alert Threshold</th><th>Pipeline ID</th><th>Pipeline Name</th></tr><tr><td>3</td><td>assays from date baseline cybersecurity</td><td>True</td><td>{"run_at": "2024-05-01T19:52:51.529708001+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.5</td><td>70</td><td>aloha-fraud-detector</td></tr><tr><td>1</td><td>assays from date baseline tutorial</td><td>False</td><td>{"run_at": "2024-04-24T16:09:37.380793174+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.5</td><td>39</td><td>houseprice-estimator</td></tr></table>

### Set Assay Active Status

Assays active status is either:

* **True**:  The assay generates analyses based on the assay configuration.
* **False**:  The assay will not generate new analyses.

Assays are set to active or not active with the `wallaroo.client.set_assay_active` which takes the following parameters.

| Parameter | Type | Description |
|---|---|---|
| *assay_id* | *Integer* | The numerical id of the assay. |
| *active* | *Boolean* | `True`: The assay status is set to `Active`.  `False`: The assay status is `Not Active`. |

First we will show the current active status.

In the following, set the assay status to `False`, then set the assay active status back to `True`.

```python
display(wl.get_assay_info(assay_id))

wl.set_assay_active(assay_id, False)
display(wl.get_assay_info(assay_id))
```

```python
display(wl.get_assay_info(assay_id))

wl.set_assay_active(assay_id, False)
display(wl.get_assay_info(assay_id))
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>active</th>
      <th>status</th>
      <th>pipeline_name</th>
      <th>last_run</th>
      <th>next_run</th>
      <th>alert_threshold</th>
      <th>baseline</th>
      <th>iopath</th>
      <th>metric</th>
      <th>num_bins</th>
      <th>bin_weights</th>
      <th>bin_mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>assays from date baseline cybersecurity</td>
      <td>True</td>
      <td>{"run_at": "2024-05-01T19:52:51.529708001+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td>
      <td>aloha-fraud-detector</td>
      <td>2024-05-01T19:52:51.529708+00:00</td>
      <td>2024-05-01T19:52:01.025108+00:00</td>
      <td>0.5</td>
      <td>Start:2024-05-01T19:20:32.870761+00:00, End:2024-05-01T19:21:34.718761+00:00</td>
      <td>output cryptolocker 0</td>
      <td>MaxDiff</td>
      <td>5</td>
      <td>None</td>
      <td>Quantile</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>active</th>
      <th>status</th>
      <th>pipeline_name</th>
      <th>last_run</th>
      <th>next_run</th>
      <th>alert_threshold</th>
      <th>baseline</th>
      <th>iopath</th>
      <th>metric</th>
      <th>num_bins</th>
      <th>bin_weights</th>
      <th>bin_mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>assays from date baseline cybersecurity</td>
      <td>False</td>
      <td>{"run_at": "2024-05-01T19:52:51.529708001+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td>
      <td>aloha-fraud-detector</td>
      <td>2024-05-01T19:52:51.529708+00:00</td>
      <td>2024-05-01T19:52:01.025108+00:00</td>
      <td>0.5</td>
      <td>Start:2024-05-01T19:20:32.870761+00:00, End:2024-05-01T19:21:34.718761+00:00</td>
      <td>output cryptolocker 0</td>
      <td>MaxDiff</td>
      <td>5</td>
      <td>None</td>
      <td>Quantile</td>
    </tr>
  </tbody>
</table>

## Cleaning up.

Now that the tutorial is complete, don't forget to undeploy your pipeline to free up the resources.

```python
# blank space to undeploy your pipeline

pipeline.undeploy()

```

<table><tr><th>name</th> <td>aloha-fraud-detector</td></tr><tr><th>created</th> <td>2024-05-01 16:30:53.995114+00:00</td></tr><tr><th>last_updated</th> <td>2024-05-01 19:51:36.073881+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>958397f7-1d3f-472c-8654-21283b8d6d85, ea59aae6-1565-4321-9086-7f8dd2a8e1c2, 41b9260e-21a8-4f16-ad56-c6267d3bae93, 46bdd5a3-fc22-41b7-b1ea-8287c99c241e, 28f5443a-ff67-4f4f-bfc3-f3f95f3c6f83, 435b73f7-76a1-4514-bd41-2cb94d3e78ff, 551242a7-fe4c-4a61-a4c7-e7fcc97509dc, d22eb0d2-9cff-4c5f-a851-10b1a19d8c44, 262909e9-8779-4c56-a994-725ddd0b58c8, 4cdf8e11-1b9c-44ab-a16d-abb054b5e9fe, 6b3529b1-1ff1-454b-8896-460c8c90d667</td></tr><tr><th>steps</th> <td>aloha-prime</td></tr><tr><th>published</th> <td>False</td></tr></table>

You have now walked through setting up a basic assay and running it over historical data.

## Congratulations!

In this tutorial you have:

* Created an assay baseline.
* Previewed the assay based on different configurations.
* Uploaded the assay.

Great job!
