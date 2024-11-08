# Tutorial Notebook 1: Deploy a Model

For this tutorial, let's pretend that you work for a production company looking to promote your movies, and want to make sure the reviews are mostly positive. You have developed a model that determines if a given review is negative or positive based on other reviews.

In this set of exercises, you will used a pre-trained model and deploy it to Wallaroo.  This will require understanding the following concepts:

* [Wallaroo Workspaces](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-workspace/):  Workspaces are environments were users upload models, create pipelines and other artifacts.  The workspace should be considered the fundamental area where work is done.  Workspaces are shared with other users to give them access to the same models, pipelines, etc.
* [Wallaroo Model Upload and Registration](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/): ML Models are uploaded to Wallaroo through the SDK or the MLOps API to a **workspace**.  ML models include default runtimes (ONNX, Python Step, and TensorFlow) that are run directly through the Wallaroo engine, and containerized runtimes (Hugging Face, PyTorch, etc) that are run through in a container through the Wallaroo engine.
* [Wallaroo Pipelines](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/): Pipelines are used to deploy models for inferencing.  Each model is a **pipeline step** in a pipelines, where the inputs of the previous step are fed into the next.  Pipeline steps can be ML models, Python scripts, or Arbitrary Python (these contain necessary models and artifacts for running a model).

For this tutorial, we will be providing pre-trained models in ONNX format.  To see how to upload and deploy your particular model, see the [Wallaroo Documentation site](https://docs.wallaroo.ai).

Before we start, let's load some libraries that we will need for this notebook (note that this may not be a complete list).

* **IMPORTANT NOTE**:  This tutorial is geared towards a Wallaroo 2023.2.1 environment.

```python
# run the following

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

## Get ready to work with Wallaroo

With the libraries loaded, you can log into Wallaroo.  This will provide access to your workspaces, workspaces shared with you from other users, and all other aspects of the Wallaroo environment.

Logging into Wallaroo via the cluster's integrated JupyterLab is quite straight forward:

```python
# Login through local Wallaroo instance 
wl = wallaroo.Client()
```

See [the documentation](https://docs.wallaroo.ai/wallaroo-101/#connect-to-the-wallaroo-instance) if you are logging into Wallaroo some other way such as from a remote location.  This tutorial assumes you're logging in through the Wallaroo JupyterHub service.

Notice that the Wallaroo client connection is stored into a variable called **wl**.  This variable can be anything you want it to be - you can have `client = wallaroo.Client()` or `myWallarooClient = wallaroo.Client()`.

This variable is about to become your best friend - a lot of the commands you'll be running will be through this variable, like `wl.list_workspaces()` to get all of the workspaces available to you in your Wallaroo environment, or `wl.list_models()` to show all of the models in your current workspace.  We'll go into these commands and more - just make sure you saved that Wallaroo client to a variable so you can use it for the other commands.

When we log into the Wallaroo through the SDK, the Client will provide a url to verify your authentication.  Either click it, or copy and past that URL, then authenticate into your Wallaroo instance with your email address and password.

### Login to Wallaroo Exercise

Time to login to your Wallaroo instance.  By now you should be logged into your Wallaroo JupyterHub service and looking at this notebook.

Copy the code below and place it into the code block and run it.  When prompted, select the authentication URL by either clicking it, or copying and pasting it into a browser.  Log into your Wallaroo instance, and then the client will be set.

```python
# Login through local Wallaroo instance 
wl = wallaroo.Client()
```

```python
## blank space to log in 

wl = wallaroo.Client()
```

## Run Client Commands

We're now logged into our Wallaroo instance, let's run some commands to get used to working within the environment.

The following are going to be very useful as you work in Wallaroo.

### List Workspaces

The command `wallaroo.Client.list_workspaces` gives a list of all of the workspaces you have access to in the Wallaroo environment.  Here's an example of running it.  For this example, our Wallaroo client is stored in the `wl` variable, but this could have been named `wallaroo_client` or whatever you like.

```python
wl.list_workspaces()
```

Name | Created At | Users | Models | Pipelines
---|---|---|---|---
john.hummel@wallaroo.ai - Default Workspace | 2023-08-21 19:06:07 | ['john.hummel@wallaroo.ai'] | 1 | 1
edge-publish-demojohn | 2023-08-21 20:54:35 | ['john.hummel@wallaroo.ai'] | 1 | 1
biolabsworkspacegomj | 2023-08-22 15:11:11 | ['john.hummel@wallaroo.ai'] | 1 | 1
biolabsworkspacedtrv | 2023-08-22 16:03:32 | ['john.hummel@wallaroo.ai'] | 1 | 1
biolabsworkspacejohn | 2023-08-22 16:07:15 | ['john.hummel@wallaroo.ai'] | 1 | 1

Listing the workspaces will show the following fields:

* **name**: The user created workspace name.  Workspace names **must** be unique across the Wallaroo instance.
* **created_at**: The date and time the workspace was created.
* **users**: The users in the workspace.  The user who created the workspace is always listed.
* **models**:  The number of models in the workspace.
* **pipelines**:  The number of pipelines in the workspace.

When you first login to Wallaroo, the workspace `{your email address} - Default Workspace` is created.  This is assigned as your **current** workspace once you have logged into Wallaroo.  We'll cover workspaces in a moment - for now, just remember that every time you do `wallaroo.Client()`, that **your** default workspace is set as your **current** workspace, and any commands you issue will be directed to that workspace.

Notice that in that example above, you only see **one** default workspace - the one of the user who ran the `list_workspaces` command.  Other users will have their own default workspaces.  You can only see workspaces **you** have access to.

### List Users

Users are listed with the `wallaroo.Client.list_users` command.  This shows all of the user's email addresses and names across the Wallaroo instance.

There's more commands, but we'll stop here until we've created our workspace and uploaded some models.

### Client Commands Exercise

For this exercise, use your Wallaroo client variable and list the current workspaces and current users.  You can do this with your Wallaroo client variable you saved in a previous step.

For example, if your Wallaroo client variable is `wl`, then the following would list the workspaces and then the users:

```python
# list the workspaces available to you
workspaces = wl.list_workspaces()
print(workspaces)

# list the users
users = wl.list_users()
print(users)
```

```python
## blank space to get workspaces and users

display(wl.list_workspaces())

display(wl.list_users())
```

<table>
    <tr>
        <th>Name</th>
        <th>Created At</th>
        <th>Users</th>
        <th>Models</th>
        <th>Pipelines</th>
    </tr>

<tr >
    <td>john.hansarick@wallaroo.ai - Default Workspace</td>
    <td>2024-10-29 16:12:39</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>0</td>
    <td>0</td>
</tr>

<tr >
    <td>tutorial-workspace-john-sentiment-analysis</td>
    <td>2024-10-29 17:03:43</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>2</td>
    <td>1</td>
</tr>

<tr >
    <td>tutorial-workspace-summarization</td>
    <td>2024-10-29 19:40:06</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>tutorial-workspace-forecast</td>
    <td>2024-10-29 20:52:00</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>3</td>
    <td>1</td>
</tr>

<tr >
    <td>tutorial-finserv-jch</td>
    <td>2024-10-30 21:03:48</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>tutorial-finserv-john</td>
    <td>2024-10-31 18:22:53</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

</table>

    [User({"id": "0e5f6913-ee32-4a03-8c8f-ab6b94d95922", "email": "admin@keycloak", "username": "admin", "enabled": "True),
     User({"id": "fca5c4df-37ac-4a78-9602-dd09ca72bc60", "email": "john.hansarick@wallaroo.ai", "username": "john.hansarick@wallaroo.ai", "enabled": "True)]

## Workspace Creation and Management

A Wallaroo workspace is place to organize the deployment artifacts for a project, and to collaborate with other team members. For more information, see [the Wallaroo 101](https://docs.wallaroo.ai/wallaroo-101/).

When you upload a ML model to Wallaroo, you upload it to a **workspace**.  When a pipeline is created and models assigned to it, that pipeline is in side of a **workspace**.  Wallaroo workload orchestrations?  They're assigned to a workspace.

{{<figure src="../images/workspace_example.png" width="800" label="Sample Wallaroo instance">}}.

When you first login, the SDK assigns you to your **default workspace**.  Most of the commands issued through your Wallaroo client will target that workspace.

You can see what workspace you are currently in with the `wallaroo.Client.get_current_workspace()` method.

This shows you the following workspace fields:

* **name**: The user created workspace name.  Workspace names **must** be unique across the Wallaroo instance.
* **id**: The numerical identifier of the workspace.
* **archived**:  Whether the workspace was archived or not.
* **created_by**:  The Keycloak ID of the user who created the workspace.  This is in UUID format, and is used to identify specific users.  Most of the time you'll refer to users by their email address.
* **
* **created_at**: The date and time the workspace was created.
* **models**: The models uploaded to the workspace and their details.
* **pipelines**:  The number of pipelines in the workspace and their details.

### Get Current Workspace Exercise

Get your current workspace with the `wallaroo.Client.get_current_workspace()`.  For example, if your Wallaroo client was saved to the variable `wl`, this command would be:

```python
wl.get_current_workspace()
```

When done, 

```python
## blank space to get your current workspace

wl.get_current_workspace()
```

    {'name': 'john.hansarick@wallaroo.ai - Default Workspace', 'id': 5, 'archived': False, 'created_by': 'john.hansarick@wallaroo.ai', 'created_at': '2024-10-29T16:12:39.233166+00:00', 'models': [], 'pipelines': []}

### Create New Workspace

Workspaces are created with the `wallaroo.Client.create_workspace(name)`, where `name` is the new name of the workspace.  When a new workspace is created, the workspace user is assigned to the user that created it (in this case - you).  For example, if the Wallaroo client is stored to the variable `wl`, then the following will create the new workspace 'sparkly-bunnies`, then store the workspace information into the variable `workspace`:

```python
workspace = wl.create_workspace('sparkly-bunnies')
```

Once this is created, this shows you the following workspace fields:

* **name**: The user created workspace name.  Workspace names **must** be unique across the Wallaroo instance, and **must** be DNS compliant.  So 'my-cool-workspace' is ok, but '?? workspace' is not.  `-` are ok, but `_` is not.
* **id**: The numerical identifier of the workspace.
* **archived**:  Whether the workspace was archived or not.
* **created_by**:  The Keycloak ID of the user who created the workspace.  This is in UUID format, and is used to identify specific users.  Most of the time you'll refer to users by their email address.
* **
* **created_at**: The date and time the workspace was created.
* **models**: The models uploaded to the workspace and their details.  For a new workspace, this will be empty `[]`.
* **pipelines**:  The number of pipelines in the workspace and their details.  For a new workspace, this will be empty `[]`.

Workspace names **must** be unique.  So the following will **fail**:

```python
wl.create_workspace('sparkly-bunnies')
wl.create_workspace('sparkly-bunnies')
Exception: Failed to create workspace.
```

### Create New Workspace Exercise

Now it's time for us to create our own workspace.  To make this easy, we'll call the workspace 'tutorial-workspace-{firstname}'.  If someone else has the same first name and is in this tutorial, each of you decide who should change their name for this to work.  Or - if it's easier - change the `firstname` to something else like `john1`.

For example, if your Wallaroo client was saved to the variable `wl`, then the command to create a new workspace `tutorial-workspace-sample` is:

```python
wl.create_workspace('tutorial-workspace-sample')
```

When you're done, list the workspaces.  You did that in a previous step, so you can copy that here.

```python
## blank space to create your workspace

print(wl.create_workspace('tutorial-workspace-sentiment-analysis'))

# list all the workspaces here

wl.list_workspaces()
```

    {'name': 'tutorial-workspace-sentiment-analysis', 'id': 11, 'archived': False, 'created_by': 'fca5c4df-37ac-4a78-9602-dd09ca72bc60', 'created_at': '2024-11-01T17:41:05.46033+00:00', 'models': [], 'pipelines': []}

<table>
    <tr>
        <th>Name</th>
        <th>Created At</th>
        <th>Users</th>
        <th>Models</th>
        <th>Pipelines</th>
    </tr>

<tr >
    <td>john.hansarick@wallaroo.ai - Default Workspace</td>
    <td>2024-10-29 16:12:39</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>0</td>
    <td>0</td>
</tr>

<tr >
    <td>tutorial-workspace-john-sentiment-analysis</td>
    <td>2024-10-29 17:03:43</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>2</td>
    <td>1</td>
</tr>

<tr >
    <td>tutorial-workspace-summarization</td>
    <td>2024-10-29 19:40:06</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>tutorial-workspace-forecast</td>
    <td>2024-10-29 20:52:00</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>3</td>
    <td>1</td>
</tr>

<tr >
    <td>tutorial-finserv-jch</td>
    <td>2024-10-30 21:03:48</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>tutorial-finserv-john</td>
    <td>2024-10-31 18:22:53</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>tutorial-workspace-sentiment-analysis</td>
    <td>2024-11-01 17:41:05</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>0</td>
    <td>0</td>
</tr>

</table>

## Retrieve Workspace

In the above example, you saw this:

```python
workspace = wl.create_workspace('sparkly-bunnies')
```

This creates the workspace `sparkly-bunnies`, then assigns it to the variable `workspace`.  We can display that workspace and see what it looks like.  For example:

```python
workspace = wl.create_workspace('sparkly-bunnies')

print(workspace)
{'name': 'sparkly-bunnies', 'id': 9, 'archived': False, 'created_by': '66d3b2c4-9b22-4429-b16e-3bcdc1ac28e3', 'created_at': '2023-08-22T17:30:40.475785+00:00', 'models': [], 'pipelines': []}
```

If we had created a workspace earlier, and want to work with it, we will have to retrieve it.  We do that with the `wallaroo.Client.get_workspace(name, create_if_not_exist: False)` method, which returns a reference to the workspace.  Then we can set the workspace we want to a variable.

Here's an example.  We start with list_workspaces:

```python
workspace = wl.get_workspace(sparkly-bunnies)
print(workspace)

{'name': 'sparkly-bunnies', 'id': 9, 'archived': False, 'created_by': '66d3b2c4-9b22-4429-b16e-3bcdc1ac28e3', 'created_at': '2023-08-22T17:30:40.475785+00:00', 'models': [], 'pipelines': []}
```

Note that the parameter `create_if_not_exist` will either retrieve the workspace with the same name, **or** if `create_if_not_exist` is set to `True`, then the workspace will be **created** if it does not already exist.

### Retrieve Workspace Exercise

Retrieve the workspace to a variable you created earlier through the following steps:

1. List the workspaces.
1. Determine the position of the workspace from the list - remember the list positions start at **0**.
1. Assign the workspace to a variable that you'll use later.

For example, if the workspace is position 1 in the `list_workspaces` list, then you would retrieve the workspace like so:

```python
workspace = wl.get_workspace("my-workspace")
```

```python
## blank space to retrieve your workspace

workspace = wl.get_workspace('tutorial-workspace-sentiment-analysis')
print(workspace)
```

    {'name': 'tutorial-workspace-sentiment-analysis', 'id': 11, 'archived': False, 'created_by': 'fca5c4df-37ac-4a78-9602-dd09ca72bc60', 'created_at': '2024-11-01T17:41:05.46033+00:00', 'models': [], 'pipelines': []}

## Set the Current Workspace

We mentioned earlier that when you login to Wallaroo, the SDK assigns you to the **default workspace** - the one named `{your email address} - Default Workspace` - replacing your email address in the front.

Usually you'll want to work in some other workspace - perhaps one that you're a part of with other users, or one you set up yourself for test purposes.  It is highly recommended that workspaces be divided by project or some specific goal where the same models are used for different purposes.

We've gone over how to create a workspace, and how to retrieve a workspace that was previously created.  Now we'll use that to set our **current workspace** with the `wallaroo.Client.set_current_workspace(workspace)`.

The **current workspace** is where your SDK commands are routed to.  When you give the upload models command - they are uploaded to your **current workspace**.  Build a pipeline?  Associated to the current workspace.

So we're going to make sure that what we're doing is done in the right workspace with two commands:

* `wallaroo.Client.get_current_workspace()`: Shows what the current workspace.
* `wallaroo.Client.set_current_workspace(workspace)`: Sets the current workspace to the target workspace.

For example, if your workspace is saved to a variable as shown in the previous step, we can change from the default workspace and set our current workspace to the new one as follows:

```python
# show the current workspace
print(wl.get_current_workspace())
{'name': 'john.hummel@wallaroo.ai - Default Workspace', 'id': 1, 'archived': False, 'created_by': '66d3b2c4-9b22-4429-b16e-3bcdc1ac28e3', 'created_at': '2023-08-21T19:06:07.404363+00:00', 'models': [{'name': 'm1', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 21, 19, 38, 36, 672465, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 21, 19, 38, 36, 672465, tzinfo=tzutc())}], 'pipelines': [{'name': 'p1', 'create_time': datetime.datetime(2023, 8, 21, 19, 38, 44, 314377, tzinfo=tzutc()), 'definition': '[]'}]}

# change to the new workspace
wl.set_current_workspace(workspace)

# show the new current workspace
print(wl.get_current_workspace())
{'name': 'tutorial-workspace-john', 'id': 12, 'archived': False, 'created_by': '66d3b2c4-9b22-4429-b16e-3bcdc1ac28e3', 'created_at': '2023-08-22T17:38:13.612187+00:00', 'models': [], 'pipelines': []}
```

Setting your current workspace to the one you want to work in is an **important** step.  We highly recommend that once a Wallaroo client connection is established, the next task should be setting whatever workspace is the proper one to work in as the current workspace, then proceeding with any other tasks.

## Set the Current Workspace Exercise

Previously you created a workspace and retrieved it to a variable.  Using the * `wallaroo.Client.get_current_workspace()` and `wallaroo.Client.set_current_workspace(workspace)` methods:

1. Get your current workspace.
1. Set your current workspace to the one created in the previous steps.
1. Get your current workspace again to verify that the change was made.

For example, if your Wallaroo client was stored as the variable `wl`, and your new workspace saved to the variable `workspace`, you can change your current workspace to the new one with the following:

```python
wl.set_current_workspace(workspace)
wl.get_current_workspace()
```

```python
## blank space to set the current workspace to the new workspace

print(wl.get_current_workspace())
wl.set_current_workspace(workspace)
print(wl.get_current_workspace())
```

    {'name': 'john.hansarick@wallaroo.ai - Default Workspace', 'id': 5, 'archived': False, 'created_by': 'john.hansarick@wallaroo.ai', 'created_at': '2024-10-29T16:12:39.233166+00:00', 'models': [], 'pipelines': []}
    {'name': 'tutorial-workspace-sentiment-analysis', 'id': 11, 'archived': False, 'created_by': 'fca5c4df-37ac-4a78-9602-dd09ca72bc60', 'created_at': '2024-11-01T17:41:05.46033+00:00', 'models': [], 'pipelines': []}

## Upload a Model

Now that we have our current workspace set, it's time to start uploading some models.

We already have some ONNX models available in the folder `./models`.  All three do the same thing:  predict house price values based on some values.  We have those values stored in the folder `./data`.

ML Models are uploaded to the **current Wallaroo workspace** the `wallaroo.Client.upload_model` method.  Wallaroo supports different model types, as well as Arbitrary Python and containerized models.

For full details, see [Wallaroo SDK Essentials Guide: Model Uploads and Registrations](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/).

Wallaroo supports ONNX models as part of the default runtime, so these will run in Wallaroo without additional configurations.

When uploading models, the following is needed:

* The **name** of the model.  This needs to be unique across the **workspace**.  
* The **path** to the ML model file.  For example, `./models/xgb_model.onnx`.
* The **framework** of the model.  These are listed through the `wallaroo.framework.Framework` list.  For these examples we will be using `wallaroo.framework.Framework.ONNX` to specify we are using ONNX models.
* The **input_schema** and **output_schema**.  For ONNX models, we can skip this.  For non-native runtime models, that has to be specified in Apache Arrow schema format.  See the [Wallaroo SDK Essentials Guide: Model Uploads and Registrations](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/) for full details.

Here's an example of uploading a model to a Wallaroo workspace and assigning it the name 'house-price-prime', with the Wallaroo client assigned to the variable `wl`, then retrieving the model version from Wallaroo once the upload it complete:

```python
house_price_model_version = wl.upload_model('house-price-prime',
                                            './models/xgb_model.onnx',
                                            framework=wallaroo.framework.Framework.ONNX)
house_price_model_version
```

Name | house-price-prime
---|---
Version | cc3ba784-ffdf-4a0f-982a-9a8ac4db8ba9
File Name | xgb_model.onnx
SHA | 31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c
Status | ready
Image Path | None
Updated At | 2023-22-Aug 19:55:26

We store this new **version** of a model to the variable `house_price_model_version`.  This is used for later processes involving pipeline deployments, generating assays, and so on.

In Wallaroo, you have the **model**, which is based on the name parameter.  Each model has one or more **versions**.

**IMPORTANT NOTE**:  Models in Wallaroo are organized by **name**.  If a model is uploaded with the same name, it will create a new **version** of the model with the same name.  For example, the following will create a model named `house-price-prime` with two versions.

```python
# set the model from the XGB model converted to ONNX
house_price_model_version = wl.upload_model('house-price-prime',
                                            './models/xgb_model.onnx',
                                            framework=wallaroo.framework.Framework.ONNX)
print(house_price_model_version)
{'name': 'house-price-prime', 'version': '83d89260-9aac-41ea-b2b4-79aae48b5a65', 'file_name': 'xgb_model.onnx', 'image_path': None, 'last_update_time': datetime.datetime(2023, 8, 22, 19, 59, 2, 26718, tzinfo=tzutc())}

# create the new model version to the model converted from an RF model
house_price_model_version = wl.upload_model('house-price-prime',
                                            './models/rf_model.onnx',
                                            framework=wallaroo.framework.Framework.ONNX)
print(house_price_model_version)
{'name': 'house-price-prime', 'version': 'c86fd309-7c28-4e95-9d3e-831fefa51a12', 'file_name': 'rf_model.onnx', 'image_path': None, 'last_update_time': datetime.datetime(2023, 8, 22, 19, 59, 3, 381581, tzinfo=tzutc())}
```

Notice that the model is the same - `house-price-prime` - but the model **version** changes each time we do an `upload_model`.  This allows you to change the model version to a totally different flavor and framework if you desire.

### Upload a Model Exercise

For this exercise, upload the model `/models/xgb_model.onnx` and assign it a name, with the framework `=wallaroo.framework.Framework.ONNX`.  For example, if the Wallaroo client is saved to the variable `wl` and we want to name out model `house-price-prime`, we would do the following:

```python
wl.upload_model('house-price-prime',
                './models/xgb_model.onnx',
                framework=wallaroo.framework.Framework.ONNX)
```

```python
## blank space to upload model, and create the pipeline

from wallaroo.framework import Framework

wl.upload_model('embedder', '../models/embedder.onnx', framework=Framework.ONNX).configure(tensor_fields=["tensor"])
wl.upload_model('sentiment', '../models/sentiment_model.onnx', framework=Framework.ONNX).configure(tensor_fields=["flatten_1"])
```

<table>
        <tr>
          <td>Name</td>
          <td>sentiment</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>ae12f23d-0597-47c0-a381-7c1e07432d91</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>sentiment_model.onnx</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>3473ea8700fbf1a1a8bfb112554a0dde8aab36758030dcde94a9357a83fd5650</td>
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
          <td>2024-01-Nov 17:41:17</td>
        </tr>
        <tr>
          <td>Workspace id</td>
          <td>11</td>
        </tr>
        <tr>
          <td>Workspace name</td>
          <td>tutorial-workspace-sentiment-analysis</td>
        </tr>
      </table>

## Retrieve Model Version

Once a model is uploaded to Wallaroo, we can list the models in a workspace with the `wallaroo.workspace.models()` method.  This returns a List of all of the models and how many versions are associated with that model.

Here's an example:

```python
workspace.models()
[{'name': 'house-price-prime', 'versions': 3, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 22, 19, 59, 3, 381581, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 22, 19, 55, 26, 603685, tzinfo=tzutc())}]
```

We can retrieve the model with the `wallaroo.client.Client.get_model(name, version)` retrieves the most recent model version that matches the model name **in the current workspace**.

The optional parameter `version` retrieves the specific model version that matches both the model name and the model version in the current workspace.

For example:

```python
my_model = wl.get_model(name="house-price-prime")
my_model
```

Name | house-price-prime
|---|---|
|# of Versions | 3|
Owner ID | ""
Last Updated | 2023-08-22 19:59:03.381581+00:00
Created At | 2023-08-22 19:55:26.603685+00:00

### Retrieve Model Version Exercise

This exercise will have you retrieving the model you uploaded earlier.  For example, if the Wallaroo client was stored as `wl`, then the command to get the current model version would be:

```python
my_model_version = wl.get_model("my-sample-model")
my_model_version
```

```python
## blank space to retrieve the model version and store it

sentiment_model_version = wl.get_model('sentiment')
embedder_version = wl.get_model('embedder')
display(embedder_version)
display(sentiment_model_version)
```

<table>
        <tr>
          <td>Name</td>
          <td>embedder</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>71aaa7f9-f754-4b71-b2e0-e030ae4d5be2</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>embedder.onnx</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>d083fd87fa84451904f71ab8b9adfa88580beb92ca77c046800f79780a20b7e4</td>
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
          <td>2024-01-Nov 17:41:16</td>
        </tr>
        <tr>
          <td>Workspace id</td>
          <td>11</td>
        </tr>
        <tr>
          <td>Workspace name</td>
          <td>tutorial-workspace-sentiment-analysis</td>
        </tr>
      </table>

<table>
        <tr>
          <td>Name</td>
          <td>sentiment</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>ae12f23d-0597-47c0-a381-7c1e07432d91</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>sentiment_model.onnx</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>3473ea8700fbf1a1a8bfb112554a0dde8aab36758030dcde94a9357a83fd5650</td>
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
          <td>2024-01-Nov 17:41:17</td>
        </tr>
        <tr>
          <td>Workspace id</td>
          <td>11</td>
        </tr>
        <tr>
          <td>Workspace name</td>
          <td>tutorial-workspace-sentiment-analysis</td>
        </tr>
      </table>

## Build a Pipeline

Pipelines are the method of taking submitting data and processing that data through the models. Each pipeline can have one or more **steps** that submit the data from the previous step to the next one. Information can be submitted to a pipeline as a file, or through the pipeline’s URL.

Each pipeline step is a **model version**.  These can be ML models like we uploaded earlier, or they can be Python scripts that manipulate the data into a format needed for another model in the chain.

When an inference is performed, data is submitted to the pipeline.  The pipeline then submits the data to the first step, receives the output, then transmits that data to the next step.  When all steps are complete, the pipeline returns the final values to the requesting client.

For this tutorial, we will focus one a very simple pipeline:  one ML model.

Pipeline are created in the **current workspace** with the `wallaroo.Client.build_pipeline(name)` command, where the name is unique to the **workspace**.  For example, to create the pipeline named `houseprice-estimator` and the Wallaroo client is saved to `wl` the command would be:

```python
pipeline = wl.build_pipeline('houseprice-estimator')
pipeline
```

name | houseprice-estimator
---|---
created | 2023-08-22 20:39:35.853683+00:00
last_updated | 2023-08-22 20:39:35.853683+00:00
deployed | (none)
tags | 
versions | f42c0457-e4f3-4370-b152-0a220347de11
steps | 

Just like models, pipelines have **version**.  Each time pipeline steps are changed, a new version is created.

See [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/) for full details.

### Build a Pipeline

Build your own pipeline!  Use the `wallaroo.Client.build_pipeline(name)` command and create a pipeline named `houseprice-pipeline`.  Recall that this creates the pipeline in the **current workspace**, so verify that the current workspace is the one you want to create a pipeline in.

For example, if the Wallaroo client is saved to the variable `wl`, the command would be:

```python
wl.build_pipeline('houseprice-estimator')
```

```python
## blank space for you to create the pipeline

wl.build_pipeline('imdb-reviewer')
```

<table><tr><th>name</th> <td>imdb-reviewer</td></tr><tr><th>created</th> <td>2024-11-01 17:41:20.204008+00:00</td></tr><tr><th>last_updated</th> <td>2024-11-01 17:41:20.204008+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>workspace_id</th> <td>11</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-sentiment-analysis</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>accel</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6a333530-d25a-4f42-9c74-97b5a96a150a</td></tr><tr><th>steps</th> <td></td></tr><tr><th>published</th> <td>False</td></tr></table>

## Retrieve a Pipeline

Pipelines that are associated with a workspace are retrieved `wallaroo.Client.list_pipelines` method - this returns a List of pipelines.  From there, a variable is assigned to the pipeline in the list we want to work with.

The `list_pipelines` method returns a list of pipelines and their details as follows:

```python
wl.list_pipelines()
```

|name | created | last_updated | deployed | tags | versions | steps
|---|---|---|---|---|---|---|
|houseprice-estimator | 2023-22-Aug 20:39:35 | 2023-22-Aug 20:39:35 | (unknown) |  | f42c0457-e4f3-4370-b152-0a220347de11 | |
|biolabspipeline | 2023-22-Aug 16:07:20 | 2023-22-Aug 16:24:40 | False |  | 4c6dceb7-e692-4b8b-b615-4f7873eb020b, 59d0babe-bc1d-4dbb-959f-711c74f7b05d, ae834c0d-7a5b-4f87-9e2e-1f06f3cd25e7, 7c438222-28d8-4fca-9a70-eabee8a0fac5 | biolabsmodel |
|biolabspipeline | 2023-22-Aug 16:03:33 | 2023-22-Aug 16:03:38 | False |  | 4e103a7d-cd4d-464b-b182-61d4041518a8, ec2a0fd6-21d4-4843-b7c3-65b1e5be1b85, 516f3848-be98-40d7-8564-a1e48eecb7a8 | biolabsmodel |
|biolabspipelinegomj | 2023-22-Aug 15:11:12 | 2023-22-Aug 15:42:44 | False |  | 1dc9f89f-82aa-4a71-b21a-75dc8d5e4e51, 152d12f2-1200-46ad-ad04-60078c5aa284, 6ca59ffd-802e-4ad5-bd9a-35146b9fbda5, bdab08cc-3e99-4afc-b22d-657e33b76f29, 3c8feb0d-3124-4018-8dfa-06162156d51e | biolabsmodelgomj |
|edge-pipeline | 2023-21-Aug 20:54:37 | 2023-22-Aug 19:06:46 | False |  | 2be013d9-a438-453c-a013-3fd8e6218394, a02b6af5-4235-42af-92c6-5ae678b35be4, e721ccad-11d8-4874-8388-4211c4957d18, d642e766-cffb-451f-b197-e058bedbdd5f, eb586aba-4908-4bff-84e1-bdeb1fa4b7d3, 2163d718-a5ea-41e3-b69f-095efa858462 | ccfraud |
|p1 | 2023-21-Aug 19:38:44 | 2023-21-Aug 19:38:44 | (unknown) |  | 5f93e90a-e8d6-4e8a-8a1a-22eee80a3e13, 5f78247f-7bf9-445b-98a6-e146fb22b8e9 | |

Just like with a model version, we can assign a variable to the pipeline via the method `wallaroo.client.Client.get_pipeline(name, version)`, which retrieves the most recent version of the pipeline **in the current workspace** that matches the pipeline name, or the pipeline name **and** the version if the `version` is supplied.

For example:

```python
this_pipeline = wl.get_pipeline(name="houseprice-estimator")
```

## Retrieve a Pipeline Exercise

For this exercise, retrieve your the pipeline you built in the previous step and store it into the variable `my_pipeline`.  You'll use the `get_pipeline` method and supply the name of the pipeline.

```python
my_pipeline = wl.get_pipeline("houseprice-estimator")
my_pipeline
```

```python
## blank space to retrieve your pipeline

my_pipeline = wl.get_pipeline("imdb-reviewer")
my_pipeline
```

<table><tr><th>name</th> <td>imdb-reviewer</td></tr><tr><th>created</th> <td>2024-11-01 17:41:20.204008+00:00</td></tr><tr><th>last_updated</th> <td>2024-11-01 17:41:20.204008+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>workspace_id</th> <td>11</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-sentiment-analysis</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>accel</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6a333530-d25a-4f42-9c74-97b5a96a150a</td></tr><tr><th>steps</th> <td></td></tr><tr><th>published</th> <td>False</td></tr></table>

## Add Model Step

Models are added to a pipeline as pipeline steps.  There are different kinds of pipeline steps that can host one or more models.

For this tutorial, we will use the method `wallaroo.pipeline.add_model_step(model_version)`.  This adds a single step to a Pipeline.  Pipeline steps start at 0 and increment from there.  We can see the steps in our pipeline with the `wallaroo.pipeline.steps()` method.

```python
# add modelA as a pipeline steps
pipeline.add_model_step(modelA)

# display the steps
pipeline.steps()

[{'ModelInference': {'models': [{'name': 'house-price-prime', 'version': 'c86fd309-7c28-4e95-9d3e-831fefa51a12', 'sha': 'e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6'}]}}]
```

For now, there's three commands you should know:

* `wallaroo.pipelineadd_model_step(model_version)`: Add a step to a pipeline from a model version.
* `wallaroo.pipeline.steps()`: Display the current steps in the pipeline.
* `wallaroo.pipeline.clear()`: Clear all pipeline steps.

Pipeline steps **are not saved in the Wallaroo instance** until the pipeline has been deployed - more on that shortly.  So you can add steps, clear them, add new ones - they all stay in your local script until you issue the command to deploy the pipeline.  More on that later.

The second thing to watch out for it every time `add_model_step` is performed on a pipeline, another step is created.  For example, if we have a pipeline with two models, `modelA` and `modelB`, then the following creates two steps in the same pipelines:

```python
# clear the steps
pipeline.clear()

# add modelA then modelB as pipeline steps
pipeline.add_model_step(modelA)
pipeline.add_model_step(modelB)

# display the steps
pipeline.steps()

[{'ModelInference': {'models': [{'name': 'house-price-prime', 'version': 'c86fd309-7c28-4e95-9d3e-831fefa51a12', 'sha': 'e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6'}]}},
 {'ModelInference': {'models': [{'name': 'house-price-gbr', 'version': '248a6eab-b159-4821-830b-22cc137a1ace', 'sha': 'ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a'}]}}]
```

This means the data from the inference is fed first to `modelA`, and that output is fed into `modelB`.  The problem is if the input data from `modelA` doesn't match what `modelB` expects, the process will fail.

Because of this, check your pipeline steps before you deploy a pipeline, clear them if you need to.

## Add Pipeline Step Exercise

We have our model version uploaded from the previous steps, and we have our pipeline.  Time to put them together and create a pipeline step with our model version.

Just for practice, do the following:

1. Clear the pipeline steps.
1. Add the sample model uploaded earlier.  In our examples, that was `my_model_version`.
1. Show the current pipeline steps.

Here's an example with the Wallaroo client stored to `wl`, with the pipeline `my_pipeline` and `my_model_version`:

```python
my_pipeline.clear()
my_pipeline.add_model_step(my_model_version)
my_pipeline.steps()
```

```python
## blank space to set the pipeline step

my_pipeline.clear()
my_pipeline.add_model_step(embedder_version)
my_pipeline.add_model_step(sentiment_model_version)
my_pipeline.steps()

```

    [{'ModelInference': {'models': [{'name': 'embedder', 'version': '71aaa7f9-f754-4b71-b2e0-e030ae4d5be2', 'sha': 'd083fd87fa84451904f71ab8b9adfa88580beb92ca77c046800f79780a20b7e4'}]}},
     {'ModelInference': {'models': [{'name': 'sentiment', 'version': 'ae12f23d-0597-47c0-a381-7c1e07432d91', 'sha': '3473ea8700fbf1a1a8bfb112554a0dde8aab36758030dcde94a9357a83fd5650'}]}}]

## Deploy a Pipeline

Now we reach what we've been aiming for:  deploying a pipeline.

By now, you've seen how workspaces contain the models, pipelines, and other artifacts.  You've uploaded a model and retrieved the latest version of the model.  You've built a pipeline and added the model version as a pipeline step.

Now we will deploy the pipeline.  Deploying a pipeline allocated resources from the cluster to that pipeline for it's use.  The amount of resources has a default value of 4 CPUs, but for this tutorial we'll be adjusting that to just 0.5 cpus and 1 GB RAM per pipeline.

* **IMPORTANT NOTE**:  Please stick to these resource configurations when using a Wallaroo instance with other users.  Otherwise, pipelines might not deploy if the available resources are used up.

To deploy a pipeline, we do two things:

* Create a deployment configuration: This is an optional step, but we will make it mandatory for this tutorial to allow other users to work in the same Wallaroo instance without running out of resources.
* Deploy the pipeline with the deployment configuration:  This is the active step that saves the pipeline steps, and allocates system resources to the pipeline for performing inferences.

Deployment configurations are made with the `wallaroo.DeploymentConfigBuilder()` class, and then we assign the resource settings from there.  This is saved to a variable so we can apply it to our pipeline deployment.

Here's an example of setting up a deployment with just 0.5 cpu and 1Gi RAM:

```python
deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
```

Notice the `replica_count(1)` configuration - this tells Wallaroo to only spin up one replica for this pipeline.  In a production environment, we could spin multiple replicas either manually or automatically as more resources are needed to improve performance.

Now we deploy the pipeline with our deployment configuration with the `wallaroo.pipeline.deploy(deploy_configuration)` method.  If our pipeline variable is `my_pipeline`, then we would deploy it as follows:

```python
deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
my_pipeline.deploy(deployment_config=deploy_config)
```

We can check the status of the pipeline deployment with the `wallaroo.pipeline.status()` method:

```python
my_pipeline.status()

{'status': 'Running',
 'details': [],
 'engines': [{'ip': '10.244.3.83',
   'name': 'engine-6d4fccf5cb-dmwfl',
   'status': 'Running',
   'reason': None,
   'details': [],
   'pipeline_statuses': {'pipelines': [{'id': 'houseprice-estimator',
      'status': 'Running'}]},
   'model_statuses': {'models': [{'name': 'house-price-prime',
      'version': 'c86fd309-7c28-4e95-9d3e-831fefa51a12',
      'sha': 'e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6',
      'status': 'Running'}]}}],
 'engine_lbs': [{'ip': '10.244.4.100',
   'name': 'engine-lb-584f54c899-sswnw',
   'status': 'Running',
   'reason': None,
   'details': []}],
 'sidekicks': []}
```

### Deploy a Pipeline Exercise

This exercise will have you deploy your pipeline with the deployment settings we listed above.  For example, if your pipeline was called `my_pipeline`, then your deployment will look like this:

```python
deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
my_pipeline.deploy(deployment_config=deploy_config)
```

```python
# run the following to set the instructor approved deployment config

deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()

```

```python
## blank space to deploy and check the status

my_pipeline.deploy(deployment_config=deploy_config)
my_pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.28.2.3',
       'name': 'engine-6585955dcf-tzwlm',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'imdb-reviewer',
          'status': 'Running',
          'version': 'f6287d00-b934-4ad8-9e92-cd20d544e6a5'}]},
       'model_statuses': {'models': [{'name': 'sentiment',
          'sha': '3473ea8700fbf1a1a8bfb112554a0dde8aab36758030dcde94a9357a83fd5650',
          'status': 'Running',
          'version': 'ae12f23d-0597-47c0-a381-7c1e07432d91'},
         {'name': 'embedder',
          'sha': 'd083fd87fa84451904f71ab8b9adfa88580beb92ca77c046800f79780a20b7e4',
          'status': 'Running',
          'version': '71aaa7f9-f754-4b71-b2e0-e030ae4d5be2'}]}}],
     'engine_lbs': [{'ip': '10.28.2.2',
       'name': 'engine-lb-6676794678-6phtf',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

## Pipeline Inference with Files

Wallaroo deployed pipelines accept three types of data:

* JSON
* pandas DataFrames
* Apache Arrow

We do this with one of two commands on a **deployed** pipeline.

* `wallaroo.pipeline.infer(input)`: Submits either JSON, a DataFrame, or Apache Arrow to the pipeline for inferences.
* `wallaroo.pipeline.infer_from_file(path)`: Submits either a JSON, a DataFrame in pandas Record format, or an Apache Arrow binary file inferences.

We'll start with a single input file:  `./data/singleton.df.json`, which contains input data as a `tensor`:

```json
[
    {
        "tensor": [
            4.0,
            3.0,
            3710.0,
            20000.0,
            2.0,
            0.0,
            2.0,
            5.0,
            10.0,
            2760.0,
            950.0,
            47.6696014404,
            -122.2610015869,
            3970.0,
            20000.0,
            79.0,
            0.0,
            0.0
        ]
    }
]
```

When we use `infer_from_file`, Wallaroo determines whether the file submitted is one of the three types above, then submits it to the pipeline to perform an inference request.

The data received through the SDK is always of the same type submitted:  Submit a DataFrame, get a DataFrame with the data back.  Submit an Arrow table file, get an Arrow table back.  Here's an example of submitting our sample file through a pipeline saved to the variable `pipeline`: 

```python
result = pipeline.infer_from_file('./data/singleton.df.json')
display(result)
```

| | time | in.tensor | out.variable | check_failures
---|---|---|---|---|
0 | 2023-08-23 15:02:41.452 | [4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696014404, -122.2610015869, 3970.0, 20000.0, 79.0, 0.0, 0.0] | [1514079.4] | 0

Let's break down each of these fields:

* Index (unnamed):  This field doesn't have a label, but in the example above this is the `index`.  We only have one submission, so we have one result.  If we had 20 inputs, we'd have 20 inference results, and each result would be aligned with each row we sent as an input.
* **time**: The date and time the inference request.
* **in.{variable}**: Every input to the inference request is listed as `in.{variable_name}`.  For example, if our inputs were `house_size_in_square_feet` and `year_house_built`, then the inputs would be listed as `in.house_size_in_square_feet` and `in.year_house_built`.
* **out.{variable}**:  Every output to the inference is listed as `out.{variable_name}`.  Our sample model outputs just one output:  `variable`, but others such might output `estimated_house_price`, `initial_offer_price`, and in the inference result those would be listed as `out.estimated_house_price` and `out.initial_offer_price`.
* **check_failures**: Indicates if any validation checks failed.  This is covered in later sessions.

There's additional data that an inference has that is retrieved by requesting it.  For full details, see the [Wallaroo SDK Essentials Guide: Inference Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-inferences/).

## Pipeline Inference with Files Exercise

Let's do an inference with two files:

* `./data/singleton.df.json`: Inference input in DataFrame records format with one input.
* `./data/test_data.df.json`: Inference input in DataFrame records format with over 4,000 inputs.

Use each and perform an inference request through your deployed pipeline with the `infer_from_file` method.  For example, if your pipeline is `my_pipeline`, here's the sample inference requests:

```python
single_result = my_pipeline.infer_from_file('./data/singleton.df.json')
display(single_result)

multiple_result = my_pipeline.infer_from_file('./data/test_data.df.json')
display(multiple_result)
```

```python
## blank space to perform sample inferences

single_result = my_pipeline.infer_from_file('../data/singleton.df.json')
display(single_result)

multiple_result = my_pipeline.infer_from_file('../data/test_data.df.json')
display(multiple_result)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-11-01 17:48:30.297</td>
      <td>[1607.0, 2635.0, 5749.0, 199.0, 49.0, 351.0, 16.0, 2919.0, 159.0, 5092.0, 2457.0, 8.0, 11.0, 1252.0, 507.0, 42.0, 287.0, 316.0, 15.0, 65.0, 136.0, 2.0, 133.0, 16.0, 4311.0, 131.0, 286.0, 153.0, 5.0, 2826.0, 175.0, 54.0, 548.0, 48.0, 1.0, 17.0, 9.0, 183.0, 1.0, 111.0, 15.0, 1.0, 17.0, 284.0, 982.0, 18.0, 28.0, 211.0, 1.0, 1382.0, 8.0, 146.0, 1.0, 19.0, 12.0, 9.0, 13.0, 21.0, 1898.0, 122.0, 14.0, 70.0, 14.0, 9.0, 97.0, 25.0, 74.0, 1.0, 189.0, 12.0, 9.0, 6.0, 31.0, 3.0, 244.0, 2497.0, 3659.0, 2.0, 665.0, 2497.0, 63.0, 180.0, 1.0, 17.0, 6.0, 287.0, 3.0, 646.0, 44.0, 15.0, 161.0, 50.0, 71.0, 438.0, 351.0, 31.0, 5749.0, 2.0, 0.0, 0.0]</td>
      <td>[0.37142318]</td>
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
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[1607.0, 2635.0, 5749.0, 199.0, 49.0, 351.0, 16.0, 2919.0, 159.0, 5092.0, 2457.0, 8.0, 11.0, 1252.0, 507.0, 42.0, 287.0, 316.0, 15.0, 65.0, 136.0, 2.0, 133.0, 16.0, 4311.0, 131.0, 286.0, 153.0, 5.0, 2826.0, 175.0, 54.0, 548.0, 48.0, 1.0, 17.0, 9.0, 183.0, 1.0, 111.0, 15.0, 1.0, 17.0, 284.0, 982.0, 18.0, 28.0, 211.0, 1.0, 1382.0, 8.0, 146.0, 1.0, 19.0, 12.0, 9.0, 13.0, 21.0, 1898.0, 122.0, 14.0, 70.0, 14.0, 9.0, 97.0, 25.0, 74.0, 1.0, 189.0, 12.0, 9.0, 6.0, 31.0, 3.0, 244.0, 2497.0, 3659.0, 2.0, 665.0, 2497.0, 63.0, 180.0, 1.0, 17.0, 6.0, 287.0, 3.0, 646.0, 44.0, 15.0, 161.0, 50.0, 71.0, 438.0, 351.0, 31.0, 5749.0, 2.0, 0.0, 0.0]</td>
      <td>[0.37142318]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[10.0, 25.0, 107.0, 11.0, 17.0, 2.0, 10.0, 119.0, 21.0, 456.0, 15.0, 11.0, 17.0, 6388.0, 10.0, 59.0, 21.0, 101.0, 41.0, 167.0, 5.0, 1447.0, 85.0, 10.0, 78.0, 21.0, 37.0, 11.0, 701.0, 2.0, 91.0, 2080.0, 4786.0, 10.0, 78.0, 21.0, 37.0, 5.0, 847.0, 782.0, 6388.0, 85.0, 10.0, 78.0, 21.0, 388.0, 65.0, 1098.0, 135.0, 59.0, 10.0, 137.0, 5.0, 2317.0, 51.0, 10.0, 244.0, 137.0, 5.0, 2316.0, 39.0, 1.0, 2346.0, 4519.0, 2316.0, 2.0, 1.0, 2346.0, 4519.0, 23.0, 1.0, 6222.0, 10.0, 8289.0, 681.0, 1.0, 7508.0, 5235.0, 78.0, 21.0, 388.0, 1.0, 782.0, 1098.0, 40.0, 37.0, 69.0, 1614.0, 10.0, 77.0, 21.0, 1411.0, 1.0, 2317.0, 1185.0, 54.0, 548.0, 48.0, 10.0, 235.0]</td>
      <td>[0.9655761]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[8.0, 3637.0, 4293.0, 1.0, 4523.0, 2.0, 2869.0, 4402.0, 2312.0, 8937.0, 4895.0, 3.0, 1872.0, 2204.0, 4.0, 695.0, 8461.0, 40.0, 5.0, 76.0, 192.0, 275.0, 5.0, 3164.0, 2.0, 8.0, 1813.0, 6501.0, 24.0, 4095.0, 2.0, 61.0, 5182.0, 6.0, 484.0, 2.0, 257.0, 1276.0, 16.0, 1639.0, 369.0, 7.0, 7.0, 6.0, 32.0, 2497.0, 1147.0, 2.0, 573.0, 355.0, 17.0, 41.0, 32.0, 9111.0, 3729.0, 12.0, 492.0, 3.0, 376.0, 4.0, 501.0, 39.0, 2477.0, 40.0, 5.0, 76.0, 192.0, 275.0, 5.0, 815.0, 4119.0, 2.0, 109.0, 1238.0, 3422.0, 685.0, 5.0, 24.0, 8670.0, 1997.0, 8.0, 16.0, 894.0, 11.0, 106.0, 59.0, 27.0, 1.0, 2606.0, 6725.0, 3528.0, 4.0, 1.0, 2132.0, 1391.0, 2.0, 445.0, 20.0, 11.0, 62.0]</td>
      <td>[0.07601619]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[11.0, 19.0, 6.0, 364.0, 16.0, 3452.0, 7794.0, 2.0, 196.0, 105.0, 560.0, 3.0, 173.0, 5.0, 27.0, 4624.0, 8.0, 1.0, 93.0, 4.0, 65.0, 285.0, 83.0, 196.0, 105.0, 23.0, 52.0, 462.0, 2033.0, 228.0, 9.0, 251.0, 5.0, 64.0, 616.0, 802.0, 1445.0, 330.0, 1080.0, 19.0, 45.0, 2571.0, 2.0, 22.0, 23.0, 914.0, 5.0, 103.0, 3.0, 2273.0, 19.0, 148.0, 4451.0, 124.0, 303.0, 5.0, 25.0, 3.0, 125.0, 4453.0, 1274.0, 10.0, 207.0, 2784.0, 2571.0, 18.0, 15.0, 1.0, 695.0, 43.0, 47.0, 11.0, 215.0, 3.0, 436.0, 131.0, 285.0, 709.0, 187.0, 23.0, 21.0, 1.0, 2154.0, 4.0, 1.0, 201.0, 19.0, 1184.0, 40.0, 1.0, 7076.0, 5002.0, 109.0, 686.0, 2745.0, 300.0, 7.0, 7.0, 14.0, 15.0]</td>
      <td>[0.24645236]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[10.0, 61.0, 927.0, 20.0, 63.0, 52.0, 49.0, 105.0, 2.0, 20.0, 2148.0, 1906.0, 58.0, 5517.0, 6.0, 5.0, 336.0, 81.0, 34.0, 178.0, 5.0, 64.0, 84.0, 105.0, 5.0, 1138.0, 65.0, 55.0, 2.0, 275.0, 6528.0, 7.0, 7.0, 10.0, 79.0, 178.0, 5.0, 567.0, 81.0, 3109.0, 65.0, 55.0, 20.0, 1240.0, 2.0, 178.0, 5.0, 1.0, 189.0, 12.0, 1.0, 164.0, 1323.0, 4.0, 131.0, 1240.0, 105.0, 188.0, 76.0, 242.0, 16.0, 9.0, 15.0, 52.0, 193.0, 72.0, 77.0, 166.0, 43.0, 34.0, 22.0, 23.0, 2.0, 77.0, 2299.0, 16.0, 43.0, 2187.0, 2.0, 7.0, 7.0, 11.0, 19.0, 692.0, 732.0, 80.0, 1.0, 1240.0, 2375.0, 7.0, 7.0, 1.0, 164.0, 2.0, 561.0, 6.0, 305.0, 42.0, 207.0, 3.0]</td>
      <td>[0.08632833]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[51.0, 22.0, 165.0, 30.0, 1.0, 1106.0, 2.0, 329.0, 535.0, 41.0, 9.0, 32.0, 1093.0, 272.0, 549.0, 4.0, 17.0, 263.0, 5.0, 327.0, 71.0, 48.0, 22.0, 76.0, 130.0, 92.0, 171.0, 276.0, 10.0, 329.0, 1.0, 2718.0, 15.0, 1.0, 82.0, 17.0, 443.0, 302.0, 14.0, 47.0, 68.0, 104.0, 99.0, 4.0, 11.0, 422.0, 622.0, 41.0, 1.0, 169.0, 55.0, 16.0, 196.0, 1889.0, 1840.0, 12.0, 66.0, 1316.0, 788.0, 8.0, 1139.0, 187.0, 883.0, 535.0, 41.0, 12.0, 17.0, 130.0, 10.0, 121.0, 10.0, 216.0, 11.0, 28.0, 2.0, 21.0, 12.0, 28.0, 2.0, 12.0, 17.0, 6.0, 57.0, 326.0, 48.0, 28.0, 59.0, 836.0, 3.0, 17.0, 16.0, 12.0, 422.0, 59.0, 27.0, 41.0, 10.0, 77.0, 27.0, 1199.0]</td>
      <td>[0.6396135]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[2108.0, 143.0, 803.0, 18.0, 57.0, 15.0, 1.0, 1351.0, 195.0, 40.0, 93.0, 96.0, 952.0, 5.0, 27.0, 2599.0, 626.0, 22.0, 67.0, 1338.0, 1.0, 201.0, 1.0, 1029.0, 558.0, 85.0, 9.0, 13.0, 151.0, 99.0, 2.0, 245.0, 284.0, 18.0, 195.0, 74.0, 221.0, 277.0, 35.0, 1.0, 751.0, 884.0, 5.0, 27.0, 3.0, 114.0, 326.0, 1247.0, 21.0, 57.0, 50.0, 7546.0, 935.0, 1.0, 83.0, 17.0, 66.0, 1.0, 1368.0, 4.0, 1375.0, 7074.0, 3.0, 129.0, 34.0, 97.0, 57.0, 94.0, 6312.0, 8.0, 5715.0, 303.0, 6977.0, 12.0, 11.0, 28.0, 5928.0, 1501.0, 35.0, 47.0, 13.0, 54.0, 8.0, 230.0, 12.0, 572.0, 9.0, 40.0, 465.0, 707.0, 7.0, 7.0, 3388.0, 10.0, 61.0, 216.0, 11.0, 277.0, 51.0, 10.0]</td>
      <td>[0.024733633]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[2535.0, 9933.0, 16.0, 4753.0, 197.0, 136.0, 2.0, 2338.0, 2104.0, 3438.0, 3.0, 173.0, 4.0, 5041.0, 5.0, 867.0, 140.0, 2.0, 77.0, 239.0, 468.0, 122.0, 88.0, 794.0, 18.0, 1.0, 412.0, 2656.0, 2794.0, 280.0, 2.0, 913.0, 34.0, 9896.0, 24.0, 645.0, 8.0, 217.0, 172.0, 133.0, 79.0, 406.0, 32.0, 1253.0, 1075.0, 236.0, 3.0, 5.0, 27.0, 249.0, 18.0, 1.0, 50.0, 4913.0, 1154.0, 90.0, 104.0, 150.0, 300.0, 6.0, 3.0, 3782.0, 1487.0, 928.0, 10.0, 1462.0, 22.0, 103.0, 12.0, 302.0, 297.0, 238.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]</td>
      <td>[0.50299007]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[51.0, 360.0, 6.0, 266.0, 5.0, 4415.0, 48.0, 32.0, 1086.0, 411.0, 6.0, 37.0, 33.0, 1851.0, 35.0, 3596.0, 1553.0, 9.0, 251.0, 1485.0, 712.0, 8.0, 1.0, 1638.0, 4.0, 1.0, 1258.0, 7.0, 7.0, 205.0, 98.0, 1258.0, 2372.0, 184.0, 266.0, 5.0, 9832.0, 8.0, 24.0, 30.0, 172.0, 1825.0, 249.0, 33.0, 3252.0, 43.0, 8.0, 5987.0, 2.0, 1259.0, 1857.0, 472.0, 42.0, 21.0, 37.0, 33.0, 23.0, 5236.0, 34.0, 112.0, 25.0, 3.0, 7.0, 7.0, 935.0, 44.0, 22.0, 23.0, 3.0, 1258.0, 22.0, 121.0, 29.0, 41.0, 2.0, 476.0, 2.0, 2414.0, 2.0, 4.0, 261.0, 332.0, 1341.0, 53.0, 5.0, 1302.0, 16.0, 2026.0, 684.0, 2.0, 3.0, 4721.0, 4.0, 95.0, 131.0, 180.0, 37.0, 29.0, 137.0, 292.0]</td>
      <td>[0.93422383]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[4955.0, 1007.0, 8902.0, 6.0, 8172.0, 43.0, 4.0, 270.0, 16.0, 11.0, 5301.0, 679.0, 248.0, 1785.0, 2.0, 7233.0, 62.0, 12.0, 2472.0, 36.0, 3.0, 563.0, 580.0, 4.0, 918.0, 8175.0, 183.0, 5.0, 25.0, 9291.0, 55.0, 2.0, 6851.0, 15.0, 1.0, 83.0, 190.0, 14.0, 829.0, 913.0, 2.0, 273.0, 114.0, 778.0, 80.0, 65.0, 552.0, 4491.0, 408.0, 2.0, 206.0, 1705.0, 7.0, 7.0, 3413.0, 7568.0, 7052.0, 5226.0, 492.0, 14.0, 32.0, 1001.0, 129.0, 20.0, 3.0, 1976.0, 289.0, 12.0, 268.0, 75.0, 2.0, 211.0, 24.0, 319.0, 554.0, 8.0, 1.0, 1768.0, 26.0, 2941.0, 36.0, 1168.0, 2.0, 1238.0, 728.0, 43.0, 5.0, 513.0, 1.0, 319.0, 4.0, 1.0, 1252.0, 34.0, 554.0, 24.0, 3099.0, 4.0, 1785.0, 23.0]</td>
      <td>[0.71775126]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[246.0, 17.0, 123.0, 107.0, 246.0, 113.0, 96.0, 10.0, 562.0, 836.0, 3.0, 17.0, 430.0, 92.0, 11.0, 161.0, 5.0, 64.0, 54.0, 113.0, 30.0, 29.0, 827.0, 1398.0, 153.0, 141.0, 165.0, 15.0, 157.0, 289.0, 10.0, 388.0, 34.0, 13.0, 375.0, 192.0, 5.0, 162.0, 273.0, 275.0, 80.0, 11.0, 17.0, 7.0, 7.0, 143.0, 803.0, 15.0, 2127.0, 3337.0, 212.0, 27.0, 1208.0, 10.0, 562.0, 836.0, 86.0, 24.0, 212.0, 27.0, 5.0, 4357.0, 653.0, 1.0, 289.0, 7.0, 7.0, 1.0, 1914.0, 8.0, 1.0, 17.0, 379.0, 33.0, 125.0, 1226.0, 5.0, 7.0, 7.0, 14.0, 15.0, 1.0, 969.0, 129.0, 48.0, 3.0, 482.0, 26.0, 125.0, 27.0, 273.0, 20.0, 3.0, 482.0, 2.0, 787.0, 47.0, 10.0, 67.0, 64.0]</td>
      <td>[0.0020476878]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[10.0, 13.0, 914.0, 5.0, 103.0, 11.0, 19.0, 15.0, 58.0, 179.0, 702.0, 11.0, 19.0, 6.0, 48.0, 6.0, 352.0, 16.0, 932.0, 637.0, 302.0, 4.0, 43.0, 1.0, 115.0, 93.0, 43.0, 4.0, 251.0, 208.0, 39.0, 1182.0, 72.0, 59.0, 244.0, 3822.0, 41.0, 86.0, 9.0, 6.0, 291.0, 5927.0, 2154.0, 11.0, 19.0, 268.0, 140.0, 1.0, 8534.0, 4.0, 2.0, 8234.0, 9.0, 1242.0, 20.0, 828.0, 8.0, 1.0, 1768.0, 4.0, 396.0, 35.0, 2566.0, 268.0, 5.0, 84.0, 5.0, 94.0, 1.0, 7097.0, 4.0, 828.0, 43.0, 5.0, 27.0, 1912.0, 40.0, 85.0, 33.0, 23.0, 396.0, 65.0, 289.0, 8.0, 3.0, 922.0, 2566.0, 105.0, 447.0, 466.0, 1.0, 19.0, 2.0, 124.0, 21.0, 123.0, 57.0, 940.0, 277.0, 44.0]</td>
      <td>[0.35586113]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[70.0, 9.0, 6.0, 41.0, 297.0, 1777.0, 150.0, 8.0, 1.0, 704.0, 2.0, 72.0, 25.0, 414.0, 4255.0, 1.0, 4299.0, 3737.0, 197.0, 18.0, 1035.0, 72.0, 128.0, 358.0, 1865.0, 12.0, 1255.0, 3599.0, 325.0, 346.0, 23.0, 128.0, 2770.0, 254.0, 82.0, 1090.0, 2.0, 394.0, 309.0, 8201.0, 994.0, 522.0, 1139.0, 2737.0, 2607.0, 325.0, 346.0, 23.0, 128.0, 394.0, 1.0, 343.0, 127.0, 4.0, 1.0, 1226.0, 8029.0, 8.0, 2704.0, 325.0, 4695.0, 4490.0, 12.0, 165.0, 37.0, 504.0, 36.0, 1493.0, 488.0, 9678.0, 16.0, 1.0, 2.0, 1048.0, 6614.0, 81.0, 128.0, 5234.0, 5.0, 845.0, 54.0, 3720.0, 39.0, 7942.0, 2989.0, 3992.0, 128.0, 269.0, 2.0, 1421.0, 992.0, 356.0, 10.0, 137.0, 20.0, 8.0, 343.0, 11.0, 845.0, 13.0]</td>
      <td>[0.24872246]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[129.0, 16.0, 1.0, 1936.0, 1221.0, 6.0, 3.0, 62.0, 4.0, 4801.0, 5508.0, 2.0, 1057.0, 8.0, 1.0, 3.0, 389.0, 510.0, 1021.0, 3656.0, 3085.0, 4214.0, 53.0, 16.0, 170.0, 4.0, 24.0, 1221.0, 2952.0, 31.0, 12.0, 4.0, 3.0, 1763.0, 7203.0, 2661.0, 1.0, 104.0, 423.0, 27.0, 50.0, 272.0, 18.0, 33.0, 1493.0, 28.0, 152.0, 196.0, 68.0, 554.0, 31.0, 1.0, 169.0, 252.0, 835.0, 142.0, 5.0, 110.0, 31.0, 3.0, 1165.0, 1654.0, 1021.0, 2.0, 808.0, 32.0, 2386.0, 5.0, 1402.0, 177.0, 65.0, 1139.0, 5458.0, 7.0, 7.0, 1474.0, 4533.0, 1743.0, 5.0, 1.0, 500.0, 186.0, 17.0, 509.0, 12.0, 516.0, 87.0, 24.0, 1212.0, 2676.0, 11.0, 55.0, 21.0, 61.0, 8.0, 1006.0, 4.0, 1.0, 265.0, 18.0]</td>
      <td>[0.27329928]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[10.0, 112.0, 329.0, 1.0, 271.0, 147.0, 10.0, 89.0, 63.0, 178.0, 5.0, 10.0, 66.0, 54.0, 2304.0, 48.0, 11.0, 17.0, 13.0, 41.0, 51.0, 10.0, 2526.0, 80.0, 1.0, 1712.0, 10.0, 128.0, 89.0, 63.0, 121.0, 48.0, 210.0, 9.0, 13.0, 420.0, 5.0, 76.0, 635.0, 18.0, 10.0, 78.0, 121.0, 12.0, 3.0, 49.0, 104.0, 631.0, 13.0, 1050.0, 36.0, 58.0, 110.0, 104.0, 3657.0, 631.0, 10.0, 67.0, 112.0, 76.0, 142.0, 7.0, 7.0, 1.0, 766.0, 13.0, 35.0, 725.0, 42.0, 1319.0, 6633.0, 39.0, 139.0, 3.0, 52.0, 6550.0, 2.0, 6270.0, 549.0, 111.0, 10.0, 8674.0, 1.0, 742.0, 674.0, 232.0, 80.0, 1.0, 17.0, 2.0, 10.0, 13.0, 2289.0, 7.0, 7.0, 1.0, 113.0, 215.0, 524.0, 1.0]</td>
      <td>[0.009601623]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[28.0, 4.0, 1.0, 99.0, 10.0, 40.0, 158.0, 178.0, 5.0, 64.0, 10.0, 185.0, 9.0, 8.0, 1.0, 4968.0, 4434.0, 18.0, 1537.0, 1.0, 113.0, 13.0, 52.0, 75.0, 30.0, 1.0, 127.0, 4.0, 1.0, 17.0, 10.0, 128.0, 241.0, 767.0, 10.0, 293.0, 1.0, 223.0, 17.0, 10.0, 3543.0, 135.0, 10.0, 293.0, 1.0, 17.0, 7.0, 7.0, 79.0, 130.0, 8.0, 1.0, 1.0, 561.0, 4.0, 11.0, 17.0, 42.0, 811.0, 36.0, 3.0, 271.0, 4.0, 194.0, 9.0, 13.0, 52.0, 75.0, 2.0, 13.0, 683.0, 12.0, 24.0, 17.0, 382.0, 43.0, 37.0, 11.0, 372.0, 55.0, 26.0, 490.0, 3.0, 214.0, 8.0, 6020.0, 81.0, 15.0, 1.0, 174.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]</td>
      <td>[0.49502048]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[70.0, 10.0, 66.0, 1.0, 577.0, 5.0, 646.0, 11.0, 19.0, 1.0, 82.0, 248.0, 10.0, 158.0, 121.0, 48.0, 5.0, 532.0, 14.0, 10.0, 112.0, 216.0, 1.0, 1469.0, 2.0, 138.0, 18.0, 48.0, 10.0, 119.0, 1971.0, 328.0, 31.0, 146.0, 1.0, 83.0, 155.0, 232.0, 6.0, 12.0, 11.0, 19.0, 6.0, 1.0, 246.0, 10.0, 25.0, 123.0, 66.0, 1.0, 6284.0, 5.0, 64.0, 7.0, 7.0, 10.0, 654.0, 10.0, 97.0, 199.0, 9.0, 11.0, 19.0, 3.0, 2244.0, 672.0, 1.0, 83.0, 155.0, 232.0, 68.0, 75.0, 18.0, 14.0, 512.0, 14.0, 9.0, 1.0, 1070.0, 133.0, 10.0, 470.0, 5.0, 40.0, 2541.0, 3.0, 7243.0, 9.0, 13.0, 63.0, 859.0, 221.0, 1.0, 153.0, 158.0, 25.0, 98.0, 455.0, 47.0, 13.0]</td>
      <td>[0.08304423]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[11.0, 6.0, 3.0, 17.0, 12.0, 141.0, 25.0, 74.0, 3.0, 2385.0, 198.0, 14.0, 9.0, 494.0, 5.0, 76.0, 96.0, 73.0, 1615.0, 8.0, 96.0, 389.0, 3.0, 831.0, 1.0, 223.0, 62.0, 6.0, 1341.0, 109.0, 16.0, 1491.0, 1840.0, 106.0, 2866.0, 2.0, 4025.0, 1322.0, 1615.0, 12.0, 137.0, 1279.0, 47.0, 6.0, 3.0, 5023.0, 111.0, 118.0, 427.0, 889.0, 3.0, 247.0, 56.0, 45.0, 5021.0, 18.0, 211.0, 1018.0, 6388.0, 2.0, 92.0, 38.0, 5021.0, 2565.0, 2.0, 56.0, 268.0, 5.0, 64.0, 44.0, 33.0, 23.0, 144.0, 33.0, 468.0, 43.0, 21.0, 5.0, 27.0, 18.0, 38.0, 656.0, 525.0, 262.0, 12.0, 56.0, 13.0, 21.0, 2.0, 38.0, 217.0, 1432.0, 149.0, 178.0, 38.0, 14.0, 56.0, 13.0, 21.0, 5.0]</td>
      <td>[0.053483546]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[462.0, 6.0, 3.0, 52.0, 361.0, 349.0, 362.0, 90.0, 31.0, 3.0, 758.0, 4.0, 4222.0, 60.0, 1025.0, 5.0, 213.0, 122.0, 14.0, 3.0, 240.0, 4.0, 2391.0, 1620.0, 17.0, 469.0, 1.0, 335.0, 299.0, 1637.0, 113.0, 2.0, 4960.0, 62.0, 163.0, 11.0, 3.0, 52.0, 1499.0, 186.0, 1175.0, 30.0, 115.0, 31.0, 54.0, 814.0, 6.0, 462.0, 1.0, 246.0, 186.0, 17.0, 204.0, 123.0, 107.0, 9.0, 40.0, 215.0, 230.0, 315.0, 2.0, 45.0, 161.0, 8.0, 9.0, 5.0, 8136.0, 3.0, 330.0, 103.0, 39.0, 437.0, 15.0, 3.0, 751.0, 22.0, 121.0, 3.0, 164.0, 45.0, 5021.0, 41.0, 65.0, 203.0, 186.0, 19.0, 51.0, 3.0, 47.0, 6.0, 46.0, 1147.0, 1003.0, 2.0, 500.0, 1.0, 1324.0, 35.0, 343.0, 33.0]</td>
      <td>[0.027423024]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[3.0, 3515.0, 4.0, 182.0, 1166.0, 2480.0, 1195.0, 5.0, 137.0, 7780.0, 16.0, 3.0, 2492.0, 3.0, 1460.0, 178.0, 5.0, 3558.0, 95.0, 5.0, 3614.0, 1.0, 7537.0, 213.0, 1.0, 288.0, 3082.0, 33.0, 79.0, 25.0, 5.0, 16.0, 32.0, 1392.0, 2.0, 3.0, 964.0, 549.0, 152.0, 287.0, 146.0, 15.0, 61.0, 339.0, 1004.0, 739.0, 7078.0, 3.0, 1035.0, 500.0, 17.0, 6.0, 20.0, 505.0, 14.0, 3.0, 2.0, 1.0, 82.0, 238.0, 4871.0, 5.0, 1.0, 1378.0, 8.0, 28.0, 4.0, 61.0, 339.0, 695.0, 1504.0, 552.0, 56.0, 66.0, 196.0, 25.0, 52.0, 389.0, 552.0, 96.0, 75.0, 282.0, 331.0, 8.0, 1.0, 17.0, 6.0, 75.0, 7.0, 7.0, 58.0, 1239.0, 1092.0, 7.0, 7.0, 285.0, 2256.0, 201.0, 1469.0, 7.0]</td>
      <td>[0.012647837]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[6.0, 267.0, 8.0, 8693.0, 118.0, 104.0, 296.0, 1166.0, 1534.0, 3019.0, 8352.0, 2860.0, 3848.0, 5.0, 2072.0, 716.0, 16.0, 1.0, 4097.0, 4.0, 716.0, 461.0, 47.0, 33.0, 848.0, 4339.0, 8.0, 3.0, 1452.0, 410.0, 1993.0, 41.0, 1.0, 1393.0, 4930.0, 4.0, 3.0, 4007.0, 1663.0, 9.0, 502.0, 43.0, 12.0, 3.0, 975.0, 442.0, 1362.0, 770.0, 453.0, 47.0, 34.0, 45.0, 2585.0, 1.0, 36.0, 24.0, 333.0, 708.0, 301.0, 3.0, 3720.0, 5.0, 1.0, 286.0, 536.0, 514.0, 2767.0, 1.0, 538.0, 4.0, 2085.0, 732.0, 15.0, 3.0, 229.0, 770.0, 485.0, 5077.0, 34.0, 40.0, 35.0, 568.0, 5.0, 27.0, 593.0, 1195.0, 5.0, 765.0, 53.0, 3741.0, 1.0, 179.0, 4.0, 1.0, 442.0, 7.0, 7.0, 523.0, 31.0, 2896.0]</td>
      <td>[0.023909122]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[6.0, 28.0, 4.0, 1.0, 99.0, 118.0, 22.0, 166.0, 1.0, 9554.0, 1510.0, 8.0, 1392.0, 435.0, 130.0, 6.0, 28.0, 49.0, 459.0, 5.0, 120.0, 86.0, 318.0, 924.0, 905.0, 2.0, 153.0, 5.0, 1.0, 1179.0, 130.0, 6.0, 28.0, 4.0, 1.0, 88.0, 386.0, 153.0, 113.0, 8.0, 3.0, 17.0, 41.0, 3.0, 144.0, 62.0, 8.0, 1.0, 8306.0, 4.0, 1.0, 19.0, 6.0, 8827.0, 16.0, 1491.0, 1270.0, 696.0, 136.0, 2.0, 687.0, 47.0, 6.0, 11.0, 106.0, 34.0, 8.0, 632.0, 6.0, 1.0, 489.0, 4.0, 1.0, 1323.0, 4.0, 1.0, 19.0, 3037.0, 26.0, 149.0, 25.0, 3.0, 1150.0, 602.0, 464.0, 24.0, 1909.0, 2252.0, 87.0, 5.0, 78.0, 35.0, 11.0, 550.0, 149.0, 121.0, 48.0, 6.0, 113.0, 2.0]</td>
      <td>[0.86372817]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[11.0, 6.0, 217.0, 1656.0, 2130.0, 3630.0, 1241.0, 12.0, 642.0, 3525.0, 4.0, 3.0, 1545.0, 964.0, 269.0, 37.0, 3.0, 1661.0, 197.0, 3.0, 2801.0, 2.0, 3.0, 75.0, 248.0, 30.0, 1.0, 1049.0, 2.0, 12.0, 478.0, 960.0, 163.0, 635.0, 3.0, 478.0, 37.0, 2.0, 135.0, 1.0, 1460.0, 9823.0, 51.0, 1.0, 2967.0, 91.0, 1482.0, 16.0, 3.0, 4919.0, 4707.0, 72.0, 76.0, 191.0, 488.0, 1970.0, 4.0, 1.0, 1555.0, 5162.0, 2019.0, 18.0, 161.0, 50.0, 10.0, 479.0, 1.0, 1179.0, 194.0, 3.0, 50.0, 5265.0, 3680.0, 191.0, 1638.0, 235.0, 1462.0, 3.0, 1514.0, 2074.0, 209.0, 2.0, 1387.0, 4.0, 1287.0, 29.0, 12.0, 1207.0, 548.0, 6.0, 2812.0, 71.0, 230.0, 8.0, 3.0, 2074.0, 17.0, 10.0, 479.0, 1.0]</td>
      <td>[0.15708977]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[10.0, 13.0, 914.0, 5.0, 329.0, 11.0, 4513.0, 116.0, 62.0, 197.0, 3.0, 1121.0, 3315.0, 288.0, 151.0, 2.0, 3.0, 1686.0, 288.0, 151.0, 9250.0, 12.0, 45.0, 990.0, 8463.0, 395.0, 29.0, 117.0, 9.0, 91.0, 240.0, 4.0, 37.0, 1.0, 17.0, 1102.0, 8.0, 60.0, 3.0, 1813.0, 490.0, 5.0, 27.0, 875.0, 18.0, 145.0, 442.0, 1527.0, 4257.0, 384.0, 9.0, 85.0, 33.0, 356.0, 9.0, 70.0, 10.0, 25.0, 49.0, 1633.0, 1.0, 1527.0, 23.0, 442.0, 8.0, 1.0, 1121.0, 1630.0, 2.0, 24.0, 1502.0, 271.0, 135.0, 3234.0, 118.0, 345.0, 5.0, 175.0, 31.0, 556.0, 2.0, 44.0, 1.0, 1527.0, 884.0, 3.0, 1813.0, 1.0, 67.0, 5459.0, 70.0, 358.0, 9.0, 8.0, 1.0, 169.0, 278.0, 1.0, 1121.0, 66.0]</td>
      <td>[0.34649062]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[1.0, 1584.0, 6.0, 3.0, 52.0, 75.0, 245.0, 17.0, 36.0, 1.0, 3468.0, 1181.0, 1.0, 92.0, 656.0, 319.0, 765.0, 4.0, 667.0, 6353.0, 2.0, 6295.0, 2130.0, 14.0, 1839.0, 56.0, 45.0, 2.0, 829.0, 3.0, 2014.0, 14.0, 3.0, 1716.0, 610.0, 561.0, 237.0, 32.0, 3553.0, 2407.0, 34.0, 6.0, 52.0, 2.0, 424.0, 9725.0, 16.0, 38.0, 65.0, 1584.0, 138.0, 14.0, 9.0, 6.0, 6.0, 4151.0, 36.0, 1.0, 377.0, 2.0, 56.0, 691.0, 9.0, 18.0, 268.0, 364.0, 16.0, 9.0, 551.0, 104.0, 180.0, 5.0, 103.0, 15.0, 44.0, 22.0, 23.0, 2603.0, 80.0, 146.0, 11.0, 1011.0, 4498.0, 12.0, 6.0, 112.0, 54.0, 548.0, 48.0, 2.0, 3.0, 3190.0, 56.0, 3135.0, 399.0, 8.0, 11.0, 2143.0, 507.0, 56.0]</td>
      <td>[0.35645902]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[100.0, 1.0, 828.0, 3.0, 19.0, 12.0, 1.0, 110.0, 4.0, 2114.0, 1.0, 828.0, 5.0, 138.0, 3825.0, 8744.0, 12.0, 9.0, 90.0, 1.0, 855.0, 411.0, 1189.0, 364.0, 213.0, 1844.0, 2.0, 24.0, 3359.0, 1143.0, 4.0, 440.0, 2212.0, 39.0, 1592.0, 2539.0, 1.0, 828.0, 1.0, 113.0, 6.0, 35.0, 75.0, 12.0, 11.0, 820.0, 450.0, 458.0, 3.0, 209.0, 15.0, 1.0, 2165.0, 4453.0, 2.0, 3.0, 1515.0, 15.0, 1.0, 1844.0, 5170.0, 334.0, 8.0, 343.0, 6.0, 1.0, 17.0, 287.0, 3.0, 165.0, 54.0, 891.0, 1204.0, 37.0, 75.0, 113.0, 16.0, 639.0, 2531.0, 2457.0, 3808.0, 249.0, 5.0, 27.0, 249.0, 5.0, 27.0, 1.0, 62.0, 6.0, 3326.0, 122.0, 36.0, 1.0, 1109.0, 1062.0, 60.0, 464.0, 6.0, 4.0]</td>
      <td>[0.07979885]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[143.0, 803.0, 491.0, 29.0, 34.0, 194.0, 11.0, 19.0, 97.0, 27.0, 139.0, 84.0, 143.0, 1590.0, 22.0, 59.0, 27.0, 683.0, 7.0, 7.0, 1.0, 1270.0, 1.0, 17.0, 470.0, 5.0, 267.0, 6.0, 337.0, 2258.0, 31.0, 46.0, 52.0, 603.0, 111.0, 35.0, 603.0, 12.0, 1.0, 17.0, 6.0, 21.0, 363.0, 1.0, 127.0, 10.0, 1800.0, 543.0, 44.0, 1.0, 111.0, 283.0, 41.0, 1.0, 202.0, 18.0, 41.0, 1.0, 290.0, 106.0, 253.0, 31.0, 4085.0, 8183.0, 18.0, 10.0, 255.0, 12.0, 1.0, 106.0, 13.0, 5560.0, 342.0, 26.0, 6.0, 3.0, 1620.0, 452.0, 39.0, 46.0, 3247.0, 2367.0, 593.0, 18.0, 196.0, 89.0, 137.0, 292.0, 85.0, 1.0, 550.0, 26.0, 494.0, 5.0, 3616.0, 6829.0, 87.0, 8.0, 1821.0, 1182.0]</td>
      <td>[0.06785953]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[10.0, 1605.0, 11.0, 533.0, 9.0, 59.0, 27.0, 181.0, 49.0, 40.0, 31.0, 1.0, 1106.0, 4.0, 1.0, 17.0, 418.0, 1910.0, 2.0, 6917.0, 644.0, 43.0, 181.0, 49.0, 452.0, 3184.0, 1.0, 129.0, 34.0, 554.0, 24.0, 319.0, 20.0, 3.0, 5470.0, 16.0, 3.0, 643.0, 1054.0, 18.0, 11.0, 17.0, 185.0, 14.0, 9.0, 432.0, 20.0, 625.0, 4935.0, 6.0, 1188.0, 281.0, 258.0, 51.0, 26.0, 295.0, 3.0, 214.0, 37.0, 11.0, 96.0, 75.0, 1.0, 17.0, 13.0, 3.0, 415.0, 4.0, 592.0, 9.0, 63.0, 1050.0, 24.0, 673.0, 1910.0, 2.0, 6917.0, 13.0, 70.0, 1041.0, 989.0, 10.0, 516.0, 9.0, 3.0, 339.0, 141.0, 25.0, 516.0, 9.0, 3.0, 238.0, 10.0, 516.0, 9.0, 32.0, 1724.0, 320.0, 40.0, 85.0]</td>
      <td>[0.0031776428]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[44.0, 9.0, 1169.0, 15.0, 1.0, 799.0, 43.0, 4.0, 3171.0, 712.0, 2.0, 3.0, 51.0, 28.0, 106.0, 199.0, 157.0, 1.0, 4398.0, 9.0, 59.0, 27.0, 773.0, 5.0, 1318.0, 11.0, 361.0, 349.0, 15.0, 3.0, 917.0, 923.0, 1305.0, 1807.0, 1.0, 111.0, 41.0, 1.0, 1024.0, 5.0, 2327.0, 3.0, 601.0, 4.0, 9.0, 1816.0, 4389.0, 14.0, 1246.0, 2874.0, 627.0, 192.0, 16.0, 1.0, 628.0, 306.0, 4914.0, 8.0, 4861.0, 4.0, 3.0, 4963.0, 5581.0, 20.0, 1057.0, 2938.0, 1.0, 17.0, 6.0, 7426.0, 355.0, 725.0, 2.0, 57.0, 4147.0, 4.0, 218.0, 315.0, 299.0, 347.0, 5168.0, 2400.0, 1790.0, 45.0, 3.0, 1401.0, 214.0, 14.0, 1.0, 442.0, 1366.0, 34.0, 1.0, 2082.0, 2.0, 147.0, 490.0, 5.0, 9587.0, 29.0]</td>
      <td>[0.43954018]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[89.0, 3233.0, 51.0, 264.0, 30.0, 1.0, 1106.0, 4.0, 1.0, 285.0, 4608.0, 61.0, 736.0, 30.0, 88.0, 454.0, 232.0, 8.0, 959.0, 8.0, 11.0, 1212.0, 353.0, 1.0, 718.0, 1104.0, 130.0, 6.0, 52.0, 309.0, 7.0, 7.0, 5.0, 401.0, 18.0, 3.0, 168.0, 4.0, 1.0, 108.0, 3574.0, 12.0, 141.0, 27.0, 1535.0, 689.0, 5.0, 7.0, 7.0, 1.0, 442.0, 2333.0, 4.0, 1.0, 2291.0, 75.0, 491.0, 1.0, 315.0, 1054.0, 1.0, 912.0, 5172.0, 2.0, 5417.0, 4.0, 1.0, 566.0, 2173.0, 1.0, 768.0, 5.0, 1271.0, 3.0, 4008.0, 2.0, 566.0, 1.0, 1560.0, 1.0, 7834.0, 4.0, 1321.0, 53.0, 661.0, 1.0, 5170.0, 133.0, 8.0, 60.0, 732.0, 8950.0, 20.0, 1.0, 887.0, 1.0, 117.0, 1588.0, 1118.0, 3015.0]</td>
      <td>[0.033311725]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[11.0, 17.0, 13.0, 40.0, 3383.0, 86.0, 97.0, 1815.0, 37.0, 11.0, 17.0, 2.0, 15.0, 1.0, 659.0, 34.0, 421.0, 9.0, 85.0, 4.0, 1.0, 638.0, 33.0, 141.0, 63.0, 190.0, 3.0, 193.0, 251.0, 165.0, 8.0, 1.0, 2908.0, 2.0, 940.0, 530.0, 44.0, 2113.0, 23.0, 21.0, 75.0, 7282.0, 1.0, 390.0, 4.0, 1.0, 4319.0, 2113.0, 11.0, 13.0, 40.0, 370.0, 9.0, 112.0, 66.0, 91.0, 385.0, 44.0, 9.0, 1535.0, 3484.0, 5.0, 5400.0, 4604.0, 9.0, 735.0, 5.0, 987.0, 46.0, 50.0, 370.0, 113.0, 391.0, 226.0, 484.0, 57.0, 15.0, 3.0, 17.0, 16.0, 39.0, 824.0, 3311.0, 11.0, 17.0, 13.0, 75.0, 36.0, 1.0, 451.0, 18.0, 81.0, 235.0, 25.0, 107.0, 1.0, 223.0, 152.0, 31.0, 1.0]</td>
      <td>[0.00014650822]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[1027.0, 5556.0, 6.0, 3.0, 1250.0, 17.0, 8.0, 58.0, 649.0, 46.0, 180.0, 78.0, 8087.0, 69.0, 41.0, 9.0, 961.0, 1330.0, 1641.0, 2357.0, 86.0, 193.0, 9.0, 559.0, 15.0, 38.0, 5.0, 76.0, 35.0, 49.0, 30.0, 1101.0, 2357.0, 7214.0, 41.0, 2484.0, 150.0, 215.0, 2357.0, 420.0, 5.0, 27.0, 3057.0, 2.0, 46.0, 180.0, 78.0, 69.0, 41.0, 9.0, 10.0, 444.0, 1.0, 441.0, 148.0, 9.0, 3692.0, 69.0, 51.0, 11.0, 17.0, 13.0, 90.0, 10.0, 283.0, 57.0, 1443.0, 10.0, 158.0, 63.0, 582.0, 1.0, 4284.0, 10.0, 13.0, 1443.0, 8.0, 2.0, 10.0, 25.0, 5.0, 591.0, 515.0, 2357.0, 59.0, 40.0, 518.0, 8.0, 270.0, 3.0, 224.0, 39.0, 1396.0, 762.0, 29.0, 117.0, 21.0, 63.0, 1101.0, 40.0]</td>
      <td>[0.73986185]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[70.0, 86.0, 2.0, 118.0, 78.0, 10.0, 377.0, 5.0, 1631.0, 11.0, 2148.0, 1831.0, 836.0, 1.0, 6717.0, 4.0, 3.0, 952.0, 360.0, 1007.0, 1396.0, 8.0, 3.0, 6772.0, 2757.0, 4.0, 1.0, 88.0, 3107.0, 1847.0, 1420.0, 766.0, 2.0, 350.0, 5.0, 1.0, 359.0, 80.0, 533.0, 91.0, 643.0, 31.0, 4864.0, 9.0, 53.0, 5.0, 27.0, 41.0, 139.0, 11.0, 19.0, 6.0, 29.0, 5002.0, 2.0, 424.0, 54.0, 2324.0, 7.0, 7.0, 9.0, 514.0, 16.0, 2332.0, 182.0, 346.0, 7805.0, 4.0, 1573.0, 2.0, 1725.0, 8.0, 1.0, 240.0, 4.0, 4775.0, 628.0, 8373.0, 313.0, 67.0, 460.0, 30.0, 100.0, 12.0, 9.0, 29.0, 268.0, 4416.0, 71.0, 3.0, 16.0, 54.0, 1.0, 133.0, 1231.0, 1.0, 83.0, 8242.0, 16.0, 4578.0]</td>
      <td>[0.15147203]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[15.0, 98.0, 4226.0, 334.0, 11.0, 6.0, 1.0, 5.0, 853.0, 54.0, 2090.0, 5.0, 1.0, 4268.0, 46.0, 1240.0, 9434.0, 1.0, 4092.0, 2.0, 1.0, 1460.0, 11.0, 13.0, 3.0, 2439.0, 139.0, 1278.0, 15.0, 1.0, 5547.0, 1206.0, 1.0, 2090.0, 4.0, 9699.0, 1571.0, 1.0, 179.0, 422.0, 30.0, 1.0, 127.0, 90.0, 54.0, 278.0, 2.0, 108.0, 81.0, 231.0, 12.0, 581.0, 336.0, 273.0, 1.0, 4418.0, 8.0, 1.0, 36.0, 1.0, 4025.0, 9434.0, 4.0, 1.0, 4099.0, 1337.0, 5.0, 328.0, 46.0, 144.0, 75.0, 4226.0, 1908.0, 4259.0, 109.0, 1.0, 246.0, 1013.0, 5.0, 328.0, 75.0, 1013.0, 228.0, 1089.0, 6989.0, 1908.0, 2043.0, 86.0, 67.0, 22.0, 25.0, 28.0, 4.0, 1.0, 830.0, 228.0, 3.0, 7267.0, 2.0, 777.0]</td>
      <td>[0.00024122]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[1062.0, 6037.0, 6.0, 3.0, 6486.0, 1499.0, 19.0, 1181.0, 6664.0, 2074.0, 14.0, 3.0, 182.0, 7523.0, 5402.0, 16.0, 3.0, 2305.0, 4.0, 3.0, 436.0, 3592.0, 1011.0, 2074.0, 45.0, 1030.0, 9355.0, 3.0, 1301.0, 1873.0, 1603.0, 36.0, 38.0, 3719.0, 2.0, 3115.0, 16.0, 1.0, 211.0, 3.0, 3914.0, 361.0, 2738.0, 965.0, 38.0, 1432.0, 1021.0, 1484.0, 14.0, 3.0, 945.0, 5807.0, 129.0, 490.0, 38.0, 5.0, 845.0, 8.0, 16.0, 87.0, 18.0, 56.0, 490.0, 38.0, 203.0, 831.0, 35.0, 56.0, 1099.0, 8.0, 2.0, 912.0, 535.0, 514.0, 1445.0, 2.0, 85.0, 11.0, 6.0, 3.0, 500.0, 1239.0, 186.0, 507.0, 222.0, 3.0, 989.0, 21.0, 5.0, 27.0, 255.0, 8.0, 632.0, 279.0, 135.0, 14.0, 1.0, 598.0, 96.0, 1769.0]</td>
      <td>[0.026909858]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[5.0, 680.0, 1742.0, 1980.0, 3.0, 1063.0, 1216.0, 353.0, 6.0, 685.0, 50.0, 5.0, 91.0, 763.0, 1302.0, 5974.0, 91.0, 1251.0, 5.0, 9910.0, 20.0, 1.0, 5797.0, 15.0, 7302.0, 1879.0, 2.0, 1.0, 1265.0, 4.0, 153.0, 34.0, 59.0, 300.0, 320.0, 8.0, 697.0, 14.0, 2016.0, 1950.0, 3242.0, 1243.0, 2.0, 7.0, 7.0, 1.0, 17.0, 407.0, 6.0, 75.0, 192.0, 5.0, 27.0, 49.0, 8033.0, 15.0, 3183.0, 2.0, 6.0, 115.0, 2396.0, 16.0, 1700.0, 36.0, 5180.0, 2.0, 1.0, 3944.0, 11.0, 6.0, 1.0, 549.0, 4.0, 17.0, 115.0, 3943.0, 5.0, 1280.0, 36.0, 1.0, 3183.0, 102.0, 139.0, 5645.0, 547.0, 1780.0, 859.0, 3745.0, 2.0, 365.0, 4.0, 2145.0, 75.0, 113.0, 134.0, 3384.0, 160.0, 535.0, 37.0, 1340.0]</td>
      <td>[0.9066123]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[11.0, 13.0, 28.0, 4.0, 1.0, 246.0, 99.0, 204.0, 123.0, 107.0, 8.0, 58.0, 110.0, 33.0, 298.0, 11.0, 13.0, 1.0, 1598.0, 1519.0, 5.0, 1061.0, 5.0, 29.0, 143.0, 167.0, 5.0, 132.0, 6.0, 12.0, 72.0, 63.0, 158.0, 8247.0, 30.0, 29.0, 10.0, 423.0, 262.0, 12.0, 9.0, 13.0, 162.0, 90.0, 1.0, 164.0, 141.0, 2253.0, 157.0, 5938.0, 85.0, 26.0, 188.0, 94.0, 3.0, 17.0, 1.0, 226.0, 283.0, 49.0, 9.0, 90.0, 54.0, 278.0, 2.0, 13.0, 52.0, 5918.0, 2135.0, 99.0, 23.0, 73.0, 125.0, 71.0, 11.0, 13.0, 2.0, 10.0, 13.0, 2364.0, 683.0, 5.0, 64.0, 1.0, 1017.0, 281.0, 8.0, 11.0, 75.0, 1338.0, 4.0, 3.0, 17.0, 44.0, 10.0, 97.0, 468.0, 142.0, 1.0, 954.0]</td>
      <td>[0.0008559227]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[86.0, 6.0, 9.0, 611.0, 5.0, 94.0, 138.0, 3.0, 75.0, 17.0, 16.0, 138.0, 153.0, 68.0, 33.0, 914.0, 80.0, 9.0, 1.0, 111.0, 45.0, 161.0, 5.0, 78.0, 16.0, 32.0, 323.0, 4.0, 86.0, 180.0, 59.0, 468.0, 43.0, 44.0, 3.0, 3008.0, 2177.0, 15.0, 2214.0, 33.0, 89.0, 57.0, 350.0, 5.0, 199.0, 32.0, 1382.0, 4.0, 12.0, 40.0, 51.0, 22.0, 194.0, 22.0, 68.0, 146.0, 3.0, 927.0, 36.0, 801.0, 20.0, 2414.0, 1.0, 83.0, 674.0, 232.0, 1.0, 17.0, 1126.0, 122.0, 1.0, 1314.0, 2.0, 80.0, 500.0, 19.0, 450.0, 41.0, 297.0, 3.0, 1219.0, 9751.0, 6191.0, 238.0, 1.0, 1984.0, 442.0, 4452.0, 3666.0, 34.0, 490.0, 5.0, 1106.0, 9.0, 53.0, 16.0, 1.0, 88.0, 4960.0, 408.0]</td>
      <td>[0.004606515]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[1.0, 408.0, 8.0, 1.0, 422.0, 4.0, 11.0, 730.0, 23.0, 1.0, 83.0, 408.0, 8.0, 11.0, 595.0, 753.0, 610.0, 3.0, 1668.0, 4716.0, 2107.0, 4.0, 1.0, 8.0, 58.0, 649.0, 524.0, 610.0, 58.0, 511.0, 180.0, 36.0, 1.0, 478.0, 4.0, 225.0, 2.0, 11.0, 250.0, 114.0, 1587.0, 215.0, 1.0, 61.0, 1248.0, 12.0, 163.0, 1.0, 645.0, 2020.0, 2539.0, 1038.0, 596.0, 4835.0, 5.0, 58.0, 1481.0, 596.0, 1154.0, 448.0, 1.0, 19.0, 60.0, 13.0, 1073.0, 217.0, 1093.0, 31.0, 1466.0, 1330.0, 1790.0, 34.0, 3004.0, 14.0, 1323.0, 561.0, 164.0, 2.0, 969.0, 129.0, 14.0, 1.0, 9657.0, 881.0, 596.0, 6.0, 592.0, 54.0, 821.0, 18.0, 9.0, 6.0, 79.0, 721.0, 821.0, 12.0, 9.0, 6.0, 1135.0, 2.0]</td>
      <td>[0.045125782]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[70.0, 86.0, 5.0, 377.0, 10.0, 216.0, 1.0, 1118.0, 289.0, 15.0, 1.0, 83.0, 55.0, 46.0, 150.0, 594.0, 2.0, 5509.0, 3.0, 2358.0, 2020.0, 10.0, 423.0, 176.0, 374.0, 135.0, 10.0, 66.0, 3.0, 75.0, 544.0, 41.0, 9.0, 147.0, 10.0, 78.0, 7.0, 7.0, 100.0, 9751.0, 15.0, 1.0, 2898.0, 15.0, 11.0, 19.0, 10.0, 216.0, 1.0, 681.0, 11.0, 19.0, 6513.0, 88.0, 5.0, 463.0, 3057.0, 536.0, 54.0, 591.0, 33.0, 158.0, 987.0, 192.0, 5.0, 5.0, 3169.0, 2.0, 10.0, 479.0, 46.0, 536.0, 3412.0, 45.0, 1146.0, 948.0, 1188.0, 7.0, 7.0, 82.0, 71.0, 12.0, 11.0, 19.0, 6.0, 337.0, 725.0, 1.0, 153.0, 23.0, 1418.0, 1433.0, 500.0, 378.0, 2.0, 57.0, 1.0, 49.0, 659.0, 23.0]</td>
      <td>[0.06713286]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[22.0, 235.0, 1779.0, 12.0, 1.0, 111.0, 4.0, 11.0, 17.0, 13.0, 395.0, 8.0, 1.0, 1768.0, 4.0, 1419.0, 9.0, 775.0, 14.0, 3.0, 3978.0, 8129.0, 17.0, 18.0, 8.0, 1.0, 652.0, 4.0, 1.0, 19.0, 1.0, 111.0, 1439.0, 6243.0, 51.0, 1.0, 8129.0, 502.0, 5.0, 27.0, 32.0, 1199.0, 129.0, 16.0, 24.0, 1199.0, 247.0, 2.0, 24.0, 1199.0, 9731.0, 2.0, 45.0, 5.0, 545.0, 1.0, 3411.0, 34.0, 178.0, 5.0, 468.0, 1.0, 9731.0, 177.0, 30.0, 98.0, 2315.0, 5.0, 1701.0, 3.0, 4481.0, 39.0, 139.0, 92.0, 1.0, 111.0, 1439.0, 171.0, 2.0, 72.0, 853.0, 41.0, 1.0, 3411.0, 491.0, 1.0, 1016.0, 147.0, 6.0, 1.0, 461.0, 4.0, 1.0, 969.0, 129.0, 34.0, 1288.0, 26.0, 6.0, 3.0]</td>
      <td>[0.3861069]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[22.0, 76.0, 32.0, 531.0, 2.0, 3.0, 317.0, 4.0, 10.0, 25.0, 5.0, 604.0, 1.0, 179.0, 202.0, 29.0, 1.0, 93.0, 28.0, 2436.0, 2.0, 3.0, 252.0, 1194.0, 8.0, 1.0, 11.0, 55.0, 26.0, 295.0, 3.0, 229.0, 34.0, 336.0, 81.0, 8.0, 1.0, 2415.0, 7525.0, 2082.0, 5.0, 3194.0, 1.0, 1106.0, 4.0, 1.0, 17.0, 26.0, 65.0, 498.0, 5.0, 604.0, 65.0, 704.0, 11.0, 25.0, 5.0, 27.0, 643.0, 10.0, 2135.0, 1151.0, 81.0, 77.0, 37.0, 9.0, 148.0, 18.0, 9.0, 6.0, 3.0, 114.0, 117.0, 1.0, 347.0, 15.0, 58.0, 1296.0, 2.0, 1.0, 28.0, 2436.0, 23.0, 63.0, 139.0, 331.0, 2.0, 10.0, 89.0, 380.0, 12.0, 8.0, 3.0, 49.0, 93.0, 100.0, 877.0, 3.0, 4668.0, 26.0]</td>
      <td>[0.27362567]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[15.0, 1.0, 83.0, 4209.0, 232.0, 39.0, 35.0, 6.0, 3.0, 144.0, 1739.0, 5.0, 103.0, 1.0, 102.0, 2.0, 65.0, 902.0, 23.0, 218.0, 2.0, 1.0, 1376.0, 2.0, 1981.0, 23.0, 304.0, 5.0, 165.0, 30.0, 92.0, 6404.0, 2144.0, 38.0, 489.0, 3876.0, 3505.0, 6.0, 769.0, 8908.0, 2.0, 1.0, 17.0, 514.0, 5.0, 8243.0, 2.0, 92.0, 458.0, 5981.0, 1319.0, 8.0, 91.0, 7845.0, 7.0, 7.0, 1641.0, 87.0, 135.0, 2.0, 51.0, 26.0, 644.0, 769.0, 1.0, 1390.0, 2.0, 56.0, 882.0, 1.0, 308.0, 123.0, 76.0, 3.0, 726.0, 1519.0, 60.0, 59.0, 27.0, 3.0, 224.0, 6022.0, 26.0, 4894.0, 21.0, 2904.0, 41.0, 230.0, 18.0, 195.0, 21.0, 205.0, 85.0, 72.0, 64.0, 692.0, 8.0, 1.0, 477.0, 399.0]</td>
      <td>[0.3874004]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[10.0, 178.0, 1.0, 3806.0, 232.0, 4.0, 58.0, 110.0, 142.0, 12.0, 13.0, 1050.0, 20.0, 11.0, 1234.0, 1338.0, 15.0, 3.0, 17.0, 1.0, 113.0, 13.0, 2993.0, 10.0, 340.0, 5.0, 27.0, 3.0, 334.0, 4.0, 4057.0, 8720.0, 2.0, 3241.0, 10.0, 77.0, 112.0, 165.0, 30.0, 95.0, 1.0, 169.0, 171.0, 8646.0, 5999.0, 2.0, 2126.0, 68.0, 21.0, 3.0, 863.0, 313.0, 691.0, 33.0, 112.0, 97.0, 508.0, 2670.0, 61.0, 2126.0, 3359.0, 32.0, 1187.0, 24.0, 1187.0, 13.0, 3.0, 1688.0, 14.0, 870.0, 10.0, 101.0, 26.0, 13.0, 342.0, 1472.0, 41.0, 1.0, 1621.0, 4.0, 1.0, 19.0, 39.0, 66.0, 112.0, 162.0, 2853.0, 5.0, 256.0, 36.0, 10.0, 437.0, 11.0, 730.0, 1523.0, 256.0, 34.0, 6.0, 41.0, 48.0]</td>
      <td>[0.19207326]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[1247.0, 538.0, 6.0, 32.0, 424.0, 2521.0, 1175.0, 19.0, 1103.0, 6095.0, 2550.0, 1.0, 4970.0, 16.0, 1.0, 2982.0, 1715.0, 1186.0, 2.0, 2894.0, 3.0, 452.0, 8.0, 3.0, 4092.0, 2377.0, 7.0, 7.0, 1.0, 349.0, 15.0, 11.0, 19.0, 212.0, 25.0, 74.0, 52.0, 361.0, 46.0, 4.0, 1.0, 153.0, 253.0, 2579.0, 528.0, 2.0, 1.0, 367.0, 340.0, 1052.0, 3.0, 428.0, 3219.0, 5.0, 1.0, 307.0, 4.0, 1.0, 201.0, 311.0, 4.0, 1.0, 578.0, 348.0, 60.0, 44.0, 1815.0, 45.0, 293.0, 12.0, 307.0, 77.0, 142.0, 69.0, 53.0, 12.0, 9.0, 6.0, 335.0, 7.0, 7.0, 11.0, 19.0, 13.0, 40.0, 35.0, 75.0, 47.0, 6.0, 161.0, 8.0, 1.0, 19.0, 57.0, 287.0, 146.0, 1.0, 52.0, 189.0, 10.0]</td>
      <td>[0.14031923]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[10.0, 66.0, 309.0, 1911.0, 15.0, 11.0, 19.0, 51.0, 10.0, 216.0, 1.0, 9609.0, 2.0, 869.0, 5.0, 103.0, 9.0, 20.0, 245.0, 31.0, 2150.0, 2009.0, 204.0, 421.0, 844.0, 1170.0, 8.0, 108.0, 99.0, 18.0, 143.0, 1590.0, 12.0, 257.0, 1.0, 82.0, 102.0, 680.0, 87.0, 1375.0, 3724.0, 149.0, 468.0, 87.0, 80.0, 57.0, 3.0, 3796.0, 4.0, 1375.0, 7.0, 7.0, 10.0, 255.0, 11.0, 17.0, 5.0, 27.0, 3.0, 597.0, 1384.0, 1.0, 225.0, 478.0, 1402.0, 800.0, 5.0, 1.0, 997.0, 8.0, 1.0, 201.0, 18.0, 9.0, 96.0, 1193.0, 5.0, 719.0, 1.0, 19.0, 53.0, 1.0, 1.0, 1270.0, 4.0, 1.0, 201.0, 1.0, 1023.0, 4.0, 1.0, 102.0, 8.0, 1.0, 201.0, 68.0, 665.0, 1009.0, 36.0, 11.0]</td>
      <td>[0.0150666535]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[1.0, 164.0, 11.0, 19.0, 16.0, 2560.0, 1134.0, 31.0, 5686.0, 3.0, 6167.0, 466.0, 1.0, 19.0, 172.0, 682.0, 321.0, 8.0, 11.0, 17.0, 6.0, 339.0, 208.0, 1205.0, 71.0, 9.0, 735.0, 5.0, 27.0, 22.0, 97.0, 711.0, 602.0, 43.0, 297.0, 454.0, 631.0, 4.0, 11.0, 193.0, 238.0, 454.0, 531.0, 19.0, 206.0, 28.0, 678.0, 4.0, 412.0, 28.0, 1457.0, 28.0, 1489.0, 39.0, 224.0, 4.0, 2419.0, 7.0, 7.0, 11.0, 13.0, 28.0, 4.0, 1.0, 88.0, 2170.0, 6994.0, 4.0, 19.0, 10.0, 25.0, 123.0, 107.0, 82.0, 1983.0, 25.0, 443.0, 9.0, 1944.0, 60.0, 6.0, 32.0, 7611.0, 6.0, 3928.0, 2775.0, 1154.0, 2426.0, 5.0, 27.0, 1207.0, 360.0, 35.0, 12.0, 1.0, 7045.0, 8758.0, 97.0, 65.0, 5746.0]</td>
      <td>[0.12673128]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[3800.0, 683.0, 1.0, 1060.0, 13.0, 547.0, 1.0, 102.0, 1295.0, 2.0, 466.0, 1.0, 19.0, 14.0, 3.0, 223.0, 40.0, 384.0, 69.0, 231.0, 1095.0, 2.0, 47.0, 13.0, 54.0, 144.0, 111.0, 12.0, 97.0, 398.0, 22.0, 7294.0, 184.0, 1.0, 19.0, 2.0, 398.0, 22.0, 925.0, 1.0, 5423.0, 407.0, 112.0, 2564.0, 98.0, 2318.0, 2.0, 158.0, 303.0, 52.0, 70.0, 148.0, 140.0, 7.0, 7.0, 47.0, 13.0, 21.0, 192.0, 1134.0, 39.0, 974.0, 5.0, 98.0, 106.0, 2.0, 106.0, 13.0, 28.0, 10.0, 7215.0, 15.0, 469.0, 45.0, 54.0, 323.0, 86.0, 5.0, 294.0, 1.0, 2.0, 6.0, 73.0, 5934.0, 14.0, 3.0, 1451.0, 106.0, 343.0, 28.0, 4.0, 58.0, 1636.0, 153.0, 8.0, 1.0, 5042.0, 384.0, 69.0, 177.0]</td>
      <td>[0.0075387955]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[51.0, 58.0, 1444.0, 288.0, 151.0, 1579.0, 2355.0, 30.0, 1.0, 748.0, 312.0, 11.0, 17.0, 12.0, 13.0, 29.0, 1.0, 10.0, 884.0, 12.0, 11.0, 19.0, 6.0, 3.0, 4225.0, 9.0, 13.0, 355.0, 1.0, 626.0, 528.0, 68.0, 355.0, 1.0, 528.0, 68.0, 355.0, 57.0, 1.0, 160.0, 528.0, 16.0, 104.0, 1401.0, 5668.0, 68.0, 355.0, 2.0, 725.0, 2.0, 119.0, 10.0, 757.0, 725.0, 2684.0, 297.0, 1.0, 8299.0, 18.0, 1708.0, 1243.0, 24.0, 7305.0, 4.0, 3.0, 550.0, 238.0, 1.0, 8299.0, 18.0, 1708.0, 1243.0, 2640.0, 1.0, 3879.0, 134.0, 1114.0, 3.0, 610.0, 339.0, 1.0, 7305.0, 4.0, 3.0, 550.0, 7036.0, 5.0, 602.0, 2.0, 518.0, 18.0, 1213.0, 5.0, 24.0, 366.0, 6417.0, 180.0, 43.0, 85.0, 33.0]</td>
      <td>[0.944641]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2024-11-01 17:48:30.543</td>
      <td>[11.0, 13.0, 90.0, 8.0, 12.0, 2756.0, 557.0, 570.0, 14.0, 1.0, 2078.0, 2.0, 321.0, 8.0, 58.0, 8103.0, 4.0, 159.0, 779.0, 540.0, 162.0, 11.0, 45.0, 410.0, 28.0, 4.0, 58.0, 511.0, 500.0, 917.0, 923.0, 99.0, 446.0, 249.0, 9.0, 63.0, 4391.0, 5.0, 309.0, 606.0, 18.0, 222.0, 35.0, 73.0, 5.0, 94.0, 250.0, 4.0, 460.0, 2.0, 354.0, 12.0, 9.0, 458.0, 50.0, 6861.0, 100.0, 172.0, 825.0, 7.0, 7.0, 138.0, 14.0, 7.0, 7.0, 350.0, 5.0, 166.0, 1.0, 4376.0, 197.0, 11.0, 2.0, 70.0, 607.0, 47.0, 6.0, 161.0, 4379.0, 75.0, 14.0, 11.0, 70.0, 546.0, 1663.0, 4.0, 7.0, 7.0, 6190.0, 67.0, 27.0, 1821.0, 5.0, 126.0, 3350.0, 7.0, 7.0, 984.0, 31.0, 6058.0]</td>
      <td>[0.007553011]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

## Batch Inferences through API

For Wallaroo instances that enable external endpoints connections to pipelines, each pipeline has it's own URL that can be used to perform inferences through an API call.

Performing an inference through an API requires the following:

* The authentication token to authorize the connection to the pipeline.
* The pipeline's inference URL.
* Inference data to sent to the pipeline - in JSON, DataFrame records format, or Apache Arrow.

Full details are available through the [Wallaroo API Connection Guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-connection-guide/) on how retrieve an authorization token and perform inferences through the pipeline's API.

For this demonstration, we'll use some Wallaroo methods to retrieve those items.  We'll use the `print` command to create the `curl` command to perform an inference on the pipeline deployment URL.

```python
deploy_url = my_pipeline._deployment._url()

headers = wl.auth.auth_header()

headers['Content-Type']='application/json; format=pandas-records'
headers['Accept']='application/json; format=pandas-records'

dataFile = '../data/test_data.df.json'

print(f'''
!curl -X POST {deploy_url} \\
    -H "Authorization:{headers['Authorization']}" \\
    -H "Content-Type:{headers['Content-Type']}" \\
    -H "Accept:{headers['Accept']}" \\
    --data @{dataFile} > curl_response.df
      ''')
```

You should have an output that looks similar to this:

```bash

!curl -X POST https://doc-test.api.wallarooexample.ai/v1/api/pipelines/infer/houseprice-estimator-89/houseprice-estimator \
    -H "Authorization:Bearer SOME TOKEN STUFF" \
    -H "Content-Type:application/json; format=pandas-records" \
    -H "Accept:application/json; format=pandas-records" \
    --data @../data/test_data.df.json > curl_response.df
```

Then you can just run the `!curl...` sample.  The inference results will be saved to the file `curl_response.df`.

### Batch Inferences through API Exercise

Perform an inferece request through the `curl` command.  Here's some sample code to run through:

```python
deploy_url = my_pipeline._deployment._url()

headers = wl.auth.auth_header()

headers['Content-Type']='application/json; format=pandas-records'
headers['Accept']='application/json; format=pandas-records'

dataFile = '../data/test_data.df.json'

print(f'''
!curl -X POST {deploy_url} \\
    -H "Authorization:{headers['Authorization']}" \\
    -H "Content-Type:{headers['Content-Type']}" \\
    -H "Accept:{headers['Accept']}" \\
    --data @{dataFile} > curl_response.df
      ''')
```

```python
# run the following to generate the curl command

deploy_url = my_pipeline._deployment._url()

headers = wl.auth.auth_header()

headers['Content-Type']='application/json; format=pandas-records'
headers['Accept']='application/json; format=pandas-records'

dataFile = '../data/test_data.df.json'

print(f'''
!curl -X POST {deploy_url} \\
    -H "Authorization:{headers['Authorization']}" \\
    -H "Content-Type:{headers['Content-Type']}" \\
    -H "Accept:{headers['Accept']}" \\
    --data @{dataFile} > curl_response.df
      ''')
```

    
    !curl -X POST https://doc-test.wallarooexample.ai/v1/api/pipelines/infer/imdb-reviewer-6/imdb-reviewer \
        -H "Authorization:Bearer eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJvTTJDR1FZTDlPbDNmQTlvN0pqdnVtVk9CeVV3R01IZ3RaNWN3a09WS3pZIn0.eyJleHAiOjE3MzA0ODMzNjIsImlhdCI6MTczMDQ4MzMwMiwianRpIjoiMjQ0MzBhNGEtN2JhMS00YTRiLWJjMGUtZTg1MWFjODA4NTA4IiwiaXNzIjoiaHR0cHM6Ly9kb2MtdGVzdC53YWxsYXJvb2NvbW11bml0eS5uaW5qYS9hdXRoL3JlYWxtcy9tYXN0ZXIiLCJhdWQiOlsibWFzdGVyLXJlYWxtIiwiYWNjb3VudCJdLCJzdWIiOiJmY2E1YzRkZi0zN2FjLTRhNzgtOTYwMi1kZDA5Y2E3MmJjNjAiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJzZGstY2xpZW50Iiwic2Vzc2lvbl9zdGF0ZSI6ImJhMTkwMzBjLTVmNTAtNDM0Mi05YzAyLTVkNmYwYjZhNDU1YSIsImFjciI6IjEiLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsiY3JlYXRlLXJlYWxtIiwiZGVmYXVsdC1yb2xlcy1tYXN0ZXIiLCJvZmZsaW5lX2FjY2VzcyIsImFkbWluIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJtYXN0ZXItcmVhbG0iOnsicm9sZXMiOlsidmlldy1pZGVudGl0eS1wcm92aWRlcnMiLCJ2aWV3LXJlYWxtIiwibWFuYWdlLWlkZW50aXR5LXByb3ZpZGVycyIsImltcGVyc29uYXRpb24iLCJjcmVhdGUtY2xpZW50IiwibWFuYWdlLXVzZXJzIiwicXVlcnktcmVhbG1zIiwidmlldy1hdXRob3JpemF0aW9uIiwicXVlcnktY2xpZW50cyIsInF1ZXJ5LXVzZXJzIiwibWFuYWdlLWV2ZW50cyIsIm1hbmFnZS1yZWFsbSIsInZpZXctZXZlbnRzIiwidmlldy11c2VycyIsInZpZXctY2xpZW50cyIsIm1hbmFnZS1hdXRob3JpemF0aW9uIiwibWFuYWdlLWNsaWVudHMiLCJxdWVyeS1ncm91cHMiXX0sImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwiLCJzaWQiOiJiYTE5MDMwYy01ZjUwLTQzNDItOWMwMi01ZDZmMGI2YTQ1NWEiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiaHR0cHM6Ly9oYXN1cmEuaW8vand0L2NsYWltcyI6eyJ4LWhhc3VyYS11c2VyLWlkIjoiZmNhNWM0ZGYtMzdhYy00YTc4LTk2MDItZGQwOWNhNzJiYzYwIiwieC1oYXN1cmEtdXNlci1lbWFpbCI6ImpvaG4uaGFuc2FyaWNrQHdhbGxhcm9vLmFpIiwieC1oYXN1cmEtZGVmYXVsdC1yb2xlIjoiYWRtaW5fdXNlciIsIngtaGFzdXJhLWFsbG93ZWQtcm9sZXMiOlsidXNlciIsImFkbWluX3VzZXIiXSwieC1oYXN1cmEtdXNlci1ncm91cHMiOiJ7fSJ9LCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJqb2huLmhhbnNhcmlja0B3YWxsYXJvby5haSIsImVtYWlsIjoiam9obi5oYW5zYXJpY2tAd2FsbGFyb28uYWkifQ.XvDxblnv8-REuOc2IvisnN3cK4uyjhKvi21uKYHmUV8OKgiZglHBhW-CAUqnLdi8t6Br-NCuOVJZxE7NN7fYfMhfG7xJf6w6Yp4lpH5cOSKkXimnZc-7_-oACnbL97z7WhEu6uzGH4DIWs3fKMlLc6xPea06oZaGSE79LjssTN3Ja0iwN7IvL1cqAtddszWp4K8sqzT4YfevRDbgti_bl_akZd1W0-PQBlraHpWlu5_eDilKrGpIIgSR1DI0Gbrmpm2bodx_3pmQoq_qTXv96sICkly6U_Wvvz3xwWrcNYxAmUKVj8-Ck_LmEZnKBMwc_VvqbN5hiIOVZQ8EbQrmKQ" \
        -H "Content-Type:application/json; format=pandas-records" \
        -H "Accept:application/json; format=pandas-records" \
        --data @../data/test_data.df.json > curl_response.df
          

```python
## blank space to execute the curl command

!curl -X POST https://doc-test.wallarooexample.ai/v1/api/pipelines/infer/imdb-reviewer-6/imdb-reviewer \
    -H "Authorization:Bearer eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJvTTJDR1FZTDlPbDNmQTlvN0pqdnVtVk9CeVV3R01IZ3RaNWN3a09WS3pZIn0.eyJleHAiOjE3MzA0ODMzNjIsImlhdCI6MTczMDQ4MzMwMiwianRpIjoiMjQ0MzBhNGEtN2JhMS00YTRiLWJjMGUtZTg1MWFjODA4NTA4IiwiaXNzIjoiaHR0cHM6Ly9kb2MtdGVzdC53YWxsYXJvb2NvbW11bml0eS5uaW5qYS9hdXRoL3JlYWxtcy9tYXN0ZXIiLCJhdWQiOlsibWFzdGVyLXJlYWxtIiwiYWNjb3VudCJdLCJzdWIiOiJmY2E1YzRkZi0zN2FjLTRhNzgtOTYwMi1kZDA5Y2E3MmJjNjAiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJzZGstY2xpZW50Iiwic2Vzc2lvbl9zdGF0ZSI6ImJhMTkwMzBjLTVmNTAtNDM0Mi05YzAyLTVkNmYwYjZhNDU1YSIsImFjciI6IjEiLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsiY3JlYXRlLXJlYWxtIiwiZGVmYXVsdC1yb2xlcy1tYXN0ZXIiLCJvZmZsaW5lX2FjY2VzcyIsImFkbWluIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJtYXN0ZXItcmVhbG0iOnsicm9sZXMiOlsidmlldy1pZGVudGl0eS1wcm92aWRlcnMiLCJ2aWV3LXJlYWxtIiwibWFuYWdlLWlkZW50aXR5LXByb3ZpZGVycyIsImltcGVyc29uYXRpb24iLCJjcmVhdGUtY2xpZW50IiwibWFuYWdlLXVzZXJzIiwicXVlcnktcmVhbG1zIiwidmlldy1hdXRob3JpemF0aW9uIiwicXVlcnktY2xpZW50cyIsInF1ZXJ5LXVzZXJzIiwibWFuYWdlLWV2ZW50cyIsIm1hbmFnZS1yZWFsbSIsInZpZXctZXZlbnRzIiwidmlldy11c2VycyIsInZpZXctY2xpZW50cyIsIm1hbmFnZS1hdXRob3JpemF0aW9uIiwibWFuYWdlLWNsaWVudHMiLCJxdWVyeS1ncm91cHMiXX0sImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwiLCJzaWQiOiJiYTE5MDMwYy01ZjUwLTQzNDItOWMwMi01ZDZmMGI2YTQ1NWEiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiaHR0cHM6Ly9oYXN1cmEuaW8vand0L2NsYWltcyI6eyJ4LWhhc3VyYS11c2VyLWlkIjoiZmNhNWM0ZGYtMzdhYy00YTc4LTk2MDItZGQwOWNhNzJiYzYwIiwieC1oYXN1cmEtdXNlci1lbWFpbCI6ImpvaG4uaGFuc2FyaWNrQHdhbGxhcm9vLmFpIiwieC1oYXN1cmEtZGVmYXVsdC1yb2xlIjoiYWRtaW5fdXNlciIsIngtaGFzdXJhLWFsbG93ZWQtcm9sZXMiOlsidXNlciIsImFkbWluX3VzZXIiXSwieC1oYXN1cmEtdXNlci1ncm91cHMiOiJ7fSJ9LCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJqb2huLmhhbnNhcmlja0B3YWxsYXJvby5haSIsImVtYWlsIjoiam9obi5oYW5zYXJpY2tAd2FsbGFyb28uYWkifQ.XvDxblnv8-REuOc2IvisnN3cK4uyjhKvi21uKYHmUV8OKgiZglHBhW-CAUqnLdi8t6Br-NCuOVJZxE7NN7fYfMhfG7xJf6w6Yp4lpH5cOSKkXimnZc-7_-oACnbL97z7WhEu6uzGH4DIWs3fKMlLc6xPea06oZaGSE79LjssTN3Ja0iwN7IvL1cqAtddszWp4K8sqzT4YfevRDbgti_bl_akZd1W0-PQBlraHpWlu5_eDilKrGpIIgSR1DI0Gbrmpm2bodx_3pmQoq_qTXv96sICkly6U_Wvvz3xwWrcNYxAmUKVj8-Ck_LmEZnKBMwc_VvqbN5hiIOVZQ8EbQrmKQ" \
    -H "Content-Type:application/json; format=pandas-records" \
    -H "Accept:application/json; format=pandas-records" \
    --data @../data/test_data.df.json > curl_response.df
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  131k  100 45953  100 88836  93551   176k --:--:-- --:--:-- --:--:--  267k 0:00:01  230k

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

my_pipeline.undeploy()

```

<table><tr><th>name</th> <td>imdb-reviewer</td></tr><tr><th>created</th> <td>2024-11-01 17:41:20.204008+00:00</td></tr><tr><th>last_updated</th> <td>2024-11-01 17:48:22.397764+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>workspace_id</th> <td>11</td></tr><tr><th>workspace_name</th> <td>tutorial-workspace-sentiment-analysis</td></tr><tr><th>arch</th> <td>x86</td></tr><tr><th>accel</th> <td>none</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>f6287d00-b934-4ad8-9e92-cd20d544e6a5, d8d77a9d-5baa-4979-9e59-d746ce94dde2, 6a333530-d25a-4f42-9c74-97b5a96a150a</td></tr><tr><th>steps</th> <td>embedder</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Congratulations!

You have now 

* Created a workspace and set it as the current workspace.
* Uploaded an ONNX model.
* Created a Wallaroo pipeline, and set the most recent version of the uploaded model as a pipeline step.
* Successfully send data to your pipeline for inference through the SDK and through an API call.

