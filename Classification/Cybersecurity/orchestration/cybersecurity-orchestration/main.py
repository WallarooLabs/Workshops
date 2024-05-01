import wallaroo
from wallaroo.object import EntityNotFoundError
import pandas as pd

wl = wallaroo.Client()

# get the arguments
arguments = wl.task_args()

if "workspace_name" in arguments:
    workspace_name = arguments['workspace_name']
else:
    workspace_name="workshop-workspace-john"

if "pipeline_name" in arguments:
    pipeline_name = arguments['pipeline_name']
else:
    pipeline_name="aloha-prime"


print(f"Workspace: {workspace_name}")
workspace = wl.get_workspace(workspace_name)

wl.set_current_workspace(workspace)
print(workspace)

# the pipeline is assumed to be deployed
print(f"Pipeline: {pipeline_name}")
pipeline = wl.get_pipeline(pipeline_name)
print(pipeline)

print(pipeline.status())

inference_result = pipeline.infer_from_file('./data/data-1k.df.json')
print(inference_result)
