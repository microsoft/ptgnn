## Run on Azure ML

To get started on using Azure ML with this library, use the following script
with the [Dockerfile](/Dockerfile) in the root of the repository.

```python
#! /usr/bin/env python
from collections import OrderedDict

from azureml.core import Workspace, Experiment
from azureml.core.compute import ComputeTarget
from azureml.core.environment import Environment, DockerSection
from azureml.train.estimator import Estimator

# AzureML Subscription Details (get details from the Azure Portal)
sub_id = "00000000-0000-0000-0000-000000000000"  # Get from Azure Portal; used for billing
res_group = "g_name"  # Name for the resource group
ws_name = "ws_name"  # Name for the workspace, which is the collection of compute clusters + experiments
compute_name = "cc_name"  # Name for computer cluster

### Get workspace and compute target
ws = Workspace.get(ws_name, subscription_id=sub_id, resource_group=res_group)
compute_target = ComputeTarget(ws, compute_name)

# The path to the dataset. If using RichPath then this should be prefixed with azure://
# otherwise this is the location where the AzureML Datastore will be mounted
datapath_prefix = "azure://account/container/path/to/dataset/"

script_folder = "."
script_params = OrderedDict(
    [
        (datapath_prefix + "train/", ""),
        (datapath_prefix + "valid/", ""),
        (datapath_prefix + "test/", ""),
        ("./model.pkl.gz", ""),
        ("--aml", ""),
        ("--quiet", ""),
        ("--azure-info", "azure_auth.json"),  # if using dpu_utils' RichPath
    ]
)

with open("./Dockerfile") as f:
    docker = DockerSection()
    docker.base_image = None
    docker.base_dockerfile = f.read()
    docker.enabled = True

environment = Environment(name="pytorchenv")
environment.docker = docker
environment.python.user_managed_dependencies = True

est = Estimator(
    source_directory=script_folder,
    script_params=script_params,
    compute_target=compute_target,
    entry_script="ptgnn/implementations/typilus/train.py",
    environment_definition=environment,
    use_docker=True,
)

### Submit the experiment
exp = Experiment(workspace=ws, name="typilus-experiments")
run = exp.submit(config=est)
print("Portal URL: ", run.get_portal_url())
run.wait_for_completion(show_output=True)
```
