FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn7-ubuntu18.04

RUN apt update && apt install -y python3.8-dev python3.8 python3.8-venv python3-pip python3-cffi
RUN python3.8 -m pip install --no-cache-dir torch torchvision


RUN python3.8 -m pip install --upgrade wheel pip cffi
RUN python3.8 -m pip install --no-cache-dir azureml-sdk chardet datasketch docopt msgpack tqdm typing_extensions dpu-utils


# ReInstall torch scatter
RUN python3.8 -m pip install --no-cache-dir --upgrade torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html


# Test installation
RUN python3.8 -c "import torch_scatter"
