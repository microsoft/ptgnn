FROM mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04

RUN apt update && apt install -y python3.8-dev python3.8 python3.8-venv python3-pip python3-cffi
RUN python3.8 -m pip install --no-cache-dir torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html


RUN python3.8 -m pip install --upgrade wheel pip cffi
RUN python3.8 -m pip install --no-cache-dir sentencepiece==0.1.90
RUN python3.8 -m pip install --no-cache-dir azureml-sdk annoy chardet datasketch docopt jedi libcst msgpack opentelemetry-api opentelemetry-exporter-jaeger opentelemetry-exporter-prometheus opentelemetry-sdk prometheus-client pystache pyzmq tqdm typing_extensions dpu-utils


# ReInstall torch scatter
RUN python3.8 -m pip install --no-cache-dir --upgrade torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html

# Test installation
RUN python3.8 -c "import torch_scatter"
