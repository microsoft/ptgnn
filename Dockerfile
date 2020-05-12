FROM mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04

RUN python3 -m pip install --upgrade --no-cache-dir numpy scipy docopt torch torchvision dpu-utils more-itertools typing_extensions sentencepiece azureml-sdk pyyaml dill jellyfish

# Install torch scatter
RUN export FORCE_CUDA=1
RUN git clone https://github.com/rusty1s/pytorch_scatter.git
RUN cd pytorch_scatter && git checkout '2.0.3' && FORCE_CUDA=1 python3 ./setup.py build && python3 ./setup.py install && cd .. && rm -rf ./pytorch_scatter

# Test installation
RUN python3 -c "import torch_scatter"
