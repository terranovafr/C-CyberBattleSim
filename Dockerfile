# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# FROM openvino/onnxruntime_ep_ubuntu20:latest
# mcr.microsoft.com/azureml/onnxruntime:latest-cuda

FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04

WORKDIR /root
# Add initial setup files
ADD *.sh ./
ADD *.txt ./
ADD *.yml ./
# Activate the conda environment
RUN conda env create -f environment.yml
RUN conda init bash
ENV PATH /opt/miniconda/envs/ccyberbattlesim/bin:$PATH
SHELL ["/bin/bash", "-c"]
RUN activate ccyberbattlesim
# Copy the rest of the application code
COPY . .
# Ensure conda activates correctly, and run the init script
RUN echo "source activate ccyberbattlesim && bash ./init.sh" >> ~/.bashrc
CMD ["bash"]

# To build the docker image:
#   docker build -t ccyberbattlesim:1.0 .
#
# To run
#   docker run -it --rm ccyberbattlesim:1.0 bash
#   conda activate ccyberbattlesim
