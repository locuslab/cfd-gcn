ARG CUDA="10.1"
ARG CUDNN="7"
FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install system utilities
RUN apt-get update -y \
    && apt-get install -y apt-utils git git-lfs ca-certificates bzip2 cmake g++ \
    && apt-get install -y libglib2.0-0 
RUN git lfs install

# Install python and pip
RUN apt-get install -y python3.7-dev python3-pip
RUN ln -s /usr/bin/python3.7 /usr/bin/python
ENV PIP37="python3.7 -m pip"
RUN $PIP37 install --upgrade pip

# Install OpenMPI
RUN apt-get install -y openmpi-bin libopenmpi-dev swig m4
RUN env MPICC=/usr/bin/mpicc $PIP37 install mpi4py

#Install pytorch and related python packages
RUN $PIP37 install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN $PIP37 install pytorch_lightning==0.8.1
RUN $PIP37 install tqdm matplotlib test-tube

# Install torch_geometric
ENV PATH=/usr/local/cuda/bin:$PATH
ENV CPATH=/usr/local/cuda/include:$CPATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
RUN $PIP37 install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
RUN $PIP37 install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
RUN $PIP37 install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
RUN $PIP37 install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
RUN $PIP37 install torch-geometric

# Install SU2
RUN git clone --branch feature_pytorch_communicator  https://github.com/su2code/SU2 \
        && cd SU2 \
        && ./preconfigure.py --enable-mpi --with-cc=/usr/bin/mpicc --with-cxx=/usr/bin/mpicxx --prefix=/SU2 --enable-autodiff --enable-PY_WRAPPER --disable-tecio --update \
        && make -j 8 install
ENV SU2_RUN="/SU2/bin"
ENV SU2_HOME="/SU2"
ENV PATH=$SU2_RUN:$PATH
ENV PYTHONPATH=$SU2_RUN:$PYTHONPATH

# Clone repository
RUN git clone https://github.com/locuslab/cfd-gcn.git
WORKDIR cfd-gcn
