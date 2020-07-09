# Combining Differentiable PDE Solvers and Graph Neural Networks for Fluid Flow Prediction


This is the repository for the ICML 2020 paper [Combining Differentiable PDE Solvers and Graph Neural Networks for Fluid Flow Prediction](https://proceedings.icml.cc/static/paper_files/icml/2020/6414-Paper.pdf) by Filipe de Avila Belbute-Peres, Thomas D. Economon and J. Zico Kolter.

The easiest way to run the experiments is to use the Dockerfile contained in this repository.  Instructions for running straight from source will be added soon.

## Dockerfile

To build the docker image, run
```
docker build --rm -t cfd-gcn .
```
This will create an image with a working version of this repository, with all dependencies installed.

To access the repository in the container you can then run
```
docker run --gpus '"device=0"' -v $PWD/logs/:/cfd-gcn/logs -u $(id -u):$(id -g) --ipc=host -it --rm cfd-gcn
```
This uses GPU 0 (`device=0`) and the current user (`-u $(id -u):$(id -g)`), which can be changed. 
To run the experiments, you can either simply run the script
```
sh run.sh
```
or copy the commands inside that script to run a particular experiment.
Logs will be saved to the `logs` directory. 
