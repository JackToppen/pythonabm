# PythonABM
This library helps develop efficient agent-based models (ABMs) by abstracting general 
ABM functionality to a Simulation class along with a collection of useful methods.
- More package information can be found on PyPI at [https://pypi.org/project/pythonabm/](https://pypi.org/project/pythonabm/)

## 
### Installation
This library ***requires*** Python 3.6-3.8 for full functionality. A CUDA compatible
GPU is necessary for enabling the optional parallelization of various simulation methods. More
information on this can be found at the bottom. You can install the latest version of PythonABM with 
```
$ pip install pythonabm
```

##

### Running a simulation
Calling the start() method of Simulation (or any subclass of Simulation) will launch the ABM 
and run it as follows.

The text-based UI will prompt for the ***name*** identifier for the simulation and
corresponding ***mode*** as described below.
- 0: New simulation
- 1: Continue a previous simulation
- 2: Turn a previous simulation's images to a video
- 3: Archive (.zip) a previous simulation's outputs

To avoid the text-based UI, the name and mode can be passed at the command line by using flags
 (without the parentheses). Note: the file does not have to be named "main.py".
```
$ python main.py -n (name) -m (mode)
```

##

### NVIDIA CUDA support
In order to use the code associated with CUDA GPU parallelization, you'll need a CUDA
compatible GPU and NVIDIA's CUDA toolkit. If you don't have the toolkit installed, make
sure you have Microsoft Visual Studio prior to installation.

Download the toolkit directly from NVIDIA [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
or with the conda command show below.
```
$ conda install cudatoolkit
```


##

