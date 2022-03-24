# PythonABM
[![Documentation Status](https://readthedocs.org/projects/pythonabm/badge/?version=latest)](https://pythonabm.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/pythonabm.svg)](https://badge.fury.io/py/pythonabm)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/JackToppen/pythonabm)



PythonABM makes complex agent-based modeling (ABM) simulations in Python accessible by providing an efficient base framework
for constructing ABMs. More information on PythonABM can be found below.

* Documentation:  [pythonabm.readthedocs.io](https://pythonabm.readthedocs.io/en/latest/index.html)
* PyPI:  [pypi.org/project/pythonabm](https://pypi.org/project/pythonabm/)

<p align="center">
    <img src="https://github.com/JackToppen/pythonabm/blob/master/docs/front_image.png?raw=true" alt="" width="500">
<p>


## 
### Installation
This library ***requires*** Python 3.7-3.10 for full functionality. A CUDA compatible GPU is necessary for enabling GPU
parallelization of various simulation methods (otherwise CPU parallelization is used). See the bottom for information
on enabling GPU parallelization. The latest version of PythonABM can be installed with the following command.
```
pip install pythonabm
```

##

### Running a simulation
Calling the start() method of Simulation class (or any subclass of Simulation) will launch the ABM and run it as follows.
(See the ***example.py*** script as a template for building a simulation.) A text-based UI will then prompt for the ***name***
identifier for the simulation and corresponding ***mode*** as
described below.
- 0: New simulation
- 1: Continue a previous simulation
- 2: Turn a previous simulation's images to a video
- 3: Archive a previous simulation's outputs to a ZIP file

To avoid the text-based UI, the name and mode can be passed at the command line by using flags
 (without the parentheses). Note: the simulation file does not have to be named "main.py".
```
python main.py -n (name) -m (mode)
```

When continuing a previous simulation (mode: 1), the UI will prompt for the updated end-step number, though this
can be passed through the command line like above.
```
python main.py -n (name) -m (mode) -es (end-step)
```

##

### NVIDIA CUDA support
In order to use the methods associated with CUDA GPU parallelization, PythonABM requires a CUDA
compatible GPU and NVIDIA's CUDA toolkit. If you don't have the toolkit, be sure to update your NVIDIA GPU drivers
[here](https://www.nvidia.com/download/index.aspx) then download the toolkit either directly from NVIDIA
[there](https://developer.nvidia.com/cuda-downloads) or with the conda ([Anaconda](https://www.anaconda.com/)) command
show below.
```
conda install cudatoolkit
```

##