# PythonABM
PythonABM makes agent-based modeling (ABM) in Python accessible by providing an efficient base framework for building
ABMs through an inheritable Simulation class. Agent values are stored with arrays to promote efficiency through
CPU/GPU parallelization. More information can be found on PyPI at [https://pypi.org/project/pythonabm/](https://pypi.org/project/pythonabm/).

<p align="center">
    <img src="./doc/front_image.png" alt="" width="500">
<p>


## 
### Installation
This library ***requires*** Python 3.7-3.10 for full functionality. A CUDA compatible GPU is necessary for enabling
the optional GPU parallelization of various simulation methods (otherwise CPU parallelization is used). More information
on this can be found at the bottom. The latest version of PythonABM can be installed with 
```
pip install pythonabm
```

##

### Running a simulation
Calling the start() method of Simulation (or any subclass of Simulation) will launch the ABM 
and run it as follows.

A text-based UI will prompt for the ***name*** identifier for the simulation and corresponding ***mode*** as
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

When continuing a previous simulation (mode: 1), the UI will prompt for the updated end step number, though this
can be passed through a command line flag like above.
```
python main.py -n (name) -m (mode) -es (end step)
```

##

### NVIDIA CUDA support
In order to use the code associated with CUDA GPU parallelization, PythonABM requires a CUDA
compatible GPU and NVIDIA's CUDA toolkit. If you don't have the toolkit, first update the NVIDIA GPU drivers
[here](https://www.nvidia.com/download/index.aspx) then download the toolkit either
directly from NVIDIA
[there](https://developer.nvidia.com/cuda-downloads) or with the conda command
show below.
```
conda install cudatoolkit
```

##