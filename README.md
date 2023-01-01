# PythonABM
[![Documentation Status](https://readthedocs.org/projects/pythonabm/badge/?version=latest)](https://pythonabm.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/pythonabm.svg)](https://badge.fury.io/py/pythonabm)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/kemplab/pythonabm)



PythonABM makes complex agent-based modeling (ABM) simulations in Python accessible by providing an efficient base framework
for constructing ABMs. More information on PythonABM can be found below.

* Documentation:  [pythonabm.readthedocs.io](https://pythonabm.readthedocs.io/en/latest/index.html)
* PyPI:  [pypi.org/project/pythonabm](https://pypi.org/project/pythonabm/)

<p align="center">
    <img src="https://github.com/kemplab/pythonabm/blob/main/docs/front_image.png?raw=true" alt="" width="500">
<p>


## 
### Installation
Assuming you have Python 3.7-3.10, the latest version of the PythonABM library can be installed using the following command.
```
pip install pythonabm
```

PythonABM can also be built from [source](https://github.com/kemplab/pythonabm) once downloaded from GitHub.
```
pip setup.py install
```

##

### Running a simulation
Calling the start() method in the Simulation class will launch the ABM platform and run it as follows.
(See the ***example.py*** script as a template for building a simulation.) A
text-based UI will prompt for a ***name*** of the simulation and a corresponding ***mode*** (described below).
- 0: New simulation
- 1: Continue a previous simulation
- 2: Convert a previous simulation’s images to a video
- 3: Archive (ZIP) a previous simulation’s outputs

To avoid the text-based UI, the name and mode can be passed at the
command line by using flags (without the parentheses). Note: the
simulation file does not have to be named “main.py”.
```
python main.py -n (name) -m (mode)
```

When continuing a previous simulation (mode=1), the ABM will prompt for
the updated end-step number, though this can be passed like above.
```
python main.py -n (name) -m (mode) -es (end-step)
```

##

### NVIDIA CUDA support
For GPU parallelization, PythonABM requires a CUDA compatible GPU and NVIDIA’s
CUDA toolkit. If you don’t have the toolkit, download the
toolkit either directly from NVIDIA
[here](https://developer.nvidia.com/cuda-downloads) or with [Anaconda's](https://www.anaconda.com/) conda
command show below.

```
conda install cudatoolkit
```

##