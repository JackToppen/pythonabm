Installation
============


Assuming you have Python 3.7-3.10, the PythonABM library can be installed using the following command.

::

   pip install pythonabm

PythonABM can also be built from `source <https://github.com/kemplab/pythonabm>`__ once downloaded from GitHub.

::

   pip setup.py install

.. note::
    A CUDA compatible GPU is necessary for enabling the GPU parallelized
    neighbor search method. See the below for information on enabling GPU
    parallelization.


CUDA Support
-------------------

For GPU parallelization, PythonABM requires a CUDA compatible GPU and NVIDIA’s
CUDA toolkit. If you don’t have the toolkit, download the
toolkit either directly from NVIDIA
`here <https://developer.nvidia.com/cuda-downloads>`__ or with `Anaconda's <https://www.anaconda.com/>`__ conda
command show below.

::

   conda install cudatoolkit