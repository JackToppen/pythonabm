Installation
============


PyPI
-----

This library **requires** Python 3.7-3.10 for full functionality. A CUDA
compatible GPU is necessary for enabling GPU parallelization of various
simulation methods (otherwise CPU parallelization is used). See the
bottom for information on enabling GPU parallelization. The latest
version of PythonABM can be installed with the following command.

::

   pip install pythonabm


NVIDIA CUDA Support
-------------------

In order to use the methods associated with CUDA GPU parallelization,
PythonABM requires a CUDA compatible GPU and NVIDIA’s CUDA toolkit. If
you don’t have the toolkit, be sure to update your NVIDIA GPU drivers
`here <https://www.nvidia.com/download/index.aspx>`__ then download the
toolkit either directly from NVIDIA
`there <https://developer.nvidia.com/cuda-downloads>`__ or with the
conda (`Anaconda <https://www.anaconda.com/>`__) command show below.

::

   conda install cudatoolkit