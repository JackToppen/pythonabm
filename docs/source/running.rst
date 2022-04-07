Running a simulation
====================

Calling the start() method in the Simulation class will launch the ABM platform and run it as follows. A
text-based UI will prompt for a **name** of the simulation and a corresponding **mode** (described below).

.. note::
    There are four different modes for running a simulation:
        | 0: New simulation
        | 1: Continue a previous simulation
        | 2: Convert a previous simulation’s images to a video
        | 3: Archive (ZIP) a previous simulation’s outputs

To avoid the text-based UI, the name and mode can be passed at the
command line by using flags (without the parentheses). Note: the
simulation file does not have to be named “main.py”.

::

   python main.py -n (name) -m (mode)

When continuing a previous simulation (mode=1), the ABM will prompt for
the updated end-step number, though this can be passed like above.

::

   python main.py -n (name) -m (mode) -es (end-step)