Running a simulation
====================

Calling the start() method of Simulation class (or any subclass of
Simulation) will launch the ABM and run it as follows. (See the
**example.py** script as a template for building a simulation.) A
text-based UI will then prompt for the **name** identifier for the
simulation and corresponding **mode** as described below.

- 0: New simulation
- 1: Continue a previous simulation
- 2: Turn a previous simulation’s images to a video
- 3: Archive a previous simulation’s outputs to a ZIP file

To avoid the text-based UI, the name and mode can be passed at the
command line by using flags (without the parentheses). Note: the
simulation file does not have to be named “main.py”.

::

   python main.py -n (name) -m (mode)

When continuing a previous simulation (mode: 1), the UI will prompt for
the updated end-step number, though this can be passed through the
command line like above.

::

   python main.py -n (name) -m (mode) -es (end-step)