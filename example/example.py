import numpy as np
import random as r

# import the Simulation class and record_time decorator from the PythonABM library
from pythonabm import Simulation, record_time


class TestSimulation(Simulation):
    """ This class inherits the Simulation class allowing it to run a
        simulation with the proper functionality.
    """
    def __init__(self):
        # initialize the Simulation object
        Simulation.__init__(self)

        # read parameters from YAML file and add them as instance variables
        self.yaml_parameters("general.yaml")

        # define instance variables outside of the YAML file
        self.move_step = 2

    def setup(self):
        """ Overrides the setup() method from the Simulation class.
        """
        # add agents to the simulation, indicate agent subtypes
        self.add_agents(self.num_green_agents, agent_type="green")
        self.add_agents(self.num_blue_agents, agent_type="blue")

        # indicate agent arrays for storing agent values
        self.indicate_arrays("locations", "radii", "colors")

        # set initial agent values
        self.locations = np.random.rand(self.number_agents, 3) * self.size
        self.radii = self.agent_array(initial=lambda: r.uniform(1, 2))
        self.colors = self.agent_array(vector=3, initial={"green": (0, 255, 0), "blue": (0, 0, 255)}, dtype=int)

        # indicate agent graphs and create a graph for holding agent neighbors
        self.indicate_graphs("neighbor_graph")
        self.neighbor_graph = self.agent_graph()

        # record initial values
        self.step_values()
        self.step_image()

    def step(self):
        """ Overrides the step() method from the Simulation class.
        """
        # get all neighbors within radius of 5, updating the graph object
        self.get_neighbors(self.neighbor_graph, 5)

        # call the following methods that update agent values
        self.die()
        self.reproduce()
        self.move()

        # add/remove agents from the simulation
        self.update_populations()

        # save data from the simulation
        self.step_values()
        self.step_image()
        self.temp()
        self.data()

    def end(self):
        """ Overrides the end() method from the Simulation class.
        """
        self.create_video()

    @record_time
    def die(self):
        """ Determine which agents will die during this step.
        """
        for index in range(self.number_agents):
            if r.random() < 0.1:
                self.mark_to_remove(index)

    @record_time
    def move(self):
        """ Assigns new locations to agents.
        """
        for index in range(self.number_agents):
            # get new location position
            new_location = self.locations[index] + self.move_step * self.random_vector()

            # check that the new location is within the space, otherwise use boundary values
            for i in range(3):
                if new_location[i] > self.size[i]:
                    self.locations[index][i] = self.size[i]
                elif new_location[i] < 0:
                    self.locations[index][i] = 0
                else:
                    self.locations[index][i] = new_location[i]

    @record_time
    def reproduce(self):
        """ Determine which agents will hatch a new agent during this step.
        """
        for index in range(self.number_agents):
            if r.random() < 0.1:
                self.mark_to_hatch(index)

if __name__ == "__main__":
    TestSimulation.start("~/Documents/Research/Outputs")
