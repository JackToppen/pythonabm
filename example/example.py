import numpy as np
import random as r

from pythonabm import Simulation, record_time


class TestSimulation(Simulation):
    """ This class inherits the Simulation class allowing it to run a
        simulation with the proper functionality.
    """
    def __init__(self):
        # initialize the Simulation object
        Simulation.__init__(self)

        # read parameters from YAML file and add them to instance variables
        self.yaml_parameters("general.yaml")

    def setup(self):
        """ Overrides the setup() method from the Simulation class.
        """
        # add agents to the simulation
        self.add_agents(self.num_to_start)

        # indicate agent arrays and create the arrays with initial conditions
        self.indicate_arrays("locations", "radii", "colors")
        self.locations = np.random.rand(self.number_agents, 3) * self.size
        self.radii = self.agent_array(initial=lambda: 5)

        # indicate agent graphs and create the graphs for holding agent neighbors
        self.indicate_graphs("neighbor_graph")
        self.neighbor_graph = self.agent_graph()

        # record initial values
        self.step_values()
        self.step_image()

    def step(self):
        """ Overrides the step() method from the Simulation class.
        """
        # get all neighbors within radius of 2
        self.get_neighbors(self.neighbor_graph, 5)

        # call the following methods that update agent values
        self.die()
        self.reproduce()
        self.move()

        # add/remove agents from the simulation
        self.update_populations()

        # get the following data
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
        """ Updates an agent based on the presence of neighbors.
        """
        # determine which agents are being removed
        for index in range(self.number_agents):
            if r.random() < 0.1:
                self.mark_to_remove(index)

    @record_time
    def move(self):
        """ Assigns new location to agent.
        """
        for index in range(self.number_agents):
            # get new location position
            new_location = self.locations[index] + 5 * self.random_vector()

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
        """ If the agent meets criteria, hatch a new agent.
        """
        # determine which agents are hatching
        for index in range(self.number_agents):
            if r.random() < 0.1:
                self.mark_to_hatch(index)

if __name__ == "__main__":
    TestSimulation.start("~/Documents/Research/ABM_outputs")

