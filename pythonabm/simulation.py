import csv
import cv2
import math
import pickle
import psutil
import shutil
import time
import re
import os
import gc
import numpy as np
import random as r
import matplotlib.pyplot as plt
from numba import cuda
from abc import ABC, abstractmethod

from pythonabm.backend import record_time, check_direct, template_params, check_existing, get_end_step, Graph, \
    progress_bar, starting_params, check_output_dir, assign_bins_jit, get_neighbors_cpu, get_neighbors_gpu


class Simulation(ABC):
    """ This class defines the necessary behavior for any Simulation
        subclass.
    """
    def __init__(self):
        # hold simulation name (will be overridden)
        self.name = "Trial"

        # hold the current number of agents and the step to begin at (updated by continuation mode)
        self.number_agents = 0
        self.current_step = 0

        # hold the real time start of the step in seconds and the total time for select methods
        self.step_start = 0
        self.method_times = dict()

        # hold the names of the agent arrays and the names of any graphs (each agent is a node)
        self.array_names = list()
        self.graph_names = list()

        # hold bounds for specifying types of agents that vary in initial values
        self.agent_types = dict()

        # default values which can be updated in the subclass
        self.num_to_start = 1000
        self.cuda = False
        self.end_step = 10
        self.size = [1000, 1000, 0]
        self.output_values = True
        self.output_images = True
        self.image_quality = 2000
        self.video_quality = 1000
        self.fps = 10
        self.tpb = 4    # blocks per grid for CUDA neighbor search, use higher number if performance is slow

    @abstractmethod
    def setup(self):
        """ Initialize the simulation prior to running the steps. Must
            be overridden.
        """
        pass

    @abstractmethod
    def step(self):
        """ Specify which methods are called during the simulation step.
            Must be overridden.
        """
        pass

    def end(self):
        """ Specify any methods to be called after all the simulation
            steps have run. Can be overridden.
        """
        self.create_video()

    def set_paths(self, output_dir):
        """ Updates simulation paths to various output directories.

            :param output_dir: Simulation output directory.
            :type output_dir: str
        """
        # get file separator
        separator = os.path.sep

        # hold path to output directory and main simulation directory
        self.output_path = output_dir
        self.main_path = output_dir + self.name + separator

        # path to image and CSV directory
        self.images_path = self.main_path + self.name + "_images" + separator
        self.values_path = self.main_path + self.name + "_values" + separator

    def info(self):
        """ Prints out info about the simulation.
        """
        # current step and number of agents
        print("Step: " + str(self.current_step))
        print("Number of agents: " + str(self.number_agents))

    def mark_to_hatch(self, index):
        """ Mark the corresponding index of the array with True to
            indicate that the agent should hatch a new agent.

            :param index: The unique index of an agent.
            :type index: int
        """
        self.hatching[index] = True

    def mark_to_remove(self, index):
        """ Mark the corresponding index of the array with True to
            indicate that the agent should be removed.

            :param index: The unique index of an agent.
            :type index: int
        """
        self.removing[index] = True

    def __setattr__(self, key, value):
        """ Overrides the __setattr__ method to make sure that agent array
            instance variables are the correct size and type.
        """
        # if instance variable is an agent array
        if hasattr(self, "array_names") and key in self.array_names:
            # if the new value is the correct type and size, set value
            if type(value) is np.ndarray and value.shape[0] == self.number_agents:
                object.__setattr__(self, key, value)
            else:
                # raise exception if incorrect
                raise Exception("Agent array should be NumPy array with length equal to number of agents.")

        # if instance variable is an agent graph
        elif hasattr(self, "graph_names") and key in self.graph_names:
            # if the new value is the correct type and size, set value
            if type(value) is Graph and value.vcount() == self.number_agents:
                object.__setattr__(self, key, value)
            else:
                # raise exception if incorrect
                raise Exception("Agent graph should be PythonABM graph with vertices equal to number of agents.")
        else:
            # otherwise set instance variable as usual
            object.__setattr__(self, key, value)

    def assign_bins(self, max_agents, distance):
        """ Generalizes agent locations to a bins within lattice imposed on
            the agent space, used for accelerating neighbor searches.

            :param max_agents: The maximum number of agents in a bin.
            :param distance: The radius of each agent's neighborhood.
            :type max_agents: int
            :type distance: float
        """
        # run until all agents have been put into bins
        while True:
            # calculate the dimensions of the bins array and the bins helper array, include extra bins for agents that
            # may fall outside of the simulation space
            bins_help_size = np.ceil(np.asarray(self.size) / distance).astype(int) + 3
            bins_size = np.append(bins_help_size, max_agents)

            # create the bins arrays
            bins_help = np.zeros(bins_help_size, dtype=int)  # holds the number of agents in each bin
            bins = np.zeros(bins_size, dtype=int)  # holds the indices of each agent in a bin

            # generalize the agent locations to bin indices and offset by 1 to prevent missing agents outside space
            bin_locations = np.floor_divide(self.locations, distance).astype(int) + 1

            # use JIT function from backend.py to speed up placement of agents
            bins, bins_help = assign_bins_jit(self.number_agents, bin_locations, bins, bins_help, max_agents)

            # break the loop if all agents were accounted for or revalue the maximum number of agents based on and run
            # one more time
            current_max_agents = np.amax(bins_help)
            if max_agents >= current_max_agents:
                break
            else:
                max_agents = current_max_agents * 2  # double to prevent continual updating

        return bins, bins_help, bin_locations, max_agents

    @record_time
    def get_neighbors(self, graph, distance, clear=True):
        """ Finds all neighbors, within fixed radius, for each each agent.

            :param graph: The graph storing the neighbor connections between agents.
            :param distance: The radius of each agent's neighborhood.
            :param clear: If true, clear the previous neighbor connections.
            :type graph: pythonabm.Graph
            :type distance: float
            :type clear: bool
        """
        # get graph object reference and if desired, remove all existing edges in the graph
        if clear:
            graph.delete_edges(None)

        # don't proceed if no agents present
        if self.number_agents == 0:
            return

        # assign each of the agents to bins, updating the max agents in a bin (if necessary)
        bins, bins_help, bin_locations, graph.max_agents = self.assign_bins(graph.max_agents, distance)

        # run until all edges are accounted for
        while True:
            # get the total amount of edges able to be stored and make the following arrays
            length = self.number_agents * graph.max_neighbors
            edges = np.zeros((length, 2), dtype=int)         # hold all edges
            if_edge = np.zeros(length, dtype=bool)                 # say if each edge exists
            edge_count = np.zeros(self.number_agents, dtype=int)   # hold count of edges per agent

            # if using CUDA GPU
            if self.cuda:
                # allow the following arrays to be passed to the GPU
                edges = cuda.to_device(edges)
                if_edge = cuda.to_device(if_edge)
                edge_count = cuda.to_device(edge_count)

                # specify threads-per-block and blocks-per-grid values
                tpb = self.tpb
                bpg = math.ceil(self.number_agents / tpb)

                # call the CUDA kernel, sending arrays to GPU
                get_neighbors_gpu[bpg, tpb](cuda.to_device(self.locations), cuda.to_device(bin_locations),
                                            cuda.to_device(bins), cuda.to_device(bins_help), distance, edges, if_edge,
                                            edge_count, graph.max_neighbors)

                # return the following arrays back from the GPU
                edges = edges.copy_to_host()
                if_edge = if_edge.copy_to_host()
                edge_count = edge_count.copy_to_host()

            # otherwise use parallelized JIT function
            else:
                edges, if_edge, edge_count = get_neighbors_cpu(self.number_agents,  self.locations, bin_locations, bins,
                                                               bins_help, distance, edges, if_edge, edge_count,
                                                               graph.max_neighbors)

            # break the loop if all neighbors were accounted for or revalue the maximum number of neighbors
            max_neighbors = np.amax(edge_count)
            if graph.max_neighbors >= max_neighbors:
                break
            else:
                graph.max_neighbors = max_neighbors * 2

        # reduce the edges to edges that actually exist and add those edges to graph
        graph.add_edges(edges[if_edge])

        # simplify the graph's edges if not clearing the graph at the start
        if not clear:
            graph.simplify()

    @record_time
    def update_populations(self):
        """ Adds/removes agents to/from the simulation by adding/removing
            indices from the cell arrays and any graphs.
        """
        # get indices of hatching/dying agents with Boolean mask
        add_indices = np.arange(self.number_agents)[self.hatching]
        remove_indices = np.arange(self.number_agents)[self.removing]

        # count how many added/removed agents
        num_added = len(add_indices)
        num_removed = len(remove_indices)

        # go through each agent array name
        for name in self.array_names:
            # copy the indices of the agent array data for the hatching agents
            copies = self.__dict__[name][add_indices]

            # add/remove agent data to/from the arrays
            self.__dict__[name] = np.concatenate((self.__dict__[name], copies), axis=0)
            self.__dict__[name] = np.delete(self.__dict__[name], remove_indices, axis=0)

        # go through each graph name
        for graph_name in self.graph_names:
            # add/remove vertices from the graph
            self.__dict__[graph_name].add_vertices(num_added)
            self.__dict__[graph_name].delete_vertices(remove_indices)

        # change total number of agents and print info to terminal
        self.number_agents += num_added - num_removed
        print("\tAdded " + str(num_added) + " agents")
        print("\tRemoved " + str(num_removed) + " agents")

        # clear the hatching/removing arrays for the next step
        self.hatching[:] = False
        self.removing[:] = False

    @record_time
    def temp(self):
        """ Pickle the current state of the simulation which can be used
            to continue a past simulation without losing information.
        """
        # get file name and save in binary mode
        file_name = f"{self.name}_temp.pkl"
        with open(self.main_path + file_name, "wb") as file:
            pickle.dump(self, file, -1)    # use the highest protocol -1 for pickling

    @record_time
    def step_values(self, arrays=None):
        """ Outputs a CSV file containing values from the agent arrays with each
            row corresponding to a particular agent index.

            :param arrays: A list of strings of agent values to record.
            :type arrays: list
        """
        # only continue if outputting agent values
        if self.output_values:
            # if arrays is None automatically output all agent arrays
            if arrays is None:
                arrays = self.array_names

            # make sure directory exists and get file name
            check_direct(self.values_path)
            file_name = f"{self.name}_values_{self.current_step}.csv"

            # open the file
            with open(self.values_path + file_name, "w", newline="") as file:
                # create CSV object and the following lists
                csv_file = csv.writer(file)
                header = list()    # header of the CSV (first row)
                data = list()    # holds the agent arrays

                # go through each of the agent arrays
                for array_name in arrays:
                    # get the agent array
                    agent_array = self.__dict__[array_name]

                    # if the array is one dimensional
                    if agent_array.ndim == 1:
                        header.append(array_name)    # add the array name to the header
                        agent_array = np.reshape(agent_array, (-1, 1))  # resize array from 1D to 2D
                    else:
                        # create name for column based on slice of array ex. locations[0], locations[1], locations[2]
                        for i in range(agent_array.shape[1]):
                            header.append(array_name + "[" + str(i) + "]")

                    # add the array to the data holder
                    data.append(agent_array)

                # write header as the first row of the CSV
                csv_file.writerow(header)

                # stack the arrays to create rows for the CSV file and save to CSV
                data = np.hstack(data)
                csv_file.writerows(data)

    @record_time
    def step_image(self, background=(0, 0, 0), origin_bottom=True):
        """ Creates an image of the simulation space.

            :param background: The 0-255 RGB color of the image background.
            :param origin_bottom: If true, the origin will be on the bottom, left of the image.
            :type background: tuple
            :type origin_bottom: bool
        """
        # only continue if outputting images
        if self.output_images:
            # get path and make sure directory exists
            check_direct(self.images_path)

            # get the size of the array used for imaging in addition to the scaling factor
            x_size = self.image_quality
            scale = x_size / self.size[0]
            y_size = math.ceil(scale * self.size[1])

            # create the agent space background image and apply background color
            image = np.zeros((y_size, x_size, 3), dtype=np.uint8)
            background = (background[2], background[1], background[0])
            image[:, :] = background

            # go through all of the agents
            for index in range(self.number_agents):
                # get xy coordinates, the axis lengths, and color of agent
                x, y = int(scale * self.locations[index][0]), int(scale * self.locations[index][1])
                major, minor = int(scale * self.radii[index]), int(scale * self.radii[index])
                color = (int(self.colors[index][2]), int(self.colors[index][1]), int(self.colors[index][0]))

                # draw the agent and a black outline to distinguish overlapping agents
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, color, -1)
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, (0, 0, 0), 1)

            # if the origin should be bottom-left flip it, otherwise it will be top-left
            if origin_bottom:
                image = cv2.flip(image, 0)

            # save the image as a PNG
            image_compression = 4  # image compression of png (0: no compression, ..., 9: max compression)
            file_name = f"{self.name}_image_{self.current_step}.png"
            cv2.imwrite(self.images_path + file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, image_compression])

    @record_time
    def step_image_3d(self):
        """ Creates an image of the 3D simulation space.
        """
        # only continue if outputting images
        if self.output_images:
            # get path and make sure directory exists
            check_direct(self.images_path)

            # use dark_background theme
            plt.style.use('dark_background')

            # dots per inch for plot resolution, self.image_quality will specify image size
            dpi = 300

            # create a new figure and add an axe subplot to the figure
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            # get x,y,z values of agents and map colors from 0-255 to 0-1
            x, y, z = self.locations[:, 0], self.locations[:, 1], self.locations[:, 2]
            colors = self.colors / 255

            # create scatter plot
            ax.scatter(x, y, z, c=colors, marker="o", alpha=1)

            # turn off gridlines and ticks
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            # set bounds of figure and aspect ratio
            ax.set_xlim([0, self.size[0]])
            ax.set_ylim([0, self.size[1]])
            ax.set_zlim([0, self.size[2]])
            ax.set_box_aspect((self.size[0], self.size[1], self.size[2]))

            # reduce margins around figure
            fig.tight_layout()

            # calculate size of figure in inches to match self.image_quality size
            inches = self.image_quality / dpi
            fig.set_size_inches(inches, inches)

            # get file name and save to image directory
            file_name = f"{self.name}_image_{self.current_step}.png"
            fig.savefig(self.images_path + file_name, dpi=dpi)

            # close figure and garbage collect to prevent memory leak
            fig.clf()
            plt.close("all")
            gc.collect(2)

    def data(self):
        """ Adds a new line to a running CSV holding data about the simulation
            such as memory, step time, number of agents and method profiling.
        """
        # get file name and open the file
        file_name = f"{self.name}_data.csv"
        with open(self.main_path + file_name, "a", newline="") as file_object:
            # create CSV object
            csv_object = csv.writer(file_object)

            # create header if this is the beginning of a new simulation
            if self.current_step == 1:
                # get list of column names for non-method values and method values
                main_header = ["Step Number", "Number Cells", "Step Time", "Memory (MB)"]
                methods_header = list(self.method_times.keys())

                # merge the headers together and write the row to the CSV
                csv_object.writerow(main_header + methods_header)

            # calculate the total step time and get memory of process in megabytes
            step_time = time.perf_counter() - self.step_start
            process = psutil.Process(os.getpid())
            memory = process.memory_info()[0] / 1024 ** 2

            # write the row with the corresponding values
            columns = [self.current_step, self.number_agents, step_time, memory]
            function_times = list(self.method_times.values())
            csv_object.writerow(columns + function_times)

    def create_video(self):
        """ Write all of the step images from a simulation to a video file in the
            main simulation directory.
        """
        # continue if there is an image directory
        if os.path.isdir(self.images_path):
            # get all of the images in the directory and count images
            file_list = [file for file in os.listdir(self.images_path) if file.endswith(".png")]
            image_count = len(file_list)

            # only continue if image directory has images in it
            if image_count > 0:
                # print statement and sort the file list so "2, 20, 3, 31..." becomes "2, 3, ..., 20, ..., 31"
                print("\nCreating video...")
                # file_list = sorted(file_list, key=sort_naturally)
                file_list = sorted(file_list, key=lambda x: int(re.split('(\d+)', x)[-2]))

                # sample the first image to get the dimensions of the image and then scale the image
                size = cv2.imread(self.images_path + file_list[0]).shape[0:2]
                scale = self.video_quality / size[1]
                new_size = (self.video_quality, int(scale * size[0]))

                # get file name and create the video object
                file_name = f"{self.name}_video.mp4"
                codec = cv2.VideoWriter_fourcc(*"mp4v")
                video_object = cv2.VideoWriter(self.main_path + file_name, codec, self.fps, new_size)

                # go through sorted image list, reading and writing each image to the video object
                for i in range(image_count):
                    image = cv2.imread(self.images_path + file_list[i])    # read image from directory
                    if size != new_size:
                        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)    # scale down if necessary
                    video_object.write(image)    # write to video
                    progress_bar(i, image_count)    # show progress

                # close the video file
                video_object.release()

        # end statement
        print("\n\nDone!\n")

    def random_vector(self):
        """ Computes a random vector on the unit sphere centered
            at the origin.
        """
        # random angle on the agent
        theta = r.random() * 2 * math.pi

        # if 2-dimensional set z=0
        if self.size[2] == 0:
            return np.array([math.cos(theta), math.sin(theta), 0])
        else:
            phi = r.random() * 2 * math.pi
            radius = math.cos(phi)
            return np.array([radius * math.cos(theta), radius * math.sin(theta), math.sin(phi)])

    def yaml_parameters(self, path):
        """ Add the instance variables to the Simulation object based
            on the keys and values from a YAML file.

            :param path: Path to YAML template file for simulation instance variables.
            :type path: str
        """
        # load the dictionary
        params = template_params(path)

        # iterate through the keys adding each instance variable
        for key in list(params.keys()):
            self.__dict__[key] = params[key]

    def add_agents(self, number, agent_type=None):
        """ Adds number of agents to the simulation.

            :param number_agents: The current number of agents in the simulation.
            :type number_agents: int
        """
        # determine bounds for array slice and increase total agents
        begin = self.number_agents
        self.number_agents += number

        # if an agent type identifier is passed, set key value to tuple of the array slice
        if agent_type is not None:
            self.agent_types[agent_type] = (begin, self.number_agents - 1)

        # go through each agent array, extending them to the new size
        for array_name in self.array_names:
            # get shape of new agent array
            shape = np.array(self.__dict__[array_name].shape)
            shape[0] = number

            # depending on array, create new array to append to the end of the old array
            if array_name == "locations":
                array = np.random.rand(number, 3) * self.size
            elif array_name == "radii":
                array = 5 * np.ones(number)
            elif array_name == "colors":
                array = np.full(shape, np.array([0, 0, 255]), dtype=int)
            else:
                # get data type and create array
                dtype = self.__dict__[array_name].dtype
                if dtype in (str, tuple, object):
                    array = np.empty(shape, dtype=object)
                else:
                    array = np.zeros(shape, dtype=dtype)

            # add array to existing agent arrays
            self.__dict__[array_name] = np.concatenate((self.__dict__[array_name], array), axis=0)

        # go through each agent graph, adding number agents to it
        for graph_name in self.graph_names:
            self.__dict__[graph_name].add_vertices(number)

    def agent_array(self, dtype=float, vector=None, initial=None):
        """ Generate NumPy array that is used to hold agent values. This allows
            one to specify initial conditions based on agent types.

            :param dtype: Data type of the array.
            :param vector: Size of agent value vector if not None.
            :param initial: Initial value of array index, can be a function.
            :type dtype: type
            :type vector: None or int
            :type initial: Object
            :returns: A NumPy array
        """
        # get shape of array
        if vector is None:
            shape = self.number_agents
        else:
            shape = (self.number_agents, vector)

        # create array based on data type
        if dtype in (str, tuple, object):
            array = np.empty(shape, dtype=object)
        else:
            array = np.zeros(shape, dtype=dtype)

        # if initial is a dict of initial conditions based on agent type
        if initial is not None:
            if type(initial) is dict:
                # go through each agent type in the dict
                for key in list(initial.keys()):
                    # get the bounds and apply the function
                    bounds = self.agent_types[key]
                    for i in range(bounds[0], bounds[1] + 1):
                        if callable(initial[key]):
                            array[i] = initial[key]()
                        else:
                            array[i] = initial[key]
            else:
                # if no dict provided, apply function for initial condition to entire array
                for i in range(0, self.number_agents):
                    if callable(initial):
                        array[i] = initial()
                    else:
                        array[i] = initial

        return array

    def indicate_arrays(self, *args):
        """ Adds agent array names to list to indicate which instance variables
            are agent arrays.

            :param args: A series of instance variable names to indicate agent arrays.
            :type args: str
        """
        # go through each indicated agent value, adding it to the agent array name list
        for array_name in args:
            if array_name not in self.array_names:
                self.array_names.append(array_name)

    def agent_graph(self):
        """ Create a graph correct number of agents.
        """
        return Graph(self.number_agents)

    def indicate_graphs(self, *args):
        """ Adds graph names to list to indicate which instance variables
            are agent graphs.

            :param args: A series of instance variable names to indicate agent graphs.
            :type args: str
        """
        # go through each indicated agent graph, adding it to the agent graph name list
        for graph_name in args:
            if graph_name not in self.graph_names:
                self.graph_names.append(graph_name)

    def full_setup(self):
        """ In addition to how the setup() method has been defined,
            this adds further hidden functionality.
        """
        # specify default array names
        self.array_names = ["locations", "radii", "colors", "hatching", "removing"]

        # make these default arrays
        self.locations = self.agent_array(vector=3)
        self.radii = self.agent_array(initial=lambda: 5)
        self.colors = self.agent_array(dtype=int, vector=3)
        self.hatching = self.agent_array(dtype=bool)
        self.removing = self.agent_array(dtype=bool)

        # call the user defined setup method
        self.setup()

    def run_simulation(self):
        """ Defines how a simulation is run and what code is run after
            the simulation.
        """
        # run through each of the steps
        for self.current_step in range(self.current_step + 1, self.end_step + 1):
            # record the starting time of the step
            self.step_start = time.perf_counter()

            # prints info about the current step
            self.info()

            # call the step methods
            self.step()

        # run any methods at the end
        self.end()

    @classmethod
    def simulation_mode_0(cls, name, output_dir):
        """ Creates a new brand new simulation and runs it through
            all defined steps.

            :param name: The name of the simulation.
            :param output_dir: Path to simulation output directory.
            :type name: str
            :type output_dir: str
        """
        # make simulation instance, update name, and add paths
        sim = cls()
        sim.name = name
        sim.set_paths(output_dir)

        # copy model files to simulation directory, ignoring __pycache__ files
        direc_path = sim.main_path + name + "_copy"
        shutil.copytree(os.getcwd(), direc_path, ignore=shutil.ignore_patterns("__pycache__", os.path.basename(output_dir[:-1])))

        # set up the simulation agents and run the simulation
        sim.full_setup()
        sim.run_simulation()

    @staticmethod
    def simulation_mode_1(name, output_dir):
        """ Opens an existing simulation and runs it for a newly
            specified number of steps.

            :param name: The name of the simulation.
            :param output_dir: Path to simulation output directory.
            :type name: str
            :type output_dir: str
        """
        # load previous simulation object from pickled temp file
        file_name = output_dir + name + os.sep + name + "_temp.pkl"
        with open(file_name, "rb") as file:
            sim = pickle.load(file)

        # update paths for the case the simulation is move to new folder
        sim.set_paths(output_dir)

        # update the end step and run the simulation
        sim.end_step = get_end_step()
        sim.run_simulation()

    @classmethod
    def simulation_mode_2(cls, name, output_dir):
        """ Turns existing simulation images into a video.

            :param name: The name of the simulation.
            :param output_dir: Path to simulation output directory.
            :type name: str
            :type output_dir: str
        """
        # make simulation object for video/path information
        sim = cls()
        sim.name = name
        sim.set_paths(output_dir)

        # compile all simulation images into a video
        sim.create_video()

    @staticmethod
    def simulation_mode_3(name, output_dir):
        """ Archives existing simulation to a ZIP file.

            :param name: The name of the simulation.
            :param output_dir: Path to simulation output directory.
            :type name: str
            :type output_dir: str
        """
        # zip a copy of the folder and save it to the output directory
        print("Compressing \"" + name + "\" simulation...")
        shutil.make_archive(output_dir + name, "zip", root_dir=output_dir, base_dir=name)
        print("Done!")

    @classmethod
    def start(cls, output_dir):
        """ Configures/runs the model based on the specified
            simulation mode.

            :param output_dir: Path to simulation output directory.
            :type output_dir: str
        """
        # get absolute path
        output_dir = os.path.abspath(os.path.expanduser(output_dir))

        # check that the output directory exists and get the name/mode for the simulation
        output_dir = check_output_dir(output_dir)
        name, mode = starting_params()

        # new simulation
        if mode == 0:
            # first check that new simulation can be made and run that mode
            name = check_existing(name, output_dir, new_simulation=True)
            cls.simulation_mode_0(name, output_dir)

        # existing simulation
        else:
            # check that previous simulation exists
            name = check_existing(name, output_dir, new_simulation=False)

            # call the corresponding mode
            if mode == 1:
                cls.simulation_mode_1(name, output_dir)    # continuation
            elif mode == 2:
                cls.simulation_mode_2(name, output_dir)    # images to video
            elif mode == 3:
                cls.simulation_mode_3(name, output_dir)    # archive simulation
            else:
                raise Exception("Mode does not exist!")
