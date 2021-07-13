import random as r
import csv
import cv2
import pickle
import math
import psutil
from abc import ABC, abstractmethod

from .backend import *


class Simulation(ABC):
    """ This class makes sure any subclasses have the necessary
        attributes to run a simulation.
    """
    def __init__(self):
        # hold name, which will be overridden
        self.name = None

        # hold the running number of agents and the step to begin at (updated by continuation mode)
        self.number_agents = 0
        self.current_step = 0

        # make default arrays for the following agent values
        self.locations = np.zeros((0, 3), dtype=float)
        self.radii = np.ones(0, dtype=float)
        self.hatching = np.zeros(0, dtype=bool)
        self.removing = np.zeros(0, dtype=bool)

        # hold the names of the agent arrays and the names of any graphs (each agent is a node)
        self.agent_array_names = ["locations", "radii", "hatching", "removing"]
        self.graph_names = list()

        # store the runtimes of methods with @record_time decorator
        self.method_times = dict()

        # default values for these often changed parameters
        self.num_to_start = 1000
        self.cuda = False
        self.end_step = 10
        self.size = [1000, 1000, 0]
        self.output_values = True
        self.output_images = True
        self.image_quality = 2000
        self.video_quality = 1000
        self.fps = 10

    @abstractmethod
    def setup(self):
        """ Initialized simulation agents.
        """
        pass

    @abstractmethod
    def step(self):
        """ Specify which methods are called during the simulation step.
        """
        pass

    def end(self):
        """ Specify any methods that are called after all the simulations
            steps have run.
        """
        self.create_video()

    def add_agents(self, number, agent_type=None):
        """ Adds number of agents to the simulation potentially with agent_type marker.

            - number: the number of agents being added
            - agent_type: string marker used to apply initial conditions to only these
              agents
        """
        # determine bounds for array slice and increase total agents
        begin = self.number_agents
        self.number_agents += number

        # extend the default arrays to the new number of agents
        self.locations = np.concatenate((self.locations, np.zeros((number, 3), dtype=float)), axis=0)
        self.radii = np.concatenate((self.radii, np.ones(number, dtype=float)))
        self.hatching = np.concatenate((self.hatching, np.zeros(number, dtype=bool)))
        self.removing = np.concatenate((self.removing, np.zeros(number, dtype=bool)))

        # if an agent type is passed
        if agent_type is not None:
            # make sure holder for types exists
            if not hasattr(self, "agent_types"):
                self.agent_types = dict()

            # set key value to tuple of the array slice
            self.agent_types[agent_type] = (begin, self.number_agents)

    def agent_array(self, array_name, agent_type=None, dtype=float, vector=None, func=None, override=None):
        """ Adds an agent array to the simulation used to hold values for all agents.

            - array_name: the name of the variable made for the agent array
            - agent_type: string marker from add_agents()
            - dtype: the data type of the array
            - vector: if 2-dimensional, the length of the vector for each agent
            - func: a function called for each index of the array to specify initial
              parameters
            - override: pass existing array instead of generating a new array
        """
        # if using existing array
        if override is not None:
            # make sure array has correct length
            if override.shape[0] != self.number_agents:
                raise Exception("Length of override array does not match number of agents in simulation!")

            # create instance variable and add array name to holder
            else:
                self.__dict__[array_name] = override
                if array_name not in self.agent_array_names:
                    self.agent_array_names.append(array_name)

        # otherwise check if instance variable exists and try to make new array
        elif not hasattr(self, array_name):
            # add array name to holder
            if array_name not in self.agent_array_names:
                self.agent_array_names.append(array_name)

            # get the dimensions of the array
            if vector is None:
                size = self.number_agents  # 1-dimensional array
            else:
                size = (self.number_agents, vector)  # 2-dimensional array (1-dimensional of vectors)

            # if using object types, make NoneType array, otherwise make array of zeros
            if dtype == str or dtype == object:
                self.__dict__[array_name] = np.empty(size, dtype=object)
            else:
                self.__dict__[array_name] = np.zeros(size, dtype=dtype)

        # only apply initial condition if not NoneType
        if func is not None:
            # get bounds for applying initial conditions to array
            if agent_type is None:
                begin = 0
                end = self.number_agents
            else:
                begin = self.agent_types[agent_type][0]
                end = self.agent_types[agent_type][1]

            # iterate through array applying function
            for i in range(begin, end):
                self.__dict__[array_name][i] = func()

    def agent_graph(self, graph_name):
        """ Adds graph to the simulation.

            - graph_name: the name of the instance variable made for the graph
        """
        # create instance variable for graph and add graph name to holder
        self.__dict__[graph_name] = Graph(self.number_agents)
        self.graph_names.append(graph_name)

    def assign_bins(self, max_agents, distance):
        """ Generalizes agent locations to a bins within lattice imposed on
            the agent space, used for accelerating neighbor searches.

            - max_agents: the current maximum number of agents that can fit
              into a bin
            - distance: the radius of search length
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
    def get_neighbors(self, graph_name, distance, clear=True):
        """ Finds all neighbors, within fixed radius, for each each agent.

            - graph_name: name of the instance variable pointing to the graph
            - distance: the radius of search length
            - clear: if removing existing edges, otherwise all edges are saved
        """
        # get graph object reference and if desired, remove all existing edges in the graph
        graph = self.__dict__[graph_name]
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
                tpb = 72
                bpg = math.ceil(self.number_agents / tpb)

                # call the CUDA kernel, sending arrays to GPU
                get_neighbors_gpu[bpg, tpb](cuda.to_device(self.locations), cuda.to_device(bin_locations),
                                            cuda.to_device(bins), cuda.to_device(bins_help), cuda.to_device(distance),
                                            edges, if_edge, edge_count, cuda.to_device(graph.max_neighbors))

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

            - arrays: a list of agent array names to output, if None then all
              arrays are outputted
        """
        # only continue if outputting agent values
        if self.output_values:
            # if arrays is None automatically output all agent arrays
            if arrays is None:
                arrays = self.agent_array_names

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
        """ Creates an image of the simulation space. Note the imaging library
            OpenCV uses BGR instead of RGB.

            - background: the color of the background image as BGR
            - origin_bottom: location of origin True -> bottom/left, False -> top/left
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
            image[:, :] = background

            # go through all of the agents
            for index in range(self.number_agents):
                # get xy coordinates, the axis lengths, and color of agent
                x, y = int(scale * self.locations[index][0]), int(scale * self.locations[index][1])
                major = int(scale * self.radii[index])
                minor = int(scale * self.radii[index])
                color = (255, 50, 50)

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

    def info(self):
        """ Records start time of the step for measuring efficiency and
            prints out info about the simulation.
        """
        # current step and number of agents
        print("Step: " + str(self.current_step))
        print("Number of agents: " + str(self.number_agents))

    def mark_to_hatch(self, index):
        """ Mark the corresponding index of the array with True to
            indicate that the agent should hatch a new agent.
        """
        self.hatching[index] = True

    def mark_to_remove(self, index):
        """ Mark the corresponding index of the array with True to
            indicate that the agent should be removed.
        """
        self.removing[index] = True

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
        for name in self.agent_array_names:
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
        """
        # load the dictionary
        params = template_params(path)

        # iterate through the keys adding each instance variable
        for key in list(params.keys()):
            self.__dict__[key] = params[key]

    def set_paths(self, output_dir):
        """ Updates simulation output paths.
        """
        separator = os.path.sep
        self.output_path = output_dir + separator  # path to output directory
        self.main_path = output_dir + self.name + separator  # path to main simulation directory
        self.images_path = self.main_path + self.name + "_images" + separator  # path to images output directory
        self.values_path = self.main_path + self.name + "_values" + separator  # path to CSV output directory

    def run_simulation(self):
        """ Defines how a simulation is run and what code is run after
            the simulation.
        """
        # run through each of the steps
        for self.current_step in range(self.current_step + 1, self.end_step + 1):
            # record the starting time of the step
            self.step_start = time.perf_counter()

            # call the step methods
            self.step()

        # run any methods at the end
        self.end()

    @classmethod
    def start(cls, output_dir):
        """ Configures/runs the model based on the specified
            simulation mode.
        """
        # check that the output directory exists and get the starting parameters for the model
        output_dir = check_output_dir(output_dir)
        name, mode, end_step = starting_params()    # end step is only for continuation mode

        # new simulation
        if mode == 0:
            # first check that new simulation can be made and create simulation output directory
            name = check_existing(name, output_dir, new_simulation=True)

            # now make simulation instance, update name, and add paths
            sim = cls()
            sim.name = name
            sim.set_paths(output_dir)

            # copy model files to simulation directory, ignoring __pycache__ files
            direc_path = sim.main_path + name + "_copy"
            shutil.copytree(os.getcwd(), direc_path, ignore=shutil.ignore_patterns("__pycache__"))

            # set up the simulation and run the simulation
            sim.setup()
            sim.run_simulation()

        # previous simulation
        else:
            # check that previous simulation exists
            name = check_existing(name, output_dir, new_simulation=False)

            # continuation
            if mode == 1:
                # load previous simulation object from pickled file
                file_name = output_dir + name + os.sep + name + "_temp.pkl"
                with open(file_name, "rb") as file:
                    sim = pickle.load(file)

                # update paths for the case the simulation is move to new folder
                sim.set_paths(output_dir)

                # update the end step and run the simulation
                sim.end_step = end_step
                sim.run_simulation()

            # images to video
            elif mode == 2:
                # make object for video/path information and create video
                sim = cls()
                sim.name = name
                sim.set_paths(output_dir)
                sim.create_video()

            # zip simulation output
            elif mode == 3:
                # zip a copy of the folder and save it to the output directory
                print("Compressing \"" + name + "\" simulation...")
                shutil.make_archive(output_dir + name, "zip", root_dir=output_dir, base_dir=name)
                print("Done!")
