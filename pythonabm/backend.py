import numpy as np
import time
import os
import sys
import yaml
import shutil
import re
import igraph
from numba import jit, cuda, prange
from functools import wraps


class Graph(igraph.Graph):
    """ This class extends the Graph class from iGraph adding
        instance variables for the bin/bucket sort algorithm.
    """
    def __init__(self, *args, **kwargs):
        # call the origin constructor from iGraph
        super().__init__(*args, **kwargs)

        # these variables are used in the bin/bucket sort for finding neighbors (values change frequently)
        self.max_neighbors = 1    # the current number of neighbors that can be stored in a holder array
        self.max_agents = 1    # the current number of agents that can be stored in a bin

    def num_neighbors(self, index):
        """ Returns the number of neighbors for the index.
        """
        return len(self.neighbors(index))


@jit(nopython=True, cache=True)
def assign_bins_jit(number_agents, bin_locations, bins, bins_help, max_agents):
    """ This just-in-time compiled method performs the actual
        calculations for the assign_bins() method.
    """
    for index in range(number_agents):
        # get the indices of the bin location
        x, y, z = bin_locations[index]

        # get the place in the bin to put the agent index
        place = bins_help[x][y][z]

        # if there is room in the bin, place the agent's index
        if place < max_agents:
            bins[x][y][z][place] = index

        # update the number of agents that should be in a bin (regardless of if they're placed there)
        bins_help[x][y][z] += 1

    return bins, bins_help


@cuda.jit(device=True)
def cuda_magnitude(vector_1, vector_2):
    """ This just-in-time compiled CUDA kernel is a device
        function for calculating the distance between vectors.
    """
    total = 0
    for i in range(0, 3):
        total += (vector_1[i] - vector_2[i]) ** 2
    return total ** 0.5


@cuda.jit
def get_neighbors_gpu(locations, bin_locations, bins, bins_help, distance, edges, if_edge, edge_count, max_neighbors):
    """ This just-in-time compiled CUDA kernel performs the actual
        calculations for the get_neighbors() method.
    """
    # get the agent index in the array
    index = cuda.grid(1)

    # double check that the index is within bounds
    if index < bin_locations.shape[0]:
        # get the starting index for writing edges to the holder array
        start = index * max_neighbors

        # hold the total amount of edges for the agent
        agent_edge_count = 0

        # get the indices of the bin location
        x, y, z = bin_locations[index]

        # go through the 9 bins that could all potential neighbors
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of agents for the current bin
                    bin_count = bins_help[x + i][y + j][z + k]

                    # go through the current bin determining if an agent is a neighbor
                    for l in range(bin_count):
                        # get the index of the current potential neighbor
                        current = bins[x + i][y + j][z + k][l]

                        # check to see if the agent is a neighbor and prevent duplicates with index condition
                        if cuda_magnitude(locations[index], locations[current]) <= distance and index < current:
                            # if there is room, add the edge
                            if agent_edge_count < max_neighbors:
                                # get the index for the edge
                                edge_index = start + agent_edge_count

                                # update the edge array and identify that this edge exists
                                edges[edge_index][0] = index
                                edges[edge_index][1] = current
                                if_edge[edge_index] = 1

                            # increase the count of edges for an agent
                            agent_edge_count += 1

        # update the array with number of edges for the agent
        edge_count[index] = agent_edge_count


@jit(nopython=True, parallel=True, cache=True)
def get_neighbors_cpu(number_agents, locations, bin_locations, bins, bins_help, distance, edges, if_edge, edge_count,
                      max_neighbors):
    """ This just-in-time compiled method performs the actual
        calculations for the get_neighbors() method.
    """
    for index in prange(number_agents):
        # get the starting index for writing edges to the holder array
        start = index * max_neighbors

        # hold the total amount of edges for the agent
        agent_edge_count = 0

        # get the indices of the bin location
        x, y, z = bin_locations[index]

        # go through the 9 bins that could all potential neighbors
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of agents for the current bin
                    bin_count = bins_help[x + i][y + j][z + k]

                    # go through the current bin determining if an agent is a neighbor
                    for l in range(bin_count):
                        # get the index of the current potential neighbor
                        current = bins[x + i][y + j][z + k][l]

                        # check to see if the agent is a neighbor and prevent duplicates with index condition
                        if np.linalg.norm(locations[current] - locations[index]) <= distance and index < current:
                            # if there is room, add the edge
                            if agent_edge_count < max_neighbors:
                                # get the index for the edge
                                edge_index = start + agent_edge_count

                                # update the edge array and identify that this edge exists
                                edges[edge_index][0] = index
                                edges[edge_index][1] = current
                                if_edge[edge_index] = 1

                            # increase the count of edges for an agent
                            agent_edge_count += 1

        # update the array with number of edges for the agent
        edge_count[index] = agent_edge_count

    return edges, if_edge, edge_count


def check_direct(path):
    """ Makes sure directory exists.
    """
    if not os.path.isdir(path):
        os.mkdir(path)


def progress_bar(progress, maximum):
    """ Makes a progress bar to show progress of output.
    """
    # length of the bar
    length = 60

    # calculate bar and percent
    progress += 1    # start at 1 not 0
    fill = int(length * progress / maximum)
    bar = '#' * fill + '.' * (length - fill)
    percent = int(100 * progress / maximum)

    # output the progress bar
    print(f"\r[{bar}] {percent}%", end="")


def normal_vector(vector):
    """ Normalizes the vector.
    """
    # get the magnitude of the vector
    mag = np.linalg.norm(vector)

    # if magnitude is 0 return zero vector, otherwise divide by the magnitude
    if mag == 0:
        return np.zeros(3)
    else:
        return vector / mag


def empty_array(shape, dtype):
    """ Create empty array based on data type.
    """
    return np.empty(shape, dtype=object) if dtype in (str, tuple, object) else np.zeros(shape, dtype=dtype)


def record_time(function):
    """ This is a decorator used to time individual methods.
    """
    @wraps(function)
    def wrap(simulation, *args, **kwargs):    # args and kwargs are for additional arguments
        # call the method and get the start/end time
        start = time.perf_counter()
        function(simulation, *args, **kwargs)
        end = time.perf_counter()

        # add the method time to the dictionary holding these times
        simulation.method_times[function.__name__] = end - start

    return wrap


# -------------------------------------------- methods for user-interface ---------------------------------------------
def commandline_param(flag, dtype):
    """ Returns the value for option passed at the
        command line.
    """
    # go through list of commandline arguments
    args = sys.argv
    for i in range(len(args)):
        # if argument matches flag, try to return value
        if args[i] == flag:
            try:
                return dtype(args[i + 1])
            except IndexError:
                raise Exception(f"No value for option: {args[i]}")

    # raise exception if option not found
    raise Exception(f"Option: {args[i]} not found")


def template_params(path):
    """ Return parameters as dict from a YAML template file.
    """
    with open(path, "r") as file:
        return yaml.safe_load(file)


def check_output_dir(output_dir):
    """ Checks that the output directory exists.
    """
    # run until directory exists
    while not os.path.isdir(output_dir):
        # prompt user input
        print("\nSimulation output directory: \"" + output_dir + "\" does not exist!")
        user = input("Do you want to make this directory? If \"n\", you can specify the correct path (y/n): ")
        print()

        # if making this directory
        if user == "y":
            os.makedirs(output_dir)
            break

        # otherwise get correct path to directory
        elif user == "n":
            output_dir = input("Correct path (absolute) to output directory: ")

            # update paths.yaml file with new output directory path
            keys["output_dir"] = output_dir
            with open("paths.yaml", "w") as file:
                keys = yaml.dump(keys, file)

        else:
            print("Either type \"y\" or \"n\"")

    # if path doesn't end with separator, add it
    separator = os.path.sep
    if output_dir[-1] != separator:
        output_dir += separator

    # return correct path
    return output_dir


def starting_params():
    """ Returns the name and mode for the simulation
        either from the commandline or a text-based UI.
    """
    # try to get the name from the commandline, otherwise run the text-based UI
    try:
        name = commandline_param("-n", str)
    except Exception:
        while True:
            # prompt user for name
            name = input("What is the \"name\" of the simulation? Type \"help\" for more information: ")
            if name == "help":
                print("\nType the name of the simulation (not a path).\n")
            else:
                break

    # try to get the mode from the commandline, otherwise run the text-based UI
    try:
        mode = commandline_param("-m", int)
    except Exception:
        while True:
            # prompt user for mode
            mode = input("What is the \"mode\" of the simulation? Type \"help\" for more information: ")
            if mode == "help":
                print("\nHere are the following modes:\n0: New simulation\n1: Continuation of past simulation\n"
                      "2: Turn simulation images to video\n3: Zip previous simulation\n")
            else:
                # make sure mode is an integer
                try:
                    mode = int(mode)
                    print()
                    break
                except ValueError:
                    print("\nInput: \"mode\" should be an integer.\n")

    return name, mode


def get_end_step():
    """ If using the continuation mode, get the last step
        number for the simulation.
    """
    try:
        end_step = commandline_param("-es", int)
    except Exception:
        while True:
            # prompt user for end step
            end_step = input("What is the last step number of this continued simulation? Type \"help\" for more"
                             " information: ")

            # keep running if "help" is typed
            if end_step == "help":
                print("\nEnter the new step number that will be the last step of the simulation.\n")
            else:
                # make sure end step is an integer
                try:
                    end_step = int(end_step)
                    print()
                    break
                except ValueError:
                    print("Input: \"last step\" should be an integer.\n")

    return end_step


def check_existing(name, output_path, new_simulation=True):
    """ Based on the mode, checks to see if an existing simulation
        in the output path has the same name.
    """
    # if running a new simulation
    if new_simulation:
        while True:
            # see if the directory exists
            if os.path.isdir(output_path + name):
                # get user input for overwriting previous simulation
                print("Simulation already exists with name: " + name)
                user = input("Would you like to overwrite that simulation? (y/n): ")
                print()

                # if not overwriting, get new name
                if user == "n":
                    name = input("New name: ")
                    print()

                # otherwise delete all files/folders in previous directory
                elif user == "y":
                    # clear current directory to prevent another possible future errors
                    files = os.listdir(output_path + name)
                    for file in files:
                        # path to each file/folder
                        path = output_path + name + os.path.sep + file
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    break
                else:
                    # inputs should either be "y" or "n"
                    print("Either type \"y\" or \"n\"")

            # if does not exist, make directory
            else:
                os.mkdir(output_path + name)
                break

    # if using an existing simulation
    else:
        while True:
            # break the loop if the simulation exists, otherwise try to get correct name
            if os.path.isdir(output_path + name):
                break
            else:
                print("No directory exists with name/path: " + output_path + name)
                name = input("\nPlease type the correct name of the simulation or type \"exit\" to exit: ")
                print()
                if name == "exit":
                    exit()

    # return correct simulation name
    return name
