'''
Run this code on Google Cloud Services or Amazon Web Services for faster processing. A process for each row in the Dataframe is created.

Originally posted by Wei-Meng Lee on towardsdatascience
For graph neural networks fastest Dijkstra routes can be found using OpenStreetMap API. 
https://towardsdatascience.com/visualization-in-python-finding-routes-between-points-2d97d4881996
Multi processing on a CPU or using a GPU is advised as the the process is very resource intensive.

Using the six sensors from the RAD dataset as an example to connect the six sensors over the road network
'''
import multiprocessing as mp
import os
import numpy as np
from itertools import islice
import pandas as pd
import osmnx as ox
import networkx as nx

# Using the cache accelerates processing for a large map
ox.config(log_console=True, use_cache=True, timeout=1000)

# find shortest route based on the mode of travel and place
def create_graph(place, mode):

    # request the graph from the inputs
    graph = ox.graph_from_place(place, network_type=mode)

    # get the largest strongly connected component
    scc_generator = nx.strongly_connected_components(graph)
    largest_scc = max(scc_generator, key=len)

    # create a new graph containing only the largest strongly connected component
    graph = graph.subgraph(largest_scc)

    return graph

##### Interface to OSMNX
def generate_adjacency_matrix(df, df_row, row_index, graph):

    # Data import path
    OS_PATH = os.path.dirname(os.path.realpath('__file__'))

    # Create the adjacency matrix
    matrix = [["detid_X, detid_Y, DISTANCE"]]

    # nested for loop to find the distances between all the sensors for every sensor
    i=0;
    for index, detector in islice(df_row.iterrows(), 0, len(list(df_row.detid))):
        j=0;
        for l, each_detector in df.iterrows():

            # coordinates from the current sensor
            start_latlng = (float(detector["lat"]), float(detector["long"]))
            # coordinates belonging to the destination sensor
            end_latlng = (float(each_detector["lat"]), float(each_detector["long"]))

            # Try catch because sometimes a node isn't on the graph
            try:
                # find the nearest node to the current sensor
                orig_node = ox.get_nearest_node(graph, start_latlng)
                # find the nearest node to the destination sensor
                dest_node = ox.get_nearest_node(graph, end_latlng)

                #find the shortest path method dijkstra or bellman-ford
                shortest_route_distance = nx.shortest_path_length(graph, orig_node,dest_node, weight="length", method="dijkstra")
            except nx.NetworkXNoPath:
                shortest_route_distance = 0
    

            matrix.append([str(detector["detid"]) + "," + str(each_detector["detid"]) + "," + str(float(shortest_route_distance))])

            print(matrix)
            
            j=+1;
        
        i=+1;

    matrix = np.array(matrix)
    matrix = np.asarray(matrix)

    # Save the dijkstra rad sensors distance matrix
    np.savetxt(OS_PATH + "/munich_adjacency_matrix_" + str(row_index) + ".csv", matrix, delimiter=",", fmt='%s')

def combine_csv_files(df):

    # Data import path
    OS_PATH = os.path.dirname(os.path.realpath('__file__'))

    # Create an empty list to store dataframes
    dfs = []

    # Loop through all the CSV files created by generate_adjacency_matrix
    for i in range(1, len(df) + 1):
        file_path = OS_PATH + "/munich_adjacency_matrix_" + str(i) + ".csv"
        if os.path.exists(file_path):
            # Read the CSV file into a dataframe and append it to the list
            df = pd.read_csv(file_path)
            dfs.append(df)

    # Concatenate all the dataframes in the list into a single dataframe
    result = pd.concat(dfs)

    # Save the concatenated dataframe as a CSV file
    result.to_csv(OS_PATH + "/munich_adjacency_matrix.csv", index=False)


if __name__ == '__main__':

    # Data import path
    OS_PATH = os.path.dirname(os.path.realpath('__file__'))
    SENSORS_CSV   = OS_PATH + '/munich_sensors.csv'

    # Data Import Path
    df = pd.read_csv(SENSORS_CSV)

    # Keep only relevant columns
    df = df.loc[:, ("detid","lat", "long")]

    # Remove missing geocoordinates
    df.dropna(subset=['lat'], how='all', inplace=True)
    df.dropna(subset=['long'], how='all', inplace=True)

    # Remove missing sensor ids
    df.dropna(subset=['detid'], how='all', inplace=True)

    num_processes = len(df)

    # Create the networkx graph
    # 'drive', 'bike', 'walk'
    graph = create_graph("Munich, Bavaria, Germany", "drive")

    # Create a process for each row in the df dataframe
    processes = []
    for i, row in df.iterrows():
        p = mp.Process(target=generate_adjacency_matrix, args=(df, df[i:i+1], i+1, graph))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()


    # Combine the CSV files
    combine_csv_files(df)
