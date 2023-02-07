'''
The idea was sparked from Wei-Meng Lee on towardsdatascience
https://towardsdatascience.com/visualization-in-python-finding-routes-between-points-2d97d4881996

For graph neural networks fastest Dijkstra routes can be found using OpenStreetMap API. 

Multi processing on a CPU or using a GPU is advised as the the process is very resource intensive.
place = 'Munich, Bavaria, Germany'

Using the six sensors from the RAD dataset as an example to connect the six sensors over the road network
'''

import os
import numpy as np
from itertools import islice
import pandas as pd
import osmnx as ox
import networkx as nx


##### Interface to OSMNX
def generate_adjacency_matrix(df):

    # Create the adjacency matrix
    matrix = [["DETEKTOR_ID_X, DETEKTOR_ID_Y, DISTANCE"]]

    # nested for loop to find the distances between all the sensors for every sensor
    i=0;
    for index, detector in islice(df.iterrows(), 0, len(list(df.DETEKTOR_ID))):
        j=0;
        for l, each_detector in df.iterrows():
        
            # Using the cache accelerates processing for a large map
            ox.config(log_console=True, use_cache=True)

            # find shortest route based on the mode of travel

            #place = 'Los Angeles, California, United States'
            place = 'Munich, Bavaria, Germany'

            # 'drive', 'bike', 'walk'
            mode = 'bike'

            # request the graph from the inputs
            graph = ox.graph_from_place(place, network_type = mode)


            # coordinates from the current sensor
            start_latlng = (float(detector["LATITUDE"]), float(detector["LONGITUDE"]))
            # coordinates belonging to the destination sensor
            end_latlng = (float(each_detector["LATITUDE"]), float(each_detector["LONGITUDE"]))

            # find the nearest node to the current sensor
            orig_node = ox.get_nearest_node(graph, start_latlng)
            # find the nearest node to the destination sensor
            dest_node = ox.get_nearest_node(graph, end_latlng)

            #find the shortest path method dijkstra or bellman-ford
            shortest_route_distance = nx.shortest_path_length(graph, orig_node,dest_node, weight="length", method="dijkstra")
    

            matrix.append([str(detector["DETEKTOR_ID"]) + "," + str(each_detector["DETEKTOR_ID"]) + "," + str(float(shortest_route_distance))])

            #print(matrix)
            
            j=+1;
        
        i=+1;

    matrix = np.array(matrix)
    matrix = np.asarray(matrix)

    # Save the dijkstra rad sensors distance matrix
    np.savetxt(OS_PATH + "output/bicycle_adjacency_matrix.csv", matrix, delimiter=",", fmt='%s')


# Data import path
OS_PATH = os.path.dirname(os.path.realpath('__file__'))
SENSORS_CSV   = OS_PATH + '/data/munich_bicycle_sensors.csv'

# Data Import Path
df = pd.read_csv(SENSORS_CSV)

# Keep only relevant columns
df = df.loc[:, ("DETEKTOR_ID","LATITUDE", "LONGITUDE")]

# Remove missing geocoordinates
df.dropna(subset=['LATITUDE'], how='all', inplace=True)
df.dropna(subset=['LONGITUDE'], how='all', inplace=True)

# Remove missing sensor ids
df.dropna(subset=['DETEKTOR_ID'], how='all', inplace=True)


generate_adjacency_matrix(df)
