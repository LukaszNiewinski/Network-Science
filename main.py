import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import collections
from pandas.plotting import register_matplotlib_converters
from collections import namedtuple
from data_tools import plot_functions as func

# maybe we can use namedtuples to store data about the nodes?
Node = namedtuple('Node', 'node_id d_in_wn d_out_wn d_in_wy d_out_wy avg_in avg_out' )

if __name__ == '__main__':
    # read file
    register_matplotlib_converters()
    data = pd.read_csv('data/soc-sign-bitcoinotc.csv', sep=',', decimal='.', parse_dates=True,
                       infer_datetime_format=True)

    # create the graph and validate number of nodes and edges
    graph = nx.from_pandas_edgelist(data, source='Source', target='Target', edge_attr='Rating',
                                            create_using=nx.DiGraph)
    n_nodes = graph.number_of_nodes()
    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())

    # convert dictionary into list of tuples
    in_degree_list = [(k, v) for k, v in nx.in_degree_centrality(graph).items()]
    out_degree_list = [(k, v) for k, v in nx.out_degree_centrality(graph).items()]
    # sort list of tuples using as key degree
    in_degree_list = sorted(in_degree_list, key=lambda tup: tup[1], reverse=True)
    out_degree_list = sorted(out_degree_list, key=lambda tup: tup[1], reverse=True)
    print("\n20 NODE WITH HIGHEST IN DEGREE \t\t\t20 NODE WITH HIGHEST OUT DEGREE\n")
    for i in range(20):
        print("IN Node:\033[1m", in_degree_list[i][0], "\033[0m\t degree:\033[1m", in_degree_list[i][1],
              "\033[0m\tOUT Node:\033[1m", out_degree_list[i][0], "\033[0m\tdegree:\033[1m", out_degree_list[i][1],
              "\033[0m")

    # count weighted degree distribution
    weighted_in_degree = dict()
    weighted_out_degree = dict()
    for instance in data.values:
        weighted_in_degree[instance[1]] = weighted_in_degree.get(instance[1], 0) + instance[2]
        weighted_in_degree[instance[0]] = weighted_in_degree.get(instance[0], 0)
        weighted_out_degree[instance[0]] = weighted_out_degree.get(instance[0], 0) + instance[2]
        weighted_out_degree[instance[1]] = weighted_out_degree.get(instance[1], 0)


    in_degree_list_weight = list()

    for key, value in weighted_in_degree.items():
        #third value is average value of ratings that node received from other nodes
        if graph.in_degree[key] != 0:
            in_degree_list_weight.append((int(key), value / n_nodes, value / graph.in_degree[key]))
        else:
            in_degree_list_weight.append((int(key), value / n_nodes, 0))

    out_degree_list_weight = list()
    for key, value in weighted_out_degree.items():
        #third value is average value of ratings that node gave to other nodes
        if graph.out_degree[key] != 0:
            out_degree_list_weight.append((int(key), value / n_nodes, value / graph.out_degree[key]))
        else:
            out_degree_list_weight.append((int(key), value / n_nodes, 0))

    in_degree_list_weight = sorted(in_degree_list_weight, key=lambda tup: tup[1], reverse=True)
    out_degree_list_weight = sorted(out_degree_list_weight, key=lambda tup: tup[1], reverse=True)

    print("\n20 NODE WITH HIGHEST IN DEGREE WEIGHTED \t\t\t20 NODE WITH HIGHEST OUT DEGREE WEIGHTED\n")
    for i in range(20):
        print("IN Node:\033[1m", in_degree_list_weight[i][0], "\033[0m\tdegree:\033[1m", out_degree_list_weight[i][1],
              "\033[0m\tOUT Node:\033[1m", out_degree_list[i][0], "\033[0m\tdegree:\033[1m", out_degree_list[i][1],
              "\033[0m", "avg_in\033[1m", in_degree_list_weight[i][2], "\t\033[0mavg_out\033[1m", out_degree_list_weight[i][2], "\033[0m")


    # prepare lists to create dataframe, each index in each lists must represent same node
    in_not_weight = sorted(in_degree_list, key=lambda tup: tup[0], reverse=True)
    out_not_weight = sorted(out_degree_list, key=lambda tup: tup[0], reverse=True)
    in_weight = sorted(in_degree_list_weight, key=lambda tup: tup[0], reverse=True)
    out_weight = sorted(out_degree_list_weight, key=lambda tup: tup[0], reverse=True)

    # create data frame
    to_data_frame = list()
    for i in range(len(in_not_weight)):
        to_data_frame.append(Node(node_id = in_not_weight[i][0],
                                  d_in_wn = in_not_weight[i][1],
                                  d_out_wn = out_not_weight[i][1],
                                  d_in_wy = in_weight[i][1],
                                  d_out_wy = out_weight[i][1],
                                  avg_in = in_weight[i][2],
                                  avg_out = out_weight[i][2]))

    # d -degree
    # in/out -in or out edge
    # wy / wn - weight yes, weight no(weight represents trust rating)
    # avg_in / avg_out - average of rate of received / given rates
    # each instance in data frame represents: node, d_in_nw, d_out_nw, d_in_wy, d_out_wy, avg_in, avg_out
    print("\n", to_data_frame[0])

    data_frame = pd.DataFrame(to_data_frame, columns=('node', 'd_in_wn', 'd_out_wn', 'd_in_wy', 'd_out_wy', 'avg_in', 'avg_out'))

    # print plots

    print("\n", data_frame.describe(include='all'))
    func.singular_boxplot(data_frame)
    func.hist_each_numeric_var(data_frame)
    # func.hist_categorical_var(sub_data, 'float64')
    func.display_best_fit_var(data_frame)
    func.fit_different_distributions(data_frame)
    func.granularity(data_frame)
    func.sparsity(data_frame)
    func.correlation_analysis(data_frame)

