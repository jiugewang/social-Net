# 用于处理社区数据
import networkx as nx


def get_graph(file_path):
    """
    通过社区的边文件获取网络
    :param file_path: 存放社区边的文件
    :return: 社区网络
    """
    my_graph = nx.Graph()
    f = open(file_path)
    line = f.readline()
    while line:
        edge = line.split()
        my_graph.add_edge(edge[0].strip(), edge[1].strip())
        line = f.readline()
    f.close()
    return my_graph


def get_standard_network(file_path):
    """
    获取标准社区的划分格式为：[[1,2,3],[4,5,6]]
    :param file_path: 标准社区的文件路径
    :return: 标准社区，格式：[[1,2,3],[4,5,6]]
    """
    f = open(file_path)
    file_context = f.read().split(';')
    standard_list = []
    for node_str in file_context:
        node = node_str.split(',')
        node_int = list(map(int, node))
        standard_list.append(node_int)
    f.close()
    return standard_list


def add_flag_graph(graph, standard_list):
    """
    为网络中的每条边添加flag属性
    :param graph: 标准网络
    :param standard_list: 标准社区
    :return: 带有标签的网络，如果一条边在社区内，则边的flag=1，否则flag=0
    """
    list_edges = list(graph.edges())
    for u, v in list_edges:
        graph[u][v]['flag'] = 0

    for u, v in list_edges:
        for social in standard_list:
            if int(u) in social and int(v) in social:
                graph[u][v]['flag'] = 1
