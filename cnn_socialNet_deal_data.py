# 处理数据，获取边的模型矩阵
import numpy as np
SIZE128 = 128
SIZE114 = 114
SIZE100 = 100
SIZE86 = 86
SIZE72 = 72
SIZE58 = 58
SIZE44 = 44
SIZE30 = 30


def get_jump2_3dimension_matrix(mygraph, edge):
    """
    获取一条边端点2跳的矩阵模型
    :param mygraph:网络
    :param edge:网络中的某条边
    :return:matrix(矩阵),tmpsize(矩阵的大小)
    """
    edges = mygraph.edges()
    source_node = edge[0]
    target_node = edge[1]
    source_node_set = set()
    target_node_set = set()

    for source_nbr in mygraph[source_node]:
        if source_nbr == target_node:
            pass
        else:
            source_node_set.add(source_nbr)
            for source_nbr1 in mygraph[source_nbr]:
                source_node_set.add(source_nbr1)
    if source_node in source_node_set:
        source_node_set.remove(source_node)
    if target_node in source_node_set:
        source_node_set.remove(target_node)

    for target_nbr in mygraph[target_node]:
        if target_nbr == source_node:
            pass
        else:
            target_node_set.add(target_nbr)
            for target_nbr1 in mygraph[target_nbr]:
                target_node_set.add(target_nbr1)
    if source_node in target_node_set:
        target_node_set.remove(source_node)
    if target_node in target_node_set:
        target_node_set.remove(target_node)

    intersection_set = source_node_set & target_node_set
    source_node_list = list(source_node_set)
    target_node_list = list(target_node_set)

    source_list_size = len(source_node_list)
    target_list_size = len(target_node_list)

    global tmp_size
    if source_list_size > target_list_size:
        tmp_size = source_list_size
    else:
        tmp_size = target_list_size

    while len(source_node_list) < tmp_size:
        source_node_list.append(-1)
    while len(target_node_list) < tmp_size:
        target_node_list.append(-1)

    # 构造矩阵
    matrix = np.zeros((tmp_size, tmp_size, 3))
    for row in range(tmp_size):
        for clown in range(tmp_size):
            if source_node_list[row] == -1 or target_node_list[clown] == -1:
                matrix[row][clown] = 0
            elif source_node_list[row] in intersection_set and target_node_list[clown] in intersection_set:
                matrix[row][clown][1] = 255
            elif (source_node_list[row], target_node_list[clown]) in edges:
                matrix[row][clown][0] = 255
            else:
                matrix[row][clown][2] = 255

    # return len(source_node_list), len(target_node_list), tmp_size
    return matrix, tmp_size


def get_jump2_3dimension_different_size_matrix(mygraph, edge):
    """
    获取一条边端点2跳的矩阵模型
    :param mygraph:网络
    :param edge:网络中的某条边
    :return:matrix(矩阵),tmpsize(矩阵的大小)
    """
    edges = mygraph.edges()
    source_node = edge[0]
    target_node = edge[1]
    source_node_set = set()
    target_node_set = set()

    for source_nbr in mygraph[source_node]:
        if source_nbr == target_node:
            pass
        else:
            source_node_set.add(source_nbr)
            for source_nbr1 in mygraph[source_nbr]:
                source_node_set.add(source_nbr1)
    if source_node in source_node_set:
        source_node_set.remove(source_node)
    if target_node in source_node_set:
        source_node_set.remove(target_node)

    for target_nbr in mygraph[target_node]:
        if target_nbr == source_node:
            pass
        else:
            target_node_set.add(target_nbr)
            for target_nbr1 in mygraph[target_nbr]:
                target_node_set.add(target_nbr1)
    if source_node in target_node_set:
        target_node_set.remove(source_node)
    if target_node in target_node_set:
        target_node_set.remove(target_node)

    intersection_set = source_node_set & target_node_set
    source_node_list = list(source_node_set)
    target_node_list = list(target_node_set)

    source_list_size = len(source_node_list)
    target_list_size = len(target_node_list)

    # 构造矩阵
    matrix = np.zeros((source_list_size, target_list_size, 3))
    for row in range(source_list_size):
        for clown in range(target_list_size):
            if source_node_list[row] in intersection_set and target_node_list[clown] in intersection_set:
                matrix[row][clown][1] = 255
            elif (source_node_list[row], target_node_list[clown]) in edges:
                matrix[row][clown][0] = 255
            else:
                matrix[row][clown][2] = 255

    # return len(source_node_list), len(target_node_list), tmp_size
    return matrix, source_list_size, target_list_size


