import cnn_socialNet_read_data
import numpy as np
import math


def nmi(A, B, G):
    """
    计算NMI
    :param A: 真实社区：[[1,2,3],[3,4,5],...]
    :param B: 自己算法检测的社区：[[1,2,3],[4,5,6],...]
    :return:
    """
    n = len(G.nodes())
    comm_count_A = len(A)  # real communities
    comm_count_B = len(B)  # resulted communities

    N_row = [0]*comm_count_A
    N_clown = [0]*comm_count_B

    N = [[0 for i in range(comm_count_B)] for i in range(comm_count_A)]

    print(N_row)
    print(N_clown)
    print(N)

    for i in range(comm_count_A):
        for j in range(comm_count_B):
            set_A = set(A[i])
            set_B = set(B[j])
            N[i][j] = len(set_A & set_B)
    print(N)
    matrix_N = np.array(N)
    N_row = matrix_N.sum(axis=1)
    N_clown = matrix_N.sum(axis=0)
    print(matrix_N)
    print(N_row)
    print(N_clown)

    Nij_sum = 0
    for i in range(comm_count_A):
        for j in range(comm_count_B):
            if N[i][j] > 0:
                Nij_sum += N[i][j]*math.log((N[i][j]*n)/(N_row[i]*N_clown[j]))
    N_i = 0
    for i in range(comm_count_A):
        if N_row[i] > 0:
            N_i += N_row[i]*math.log(N_row[i]/n)

    N_j = 0
    for j in range(comm_count_B):
        if N_clown[j] > 0:
            N_j += N_clown[j]*math.log(N_clown[j]/n)

    nmi = -2*Nij_sum/(N_i+N_j)
    return nmi

    '''
    Hx = 0
    Hy = 0

    for m in range(comm_count_B):
        if X[m] > 0:
            Hx += h(X[m]/node_count)
    for m in range(comm_count_A):
        if Y[m] > 0:
            Hy += h(Y[m]/node_count)

    InormXy = 2*Ixy/(Hx + Hy)
    return InormXy
    '''


community1 = [['1','2','3','4','5','6','7'], ['8','9','10','11','12','13'], ['14','15','16','17','18']]
community2 = [['1', '2', '3', '4', '5', '7'], ['8', '9', '10', '11', '12', '13'],
                  ['14', '15', '16', '17'],['6','18']]
community3 = [['1', '2', '3', '4', '5', '7'], ['8', '9', '10', '11', '13'],
                  ['14', '15', '16', '17', '18', '6', '12']]
test_network = './0814data/test_nodes.txt'
# network_file_path = '0814data/1/network.txt'
test_G = cnn_socialNet_read_data.get_graph(test_network)
# print("nmi:", nmi(community1, community1, test_G))