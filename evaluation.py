import numpy as np
import math
import cnn_socialNet_read_data


# NMI
'''
def NMI(A,B):
    # len(A) should be equal to len(B)
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    # Mutual information
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur, idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps, 2)
    # Normalized Mutual information
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A == idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps, 2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps, 2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat
'''


def h(p):
    return -p * (math.log(p)/math.log(2.0))


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

    X = [0]*comm_count_B
    Y = [0]*comm_count_A

    XY = [[0 for i in range(comm_count_B)] for i in range(comm_count_A)]

    i = 0
    j = 0
    for com1 in list2:
        j=0
        com1_set = set(com1)
        for com2 in list:
            com2_set = set(com2)
            XY[i][j] = len(com1_set & com2_set)
            X[i] += XY[i][j]
            Y[j] += XY[i][j]
            j += 1
        i += 1
    Ixy = 0
    for m in range(comm_count_B):
        for n in range(comm_count_A):
            if XY[m][n] > 0:
                Ixy += (XY[m][n]/node_count)*math.log(XY[m][n]*node_count/(X[m]*Y[n])/math.log(2.0))

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


def precision(A, B):
    """
    计算准确率
    :param A: 划分的社区
    :param B: 标准社区
    :return:
    """
    F = set(A)
    T = set(B)
    a = len(F & T)
    b = len(F)
    # print('a', a)
    # print('b', b)
    p = a/b
    return p


def recall(A, B):
    """
        计算准确率
        :param A: 划分的社区
        :param B: 标准社区
        :return:
        """
    F = set(A)
    T = set(B)
    a = len(F & T)
    b = len(T)
    # print('a', a)
    # print('b', b)
    r = a / b
    return r


def f1(A, B):
    f = 2*precision(A, B)*recall(A, B)/(precision(A, B) + recall(A, B))
    return f


'''

输入参数格式说明

comm=[[1,2,3],[4,5,6],...]

G=nx.Graph()



Q1、Q2、Q3是采用三种不同方法来计算模块度，其结果相同



G中边(a,b)都是a<b,

如果查询(b,a) 是不是图的边，返回的结果为false（得到的返回结果是错的，这是个坑，要注意）

'''


def Q1(comm, G):
    # 边的个数

    edges = G.edges()

    m = len(edges)

    # print 'm',m

    # 每个节点的度

    du = G.degree()

    # print('du', du)

    # 通过节点对（同一个社区内的节点对）计算

    ret = 0.0

    for c in comm:

        for x in c:

            for y in c:

                # 边都是前小后大的

                # 不能交换x，y，因为都是循环变量

                if x <= y:

                    if (x, y) in edges:

                        aij = 1.0

                    else:

                        aij = 0.0

                else:

                    if (y, x) in edges:

                        aij = 1.0

                    else:

                        aij = 0

                # print(x,' ',y,' ',aij)

                # cprint(du[x], ' ', du[y])

                tmp = aij - du[x] * du[y] * 1.0 / (2 * m)


                # print tmp

                ret = ret + tmp

                # print ret

                # print ' '

    ret = ret * 1.0 / (2 * m)

    # print 'ret ',ret

    return ret


def R(comm,G):
    """
    计算局部模块度R
    :param comm:[1,2,3,4,5,6]
    :param G: nx.Graph()
    :return:
    """
    b_in = 0
    b_out = 0
    edges = G.edges()
    for edge in edges:
        if edge[0] in comm and edge[1] in comm:
            # print(edge)
            b_in += 1
        elif edge[0] in comm or edge[1] in comm:
            b_out += 1
    r = b_in/(b_in+b_out)
    return r


if __name__ == '__main__':
    A = np.array([12,15,19,21,26,33,38,41,42,45,48,55,58,59,60,61,71,76,80,82,83,89,102,104,109,110,117])
    B = np.array([1,193,121,48,58,41,104,82,55,26,109,117,60,102,42,83,110,71,89,33,38,80,21,76,59,61,15,45])
    # A = np.array([3,5,6,7],[1,2])
    # B = np.array([3,5,7],[1,2,6])
    # print('NMI:', NMI(A, B))
    print('P:', precision(A, B))
    print('R:', recall(A, B))
    print('F1:',f1(A, B))
    community1 = [['1','2','3','4','5','6','7'], ['8','9','10','11','12','13'], ['14','15','16','17','18']]
    community2 = [['1', '2', '3', '4', '5', '7'], ['8', '9', '10', '11', '12', '13'],
                  ['14', '15', '16', '17', '18','6']]
    community3 = [['1', '2', '3', '4', '5', '7'], ['8', '9', '10', '11', '13'],
                  ['14', '15', '16', '17', '18', '6', '12']]
    test_network = './0814data/test_nodes.txt'
    # network_file_path = '0814data/1/network.txt'
    test_G = cnn_socialNet_read_data.get_graph(test_network)
    print('Q:',Q1(community1, test_G))
    print('Q:', Q1(community2, test_G))
    print('Q:', Q1(community3, test_G))

    karate_comm = [['1', '2', '3', '4', '5', '6', '7', '8', '11', '12', '13', '14', '17', '18', '20', '22'],
                   ['9', '10', '15', '16', '19', '21', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34']]
    karate_network = './small_data/karate-edges.txt'
    karate_G = cnn_socialNet_read_data.get_graph(karate_network,split=',')
    print('karate_Q:',Q1(karate_comm,karate_G))

    comm1 = ['1', '2', '3', '4', '5', '6', '7']  # 13/(13+2) = 0.86
    comm2 = ['8', '9', '10', '11', '12', '13']  # 9/(9+2) = 0.81
    comm3 = ['14', '15', '16', '17', '18']  # 8/(8+2) = 0.8
    print('局部模块度：', R(comm1, test_G))
    print('局部模块度：', R(comm2, test_G))
    print('局部模块度：', R(comm3 , test_G))
    print("nmi:", nmi(community1, community1, test_G))
