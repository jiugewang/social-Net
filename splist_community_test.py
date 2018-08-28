import networkx as nx
import cnn_socialNet_read_data


file_path = './0814data/1/community-standard.txt'
test_file_path = './0814data/test_community.txt'
social_list = cnn_socialNet_read_data.get_standard_network(test_file_path)
print(social_list)
# 测试获取网络
test_network = './0814data/test_nodes.txt'
network_file_path = '0814data/1/network.txt'
test_G = cnn_socialNet_read_data.get_graph(test_network)
print(test_G.edges())

# 测试为网络中每条边添加flag属性
edges = []
cnn_socialNet_read_data.add_flag_graph(test_G, social_list)
for (u, v, flag) in test_G.edges.data('flag'):
    print(u, v, flag)
    edges.append(((u, v), int(flag)))

# 划分社区算法
social_set = set()
social_set.add('1')
for node in test_G.nodes:
    print(node)
    node_set = set()
    for nbr in test_G[node]:
        if test_G[node][nbr]['flag'] == 1:
            social_set.add(nbr)
        if nbr in social_set:
            pass
        else:
            node_set.add(nbr)
print(social_set)

