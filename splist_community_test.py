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
social1 = set()
social2 = set()
social1.add(edges[0][0][0])
for edge in edges:
    print(edge)
    if edge[1] == 1 and (edge[0][0] in social1 or edge[0][1] in social1):
        social1.add(edge[0][0])
        social1.add(edge[0][1])
    if edge[1] == 1 and (edge[0][0] not in social1 and edge[0][1] not in social1):
        social2.add(edge[0][0])
        social2.add(edge[0][1])
print(social1)
print(social2)

for node in test_G.nodes:
    print(node)