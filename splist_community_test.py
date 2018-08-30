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
visited = {}


def breadth_first_search(root=None):
    queue = []
    social = []
    nodes = test_G.nodes

    def bfs(first_node):
        order = [first_node]
        while len(queue) > 0:
            node = queue.pop(0)

            visited[node] = True
            for n in test_G[node]:
                if (not n in visited) and (not n in queue) and test_G[node][n]['flag'] == 1:
                    queue.append(n)
                    order.append(n)
        social.append(order)

    if root:
        queue.append(root)
        # order.append(root)
        bfs(root)

    for node in test_G.nodes:
        if not node in visited:
            queue.append(node)
            bfs(node)

    # print(social)
    # print(queue)
    # print(visited)
    """
    order1 = []
    for node in test_G.nodes:
        if not node in visited:
            queue.append(node)
            order1.append(node)
            bfs()
    print(order1)
    """

    return social


social = breadth_first_search('1')
print(social)
