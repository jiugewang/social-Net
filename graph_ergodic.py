# 图的遍历
import cnn_socialNet_read_data

test_network = './0814data/test_nodes.txt'
test_G = cnn_socialNet_read_data.get_graph(test_network)
print(test_G.edges())
test_G.remove_edge('12', '14')
test_G.remove_edge('4', '8')
test_G.remove_edge('6', '14')
print(test_G.edges())

# 遍历图
# 划分社区算法


def breadth_first_search(root=None):
    visited = {}
    queue = []
    social = []
    nodes = test_G.nodes

    def bfs(first_node):
        order = [first_node]
        while len(queue) > 0:
            node = queue.pop(0)

            visited[node] = True
            for n in test_G[node]:
                if (not n in visited) and (not n in queue):
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