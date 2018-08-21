import numpy as np
import networkx as nx
import tensorflow as tf

import cnn_socialNet_read_data
import cnn_socialNet_deal_data

matrix = np.array([3, 4])
print(matrix)

G = nx.Graph()
G.add_edge(1, 2)
print(G.edges())

a = tf.Variable(3)
print(a)


# 测试获取标准社区数据
file_path = './0814data/1/community-standard.txt'
test_file_path = './0814data/test_community.txt'
social_list = cnn_socialNet_read_data.get_standard_network(file_path)
print(social_list)

# 测试获取网络
test_network = './0814data/test_nodes.txt'
network_file_path = '0814data/1/network.txt'
test_G = cnn_socialNet_read_data.get_graph(network_file_path)
print(test_G.edges())

# 测试为网络中每条边添加flag属性
edges = []
cnn_socialNet_read_data.add_flag_graph(test_G, social_list)
for (u, v, flag) in test_G.edges.data('flag'):
    # print(u, v, flag)
    edges.append((u, v))

# 获取每条边的矩阵
train_x = []
for i in range(len(edges)):
    matrix, row, clown = cnn_socialNet_deal_data.get_jump2_3dimension_different_size_matrix(test_G, edges[i])
    train_x.append(matrix)
    print("matrix", matrix)
    print("矩阵的大小 %d %d" % (row, clown))

# 将矩阵构造的图片写入tensorboard，进行可视化
sess = tf.Session()
writer = tf.summary.FileWriter('./log')
for i in range(len(train_x)):
    image = tf.convert_to_tensor(train_x[i])
    image = tf.image.convert_image_dtype(image, tf.float32)
    summary_op = tf.summary.image("image%d" % i, tf.expand_dims(image, 0))
    summary = sess.run(summary_op)
    writer.add_summary(summary)
    print('write %d image' % i)
writer.close()
sess.close()
