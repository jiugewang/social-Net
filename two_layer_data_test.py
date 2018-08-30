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
    # print(u, v, flag)
    edges.append(((u, v), int(flag)))

# 获取每条边的矩阵
train_x = []
for i in range(len(edges)):
    matrix1, row1, clown1 = cnn_socialNet_deal_data.get_jump1_3dimension_different_size_matrix(test_G, edges[i][0])
    matrix2, row2, clown2 = cnn_socialNet_deal_data.get_jump2_3dimension_different_size_matrix(test_G, edges[i][0])
    train_x.append((matrix1, matrix2, edges[i][1], edges[i][0]))
    # print("matrix", matrix)
    # print("flag", edges[i][1])
    # print("矩阵的大小 %d %d" % (row, clown))


# 将矩阵构造的图片写入tensorboard，进行可视化
sess = tf.Session()
writer = tf.summary.FileWriter('./log_2picture_diff_jump1')
images = tf.placeholder(tf.float32, [None, 128, 128, 3])
for i in range(len(train_x)):
    # print(train_x[i][0].shape)
    image1 = tf.convert_to_tensor(train_x[i][0])
    image1 = tf.image.convert_image_dtype(image1, tf.float32)
    image2 = tf.convert_to_tensor(train_x[i][1])
    image2 = tf.image.convert_image_dtype(image2, tf.float32)
    resize_image1 = tf.image.resize_images(image1, [128, 128], method=3)
    resize_image2 = tf.image.resize_images(image2, [128, 128], method=3)
    print(type(resize_image1))
    print(resize_image1.shape)
    tmp_images = [resize_image1, resize_image2]
    images = tf.convert_to_tensor(tmp_images)
    print(images.shape)
    summary_op = tf.summary.image("image%d-%d-(%s,%s)" % (i, train_x[i][2], train_x[i][3][0], train_x[i][3][1]), images)
    summary = sess.run(summary_op)
    writer.add_summary(summary)
    print('write %d image' % i)
writer.close()
sess.close()
