# 使用cnn进行训练，识别模型边
import tensorflow as tf

import cnn_socialNet_read_data
import cnn_socialNet_deal_data


# 获取数据
def get_train_data():
    train_data = []
    global flag0_count
    global flag1_count
    flag0_count = 0
    flag1_count = 0
    for i in range(1, 21):
        file_path_community = './0814data/%d/community-standard.txt' % i
        file_path_network = './0814data/%d/network.txt' % i
        # print(file_path_network)
        social_list = cnn_socialNet_read_data.get_standard_network(file_path_community)
        my_graph = cnn_socialNet_read_data.get_graph(file_path_network)
        cnn_socialNet_read_data.add_flag_graph(my_graph, social_list)
        edges = []
        for (u, v, flag) in my_graph.edges.data('flag'):
            # print(u, v, flag)
            if int(flag) == 0:
                flag0_count = flag0_count + 1
            else:
                flag1_count = flag1_count + 1
            edges.append(((u, v), flag))
        for j in range(len(edges)):
            matrix, row, clown = cnn_socialNet_deal_data.get_jump2_3dimension_different_size_matrix(my_graph, edges[j][0])
            # print(edges[j][1])
            if int(edges[j][1]) == 1:
                label = [1, 0]
            else:
                label = [0, 1]
            train_data.append((matrix, label))
    return train_data, flag0_count, flag1_count


train_x, count0, count1 = get_train_data()
print(train_x[0])
print("所有边的个数", len(train_x))
print("社区内边的个数", count1)
print("社区间边的个数", count0)

# 构建网络



