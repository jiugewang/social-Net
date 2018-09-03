# 使用cnn进行训练，识别模型边
import tensorflow as tf
import numpy as np

import cnn_socialNet_read_data
import cnn_socialNet_deal_data

SIZE = 128


# 获取数据
def get_train_data():
    train_x_y = []
    global flag0_count
    global flag1_count
    flag0_count = 0
    flag1_count = 0
    sess = tf.InteractiveSession()
    for i in range(1, 10):
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
            matrix, row, clown = cnn_socialNet_deal_data.get_jump1_3dimension_different_size_matrix(my_graph, edges[j][0])
            image = tf.convert_to_tensor(matrix)
            image = tf.image.convert_image_dtype(image, tf.float32)
            resize_image = tf.image.resize_images(image, [128, 128], method=3)
            img_numpy = resize_image.eval(session=sess)
            # print('resize_iamge', img_numpy)
            # matrix1 = tf.constant(resize_image).eval()
            # print(edges[j][1])
            if int(edges[j][1]) == 1:
                label = [1, 0]
            else:
                label = [0, 1]
            train_x_y.append((img_numpy, label))
    sess.close()
    return train_x_y, flag0_count, flag1_count


train_data, count0, count1 = get_train_data()
print(train_data[0])
print("所有边的个数", len(train_data))
print("社区内边的个数", count1)
print("社区间边的个数", count0)
train_x = []
train_y = []
for i in range(len(train_data)):
    train_x.append(train_data[i][0])
    train_y.append(train_data[i][1])

# 构建网络

x_data = tf.placeholder(tf.float32, [None, SIZE, SIZE, 3])
y_data = tf.placeholder(tf.float32, [None, None])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)


def weight_variable(shape):
    """构建权重"""
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)


def bias_variable(shape):
    """构建偏移量"""
    init = tf.random_normal(shape)
    return tf.Variable(init)


def conv2d(x, weight):
    """x是输入的样本，在这里就是图像，x的shape=[batch, height, width, channels]"""

    # - batch是输入样本的数量
    # - height,width是每张图片的高和宽
    # - channels是输入的通道，比如输入的是灰色图像，那么channels=1，如果是rgb，那么channels=3

    "W表示卷积核的参数，W的shape=[height,width,in_channels,out_channels]"

    """
    strides参数表示的是卷积核在输入x的各个维度下移动的步长。了解cnn的都知道，在宽和高方向stride的大小
    决定了卷积后图像的size。这里为什么有4个维度呢？因为strides对应的是输入x的维度，所以第一个参数表示
    在batch方向移动的步长，第四个参数表示在channels上移动的步长，这两个参数都设置为1就好。重点是第二个
    ，第三个参数的意义，也就是在height和width方向上的步长，这里也都设置为1。
    """
    return tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    """这里用2*2的max_pool。参数ksize定义pool窗口的大小，每个维度的意义与之前的strides相同"""
    # - 第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch,height,width,channels]
    # - 第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1,height,width,1]，因为不想在batch和channels上做池化，所以这两个维度设为了1
    # - 第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1,stride,stride,1]
    # - 第四个参数padding：和卷积类似，可以取'VALID'或者'SAME'
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def dropout(x, keep):
    return tf.nn.dropout(x, keep)


def cnn_1jump_layer(classnum):
    """create cnn layer"""
    W1 = weight_variable([7, 7, 3, 64])  # 卷积核大小(7,7)， 输入通道(3)， 输出通道(64)
    b1 = bias_variable([64])
    # conv1
    conv1 = tf.nn.relu(conv2d(x_data, W1) + b1)
    # pool1
    pool1 = max_pool(conv1)
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    # 减少过拟合，随机让某些权重不更新
    # drop1 = dropout(norm1, keep_prob_5)  # 32 * 64 * 64 多个输入channel 被filter内积掉了

    W2a = weight_variable([1, 1, 64, 64])
    b2a = bias_variable([64])
    W2 = weight_variable([3, 3, 64, 192])
    b2 = bias_variable([192])
    # conv2a
    conv2a = tf.nn.relu(conv2d(norm1, W2a) + b2a)
    # conv2
    conv2 = tf.nn.relu(conv2d(conv2a, W2) + b2)
    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    # pool2
    pool2 = max_pool(norm2)  # 32 * 32

    W3a = weight_variable([1, 1, 192, 192])
    b3a = bias_variable([192])
    W3 = weight_variable([3, 3, 192, 384])
    b3 = bias_variable([384])
    # conv3a
    conv3a = tf.nn.relu(conv2d(pool2, W3a) + b3a)
    # conv3
    conv3 = tf.nn.relu(conv2d(conv3a, W3) + b3)
    # pool3
    pool3 = max_pool(conv3)  # 16 * 16

    W4a = weight_variable([1, 1, 384, 384])
    b4a = bias_variable([384])
    W4 = weight_variable([3, 3, 384, 256])
    b4 = bias_variable([256])
    # conv4a
    conv4a = tf.nn.relu(conv2d(pool3, W4a) + b4a)
    # conv4
    conv4 = tf.nn.relu(conv2d(conv4a, W4) + b4)

    W5a = weight_variable([1, 1, 256, 256])
    b5a = bias_variable([256])
    W5 = weight_variable([3, 3, 256, 256])
    b5 = bias_variable([256])
    # conv4a
    conv5a = tf.nn.relu(conv2d(conv4, W5a) + b5a)
    # conv4
    conv5 = tf.nn.relu(conv2d(conv5a, W5) + b5)

    W6a = weight_variable([1, 1, 256, 256])
    b6a = bias_variable([256])
    W6 = weight_variable([3, 3, 256, 256])
    b6 = bias_variable([256])
    # conv4a
    conv6a = tf.nn.relu(conv2d(conv5, W6a) + b6a)
    # conv4
    conv6 = tf.nn.relu(conv2d(conv6a, W6) + b6)
    pool4 = max_pool(conv6)  # 8 * 8


def cnn_2jump_layer(classnum):
    """create cnn layer"""
    ksize = 1
    ksize_a = 3
    ksize_b = 5

    W1 = weight_variable([7, 7, 3, 64])  # 卷积核大小(7,7)， 输入通道(3)， 输出通道(64)
    b1 = bias_variable([64])
    # conv1
    conv1 = tf.nn.relu(conv2d(x_data, W1) + b1)
    # pool1
    pool1 = max_pool(conv1)
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    # 减少过拟合，随机让某些权重不更新
    # drop1 = dropout(norm1, keep_prob_5)  # 32 * 64 * 64 多个输入channel 被filter内积掉了

    """第二层"""
    W2a = weight_variable([1, 1, 64, 64])
    b2a = bias_variable([64])

    W2b = weight_variable([1, 1, 64, 64])
    b2b = bias_variable([64])

    W2 = weight_variable([3, 3, 64, 192])
    b2 = bias_variable([192])

    # conv2a
    conv2a = tf.nn.relu(conv2d(norm1, W2a) + b2a)

    # conv2b
    conv2b = tf.nn.relu(conv2d(conv2a, W2b) + b2b)

    # conv2
    conv2 = tf.nn.relu(conv2d(conv2b, W2) + b2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    # pool2
    pool2 = max_pool(norm2)  # 32 * 32

    """第三层"""
    W3a = weight_variable([1, 1, 192, 192])
    b3a = bias_variable([192])

    W3b = weight_variable([1, 1, 192, 192])
    b3b = bias_variable([192])

    W3 = weight_variable([3, 3, 192, 384])
    b3 = bias_variable([384])

    # conv3a
    conv3a = tf.nn.relu(conv2d(pool2, W3a) + b3a)

    # conv3b
    conv3b = tf.nn.relu(conv2d(conv3a, W3b) + b3b)

    # conv3
    conv3 = tf.nn.relu(conv2d(conv3b, W3) + b3)
    # pool3
    pool3 = max_pool(conv3)  # 16 * 16

    """第四层"""
    W4a = weight_variable([1, 1, 384, 384])
    b4a = bias_variable([384])

    W4b = weight_variable([1, 1, 384, 384])
    b4b = bias_variable([384])

    W4 = weight_variable([3, 3, 384, 256])
    b4 = bias_variable([256])

    # conv4a
    conv4a = tf.nn.relu(conv2d(pool3, W4a) + b4a)

    # conv4b
    conv4b = tf.nn.relu(conv2d(conv4a, W4b) + b4b)

    # conv4
    conv4 = tf.nn.relu(conv2d(conv4b, W4) + b4)

    """第五层"""
    W5a = weight_variable([1, 1, 256, 256])
    b5a = bias_variable([256])

    W5b = weight_variable([1, 1, 256, 256])
    b5b = bias_variable([256])

    W5 = weight_variable([3, 3, 256, 256])
    b5 = bias_variable([256])

    # conv5a
    conv5a = tf.nn.relu(conv2d(conv4, W5a) + b5a)

    # conv5b
    conv5b = tf.nn.relu(conv2d(conv5a, W5b) + b5b)

    # conv5
    conv5 = tf.nn.relu(conv2d(conv5b, W5) + b5)

    """第六层"""
    W6a = weight_variable([1, 1, 256, 256])
    b6a = bias_variable([256])

    W6b = weight_variable([1, 1, 256, 256])
    b6b = bias_variable([256])

    W6 = weight_variable([3, 3, 256, 256])
    b6 = bias_variable([256])

    # conv6a
    conv6a = tf.nn.relu(conv2d(conv5, W6a) + b6a)

    # conv6b
    conv6b = tf.nn.relu(conv2d(conv6a, W6b) + b6b)

    # conv6
    conv6 = tf.nn.relu(conv2d(conv6b, W6) + b6)
    pool4 = max_pool(conv6)  # 8 * 8

    # 全连接层
    Wf = weight_variable([8 * 8 * 256, 1024])
    bf = bias_variable([1024])
    drop3_flat = tf.reshape(pool4, [-1, 8 * 8 * 256])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weight_variable([1024, classnum])
    bout = weight_variable([classnum])
    # out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out


def train(train_x, train_y, tfsavepath):
    # log.debug('train')
    out = cnn_2jump_layer(2)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_data))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_data, 1)), tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_size = 40
        num_batch = len(train_x) // batch_size
        for n in range(40):
            #r = np.random.permutation(train_x)
            #train_x = train_x[r, :]
            #train_y = train_y[r, :]

            for i in range(num_batch):
                batch_x = train_x[i * batch_size: (i + 1) * batch_size]
                batch_y = train_y[i * batch_size: (i + 1) * batch_size]
                _, loss = sess.run([train_step, cross_entropy], \
                                   feed_dict={x_data: batch_x, y_data: batch_y,
                                              keep_prob_5: 0.75, keep_prob_75: 0.75})

                print(n * num_batch + i, loss)
                if((n * num_batch + i) % 100 == 0):
                    # 获取测试数据的准确率
                    acc = accuracy.eval({x_data: batch_x, y_data: batch_y, keep_prob_5: 1.0, keep_prob_75: 1.0})
                    print('accuracy is ', acc)
        saver.save(sess, tfsavepath)


if __name__ == '__main__':
    train(train_x, train_y, './checkpoint/social.ckpt')