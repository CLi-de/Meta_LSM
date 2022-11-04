import numpy as np
import tensorflow as tf


#  transform
def transform_softplus(inputX, weights, bias, activations=tf.nn.softplus):
    return activations(tf.transpose(a=tf.matmul(weights, tf.transpose(a=inputX))) + bias)


def transform_relu(inputX, weights, bias, activations=tf.nn.relu):
    return activations(tf.transpose(a=tf.matmul(weights, tf.transpose(a=inputX))) + bias)


def transform_sigmoid(inputX, weights, bias, activations=tf.nn.sigmoid):
    return activations(tf.transpose(a=tf.matmul(weights, tf.transpose(a=inputX))) + bias)


#  initial weights of the unsupervised model
def weights_append(weights_curent, weights_added):
    return weights_curent.append(weights_added)


def weight_variable(func, shape, stddev, dtype=tf.float32):
    initial = func(shape, stddev=stddev, dtype=dtype)
    return tf.Variable(initial)


def bias_variable(value, shape, dtype=tf.float32):
    initial = tf.constant(value, shape=shape, dtype=dtype)
    return tf.Variable(initial)


def _initialize_softmax_weights(input_units, output_units):  # output_units: class_num
    weight = dict()
    stddev = 0.1 / np.sqrt(input_units)
    weight['w'] = weight_variable(tf.random.truncated_normal, [input_units, output_units], stddev)
    weight['b'] = bias_variable(stddev, [output_units])
    return weight
    # self._activation_function_class = tf.nn.relu


def _initialize_weights(input_units, output_units):  # output_units: class_num
    weight = dict()
    stddev = 0.1 / np.sqrt(input_units)
    weight['w'] = weight_variable(tf.random.truncated_normal, [input_units, output_units], stddev)
    weight['b'] = bias_variable(stddev, [output_units])
    return weight


def build_and_train_SVmodel(X, Y, weights_initial, input_units, n_iter_backprop, batch_size, p):
    with tf.compat.v1.variable_scope('weights2merge'):
        num_classes = _determine_num_output_neurons(Y)  # 计算类型数目
        ############################################## build ##########################################################
        visible_units_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[None, input_units])
        keep_prob = tf.compat.v1.placeholder(tf.float32)
        visible_units_placeholder_drop = tf.nn.dropout(visible_units_placeholder, rate=1 - (keep_prob))
        keep_prob_placeholders = [keep_prob]

        activation = visible_units_placeholder_drop
        for layer_index in range(len(weights_initial)):  # 这里需要改变激活函数
            activation = tf.nn.relu(
                tf.transpose(a=tf.matmul(weights_initial[layer_index]['w'], tf.transpose(a=activation)))
                + weights_initial[layer_index]['b'])
            keep_prob = tf.compat.v1.placeholder(tf.float32)  # 每一层使用不同的keep_prob
            keep_prob_placeholders.append(keep_prob)
            activation = tf.nn.dropout(activation, rate=1 - (keep_prob))

        transform_op = activation  # 无监督最后一层
        softmax_input = int(transform_op.shape[1])  # 输入节点数

        # operations
        softmax_weight = _initialize_softmax_weights(softmax_input, num_classes)  # 初始化softmax层权参
        current_weights = tf.compat.v1.global_variables()  # just see if right exist the weights

        # y = transpose(tf.matmul(tf.transpose(softmax_weight['w']), tf.transpose(transform_op))) + softmax_weight['b']  # 预测值
        y = tf.matmul(transform_op, softmax_weight['w']) + softmax_weight['b']  # 预测值
        y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, num_classes])  # 标记值

        output = tf.nn.softmax(y)
        cost_function = tf.reduce_mean(
            input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_))
        # tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=tf.stop_gradient(y_)))
        cost = tf.compat.v1.summary.scalar('cost', cost_function)  # 用于显示
        train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001).minimize(cost_function)  # 计算gradient并更新
        ############################################## train ##########################################################
        with tf.compat.v1.Session() as sess:
            # summaries合并
            merged = tf.compat.v1.summary.merge([cost])
            # 写到指定的磁盘路径中
            train_writer = tf.compat.v1.summary.FileWriter('logs/train', sess.graph)
            test_writer = tf.compat.v1.summary.FileWriter('logs/test')
            step_cnt = 0

            init = tf.compat.v1.global_variables(scope='weights2merge')
            sess.run(tf.compat.v1.initialize_variables(var_list=init))
            labels = _transform_labels_to_network_format(Y, num_classes)
            print("[START] Fine tuning step:")
            error = 1
            iteration = 0
            # for iteration in range(n_iter_backprop):
            while (error > 0.53 or iteration < 100):
                for batch_data, batch_labels in batch_generator(batch_size, X, labels):
                    feed_dict = {visible_units_placeholder: batch_data,
                                 y_: batch_labels}
                    feed_dict.update({placeholder: p for placeholder in keep_prob_placeholders})
                    sess.run(train_step, feed_dict=feed_dict)  # train（更新）权值
                # if self.verbose:
                feed_dict = {visible_units_placeholder: X, y_: labels}
                feed_dict.update({placeholder: 1.0 for placeholder in keep_prob_placeholders})
                summary, error = sess.run([merged, cost_function], feed_dict=feed_dict)  # 计算loss(error)
                train_writer.add_summary(summary, step_cnt)
                step_cnt += 1
                iteration += 1
                print(">> Epoch %d finished \tDAS training loss %f" % (iteration, error))
                # rbm_weights = sess.run(weights_initial[0]['w'])
                softmax_weights = sess.run(softmax_weight['w'])
                k = 1
                # 计算accuracy

            # 输出权值信息
            with tf.compat.v1.variable_scope('weights_for_predicting'):
                SV_weights = []
                for i in range(len(weights_initial)):
                    temp_dict = dict()
                    temp_dict['w'] = sess.run(weights_initial[i]['w'])
                    temp_dict['b'] = sess.run(weights_initial[i]['b'])
                    # temp_dict['w'] = weights_initial[i]['w']
                    # temp_dict['b'] = weights_initial[i]['b']
                    SV_weights.append(temp_dict)
                temp_dict = dict()
                temp_dict['w'] = sess.run(tf.transpose(a=softmax_weight['w']))  # 'w'为Tensor而非Variable
                temp_dict['b'] = sess.run(softmax_weight['b'])
                SV_weights.append(temp_dict)
                return SV_weights


def build_and_train_SVmodel1(X, Y, weights_initial, input_units, n_iter_backprop, batch_size, p):
    with tf.compat.v1.variable_scope('weights2merge'):
        num_classes = _determine_num_output_neurons(Y)  # 计算类型数目
        ############################################## build ##########################################################
        visible_units_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[None, input_units])
        keep_prob = tf.compat.v1.placeholder(tf.float32)
        visible_units_placeholder_drop = tf.nn.dropout(visible_units_placeholder, rate=1 - (keep_prob))
        keep_prob_placeholders = [keep_prob]

        activation = visible_units_placeholder_drop
        for layer_index in range(len(weights_initial)):  # 这里需要改变激活函数
            activation = tf.nn.relu(
                tf.transpose(a=tf.matmul(weights_initial[layer_index]['w'], tf.transpose(a=activation)))
                + weights_initial[layer_index]['b'])
            keep_prob = tf.compat.v1.placeholder(tf.float32)  # 每一层使用不同的keep_prob
            keep_prob_placeholders.append(keep_prob)
            activation = tf.nn.dropout(activation, rate=1 - (keep_prob))

        transform_op = activation  # 无监督最后一层
        softmax_input = int(transform_op.shape[1])  # 输入节点数

        # operations
        softmax_weight = _initialize_softmax_weights(softmax_input, num_classes)  # 初始化softmax层权参
        current_weights = tf.compat.v1.global_variables()  # just see if right exist the weights

        # y = transpose(tf.matmul(tf.transpose(softmax_weight['w']), tf.transpose(transform_op))) + softmax_weight['b']  # 预测值
        y = tf.matmul(transform_op, softmax_weight['w']) + softmax_weight['b']  # 预测值
        y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, num_classes])  # 标记值

        output = tf.nn.softmax(y)
        cost_function = tf.reduce_mean(
            input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_))
        # tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=tf.stop_gradient(y_)))
        cost = tf.compat.v1.summary.scalar('cost', cost_function)  # 用于显示
        train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001).minimize(cost_function)  # 计算gradient并更新
        ############################################## train ##########################################################
        with tf.compat.v1.Session() as sess:
            # summaries合并
            merged = tf.compat.v1.summary.merge([cost])
            # 写到指定的磁盘路径中
            train_writer = tf.compat.v1.summary.FileWriter('logs/train', sess.graph)
            test_writer = tf.compat.v1.summary.FileWriter('logs/test')
            step_cnt = 0

            init = tf.compat.v1.global_variables(scope='weights2merge')
            sess.run(tf.compat.v1.initialize_variables(var_list=init))
            labels = _transform_labels_to_network_format(Y, num_classes)
            print("[START] Fine tuning step:")
            for iteration in range(n_iter_backprop):
                for batch_data, batch_labels in batch_generator(batch_size, X, labels):
                    feed_dict = {visible_units_placeholder: batch_data,
                                 y_: batch_labels}
                    feed_dict.update({placeholder: p for placeholder in keep_prob_placeholders})
                    sess.run(train_step, feed_dict=feed_dict)  # train（更新）权值
                # if self.verbose:
                feed_dict = {visible_units_placeholder: X, y_: labels}
                feed_dict.update({placeholder: 1.0 for placeholder in keep_prob_placeholders})
                summary, error = sess.run([merged, cost_function], feed_dict=feed_dict)  # 计算loss(error)
                train_writer.add_summary(summary, step_cnt)
                step_cnt += 1
                print(">> Epoch %d finished \tDAS training loss %f" % (iteration, error))
                # rbm_weights = sess.run(weights_initial[0]['w'])
                softmax_weights = sess.run(softmax_weight['w'])
                k = 1
                # 计算accuracy

            # 输出权值信息
            with tf.compat.v1.variable_scope('weights_for_predicting'):
                SV_weights = []
                for i in range(len(weights_initial)):
                    temp_dict = dict()
                    temp_dict['w'] = sess.run(weights_initial[i]['w'])
                    temp_dict['b'] = sess.run(weights_initial[i]['b'])
                    # temp_dict['w'] = weights_initial[i]['w']
                    # temp_dict['b'] = weights_initial[i]['b']
                    SV_weights.append(temp_dict)
                temp_dict = dict()
                temp_dict['w'] = sess.run(tf.transpose(a=softmax_weight['w']))  # 'w'为Tensor而非Variable
                temp_dict['b'] = sess.run(softmax_weight['b'])
                SV_weights.append(temp_dict)
                return SV_weights


def batch_generator(batch_size, data, labels=None):
    """
    Generates batches of samples
    :param data: array-like, shape = (n_samples, n_features)
    :param labels: array-like, shape = (n_samples, )
    :return:
    """
    n_batches = int(np.ceil(len(data) / float(batch_size)))
    idx = np.random.permutation(len(data))
    data_shuffled = data[idx]
    if labels is not None:
        labels_shuffled = labels[idx]
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        if labels is not None:
            yield data_shuffled[start:end, :], labels_shuffled[start:end]
        else:
            yield data_shuffled[start:end, :]


def get_Batch(data, label, batch_size):
    print(data.shape, label.shape)
    input_queue = tf.compat.v1.train.slice_input_producer([data, label], num_epochs=1, shuffle=True, capacity=32)
    x_batch, y_batch = tf.compat.v1.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=32,
                                                allow_smaller_final_batch=False)
    return x_batch, y_batch


# compute Update_weight with batch
# def _stochastic_gradient_descent(data, labels, n_iter_backprop, batch_size, ):
#     for iteration in range(n_iter_backprop):
#         for batch_data, batch_labels in batch_generator(batch_size, data, labels):
#             feed_dict = {self.visible_units_placeholder: batch_data,
#                          self.y_: batch_labels}
#             feed_dict.update({placeholder: self.p for placeholder in self.keep_prob_placeholders})
#             sess.run(self.train_step, feed_dict=feed_dict)  # train（更新）权值
#
#         if self.verbose:
#             feed_dict = {self.visible_units_placeholder: data, self.y_: labels}
#             feed_dict.update({placeholder: 1.0 for placeholder in self.keep_prob_placeholders})
#             error = sess.run(self.cost_function, feed_dict=feed_dict)  # 计算loss(error)
#             print(">> Epoch %d finished \tANN training loss %f" % (iteration, error))

#  将label从dim(1)升到dim(10)

def to_categorical(labels, num_classes):
    """
    Converts labels as single integer to row vectors. For instance, given a three class problem, labels would be
    mapped as label_1: [1 0 0], label_2: [0 1 0], label_3: [0, 0, 1] where labels can be either int or string.
    :param labels: array-like, shape = (n_samples, )
    :return:
    """
    new_labels = np.zeros([len(labels), num_classes])
    label_to_idx_map, idx_to_label_map = dict(), dict()
    idx = 0
    for i, label in enumerate(labels):
        # if label not in label_to_idx_map:
        #     label_to_idx_map[label] = idx
        #     idx_to_label_map[idx] = label
        #     idx += 1
        # new_labels[i][label_to_idx_map[label]] = 1
        new_labels[i][label] = 1
    return new_labels, label_to_idx_map, idx_to_label_map


def to_categorical1(labels, num_classes):
    """
    Converts labels as single integer to row vectors. For instance, given a three class problem, labels would be
    mapped as label_1: [1 0 0], label_2: [0 1 0], label_3: [0, 0, 1] where labels can be either int or string.
    :param labels: array-like, shape = (n_samples, )
    :return:
    """
    lables_int = labels.astype(np.int)
    new_labels = np.zeros([len(labels), num_classes], dtype=np.float)
    # label_to_idx_map, idx_to_label_map = dict(), dict()
    # idx = 0
    for i in range(len(lables_int)):
        j = lables_int[i]
        new_labels[i][j] = 1  # (1,0)代表为非滑坡
    # labels.astype(np.float)
    return new_labels


def _transform_labels_to_network_format(labels, num_classes):
    new_labels = to_categorical1(labels, num_classes)
    return new_labels


# 返回labels里类型数目
def _determine_num_output_neurons(labels):
    return len(np.unique(labels))


def relu(x):
    s = np.where(x < 0, 0, x)
    return s
