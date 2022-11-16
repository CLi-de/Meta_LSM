""" Utility functions. """
import numpy as np
import tensorflow as tf
import pandas as pd

import tf_slim as slim
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def normalize(inp, activation, reuse, scope):
    if FLAGS.norm == 'batch_norm':
        return slim.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'layer_norm':
        return slim.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp


## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(input_tensor=tf.square(pred - label))


def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / tf.cast(tf.shape(input=label)[0],
                                                                                        dtype=tf.float32)  # 注意归一


def tasksbatch_generator(data, batch_size, num_samples, dim_input, dim_output):
    """generate batch tasks"""
    init_inputs = np.zeros([batch_size, num_samples, dim_input], dtype=np.float32)
    labels = np.zeros([batch_size, num_samples, dim_output], dtype=np.float32)

    np.random.shuffle(data)
    start_index = np.random.randint(0, len(data) - batch_size)
    batch_tasks = data[start_index:(start_index + batch_size)]

    cnt_sample = []
    for i in range(batch_size):
        cnt_sample.append(len(batch_tasks[i]))

    for i in range(batch_size):
        np.random.shuffle(batch_tasks[i])  # shuffle samples in each task
        start_index1 = np.random.randint(0, len(batch_tasks[i]) - num_samples)
        task_samples = batch_tasks[i][start_index1:(start_index1 + num_samples)]
        for j in range(num_samples):
            init_inputs[i][j] = task_samples[j][0]
            if task_samples[j][1] == 1:
                labels[i][j][0] = 1  # 滑坡
            else:
                labels[i][j][1] = 1  # 非滑坡
    return init_inputs, labels, np.array(cnt_sample).astype(np.float32)


# for each task
# def sample_generator(one_task, dim_input, dim_output):
#     """generate samples from one tasks"""
#     np.random.shuffle(one_task)
#     num_samples = len(one_task)
#     init_inputs = np.zeros([1, num_samples, dim_input], dtype=np.float32)
#     labels = np.zeros([1, num_samples, dim_output], dtype=np.float32)
#     for i in range(num_samples):
#         init_inputs[0][i] = one_task[i][0]
#         if one_task[i][1] == 1:
#             labels[0][i][0] = 1
#         else:
#             labels[0][i][1] = 1
#     return init_inputs, labels

def batch_generator(one_task, dim_input, dim_output, batch_size):
    """generate samples from one tasks"""
    np.random.shuffle(one_task)
    batch_ = one_task[:batch_size]
    init_inputs = np.zeros([batch_size, dim_input], dtype=np.float32)
    labels = np.zeros([batch_size, dim_output], dtype=np.float32)
    for i in range(batch_size):
        init_inputs[i] = batch_[i][0]
        if batch_[i][1] == 1:
            labels[i][0] = 1
        else:
            labels[i][1] = 1
    return init_inputs, labels


# for each region (e.g., FJ&FL)
def sample_generator_(tasks, dim_input, dim_output):
    all_samples = np.array(tasks[0])
    num_samples = int(FLAGS.num_samples_each_task / 2)
    for i in range(len(tasks) - 1):
        if len(tasks[i + 1]) > 0:
            all_samples = np.vstack((all_samples, np.array(tasks[i + 1])))
    init_inputs = np.zeros([1, num_samples, dim_input], dtype=np.float32)
    labels = np.zeros([1, num_samples, dim_output], dtype=np.float32)
    for i in range(num_samples):
        init_inputs[0][i] = all_samples[i][:-1]
    if all_samples[i][-1] == 1:
        labels[0][i][0] = 1
    else:
        labels[0][i][1] = 1
    return init_inputs, labels


def meta_train_test(fj_tasks, fl_tasks, mode=0):
    test1_fj_tasks, test1_fl_tasks, read_tasks, one_test_tasks = [], [], [], []
    _train, _test = [], []
    # np.random.shuffle(tasks)
    if mode == 0:
        elig_tasks = []
        for i in range(len(fj_tasks)):
            if len(fj_tasks[i]) > FLAGS.num_samples_each_task:
                elig_tasks.append(fj_tasks[i])
            elif len(fj_tasks[i]) > 10:  # set 10 to test K=10-shot learning
                test1_fj_tasks.append(fj_tasks[i])
            else:
                read_tasks.append(fj_tasks[i])
        _train = elig_tasks[:int(len(elig_tasks) / 4 * 3)]
        _test = elig_tasks[int(len(elig_tasks) / 4 * 3):] + test1_fj_tasks
        for i in range(len(read_tasks)):  # read_tasks暂时不用
            one_test_tasks.extend(read_tasks[i])
        return _train, _test

    if mode == 1:
        for i in range(len(fj_tasks)):
            if len(fj_tasks[i]) > FLAGS.num_samples_each_task:
                _train.append(fj_tasks[i])
        for i in range(len(fl_tasks)):
            if len(fl_tasks[i]) > 10:
                _test.append(fl_tasks[i])
        return _train, _test

    if mode == 2 or mode == 3:
        elig_fj_tasks, elig_fl_tasks = [], []
        for i in range(len(fj_tasks)):
            if len(fj_tasks[i]) > FLAGS.num_samples_each_task:
                elig_fj_tasks.append(fj_tasks[i])
            elif len(fj_tasks[i]) > 10:
                test1_fj_tasks.append(fj_tasks[i])
        for i in range(len(fl_tasks)):
            if len(fl_tasks[i]) > FLAGS.num_samples_each_task:
                elig_fl_tasks.append(fl_tasks[i])
            elif len(fl_tasks[i]) > 10:
                test1_fl_tasks.append(fl_tasks[i])
        if mode == 2:
            _train = elig_fj_tasks[:int(len(elig_fj_tasks) / 4 * 3)] + elig_fl_tasks
            _test = elig_fj_tasks[int(len(elig_fj_tasks) / 4 * 3):] + test1_fj_tasks
            return _train, _test
        elif mode == 3:
            _train = elig_fj_tasks + elig_fl_tasks[:int(len(elig_fj_tasks) / 2)]
            _test = elig_fl_tasks[int(len(elig_fl_tasks) / 2):] + test1_fl_tasks
            return _train, _test
    # _test.extend(resid_tasks)


def meta_train_test1(HK_tasks):
    test_hk_tasks, one_test_tasks, remain_tasks, elig_tasks = [], [], [], []
    for i in range(len(HK_tasks)):
        if len(HK_tasks[i]) > FLAGS.num_samples_each_task:
            elig_tasks.append(HK_tasks[i])
        # elif len(HK_tasks[i]) > 10:  # set 10 to test K=10-shot learning
        #     test_hk_tasks.append(HK_tasks[i])
        else:
            remain_tasks.append(HK_tasks[i])
    np.random.shuffle(elig_tasks)
    _train = elig_tasks[:int(len(elig_tasks) / 4 * 3)]
    _test = elig_tasks[int(len(elig_tasks) / 4 * 3):]
    # _test = elig_tasks[int(len(elig_tasks) / 4 * 3):] + test_hk_tasks
    # for i in range(len(remian_tasks)):  # read_tasks暂时不用
    #     one_test_tasks.extend(read_tasks[i])
    return _train, _test


def save_tasks(tasks):
    """将tasks存到csv中"""
    writer = pd.ExcelWriter('./seg_output/' + FLAGS.str_region + '_tasks.xlsx')
    for i in range(len(tasks)):
        task_sampels = []
        for j in range(len(tasks[i])):
            attr_lb = np.append(tasks[i][j][0], tasks[i][j][1])
            task_sampels.append(attr_lb)
        data_df = pd.DataFrame(task_sampels)
        data_df.to_excel(writer, 'task_' + str(i), float_format='%.5f', header=False, index=False)
        writer.save()
    writer.close()


def read_tasks(file):
    """获取tasks"""
    f = pd.ExcelFile(file)
    tasks = [[] for i in range(len(f.sheet_names))]
    k = 0  # count task
    for sheetname in f.sheet_names:
        attr = pd.read_excel(file, usecols=[i for i in range(FLAGS.dim_input)], sheet_name=sheetname,
                             header=None).values.astype(np.float32)
        label = pd.read_excel(file, usecols=[FLAGS.dim_input], sheet_name=sheetname, header=None).values.reshape(
            (-1,)).astype(np.float32)
        for j in range(np.shape(attr)[0]):
            tasks[k].append([attr[j], label[j]])
        k += 1
    return tasks


def savepts_fortask(clusters, file):
    writer = pd.ExcelWriter(file)
    count = 0
    for cluster in clusters:
        pts = []
        for pixel in cluster.pixels:
            pts.append(pixel)
        data_df = pd.DataFrame(pts)
        data_df.to_excel(writer, 'task_' + str(count), float_format='%.5f', header=False, index=False)
        count = count + 1
        writer.save()
    writer.close()


def read_pts(file):
    """获取tasks"""
    f = pd.ExcelFile(file)
    tasks = []
    for sheetname in f.sheet_names:
        arr = pd.read_excel(file, sheet_name=sheetname).values.astype(np.float32)
        tasks.append(arr)
    return tasks


def cal_measure(pred, y_test):
    TP = ((pred == 1) * (y_test == 1)).astype(int).sum()
    FP = ((pred == 1) * (y_test == 0)).astype(int).sum()
    FN = ((pred == 0) * (y_test == 1)).astype(int).sum()
    TN = ((pred == 0) * (y_test == 0)).astype(int).sum()
    # statistical measure
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F_measures = 2 * Precision * Recall / (Precision + Recall)
    print('Precision: %f' % Precision, '\nRecall: %f' % Recall, '\nF_measures: %f' % F_measures)