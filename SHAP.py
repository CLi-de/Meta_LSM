#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author  : CHEN Li
# @Time    : 2022/12/2 14:55
# @File    : SHAP.py
# @annotation

import tensorflow as tf
import xgboost
import shap
import warnings
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import pandas as pd
from meta_LSM import FLAGS
from modeling import MAML

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# construct model
def init_weights(file):
    """读取DAS权参"""
    with tf.compat.v1.variable_scope('model'):  # get variable in 'model' scope, to reuse variables
        npzfile = np.load(file)
        weights = {}
        weights['w1'] = npzfile['arr_0']
        weights['b1'] = npzfile['arr_1']
        weights['w2'] = npzfile['arr_2']
        weights['b2'] = npzfile['arr_3']
        weights['w3'] = npzfile['arr_4']
        weights['b3'] = npzfile['arr_5']
        weights['w4'] = npzfile['arr_6']
        weights['b4'] = npzfile['arr_7']
    return weights


# define model.pred_prob() for shap.KernelExplainer(model, data)
def pred_prob(X):
    with tf.compat.v1.variable_scope('model', reuse=True):
        return sess.run(tf.nn.softmax(model.forward(X, model.weights, reuse=True)))


# read subtasks
def read_tasks(file):
    """获取tasks"""
    f = pd.ExcelFile(file)
    tasks = [[] for i in range(len(f.sheet_names))]
    k = 0
    for sheetname in f.sheet_names:
        # attr = pd.read_excel(file, usecols=[i for i in range(FLAGS.dim_input)], sheet_name=sheetname,
        #                      header=None).values.astype(np.float32)
        arr = pd.read_excel(file, sheet_name=sheetname,
                            header=None).values.astype(np.float32)
        tasks[k] = arr
        k = k + 1
    return tasks


"""construct model"""
tf.compat.v1.disable_eager_execution()
model = MAML(FLAGS.dim_input, FLAGS.dim_output, test_num_updates=5)
input_tensors_input = (FLAGS.meta_batch_size, int(FLAGS.num_samples_each_task / 2), FLAGS.dim_input)
input_tensors_label = (FLAGS.meta_batch_size, int(FLAGS.num_samples_each_task / 2), FLAGS.dim_output)
model.construct_model(input_tensors_input=input_tensors_input, input_tensors_label=input_tensors_label,
                      prefix='metatrain_')

tmp = np.loadtxt('./src_data/samples_HK.csv', dtype=str, delimiter=",", encoding='UTF-8')
feature_names = tmp[0, :-3].astype(np.str)
task = read_tasks('./metatask_sampling/HK_tasks_K{k}.xlsx'.format(k=FLAGS.K))

sess = tf.compat.v1.InteractiveSession()
init = tf.compat.v1.global_variables()  # optimizer里会有额外variable需要初始化
sess.run(tf.compat.v1.variables_initializer(var_list=init))

# eligible i: [11, 31, 81, ],['planting area', 'catchment', 'mountainous areas with severe deformation', '']
# SHAP for ith subtasks
for i in range(181, len(task), 10):
    model.weights = init_weights('./models_of_blocks/HK/model' + str(i) + '.npz')

    tmp_ = task[i]
    np.random.shuffle(tmp_)  # shuffle
    # 训练集
    x_train = tmp_[:int(tmp_.shape[0] / 2), :-1]  # 加载i行数据部分
    y_train = tmp_[:int(tmp_.shape[0] / 2), -1]  # 加载类别标签部分
    x_train = x_train / x_train.max(axis=0)
    # 测试集
    # x_test = tmp_[int(tmp_.shape[0] / 2):, :-1]  # 加载i行数据部分
    # y_test = tmp_[int(tmp_.shape[0] / 2):, -1]  # 加载类别标签部分
    x_test = tmp_[:, :-1]  # 加载i行数据部分
    y_test = tmp_[:, -1]  # 加载类别标签部分
    x_test = x_test / x_test.max(axis=0)

    shap.initjs()
    # SHAP demo are using dataframe instead of nparray
    x_train = pd.DataFrame(x_train)  # convert np.array to pd.dataframe
    x_test = pd.DataFrame(x_test)
    x_train.columns = feature_names  # 添加特征名称
    x_test.columns = feature_names

    # explainer = shap.KernelExplainer(pred_prob, shap.kmeans(x_train, 80))
    explainer = shap.KernelExplainer(pred_prob, x_train)
    shap_values = explainer.shap_values(x_test, nsamples=100)  # shap_values
    # (_prob, n_samples, features)
    # TODO: refer https://shap-lrjball.readthedocs.io/en/latest/generated/shap.summary_plot.html to change plot style
    # shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], x_test.iloc[0, :], show=True, matplotlib=True)  # single feature
    shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
    plt.savefig('tmp/bar_'+str(i)+'.pdf')
    plt.close()
    shap.summary_plot(shap_values[1], x_test, plot_type="violin", show=False)
    plt.savefig('tmp/violin_'+str(i)+'.pdf')
    plt.close()
# shap.summary_plot(shap_values[1], x_test, plot_type="compact_dot")

# shap.force_plot(explainer.expected_value[1], shap_values[1], x_test, link="logit")

# shap.dependence_plot('DV', shap_values[1], x_test, interaction_index=None)
# shap.dependence_plot('SPI', shap_values[1], x_test, interaction_index='DV')
# shap.plots.beeswarm(shap_values[0])  # the beeswarm plot requires Explanation object as the `shap_values` argument
