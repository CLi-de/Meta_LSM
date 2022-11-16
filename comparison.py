# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics._classification import accuracy_score

from utils_v2 import cal_measure


# import matplotlib.pyplot as plt


def SVM_compare(x_train, y_train, x_test, y_test):
    """predict and test"""
    print('start SVM evaluation...')
    clf = svm.SVC(C=5, kernel='rbf', gamma=1 / (2 * x_train.var()), decision_function_shape='ovr', probability=True)
    # clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    clf.fit(x_train, y_train)
    # train accuracy
    predict_results = clf.predict(x_train)
    print('train accuracy:' + str(metrics.accuracy_score(y_train, predict_results)))  # 0.69-0.71
    # test accuracy
    predict_results1 = clf.predict(x_test)
    print('test accuracy:' + str(metrics.accuracy_score(y_test, predict_results1)))  # 0.69-0.71
    # Precision, Recall, F1-score
    cal_measure(predict_results1, y_test)

    """LSM prediction"""
    grid_f = np.loadtxt('./src_data/grid_samples_HK.csv', dtype=str, delimiter=",",
                        encoding='UTF-8')
    samples_f = grid_f[1:, :-2].astype(np.float32)
    xy = grid_f[1:, -2:].astype(np.float32)
    samples_f = samples_f / samples_f.max(axis=0)

    predict_results2 = clf.predict_proba(samples_f)

    # save the prediction result
    data = np.hstack((xy, predict_results2))
    data_df = pd.DataFrame(data)
    writer = pd.ExcelWriter('./tmp/SVM_prediction_HK.xlsx')
    data_df.to_excel(writer, 'page_1', float_format='%.5f')
    writer.close()
    print('done SVM LSM prediction! \n')


def ANN_compare(x_train, y_train, x_test, y_test):
    """predict and test"""
    print('start ANN evaluation...')
    model = MLPClassifier(hidden_layer_sizes=(32, 32, 16), activation='relu', solver='adam', alpha=0.0001,
                          batch_size=64, max_iter=1000)
    model.fit(x_train, y_train)
    predict1 = model.predict(x_train)
    print('Train Accuracy: %f' % accuracy_score(y_train, predict1))  # 奉节，在0.82 - 0.90；
    # test
    predict = model.predict(x_test)
    print('Test Accuracy: %f' % accuracy_score(y_test, predict))  # 奉节，在0.72 - 0.76间；涪陵，在0.76-0.81之间
    # Precision, Recall, F1-score
    cal_measure(predict, y_test)

    """LSM prediction"""
    grid_f = np.loadtxt('./src_data/grid_samples_HK.csv', dtype=str, delimiter=",",
                        encoding='UTF-8')
    samples_f = grid_f[1:, :-2].astype(np.float32)
    xy = grid_f[1:, -2:].astype(np.float32)
    samples_f = samples_f / samples_f.max(axis=0)

    predict_results = model.predict_proba(samples_f)

    # save the prediction result
    data = np.hstack((xy, predict_results))
    data_df = pd.DataFrame(data)
    writer = pd.ExcelWriter('./tmp/MLP_prediction_HK.xlsx')
    data_df.to_excel(writer, 'page_1', float_format='%.5f')
    writer.close()
    print('done MLP LSM prediction! \n')


def RF_compare(x_train, y_train, x_test, y_test):
    """predict and test"""
    print('start RF evaluation...')
    clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2,
                                 bootstrap=True)

    clf.fit(x_train, y_train)
    pred_train = clf.predict(x_train)
    pred_test = clf.predict(x_test)
    # 训练精度
    print('train_Accuracy: %f' % accuracy_score(y_train, pred_train))
    # 测试精度
    print('test_Accuracy: %f' % accuracy_score(y_test, pred_test))  # 0.71 - 0.77
    # pred1 = clf2.predict_proba() # 预测类别概率
    cal_measure(pred_test, y_test)

    """"LSM prediction"""
    grid_f = np.loadtxt('./src_data/grid_samples_HK.csv', dtype=str, delimiter=",",
                        encoding='UTF-8')
    samples_f = grid_f[1:, :-2].astype(np.float32)
    xy = grid_f[1:, -2:].astype(np.float32)
    samples_f = samples_f / samples_f.max(axis=0)

    predict_results = clf.predict_proba(samples_f)
    # save the prediction result
    data = np.hstack((xy, predict_results))
    data_df = pd.DataFrame(data)
    writer = pd.ExcelWriter('./tmp/RF_prediction_HK.xlsx')
    data_df.to_excel(writer, 'page_1', float_format='%.5f')
    writer.close()
    print('done RF LSM prediction! \n')


"""SVM"""
# Input data
pd.read_excel('./src_data/samples_HK.xlsx', 'Sheet1', index_col=0) \
    .to_csv('./tmp/data.csv', encoding='utf-8')
tmp = np.loadtxt('./tmp/data.csv', dtype=str, delimiter=",", encoding='UTF-8')
tmp_ = np.hstack((tmp[1:, :-3], tmp[1:, -1].reshape(-1, 1))).astype(np.float32)
np.random.shuffle(tmp_)  # shuffle
# 训练集
x_train = tmp_[:int(tmp_.shape[0] / 2), :-1]  # 加载i行数据部分
y_train = tmp_[:int(tmp_.shape[0] / 2), -1]  # 加载类别标签部分
x_train = x_train / x_train.max(axis=0)
# 测试集
x_test = tmp_[int(tmp_.shape[0] / 2):, :-1]  # 加载i行数据部分
y_test = tmp_[int(tmp_.shape[0] / 2):, -1]  # 加载类别标签部分
x_test = x_test / x_test.max(axis=0)

SVM_compare(x_train, y_train, x_test, y_test)

ANN_compare(x_train, y_train, x_test, y_test)

RF_compare(x_train, y_train, x_test, y_test)