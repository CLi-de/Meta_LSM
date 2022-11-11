# -*- coding:utf-8 -*-
from sklearn import svm
from sklearn import metrics
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics._classification import accuracy_score

import matplotlib as mpl
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

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


def SVM_compare(x_train, y_train, x_test, y_test):
    """predict and test"""
    print('start SVM evaluation...')
    clf = svm.SVC(C=1, kernel='rbf', gamma=1 / (2 * x_train.var()), decision_function_shape='ovr', probability=True)
    # clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    clf.fit(x_train, y_train)
    # train accuracy
    predict_results1 = clf.predict(x_train)
    print('train accuracy:' + str(metrics.accuracy_score(predict_results1, y_train)))  # 0.69-0.71
    # test accuracy
    predict_results = clf.predict(x_test)
    print('test accuracy:' + str(metrics.accuracy_score(y_test, predict_results)))  # 0.69-0.71
    # Precision, Recall, F1-score
    cal_measure(predict_results, y_test)

    # # train AUROC
    # fpr1, tpr1, threshold1 = metrics.roc_curve(y_train, predict_results1, pos_label=1)
    # print(metrics.auc(fpr1, tpr1))
    # # Test AUROC
    # fpr, tpr, threshold = metrics.roc_curve(y_test, predict_results, pos_label=1)
    # print(metrics.auc(fpr, tpr))

    """LSM prediction"""
    grid_f = np.loadtxt('./src_data/grid_samples_HK.csv', dtype=str, delimiter=",",
                        encoding='UTF-8')
    samples_f = grid_f[1:, :-2].astype(np.float32)
    samples_f = samples_f / samples_f.max(axis=0)

    predict_results2 = clf.predict_proba(samples_f)

    # save the prediction result
    data_df = pd.DataFrame(predict_results2)
    writer = pd.ExcelWriter('./tmp/SVM_prediction_HK.xlsx')
    data_df.to_excel(writer, 'page_1', float_format='%.5f')
    writer.close()
    print('done SVM LSM prediction!')


def ANN_compare(x_train, y_train, x_test, y_test):
    """predict and test"""
    print('start ANN evaluation...')
    model = MLPClassifier(hidden_layer_sizes=(32, 32, 16), activation='relu', solver='adam', alpha=0.0001,
                          batch_size=32, max_iter=1000)
    model.fit(x_train, y_train)
    predict1 = model.predict(x_train)
    print('Done.\n Train Accuracy: %f' % accuracy_score(y_train, predict1))  # 奉节，在0.82 - 0.90；
    # test
    predict = model.predict(x_test)
    print('Done.\n Test Accuracy: %f' % accuracy_score(y_test, predict))  # 奉节，在0.72 - 0.76间；涪陵，在0.76-0.81之间
    # Precision, Recall, F1-score
    cal_measure(predict, y_test)

    # # train AUROC
    # fpr1,tpr1,threshold1 = metrics.roc_curve(y_train,predict1,pos_label=1)
    # print(metrics.auc(fpr1,tpr1))
    #
    # # Test AUROC
    # fpr,tpr,threshold = metrics.roc_curve(y_test,predict,pos_label=1)
    # print(metrics.auc(fpr,tpr))

    # # Compute test ROC curve and ROC area for each class
    # fpr, tpr, threshold = metrics.roc_curve(y_test, predict)  ###计算真正率和假正率
    # roc_auc = metrics.auc(fpr, tpr)  ###计算auc的值
    # plt.figure()
    # lw = 2
    # plt.figure(figsize=(10, 10))
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # # plt.xlim([0.0, 1.0])
    # # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()

    """LSM prediction"""
    grid_f = np.loadtxt('./src_data/grid_samples_HK.csv', dtype=str, delimiter=",",
                        encoding='UTF-8')
    samples_f = grid_f[1:, :-2].astype(np.float32)
    samples_f = samples_f / samples_f.max(axis=0)

    predict_results = model.predict_proba(samples_f)

    # save the prediction result
    data_df = pd.DataFrame(predict_results)
    writer = pd.ExcelWriter('./tmp/MLP_prediction_HK.xlsx')
    data_df.to_excel(writer, 'page_1', float_format='%.5f')
    writer.close()
    print('done RF LSM prediction!')


def RF_compare(x_train, y_train, x_test, y_test):
    """predict and test"""
    print('start RF evaluation...')
    clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2,
                                  bootstrap=True)

    clf.fit(x_train, y_train)
    pred_train = clf.predict(x_train)
    pred_test = clf.predict(x_test)
    # 训练精度
    print('Done.\ntrain_Accuracy: %f' % accuracy_score(y_train, pred_train))
    # 测试精度
    print('Done.\ntest_Accuracy: %f' % accuracy_score(y_test, pred_test))  # 0.71 - 0.77
    # pred1 = clf2.predict_proba() # 预测类别概率
    cal_measure(pred_test, y_test)

    """"LSM prediction"""
    grid_f = np.loadtxt('./src_data/grid_samples_HK.csv', dtype=str, delimiter=",",
                        encoding='UTF-8')
    samples_f = grid_f[1:, :-2].astype(np.float32)
    samples_f = samples_f / samples_f.max(axis=0)

    predict_results = clf.predict_proba(samples_f)
    # save the prediction result
    data_df = pd.DataFrame(predict_results)
    writer = pd.ExcelWriter('./tmp/RF_prediction_HK.xlsx')
    data_df.to_excel(writer, 'page_1', float_format='%.5f')
    writer.close()
    print('done MLP LSM prediction!')

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

# SVM_compare(x_train, y_train, x_test, y_test)

# ANN_compare(x_train, y_train, x_test, y_test)

RF_compare(x_train, y_train, x_test, y_test)
