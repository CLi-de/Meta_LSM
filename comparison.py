# -*- coding:utf-8 -*-
from sklearn import svm
from sklearn import metrics
import numpy as np
import pandas as pd

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


def SVM_compare(x_train, y_train, x_test, y_test):
    """predict and test"""
    clf = svm.SVC(C=1, kernel='rbf', gamma=1 / (2 * x_train.var()), decision_function_shape='ovr', probability=True)
    # clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    clf.fit(x_train, y_train)
    # train accuracy
    predict_results1 = clf.predict(x_train)
    print('train accuracy:' + str(metrics.accuracy_score(predict_results1, y_train)))  # 0.69-0.71
    # test accuracy
    predict_results = clf.predict(x_test)
    print('test accuracy:' + str(metrics.accuracy_score(y_test, predict_results)))  # 0.69-0.71

    TP = ((predict_results == 1) * (y_test == 1)).astype(int).sum()
    FP = ((predict_results == 1) * (y_test == 0)).astype(int).sum()
    FN = ((predict_results == 0) * (y_test == 1)).astype(int).sum()
    TN = ((predict_results == 0) * (y_test == 0)).astype(int).sum()
    # statistical measure
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F_measures = 2 * Precision * Recall / (Precision + Recall)
    print('Precision: %f' % Precision, '\nRecall: %f' % Recall, '\nF_measures: %f' % F_measures)

    # # train AUROC
    # fpr1, tpr1, threshold1 = metrics.roc_curve(y_train, predict_results1, pos_label=1)
    # print(metrics.auc(fpr1, tpr1))
    # # Test AUROC
    # fpr, tpr, threshold = metrics.roc_curve(y_test, predict_results, pos_label=1)
    # print(metrics.auc(fpr, tpr))

    """For LSM"""
    grid_f = np.loadtxt('./src_data/grid_samples_HK.csv', dtype=str, delimiter=",",
                        encoding='UTF-8')
    samples_f = grid_f[1:, :-2].astype(np.float32)
    samples_f = samples_f / samples_f.max(axis=0)

    predict_results2 = clf.predict_proba(samples_f)

    # save the prediction result
    data_df = pd.DataFrame(predict_results2)
    writer = pd.ExcelWriter('./tmp/SVM_prediction_HK.xlsx')
    data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
    writer.close()
    print('done SVM LSM prediction!')


SVM_compare(x_train, y_train, x_test, y_test)
