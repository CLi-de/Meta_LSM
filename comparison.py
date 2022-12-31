# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from unsupervised_pretraining.dbn_.models import SupervisedDBNClassification

from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from utils_v2 import cal_measure
import shap
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

"""model-agnostic SHAP"""


def SHAP_(predict_proba, x_train, x_test, f_name):
    shap.initjs()
    # SHAP demo are using dataframe instead of nparray
    x_train = pd.DataFrame(x_train)  # 将numpy的array数组x_test转为dataframe格式。
    x_test = pd.DataFrame(x_test)
    x_train.columns = f_name  # 添加特征名称
    x_test.columns = f_name

    explainer = shap.KernelExplainer(predict_proba, shap.kmeans(x_train, 100))
    x_ = x_test
    shap_values = explainer.shap_values(x_, nsamples=100)  # shap_values(_prob, n_samples, features)
    # shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], x_test.iloc[0, :], show=True, matplotlib=True)  # single feature
    shap.summary_plot(shap_values, x_, plot_type="bar")
    shap.summary_plot(shap_values[1], x_, plot_type="violin")  # shap_values[k], k表类别，k=1（landslides）
    # shap.summary_plot(shap_values[1], x_test, plot_type="compact_dot")
    shap.dependence_plot('lithology', shap_values[1], x_, interaction_index=None)
    shap.dependence_plot('SPI', shap_values[1], x_, interaction_index='DV')
    # shap.plots.beeswarm(shap_values[0])  # the beeswarm plot requires Explanation object as the `shap_values` argument


def pred_LSM(trained_model, xy, samples, name):
    """LSM prediction"""
    pred = trained_model.predict_proba(samples)
    data = np.hstack((xy, pred))
    data_df = pd.DataFrame(data)
    writer = pd.ExcelWriter('./tmp/'+name+'_prediction_HK.xlsx')
    data_df.to_excel(writer, 'page_1', float_format='%.5f')
    writer.close()


def SVM_(x_train, y_train, x_test, y_test):
    """predict and test"""
    print('start SVM evaluation...')
    model = svm.SVC(C=1, kernel='rbf', gamma=1 / (2 * x_train.var()), decision_function_shape='ovr', probability=True)
    # clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    print('train accuracy:' + str(metrics.accuracy_score(y_train, pred_train)))
    pred_test = model.predict(x_test)
    print('test accuracy:' + str(metrics.accuracy_score(y_test, pred_test)))
    # Precision, Recall, F1-score
    cal_measure(pred_test, y_test)
    kappa_value = cohen_kappa_score(pred_test, y_test)
    print('Cohen_Kappa: %f' % kappa_value)

    # feature permutation
    print('SHAP...')
    # SHAP_(model.predict_proba, x_train, x_test, f_names)

    return model


# can be deprecated
def ANN_(x_train, y_train, x_test, y_test):
    """predict and test"""
    print('start ANN evaluation...')
    model = MLPClassifier(hidden_layer_sizes=(32, 32, 16), activation='relu', solver='adam', alpha=0.01,
                          batch_size=32, max_iter=1000)
    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    print('Train Accuracy: %f' % accuracy_score(y_train, pred_train))
    pred_test = model.predict(x_test)
    print('Test Accuracy: %f' % accuracy_score(y_test, pred_test))
    # Precision, Recall, F1-score
    cal_measure(pred_test, y_test)
    kappa_value = cohen_kappa_score(pred_test, y_test)
    print('Cohen_Kappa: %f' % kappa_value)

    # SHAP
    print('SHAP...')
    # SHAP_(model.predict_proba, x_train, x_test, f_names)

    return model


def DBN_(x_train, y_train, x_test, y_test):
    print('start DBN evaluation...')
    # Training
    model = SupervisedDBNClassification(hidden_layers_structure=[32, 32],
                                        learning_rate_rbm=0.001,
                                        learning_rate=0.5,
                                        n_epochs_rbm=10,
                                        n_iter_backprop=200,
                                        batch_size=64,
                                        activation_function='relu',
                                        dropout_p=0.1)
    model.fit(x_train, y_train)

    pred_train = np.array(model.predict(x_train))
    pred_test = np.array(model.predict(x_test))
    # 训练精度
    print('train_Accuracy: %f' % accuracy_score(y_train, pred_train))
    # 测试精度
    print('test_Accuracy: %f' % accuracy_score(y_test, pred_test))
    # pred1 = clf2.predict_proba() # 预测类别概率
    cal_measure(pred_test, y_test)
    kappa_value = cohen_kappa_score(pred_test, y_test)
    print('Cohen_Kappa: %f' % kappa_value)

    # SHAP
    print('SHAP...')
    # SHAP_(model.predict_proba, x_train, x_test, f_names)
    return model


def RF_(x_train, y_train, x_test, y_test):
    """predict and test"""
    print('start RF evaluation...')
    model = RandomForestClassifier(n_estimators=200, max_depth=None)

    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    # 训练精度
    print('train_Accuracy: %f' % accuracy_score(y_train, pred_train))
    # 测试精度
    print('test_Accuracy: %f' % accuracy_score(y_test, pred_test))
    # pred1 = clf2.predict_proba() # 预测类别概率
    cal_measure(pred_test, y_test)
    kappa_value = cohen_kappa_score(pred_test, y_test)
    print('Cohen_Kappa: %f' % kappa_value)

    # # SHAP
    # print('SHAP...')
    # # TODO: SHAP for RF
    # # SHAP_(model.predict_proba, x_train, x_test, f_names)
    # shap.initjs()
    # explainer = shap.Explainer(model)
    # shap_values = explainer(x_train)
    # shap.plots.bar(shap_values[:100, :, 0])  # shap_values(n_samples, features, _prob)

    return model


if __name__ == "__main__":
    """Input data"""
    tmp = np.loadtxt('./src_data/samples_HK_noTS.csv', dtype=str, delimiter=",", encoding='UTF-8')
    f_names = tmp[0, :-3].astype(np.str)
    tmp_ = np.hstack((tmp[1:, :-3], tmp[1:, -1].reshape(-1, 1))).astype(np.float32)
    np.random.shuffle(tmp_)  # shuffle
    # 训练集
    x_train = tmp_[:int(tmp_.shape[0] / 4 * 3), :-1]  # 加载i行数据部分
    y_train = tmp_[:int(tmp_.shape[0] / 4 * 3), -1]  # 加载类别标签部分
    x_train = x_train / x_train.max(axis=0)
    # 测试集
    x_test = tmp_[int(tmp_.shape[0] / 4 * 3):, :-1]  # 加载i行数据部分
    y_test = tmp_[int(tmp_.shape[0] / 4 * 3):, -1]  # 加载类别标签部分
    x_test = x_test / x_test.max(axis=0)
    #
    grid_f = np.loadtxt('./src_data/grid_samples_HK.csv', dtype=str, delimiter=",", encoding='UTF-8')
    samples_f = grid_f[1:, :-2].astype(np.float32)
    xy = grid_f[1:, -2:].astype(np.float32)
    samples_f = samples_f / samples_f.max(axis=0)

    """evaluate and save LSM result"""
    # SVM-based
    model_svm = SVM_(x_train, y_train, x_test, y_test)
    pred_LSM(model_svm, xy, samples_f, 'SVM')
    print('done SVM-based LSM prediction! \n')
    # MLP_based
    model_mlp = ANN_(x_train, y_train, x_test, y_test)
    pred_LSM(model_mlp, xy, samples_f, 'MLP')
    print('done MLP-based LSM prediction! \n')

    # DBN-based
    model_dbn = DBN_(x_train, y_train, x_test, y_test)
    pred_LSM(model_dbn, xy, samples_f, 'DBN')
    print('done DBN-based LSM prediction! \n')

    #RF-based
    model_rf = RF_(x_train, y_train, x_test, y_test)
    pred_LSM(model_rf, xy, samples_f, 'RF')
    print('done RF-based LSM prediction! \n')


