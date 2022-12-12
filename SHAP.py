#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author  : CHEN Li
# @Time    : 2022/12/2 14:55
# @File    : SHAP.py
# @annotation

import xgboost
import shap
import warnings
# import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

"""test tree ensemble"""
#
# X, y = shap.datasets.boston()
# model = xgboost.XGBRegressor().fit(X, y)
#
# explainer = shap.Explainer(model)
# shap_values = explainer(X)
#
# shap.plots.waterfall(shap_values[105])
# # shap.plots.force(shap_values[0], matplotlib=True, show=False).savefig('xgboost_force.png')
# shap.plots.force(shap_values[0], matplotlib=True)
# # shap.plots.force(shap_values)  #
#
# shap.plots.scatter(shap_values[:, "RM"], color=shap_values)
#
# shap.plots.beeswarm(shap_values)
#
# shap.plots.bar(shap_values)

"""test model agnostic example"""
# iris data
# X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)

# Input data
tmp = np.loadtxt('./src_data/samples_HK_noTS.csv', dtype=str, delimiter=",", encoding='UTF-8')
feature_names = tmp[0, :-3].astype(np.str)
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

model = svm.SVC(C=1, kernel='rbf', gamma=1 / (2 * x_train.var()), decision_function_shape='ovr', probability=True)
model.fit(x_train, y_train)

shap.initjs()
# SHAP demo are using dataframe instead of nparray
x_train = pd.DataFrame(x_train)  # 将numpy的array数组x_test转为dataframe格式。
x_test = pd.DataFrame(x_test)
x_train.columns = feature_names  # 添加特征名称
x_test.columns = feature_names

explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(x_train, 100))
shap_values = explainer.shap_values(x_test, nsamples=100)  # shap_values(_prob, n_samples, features)
shap.force_plot(explainer.expected_value[0], shap_values[0][0, :], x_test.iloc[0, :], show=True, matplotlib=True)
# shap.force_plot(explainer.expected_value[0], shap_values[0], x_test, show=False, matplotlib=True)
shap.summary_plot(shap_values, x_test, plot_type="bar")
shap.summary_plot(shap_values[0], x_test, plot_type="violin")
shap.summary_plot(shap_values[0], x_test, plot_type="compact_dot")
shap.dependence_plot('lithology', shap_values[0], x_test, interaction_index=None)
shap.dependence_plot('lithology', shap_values[0], x_test, interaction_index='DEM')
# shap.plots.beeswarm(shap_values[0])  # the beeswarm plot requires Explanation object as the `shap_values` argument