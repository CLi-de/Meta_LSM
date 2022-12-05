#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author  : CHEN Li
# @Time    : 2022/12/2 14:55
# @File    : SHAP.py
# @annotation
import xgboost
import shap

# train an XGBoost model
X, y = shap.datasets.boston()
model = xgboost.XGBRegressor().fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)

# visualize the first prediction's explanation
# shap.plots.waterfall(shap_values[0])

# visualize the first prediction's explanation with a force plot
# shap.plots.force(shap_values[0])

shap.plots.force(shap_values)