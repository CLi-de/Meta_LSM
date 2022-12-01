import numpy as np

import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA
import umap

from mpl_toolkits.mplot3d import Axes3D  # 3D plot
import pandas as pd
import tensorflow as tf

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from scipy import interp
from sklearn.metrics import roc_auc_score

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics._classification import accuracy_score

"""for visiualization"""


def read_tasks(file, dim_input=16):
    """read csv and obtain tasks"""
    f = pd.ExcelFile(file)
    tasks = []
    for sheetname in f.sheet_names:
        attr = pd.read_excel(file, usecols=dim_input - 1, sheet_name=sheetname).values.astype(np.float32)
        label = pd.read_excel(file, usecols=[dim_input], sheet_name=sheetname).values.reshape((-1, 1)).astype(
            np.float32)
        tasks.append([attr, label])
    return tasks


def read_csv(path):
    tmp = np.loadtxt(path, dtype=np.str, delimiter=",", encoding='UTF-8')
    tmp_feature = tmp[1:, :]
    np.random.shuffle(tmp_feature)  # shuffle
    label_attr = tmp_feature[:, -1].astype(np.float32)  #
    data_atrr = tmp_feature[:, :-1].astype(np.float32)  #
    return data_atrr, label_attr


def load_weights(npzfile):
    npzfile = np.load(npzfile)
    weights = {}
    weights['w0'] = npzfile['arr_0']
    weights['b0'] = npzfile['arr_1']
    weights['w1'] = npzfile['arr_2']
    weights['b1'] = npzfile['arr_3']
    weights['w2'] = npzfile['arr_4']
    weights['b2'] = npzfile['arr_5']
    return weights


def transform_relu(inputX, weights, bias, activations=tf.nn.relu):
    return activations(tf.transpose(a=tf.matmul(weights, tf.transpose(a=inputX))) + bias)


def forward(inp, weights, sess):
    for i in range(int(len(weights) / 2)):  # 3 layers
        inp = transform_relu(inp, tf.transpose(a=weights['w' + str(i)]), weights['b' + str(i)])
    return sess.run(inp)


def _PCA(X, y, figsavename):
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    x_min, x_max = X_pca.min(0), X_pca.max(0)
    X_norm = (X_pca - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.scatter(x1,x2,x3,c=pre

    landslide_pts_x = []
    landslide_pts_y = []
    landslide_pts_z = []
    nonlandslide_pts_x = []
    nonlandslide_pts_y = []
    nonlandslide_pts_z = []

    for i in range(len(y)):
        if y[i] == 0:
            nonlandslide_pts_x.append(X_norm[i][0])
            nonlandslide_pts_y.append(X_norm[i][1])
            nonlandslide_pts_z.append(X_norm[i][2])
        if y[i] == 1:
            landslide_pts_x.append(X_norm[i][0])
            landslide_pts_y.append(X_norm[i][1])
            landslide_pts_z.append(X_norm[i][2])

    type_landslide = ax.scatter(landslide_pts_x, landslide_pts_y, landslide_pts_z, c='red')
    type_nonlandslide = ax.scatter(nonlandslide_pts_x, nonlandslide_pts_y, nonlandslide_pts_z, c='blue')

    ax.legend((type_landslide, type_nonlandslide), ('landslide points', 'nonlandslide points'), loc=2)
    # plt.legend( bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # 设置坐标标签
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')

    # 设置标题
    plt.title("Visualization with PCA")
    plt.savefig(figsavename)
    # 显示图形
    plt.show()


def ISOMAP(X, y, figsavename):
    isomap = manifold.Isomap(n_components=3)
    X_isomap = isomap.fit_transform(X)

    x_min, x_max = X_isomap.min(0), X_isomap.max(0)
    X_norm = (X_isomap - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.scatter(x1,x2,x3,c=pre

    landslide_pts_x = []
    landslide_pts_y = []
    landslide_pts_z = []
    nonlandslide_pts_x = []
    nonlandslide_pts_y = []
    nonlandslide_pts_z = []

    for i in range(len(y)):
        if y[i] == 0:
            nonlandslide_pts_x.append(X_norm[i][0])
            nonlandslide_pts_y.append(X_norm[i][1])
            nonlandslide_pts_z.append(X_norm[i][2])
        if y[i] == 1:
            landslide_pts_x.append(X_norm[i][0])
            landslide_pts_y.append(X_norm[i][1])
            landslide_pts_z.append(X_norm[i][2])

    type_landslide = ax.scatter(landslide_pts_x, landslide_pts_y, landslide_pts_z, c='red')
    type_nonlandslide = ax.scatter(nonlandslide_pts_x, nonlandslide_pts_y, nonlandslide_pts_z, c='blue')

    ax.legend((type_landslide, type_nonlandslide), ('landslide points', 'nonlandslide points'), loc=2)
    # plt.legend( bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # 设置坐标标签
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    # 设置标题
    plt.title("Visualization with Isomap")
    plt.savefig(figsavename)
    # 显示图形
    plt.show()


def t_SNE(X, y, figsavename):
    tsne = manifold.TSNE(n_components=3, init='random', random_state=501)
    X_tsne = tsne.fit_transform(X)

    """嵌入空间可视化"""
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.scatter(x1,x2,x3,c=pre

    landslide_pts_x = []
    landslide_pts_y = []
    landslide_pts_z = []
    nonlandslide_pts_x = []
    nonlandslide_pts_y = []
    nonlandslide_pts_z = []

    for i in range(len(y)):
        if y[i] == 0:
            nonlandslide_pts_x.append(X_norm[i][0])
            nonlandslide_pts_y.append(X_norm[i][1])
            nonlandslide_pts_z.append(X_norm[i][2])
        if y[i] == 1:
            landslide_pts_x.append(X_norm[i][0])
            landslide_pts_y.append(X_norm[i][1])
            landslide_pts_z.append(X_norm[i][2])

    type_landslide = ax.scatter(landslide_pts_x, landslide_pts_y, landslide_pts_z, c='red')
    type_nonlandslide = ax.scatter(nonlandslide_pts_x, nonlandslide_pts_y, nonlandslide_pts_z, c='blue')

    ax.legend((type_landslide, type_nonlandslide), ('landslide points', 'nonlandslide points'), loc=2)
    # plt.legend( bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # 设置坐标标签
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    # 设置标题
    plt.title("Visualization with t-SNE")
    plt.savefig(figsavename)
    # 显示图形
    plt.show()


def UMAP(X, y, figsavename):
    reducer = umap.UMAP(n_components=3)
    X_umap = reducer.fit_transform(X)

    x_min, x_max = X_umap.min(0), X_umap.max(0)
    X_norm = (X_umap - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.scatter(x1,x2,x3,c=pre

    landslide_pts_x = []
    landslide_pts_y = []
    landslide_pts_z = []
    nonlandslide_pts_x = []
    nonlandslide_pts_y = []
    nonlandslide_pts_z = []

    for i in range(len(y)):
        if y[i] == 0:
            nonlandslide_pts_x.append(X_norm[i][0])
            nonlandslide_pts_y.append(X_norm[i][1])
            nonlandslide_pts_z.append(X_norm[i][2])
        if y[i] == 1:
            landslide_pts_x.append(X_norm[i][0])
            landslide_pts_y.append(X_norm[i][1])
            landslide_pts_z.append(X_norm[i][2])

    type_landslide = ax.scatter(landslide_pts_x, landslide_pts_y, landslide_pts_z, c='red')
    type_nonlandslide = ax.scatter(nonlandslide_pts_x, nonlandslide_pts_y, nonlandslide_pts_z, c='blue')

    ax.legend((type_landslide, type_nonlandslide), ('landslide points', 'nonlandslide points'), loc=2)
    # plt.legend( bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # 设置坐标标签
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    # 设置标题
    plt.title("Visualization with UMAP")
    plt.savefig(figsavename)
    # 显示图形
    plt.show()


def visualization():
    fj_tasks = read_tasks('./seg_output/FJ_tasks.xlsx')  # task里的samplles
    # fl_tasks = read_tasks('./seg_output/FL_tasks.xlsx')  # num_samples of FL is too scarce to visualize

    """select part of FJ and FL data for visualization"""
    # def get_oriandinf_Xs(tasks, regionname):
    #     ori_Xs, inf_Xs, Ys = [], [], []
    #     for i in range(len(tasks)):
    #         if tasks[i][0].shape[0] > 30:
    #             """download model parameters"""
    #             w = load_weights('models_of_blocks/'+ regionname +'/model'+str(i) + '.npz')
    #             ori_Xs.append(tasks[i][0])
    #             inf_Xs.append(forward(tasks[i][0], w))
    #             Ys.append(tasks[i][1])
    #     return ori_Xs, inf_Xs, Ys
    """overall FJ and FL data for visualization"""

    def get_oriandinf_Xs(tasks, regionname):
        with tf.compat.v1.Session() as sess:  # for tf calculation
            w = load_weights('models_of_blocks/' + 'overall_' + regionname + '/model_MAML' + '.npz')
            ori_Xs = tasks[0][0]
            inf_Xs = forward(tasks[0][0], w, sess)
            Ys = tasks[0][1]
            for i in range(len(tasks) - 1):
                if len(tasks[i + 1][0]) > 0:
                    ori_Xs = np.vstack((ori_Xs, tasks[i + 1][0]))
                    inf_Xs = np.vstack((inf_Xs, forward(tasks[i + 1][0], w, sess)))
                    Ys = np.vstack((Ys, tasks[i + 1][1]))
            return ori_Xs, inf_Xs, Ys

    ori_FJ_Xs, inf_FJ_Xs, FJ_Ys = get_oriandinf_Xs(fj_tasks, 'FJ')

    # ori_FL_Xs, inf_FL_Xs, FL_Ys = get_oriandinf_Xs(fl_tasks, 'FL')

    # ori_X, y = read_csv('src_data/FJ_FL.csv')
    #
    # tmp = np.loadtxt('src_data/FJ_FL.csv', dtype=np.str, delimiter=",",encoding='UTF-8')
    # w = load_weights('DAS_logs/savedmodel.npz')
    # unsupervised_X = forward(ori_X, w)
    def plot_points(ori_X, inf_X, Y, regionname):
        _PCA(ori_X, Y, './figs/' + regionname + '_ori_PCA.pdf')
        _PCA(inf_X, Y, './figs/' + regionname + '_inf_PCA.pdf')

        t_SNE(ori_X, Y, './figs/' + regionname + '_ori_t_SNE.pdf')
        t_SNE(inf_X, Y, './figs/' + regionname + '_inf_SNE.pdf')

        ISOMAP(ori_X, Y, './figs/' + regionname + '_ori_Isomap.pdf')
        ISOMAP(inf_X, Y, './figs/' + regionname + '_inf_Isomap.pdf')

        UMAP(ori_X, Y, './figs/' + regionname + '_ori_UMAP.pdf')
        UMAP(inf_X, Y, './figs/' + regionname + '_inf_UMAP.pdf')

    plot_points(ori_FJ_Xs, inf_FJ_Xs, FJ_Ys, 'FJ')
    # plot_points(ori_FL_Xs, inf_FL_Xs, FL_Ys, 'FL')


"""for figure plotting"""


def read_statistic(file):
    """读取csv获取statistic"""
    f = pd.ExcelFile(file)
    K, meanOA, maxOA, minOA, std = [], [], [], [], []
    for sheetname in f.sheet_names:
        tmp_K, tmp_meanOA, tmp_maxOA, tmp_minOA, tmp_std = np.transpose(
            pd.read_excel(file, sheet_name=sheetname).values)
        K.append(tmp_K)
        meanOA.append(tmp_meanOA)
        maxOA.append(tmp_maxOA)
        minOA.append(tmp_minOA)
        std.append(tmp_std)
    return K, meanOA, maxOA, minOA, std


def read_statistic1(file):
    """读取csv获取statistic"""
    f = pd.ExcelFile(file)
    K, meanOA = [], []
    for sheetname in f.sheet_names:
        tmp_K, tmp_meanOA = np.transpose(pd.read_excel(file, sheet_name=sheetname).values)
        K.append(tmp_K)
        meanOA.append(tmp_meanOA)
    return K, meanOA


def read_statistic2(file):
    """读取csv获取statistic"""
    f = pd.ExcelFile(file)
    measures = []
    for sheetname in f.sheet_names:
        temp = pd.read_excel(file, sheet_name=sheetname).values
        measures.append(temp[:, 1:].tolist())
    return measures


def plot_candle(methodname, K, meanOA, maxOA, minOA, std):
    # 设置框图
    plt.figure("", facecolor="lightgray")
    # plt.style.use('ggplot')
    # 设置图例并且设置图例的字体及大小
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 18,
             }

    # legend = plt.legend(handles=[A,B],prop=font1)
    plt.title(methodname, fontdict=font2)
    plt.xlabel("Number of samples for adaption", fontdict=font1)
    plt.ylabel("OA(%)", fontdict=font1)

    my_x_ticks = [1, 2, 3, 4, 5, 6]
    my_x_ticklabels = ['1', '2', '3', '4', '5', 'M/2']
    plt.xticks(ticks=my_x_ticks, labels=my_x_ticklabels)

    plt.ylim((50, 90))
    my_y_ticks = np.arange(50, 90, 5)
    plt.yticks(my_y_ticks)

    colors = np.zeros(5, dtype="U5")
    colors[:] = 'white'
    edge_colors = np.zeros(5, dtype="U1")
    edge_colors[:] = 'b'

    '''格网设置'''
    plt.grid(linestyle="--", zorder=-1)

    # draw line
    # plt.plot(K[0:-1], meanOA[0:-1], color="b", linestyle='solid',
    #         linewidth=1, label="open", zorder=1)
    # plt.plot(K[-2:], meanOA[-2:], color="b", linestyle="--",
    #          linewidth=1, label="open", zorder=1)
    # draw bar
    barwidth = 0.4
    plt.bar(K, 2 * std, barwidth, bottom=meanOA - std, color=colors,
            edgecolor=edge_colors, linewidth=1, zorder=20)
    # draw vertical line
    plt.vlines(K, minOA, maxOA, color='black', linestyle='solid', zorder=10)
    plt.hlines(meanOA, K - barwidth / 2, K + barwidth / 2, color='r', linestyle='solid', zorder=30)
    plt.hlines(minOA, K - barwidth / 4, K + barwidth / 4, color='black', linestyle='solid', zorder=10)
    plt.hlines(maxOA, K - barwidth / 4, K + barwidth / 4, color='black', linestyle='solid', zorder=10)


def plot_brokenline(K, meanOA):
    '''设置框图'''
    plt.figure("", facecolor="lightgray")  # 设置框图大小
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 12,
             }
    plt.xlabel("Number of samples for adaption", fontdict=font1)
    plt.ylabel("Accuracy(%)", fontdict=font1)

    '''设置刻度'''
    plt.ylim((60, 80))
    my_y_ticks = np.arange(60, 80, 2)
    plt.yticks(my_y_ticks)

    my_x_ticks = [1, 2, 3, 4, 5, 6]
    my_x_ticklabels = ['1', '2', '3', '4', '5', 'M/2']
    plt.xticks(ticks=my_x_ticks, labels=my_x_ticklabels)

    '''格网设置'''
    plt.grid(linestyle="--")

    '''draw line'''
    line_MLP = plt.plot(K[0], meanOA[0], color="r", linestyle="solid",
                        linewidth=3, label="line_MLP", marker='^', markerfacecolor='white', ms=10)
    line_RL = plt.plot(K[1], meanOA[1], color="b", linestyle="solid",
                       linewidth=3, label="line_RL", marker='x', markerfacecolor='white', ms=10)
    line_MAML = plt.plot(K[2], meanOA[2], color="orange", linestyle="solid",
                         linewidth=3, label="line_MAML", marker='*', markerfacecolor='white', ms=12)
    line_proposed = plt.plot(K[3], meanOA[3], color="black", linestyle="solid",
                             linewidth=3, label="line_proposed", marker='s', markerfacecolor='white', ms=10)

    '''设置图例'''
    legend = plt.legend(loc="upper left", prop=font2, ncol=2)
    # plt.savefig("C:\\Users\\hj\\Desktop\\brokenline_A")
    # plt.show()


def plot_histogram(region, measures):
    '''设置框图'''
    plt.figure("", facecolor="lightgray")  # 设置框图大小
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 14,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 18,
             }
    # plt.xlabel("Statistical measures", fontdict=font1)
    plt.ylabel("Performance(%)", fontdict=font1)
    plt.title(region, fontdict=font2)

    '''设置刻度'''
    plt.ylim((60, 90))
    my_y_ticks = np.arange(60, 90, 3)
    plt.yticks(my_y_ticks)

    my_x_ticklabels = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    bar_width = 0.3
    interval = 0.2
    my_x_ticks = np.arange(bar_width / 2 + 2.5 * bar_width, 4 * 5 * bar_width + 1, bar_width * 6)
    plt.xticks(ticks=my_x_ticks, labels=my_x_ticklabels, fontproperties='Times New Roman', size=14)

    '''格网设置'''
    plt.grid(linestyle="--")

    '''draw bar'''
    rects1 = plt.bar([x - 2 * bar_width for x in my_x_ticks], height=measures[0], width=bar_width, alpha=0.8,
                     color='dodgerblue', label="MLP")
    rects2 = plt.bar([x - 1 * bar_width for x in my_x_ticks], height=measures[1], width=bar_width, alpha=0.8,
                     color='yellowgreen', label="RF")
    rects3 = plt.bar([x for x in my_x_ticks], height=measures[2], width=bar_width, alpha=0.8, color='gold', label="RL")
    rects4 = plt.bar([x + 1 * bar_width for x in my_x_ticks], height=measures[3], width=bar_width, alpha=0.8,
                     color='peru', label="MAML")
    rects5 = plt.bar([x + 2 * bar_width for x in my_x_ticks], height=measures[4], width=bar_width, alpha=0.8,
                     color='crimson', label="proposed")

    '''设置图例'''
    legend = plt.legend(loc="upper left", prop=font1, ncol=3)

    '''add text'''
    # for rect in rects1:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height)+'%', ha="center", va="bottom")
    # for rect in rects2:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height)+'%', ha="center", va="bottom")
    # for rect in rects3:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height)+'%', ha="center", va="bottom")
    # for rect in rects4:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height)+'%', ha="center", va="bottom")
    # for rect in rects5:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height)+'%', ha="center", va="bottom")

    plt.savefig("C:\\Users\\hj\\Desktop\\histogram" + region + '.pdf')
    plt.show()


"""for AUROC plotting"""


def load_data(filepath, dim_input):
    data = pd.read_excel(filepath).values.astype(np.float32)
    attr = data[:, :dim_input]
    attr = attr / attr.max(axis=0)
    label = data[:, -1].astype(np.int32)
    return attr, label


def MLP_fit_pred(X_train, X_test, y_train, y_test):
    classifier = MLPClassifier(hidden_layer_sizes=(32, 32, 16), activation='relu', solver='adam', alpha=0.0001,
                               batch_size=32, max_iter=1000)
    classifier.fit(X_train, y_train)
    y_score = classifier.predict_proba(X_test)
    pred_test = classifier.predict(X_test)
    print('Done.\n MLP Test Accuracy: %f' % accuracy_score(y_test, pred_test))  # 奉节，在0.72 - 0.76间；涪陵，在0.76-0.81之间
    return y_score


def RF_fit_pred(X_train, X_test, y_train, y_test):
    classifier = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2,
                                        bootstrap=True)
    classifier.fit(X_train, y_train[:, 1])
    y_score = classifier.predict_proba(X_test)
    pred_test = classifier.predict(X_test)
    print('Done.\n RF Test_Accuracy: %f' % accuracy_score(y_test[:, -1], pred_test))  # 0.71 - 0.77
    return y_score


def RL_fit_pred(X, y):
    pass  # realized in other place, we read predictions and labels from csv file instead.


def MAML_fit_pred(X, y, ):
    pass  # realized in other place, we read predictions and labels from csv file instead.


def proposed_fit_pred(X, y, ):
    pass  # realized in other place, we read predictions and labels from csv file instead.


def plot_auroc(n_classes, y_score, y_test, title):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # %%
    # Plot ROC curves for the multilabel problem
    # ..........................................
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average (area = {0:0.3f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=3)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average (area = {0:0.3f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=3)

    colors = [(0, 1, 0), (0, 0, 1)]
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.3f})'
                       ''.format(i, roc_auc[i]))

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 12,
             }
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontdict=font1)
    plt.ylabel('True Positive Rate', fontdict=font1)
    plt.title('ROC curve by ' + title, fontdict=font1)
    plt.legend(loc="lower right", prop=font2)
    # plt.show()


if __name__ == "__main__":
    """space visualization"""
    # visualization()

    """draw histogram"""
    # regions = ['FJ', 'FL']
    # measures = read_statistic2("C:\\Users\\hj\\Desktop\\performance.xlsx")
    # for i in range(len(regions)):
    #     plot_histogram(regions[i], measures[i])

    """draw candle"""
    methods = ['MLP', 'RL', 'MAML', 'proposed']
    for i in range(len(methods)):
        K, meanOA, maxOA, minOA, std = read_statistic("C:\\Users\\hj\\Desktop\\statistics.xlsx")
        plot_candle(methods[i], K[i], meanOA[i], maxOA[i], minOA[i], std[i])
        plt.savefig("C:\\Users\\hj\\Desktop\\" + methods[i] + '_' + 'candle.pdf')
        plt.show()

    """draw broken line"""
    # Experimentname = ['A', 'B', 'C', 'D']
    # for i in range(len(Experimentname)):
    #     filename = "C:\\Users\\hj\\Desktop\\" + 'statistics' + str(i+1) + '.xlsx'
    #     K, meanOA = read_statistic1(filename)
    #     plot_brokenline(K, meanOA)
    #     plt.savefig("C:\\Users\\hj\\Desktop\\"+Experimentname[i]+'_'+'broken.pdf')
    #     plt.show()

    """draw AUROC"""
    # mode = 'mode3'
    # X, y = load_data('src_data/FJ_FL.xlsx', 16)
    # X, y = X[:int(len(X)/8), :], y[:int(len(y)/8)]  # few samples for transfering-based methods
    # # Binarize the output
    # y = label_binarize(y, classes=[0, 1, 3])[:, :2]  # 0: nonlandside ; 1: landslide
    # n_classes = y.shape[1]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.8, shuffle=True)
    # # fit and predict
    # MLP_y_score = MLP_fit_pred(X_train, X_test, y_train, y_test)
    # RF_y_score = RF_fit_pred(X_train, X_test, y_train, y_test)
    # # RL_y_score, RL_y_test = pd.read_excel('').values.astype(np.float32)
    # # MAML_y_score, MAML_y_test = pd.read_excel('').values.astype(np.float32)
    # tmp = pd.read_excel(mode + 'predict.xlsx').values.astype(np.float32)
    # proposed_y_score, proposed_y_test = tmp[:, 1:3], tmp[:, 3:5]
    #
    # # draw auroc
    # roc_aucs = [dict() for i in range(5)]
    # plot_auroc(n_classes, MLP_y_score, y_test, 'MLP')
    # plt.savefig("C:\\Users\\hj\\Desktop\\" + mode +'_MLP.pdf')
    # plt.show()
    # plot_auroc(n_classes, RF_y_score, y_test, 'RF')
    # plt.savefig("C:\\Users\\hj\\Desktop\\" + mode +'_RF.pdf')
    # plt.show()
    # # plot_auroc(n_classes, RL_y_score, RL_y_test)
    # # plot_auroc(n_classes, MAML_y_score, MAML_y_test)
    # plot_auroc(n_classes, proposed_y_score, proposed_y_test, 'proposed')
    # plt.savefig("C:\\Users\\hj\\Desktop\\" + mode +'_proposed.pdf')
    # plt.show()
