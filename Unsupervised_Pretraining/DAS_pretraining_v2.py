# Author: C.L
# Network: RBMs + DAE + classifier

from sklearn.metrics._classification import accuracy_score
import pandas as pd

from .denoising_AE_v2 import *
from .utils_v2 import *
from .dbn.tensorflow import SupervisedDBNClassification

# define proposed algorithm
"""仅做无监督"""


def DAS(tmp_feature):
    label_attr = tmp_feature[:, -1].astype(np.float32)  # 加载类别标签部分
    data_atrr = tmp_feature[:, :-1].astype(np.float32)  # 加载i行数据部分
    data_atrr = data_atrr / data_atrr.max(axis=0)

    # Pretrain(Graph 0)
    weights = []
    reconstruction_error = []
    with tf.compat.v1.variable_scope('deepBM'):
        classifier = SupervisedDBNClassification(hidden_layers_structure=[32, 32],  # rbm隐藏层列表
                                                 learning_rate_rbm=0.001,
                                                 learning_rate=0.01,
                                                 n_epochs_rbm=20,
                                                 n_iter_backprop=100,
                                                 batch_size=32,
                                                 activation_function='relu',  # 多层rbm时relu比sigmoid合适
                                                 dropout_p=0.1)
        # RBM fit
        classifier.fit(data_atrr, weights, reconstruction_error)
    """infer"""
    activations = data_atrr
    for i in range(len(weights)):
        activations = transform_relu(activations, weights[i]['w'], weights[i]['b'])
    with tf.compat.v1.Session() as sess:
        X_train_dae = sess.run(activations)

    # 超参数设置
    weights1 = {'w': [], 'b': []}
    input_units = int(X_train_dae.shape[1])  # dae输入节点
    structure = [16]
    n_samples = int(X_train_dae.shape[0])
    training_epochs = 20
    batch_size = 16
    display_step = 1
    dae_weights = []  # 存储dae预训练权重参数
    dae_bias = []  # 存储dae预训练偏置参数
    activations = X_train_dae
    # build and train DAE
    for hidden_units in structure:
        with tf.compat.v1.variable_scope('DAE'):
            autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=input_units, n_hidden=hidden_units,
                                                           transfer_function=tf.nn.softplus,
                                                           optimizer=tf.compat.v1.train.AdamOptimizer(
                                                               learning_rate=0.00005), scale=0.01)
        print("[START] DAE training step:")
        current_weights2 = tf.compat.v1.global_variables()  # just see if right exist the weights
        for epoch in range(training_epochs):
            cost = 0.
            total_batch = int(n_samples / batch_size)
            for i in range(total_batch):
                batch_xs = get_random_block_from_data(activations, batch_size)  # 不放回抽样
                cost = autoencoder.partial_fit(batch_xs)  # 此处计算loss并优化权参
            print(">> Epoch %d finished \tDAE training loss %f" % (epoch, cost))

        weights1['w'].append(autoencoder.sess.run(tf.transpose(a=autoencoder.weights['w1'])))
        weights1['b'].append(autoencoder.sess.run(autoencoder.weights['b1']))

        input_units = hidden_units
    # 保存权值信息
    np.savez('./DAS_logs/savedmodel',
             weights[0]['w'], weights[0]['b'],
             weights[1]['w'], weights[1]['b'],
             weights1['w'][0], weights1['b'][0])
