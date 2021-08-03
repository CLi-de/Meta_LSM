'''先导入常用库NumPy，还有Scikit-learn中的preprocessing模块，这是一个对数据进行预处理的常用模块。'''

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
'''自编码器中会使用一种参数初始化方法xavier initialization，它的特点是会根据某一层网络的输入，输出节点数量自动调整最合适的分布。
    如果深度学习模型的权重初始化得太小，那信号将在每层间传递时逐渐缩小而难以产生作用，但如果权重初始化得太大，
    那信号将在每层间传递时逐渐放大并导致发散和失效。而Xaiver初始化器做的事情就是让权重被初始化得不大不小，正好合适。
    即让权重满足0均值，同时方差为2／（n（in）+n(out)），分布可以用均匀分布或者高斯分布。
    下面fan_in是输入节点的数量，fan_out是输出节点的数量。'''
def xavier_init(fan_in,fan_out,constant=1):
    low = -constant * np.sqrt(6.0/(fan_in+fan_out))
    high = constant * np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)
'''下面是一个去噪自编码的class，包含一个构建函数_init_()：n_input（输入变量数），n_hidden（隐含层节点数），transfer_function（隐含层激活函数，默认为softplus）
    optimizer（优化器，默认为Adam），scale（高斯噪声系数，默认为0.1）。其中，class内的scale参数做成了一个占位符
    参数初始化则使用_initialize_weights函数。'''
class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.relu,
                 optimizer = tf.train.AdamOptimizer(),scale=0.1):
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.transfer=transfer_function
        self.scale=tf.placeholder(tf.float32)
        self.training_scale=scale
        network_weights=self._initialize_weights()
        self.weights=network_weights
        #      接下来开始定义网络结构，为x创建一个维度为n_input的占位符，然后建立一个能提取特征的隐含层，先将输入x加上噪声，然后用tf.matmul将加了噪声的输入与隐含层的权重相乘
        #      并使用tf.add加上隐含层的偏置，最后对结果进行激活函数处理。经过隐含层后，需要在输出层进行数据复原，重建操作。
        self.x=tf.placeholder(tf.float32,[None,self.n_input])
        self.hidden=self.transfer(tf.add(tf.matmul(self.x+scale * tf.random_normal((n_input,)),self.weights['w1']),self.weights['b1']))
        self.reconstruction=tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])
        #      接下来定义自编码器的损失函数，这里使用平方误差作为损失函数，再定义训练操作作为优化器对损失进行优化，最后创建Session并初始化自编码器的全部模型参数。
        self.cost=0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))
        self.optimizer=optimizer.minimize(self.cost)
        #init=tf.global_variables_initializer()
        local_variables = tf.global_variables(scope='DAE')
        global_variables = tf.global_variables()
        init=tf.variables_initializer(var_list=tf.global_variables(scope='DAE'))
        #init=tf.initialize_variables([self.weights['w1'], self.weights['b1'], self.weights['w2'], self.weights['b2']])
        self.sess=tf.Session()
        self.sess.run(init)
    #      下面是参数初始化函数
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype= tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],dtype=tf.float32))
        return all_weights
    #      定义计算损失cost及执行一步训练的函数partial_fit。函数里只需让Session执行两个计算图的节点，分别是损失cost和训练过程optimizer，输入的feed_dict包括输入数据x，
    #      以及噪声的系数scale。函数partial_fit做的就是用一个batch数据进行训练并返回当前的损失cost。
    def partial_fit(self,X):
        cost, op = self.sess.run((self.cost,self.optimizer),feed_dict={self.x:X,self.scale:self.training_scale})
        # weights_layer = dict()
        # weights_layer['w'] = self.sess.run(self.weights['w1'])
        # weights_layer['b'] = self.sess.run(self.weights['b1'])
        return cost
    #      下面为一个只求损失cost的函数，这个函数是在自编码器训练完毕后，在测试集上对模型性能进行评测时会用到的。
    def calc_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict={self.x:X,self.scale:self.training_scale})
    #      定义transform函数，返回自编码器隐含层的输出结果，它的目的是提供一个接口来获取抽象后的特征，自编码器的隐含层的最主要功能就是学习出数据中的高阶特征。
    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x:X,self.scale:self.training_scale})
    #      定义generate函数，将隐含层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据
    def generate(self,hidden=None):
        if hidden is None:
            hidden=np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})
    #      定义reconstruct函数，它整体运行一遍复原过程，包括提取高阶特征和通过高阶特征复原数据
    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,feed_dict={self.x:X,self.scale:self.training_scale})
    #      定义getWeights函数的作用是获取隐含层的权重w1
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    #       定义getBiases函数则是获取隐含层的偏置系数b1
    def getBiases(self):
        return self.sess.run(self.weights['b1'])
#       下面使用定义好的自编码器在MINIST数据集上进行一些简单的性能测试
#mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
#       先定义一个对训练、测试数据进行标准化处理的函数，标准化即让数据变成0均值且标准差为1的分布。方法就是先减去均值，再除以标准差。
def standard_scale(X_train,X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train=preprocessor.transform(X_train)
    X_test=preprocessor.transform(X_test)
    return X_train,X_test
#       再定义一个获取随机block数据的函数：取一个从0到len(data)-batch_size之间的随机整数，再以这个随机数作为block的起始位置，然后顺序取到一个batch size的数据。
#       需要注意的是，这属于不放回抽样，可以提高数据的利用效率
def get_random_block_from_data(data,batch_size):
    start_index = np.random.randint(0, int(data.shape[0]) - batch_size)
    return data[start_index:(start_index+batch_size)]
#       使用之前定义的standard_scale函数对训练集、测试机进行标准化变换
# X_train,X_test=standard_scale(mnist.train.images,mnist.test.images)
# #       下面定义几个常用参数，总训练样本数，最大训练的轮数(epoch)设为20，batch_size设为128，并设置每隔一轮(epoch)就显示一次损失cost
# n_samples=int(mnist.train.num_examples)
# training_epochs=20
# batch_size=128
# display_step=1
# #       创建一个自编码器的实例，定义模型输入节点数n_input为784，自编码器的隐含层点数n_hidden为200，隐含层的激活函数transfer_function为softplus，优化器optimizer为Adam
# #       且学习速率为0。001，同时将噪声的系数设为0.01
# autoencoder=AdditiveGaussianNoiseAutoencoder(n_input=784,n_hidden=200,transfer_function=tf.nn.softplus,
#                                              optimizer=tf.train.AdamOptimizer(learning_rate=0.001),scale=0.01)
# #       下面开始训练过程，在每一轮(epoch)循环开始时，将平均损失avg_cost设为0，并计算总共需要的batch数（通过样本总数除以batch大小），在每一轮迭代后，显示当前的迭代数和这一轮迭代的平均cost。
# for epoch in range(training_epochs):
#     avg_cost=0.
#     total_batch=int(n_samples/batch_size)
#     for i in range(total_batch):
#         batch_xs=get_random_block_from_data(X_train,batch_size)
#         cost=autoencoder.partial_fit(batch_xs)  # 此处计算loss并优化权参
#         avg_cost += cost/n_samples * batch_size
#     if epoch % display_step == 0:
#         print("Epoch:", '%04d' % (epoch + 1), "cost=","{:.9f}".format(avg_cost))
# #       最后对训练完的模型进行性能测试，使用的评价指标是平方误差。
# print("Total cost:" + str(autoencoder.calc_total_cost(X_test)))