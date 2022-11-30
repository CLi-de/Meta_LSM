# This file will be deprecated in the future

# -*- coding:utf-8 -*-

import tensorflow as tf
from modeling import MAML
import numpy as np

from utils_v2 import read_pts, sample_generator_
from meta_learner_v2 import FLAGS


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    exp_string = ".mbs16.ubs_16.numstep5.updatelr0.01.meta_lr0.001"  # if FJ, mode 2; if FL, mode 3; if HK, ...;
    model = MAML(FLAGS.dim_input, FLAGS.dim_output, test_num_updates=5)
    input_tensors_input = (FLAGS.meta_batch_size, int(FLAGS.num_samples_each_task / 2), FLAGS.dim_input)
    input_tensors_label = (FLAGS.meta_batch_size, int(FLAGS.num_samples_each_task / 2), FLAGS.dim_output)
    model.construct_model(input_tensors_input=input_tensors_input, input_tensors_label=input_tensors_label,
                          prefix='metatrain_')

    var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    # saver = tf.train.import_meta_graph('./checkpoint_dir/' + exp_string + '/model4999.meta')
    saver = tf.compat.v1.train.Saver(var_list)
    sess = tf.compat.v1.InteractiveSession()
    init = tf.compat.v1.global_variables()  # extra variables require initialization in the optimizer
    sess.run(tf.compat.v1.variables_initializer(var_list=init))

    model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
    if model_file:
        print("Restoring model weights from " + model_file)
        saver.restore(sess, model_file)  # use model_file initialize sess graph
    else:
        print("no intermediate model found!")

    """获取tasks"""

    """adapting within a region"""


    def overall_adapting(taskfile, num_updates=5):
        tasks = read_pts(taskfile)
        inputa, labela = sample_generator_(tasks, FLAGS.dim_input, FLAGS.dim_output)
        with tf.compat.v1.variable_scope('model', reuse=True):  # Variable reuse in np.normalize()
            task_output = model.forward(inputa[0], model.weights, reuse=True)
            task_loss = model.loss_func(task_output, labela)
            grads = tf.gradients(ys=task_loss, xs=list(model.weights.values()))
            gradients = dict(zip(model.weights.keys(), grads))
            fast_weights = dict(zip(model.weights.keys(), [model.weights[key] -
                                                           model.update_lr * gradients[key] for key in
                                                           model.weights.keys()]))
            for j in range(num_updates - 1):
                # fast_weight is related to grads (stopped), but doesn't affect the gradient computation
                loss = model.loss_func(model.forward(inputa[0], fast_weights, reuse=True), labela)
                grads = tf.gradients(ys=loss, xs=list(fast_weights.values()))
                gradients = dict(zip(fast_weights.keys(), grads))
                fast_weights = dict(zip(fast_weights.keys(),
                                        [fast_weights[key] - model.update_lr * gradients[key] for key in
                                         fast_weights.keys()]))
            adapted_weights = sess.run(fast_weights)
            np.savez('models_of_blocks/overall_HK/model_MAML', adapted_weights['w1'], adapted_weights['b1'],
                     adapted_weights['w2'], adapted_weights['b2'],
                     adapted_weights['w3'], adapted_weights['b3'],
                     adapted_weights['w4'], adapted_weights['b4'])
            print('overall model saved')


    HK_taskfile = './seg_output/HK_tasks_K512.xlsx'

    overall_adapting(HK_taskfile)
    sess.close()
