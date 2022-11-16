import tensorflow as tf

from maml_v2 import MAML

from utils_v2 import batch_generator, read_pts, read_tasks

import pandas as pd
import numpy as np
from osgeo import gdal

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('dim_input', 13, 'dim of input data')
flags.DEFINE_integer('dim_output', 2, 'dim of output data')
flags.DEFINE_float('update_lr', 1e-2, 'learning rate in meta-learning task')
flags.DEFINE_float('meta_lr', 1e-3, 'the base learning rate of meta learning process')
flags.DEFINE_string('basemodel', 'DAS', 'MLP: no unsupervised pretraining; DAS: pretraining with DAS')
flags.DEFINE_integer('num_updates', 5, 'number of inner gradient updates during training.')
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_samples_each_task', 16,
                     'number of samples sampling from each task when training, inner_batch_size')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')
flags.DEFINE_integer('meta_batch_size', 16, 'number of tasks sampled per meta-update, not nums tasks')
flags.DEFINE_string('logdir', './checkpoint_dir', 'directory for summaries and checkpoints.')
# flags.DEFINE_integer('num_samples', 18469, 'total number of samples in HK.')
flags.DEFINE_integer('test_update_batch_size', 8,
                     'number of examples used for gradient update during adapting (K=1,3,5 in experiment, K-shot).')


def readpts(filepath):
    tmp = np.loadtxt(filepath, dtype=str, delimiter=",", encoding='UTF-8')
    features = tmp[1:, :-2].astype(np.float32)
    features = features / features.max(axis=0)  # Normalization
    xy = tmp[1:, -2:].astype(np.float32)
    return features, xy


def getclusters(gridpts_xy, taskpts, tifformat_path):
    dataset = gdal.Open(tifformat_path)
    if dataset == None:
        print("can not open *.tif file!")
    im_geotrans = dataset.GetGeoTransform()
    gridcluster = [[] for i in range(len(taskpts))]
    for i in range(np.shape(gridpts_xy)[0]):
        height = int((gridpts_xy[i][1] - im_geotrans[3]) / im_geotrans[5])
        width = int((gridpts_xy[i][0] - im_geotrans[0]) / im_geotrans[1])
        for j in range(len(taskpts)):
            if [height, width] in taskpts[j].tolist():
                gridcluster[j].append(i)
                break
    return gridcluster


def predict_LSM(tasks_samples, features, xy, indexes, savename, num_updates=5):
    """restore model from checkpoint"""
    tf.compat.v1.disable_eager_execution()
    model = MAML(FLAGS.dim_input, FLAGS.dim_output, test_num_updates=5)
    input_tensors_input = (FLAGS.meta_batch_size, int(FLAGS.num_samples_each_task / 2), FLAGS.dim_input)
    input_tensors_label = (FLAGS.meta_batch_size, int(FLAGS.num_samples_each_task / 2), FLAGS.dim_output)
    model.construct_model(input_tensors_input=input_tensors_input, input_tensors_label=input_tensors_label,
                          prefix='metatrain_')
    exp_string = ".mbs16.ubs_16.numstep5.updatelr0.01.meta_lr0.001"
    saver = tf.compat.v1.train.Saver(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES))
    sess = tf.compat.v1.InteractiveSession()
    init = tf.compat.v1.global_variables()  # optimizer里会有额外variable需要初始化
    sess.run(tf.compat.v1.variables_initializer(var_list=init))
    model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
    if model_file:
        print("Restoring model weights from " + model_file)
        saver.restore(sess, model_file)  # 以model_file初始化sess中图
    else:
        print("no intermediate model found!")

    savearr = np.arange(4, dtype=np.float32).reshape((1, 4))  # save predicting result

    for i in range(len(tasks_samples)):
        np.random.shuffle(tasks_samples[i])
        with tf.compat.v1.variable_scope('model', reuse=True):  # Variable reuse in np.normalize()
            if len(tasks_samples[i]) > FLAGS.num_samples_each_task:
                train_ = tasks_samples[i][:int(len(tasks_samples[i]) / 2)]
                batch_size = FLAGS.test_update_batch_size
            else:
                train_ = tasks_samples[i]
                batch_size = int(len(train_) / 2)
            fast_weights = model.weights
            for j in range(num_updates):
                inputa, labela = batch_generator(train_, FLAGS.dim_input, FLAGS.dim_output,
                                                 batch_size)
                loss = model.loss_func(model.forward(inputa, fast_weights, reuse=True), labela)
                grads = tf.gradients(ys=loss, xs=list(fast_weights.values()))
                gradients = dict(zip(fast_weights.keys(), grads))
                fast_weights = dict(zip(fast_weights.keys(),
                                        [fast_weights[key] - model.update_lr * gradients[key] for key in
                                         fast_weights.keys()]))

                """predict LSM"""
            if len(indexes[i]):
                features_arr = np.array([features[index] for index in indexes[i]])
                xy_arr = np.array([xy[index] for index in indexes[i]])
                pred = model.forward(features_arr, fast_weights, reuse=True)
                pred = sess.run(tf.nn.softmax(pred))
                tmp = np.hstack(
                    (xy_arr[:, 0].reshape(xy_arr.shape[0], 1), xy_arr[:, 1].reshape(xy_arr.shape[0], 1), pred))
                savearr = np.vstack((savearr, tmp))
            """save model parameters to npz file"""
            adapted_weights = sess.run(model.weights)
            np.savez('models_of_blocks/HK/model' + str(i), adapted_weights['w1'], adapted_weights['b1'],
                     adapted_weights['w2'], adapted_weights['b2'],
                     adapted_weights['w3'], adapted_weights['b3'],
                     adapted_weights['w4'], adapted_weights['b4'])

    writer = pd.ExcelWriter(savename)
    data_df = pd.DataFrame(savearr)
    data_df.to_excel(writer)
    writer.close()

    print('save LSM successfully')
    sess.close()


if __name__ == "__main__":
    print('grid points assignment...')
    HK_tasks = read_tasks('./seg_output/HK_tasks.xlsx')
    HK_taskpts = read_pts('./seg_output/HKpts_tasks.xlsx')
    HK_gridpts_feature, HK_gridpts_xy = readpts('./src_data/grid_samples_HK.csv')
    HK_gridcluster = getclusters(HK_gridpts_xy, HK_taskpts, './seg_output/HK_Elegent_Girl_M250.0_K512_loop0.tif')
    print('adapt and predict...')
    predict_LSM(HK_tasks, HK_gridpts_feature, HK_gridpts_xy, HK_gridcluster, 'HK_LSpred.xlsx')