
import tensorflow as tf

from maml import MAML

from utils import sample_generator, read_pts, read_tasks

from scene_sampling import SLICProcessor

import pandas as pd
import numpy as np
from osgeo import gdal

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('dim_input', 16, 'dim of input data')
flags.DEFINE_integer('dim_output', 2, 'dim of output data')
flags.DEFINE_float('update_lr', 1e-3, 'learning rate in meta-learning task')
flags.DEFINE_float('meta_lr', 1e-4, 'the base learning rate of meta learning process')
flags.DEFINE_string('basemodel', 'DAS', 'MLP: no unsupervised pretraining; DAS: pretraining with DAS')
flags.DEFINE_integer('num_updates', 5, 'number of inner gradient updates during training.')
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_samples_each_task', 12, 'number of samples sampling from each task when training, inner_batch_size')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')
flags.DEFINE_integer('meta_batch_size', 16, 'number of tasks sampled per meta-update, not nums tasks')
flags.DEFINE_string('logdir', './checkpoint_dir', 'directory for summaries and checkpoints.')
flags.DEFINE_integer('num_samples', 2637, 'total number of number of samples in FJ and FL.')
flags.DEFINE_integer('test_update_batch_size', 5, 'number of examples used for gradient update during adapting (K=1,3,5 in experiment, K-shot).')

def readpts(filepath):
    tmp = np.loadtxt(filepath, dtype=np.str, delimiter=",", encoding='UTF-8')
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
    model = MAML(FLAGS.dim_input, FLAGS.dim_output, test_num_updates=5)
    input_tensors = None
    model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    exp_string = "mode3.mbs16.ubs_12.numstep5.updatelr0.1.meta_lr0.0001"
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
        # TODO: 1.考虑few-shot样本数量； 2. 考虑少于6个样本地区的LSM预测
        batch_x, batch_y = sample_generator(tasks_samples[i], FLAGS.dim_input, FLAGS.dim_output)   # only one task samples
        inputa = batch_x[:, :FLAGS.test_update_batch_size, :]  # setting K-shot K here
        labela = batch_y[:, :FLAGS.test_update_batch_size, :]
        # inputb = batch_x[:, FLAGS.test_update_batch_size:, :]
        # labelb = batch_y[:, FLAGS.test_update_batch_size:, :]
        with tf.compat.v1.variable_scope('model', reuse=True):  # Variable reuse in np.normalize()
            task_output = model.forward(inputa[0], model.weights, reuse=True)
            task_loss = model.loss_func(task_output, labela)
            grads = tf.gradients(ys=task_loss,xs=list(model.weights.values()))
            gradients = dict(zip(model.weights.keys(), grads))
            fast_weights = dict(zip(model.weights.keys(), [model.weights[key] -
                                                           model.update_lr*gradients[key] for key in model.weights.keys()]))
            for j in range(num_updates - 1):
                loss = model.loss_func(model.forward(inputa[0], fast_weights, reuse=True), labela)
                grads = tf.gradients(ys=loss, xs=list(fast_weights.values()))
                gradients = dict(zip(fast_weights.keys(), grads))
                fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - model.update_lr*gradients[key] for key in fast_weights.keys()]))
            """predict LSM"""
            if len(indexes[i]):
                features_arr = np.array([features[index] for index in indexes[i]])
                xy_arr = np.array([xy[index] for index in indexes[i]])
                pred = model.forward(features_arr, fast_weights, reuse=True)
                pred = sess.run(tf.nn.softmax(pred))
                tmp = np.hstack((xy_arr[:, 0]. reshape(xy_arr.shape[0], 1), xy_arr[:, 1].reshape(xy_arr.shape[0], 1), pred))
                savearr=np.vstack((savearr, tmp))
            """save model parameters to npz file"""
            adapted_weights = sess.run(fast_weights)
            np.savez('models_of_blocks/FL/model'+str(i), adapted_weights['w1'],adapted_weights['b1'],
                     adapted_weights['w2'],adapted_weights['b2'],
                     adapted_weights['w3'],adapted_weights['b3'],
                     adapted_weights['w4'],adapted_weights['b4'])

    writer = pd.ExcelWriter(savename)
    data_df = pd.DataFrame(savearr)
    data_df.to_excel(writer)
    writer.save()

    print('save LSM successfully')


if __name__ == "__main__":
    fj_tasks = read_tasks('./seg_output/FJ_tasks.xlsx')  # task里的samplles
    fl_tasks = read_tasks('./seg_output/FL_tasks.xlsx')
    print('done read tasks')
    FJ_taskpts = read_pts('./seg_output/FJpts_tasks.xlsx')  # task里的grid pts
    FL_taskpts = read_pts('./seg_output/FLpts_tasks.xlsx')
    print('done read pts')
    FJ_gridpts_feature, FJ_gridpts_xy = readpts('./src_data/grid_samples_fj.csv')  # study area的grid pts
    FL_gridpts_feature, FL_gridpts_xy = readpts('./src_data/grid_samples_fl.csv')
    print('done read grid pts features and xy')

    # clusters of grid points (points from excel)
    FJ_gridcluster = getclusters(FJ_gridpts_xy, FJ_taskpts, './seg_output/FJ_Elegent_Girl_M250.0_K256_loop0.tif')
    FL_gridcluster = getclusters(FL_gridpts_xy, FL_taskpts, './seg_output/FL_Elegent_Girl_M250.0_K96_loop0.tif')

    # please note that 'FJ_LSpred.xlsx' and 'FL_LSpred.xlsx' should be predicted separately.
    print('start predicting FJ LSM...')
    # predict_LSM(fj_tasks, FJ_gridpts_feature, FJ_gridpts_xy, FJ_gridcluster, 'FJ_LSpred.xlsx')
    # print('start predicting FL LSM...')
    predict_LSM(fl_tasks, FL_gridpts_feature, FL_gridpts_xy, FL_gridcluster, 'FL_LSpred.xlsx')

    # for each task， predict LS of each pt

