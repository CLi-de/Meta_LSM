"""
Usage Instructions:

"""
import numpy as np
import tensorflow as tf
import pandas as pd
from maml_v2 import MAML
from scene_sampling_v2 import SLICProcessor, TaskSampling
from tensorflow.python.platform import flags
from utils_v2 import tasksbatch_generator, batch_generator, meta_train_test, meta_train_test1, save_tasks, \
    read_tasks, savepts_fortask, cal_measure
from Unsupervised_Pretraining.DAS_pretraining_v2 import Unsupervise_pretrain
from sklearn.metrics._classification import accuracy_score
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = flags.FLAGS

"""for task sampling"""
flags.DEFINE_float('M', 250, 'determine how distance influence the segmentation')
flags.DEFINE_integer('K', 512, 'number of superpixels')
flags.DEFINE_integer('loop', 5, 'number of SLIC iterations')
flags.DEFINE_string('str_region', 'HK', 'the study area')
flags.DEFINE_string('landslide_pts', './src_data/samples_HK.xlsx', 'path to (non)landslide samples')

"""for meta-train"""
flags.DEFINE_string('basemodel', 'DAS', 'MLP: no unsupervised pretraining; DAS: pretraining with DAS')
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_string('log', './tmp/data', 'batch_norm, layer_norm, or None')
flags.DEFINE_string('logdir', './checkpoint_dir', 'directory for summaries and checkpoints.')

flags.DEFINE_integer('dim_input', 13, 'dim of input data')
flags.DEFINE_integer('dim_output', 2, 'dim of output data')
flags.DEFINE_integer('meta_batch_size', 16, 'number of tasks sampled per meta-update, not nums tasks')
flags.DEFINE_integer('num_samples_each_task', 16,
                     'number of samples sampling from each task when training, inner_batch_size')
flags.DEFINE_integer('test_update_batch_size', 8,
                     'number of examples used for gradient update during adapting (K=1,3,5 in experiment, K-shot); -1: M.')
flags.DEFINE_integer('metatrain_iterations', 5001, 'number of meta-training iterations.')
flags.DEFINE_integer('num_updates', 5, 'number of inner gradient updates during training.')
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
# flags.DEFINE_integer('num_samples', 18469, 'total number of samples in HK.')
flags.DEFINE_float('update_lr', 1e-2, 'learning rate of single task objective (inner)')
flags.DEFINE_float('meta_lr', 1e-3, 'the base learning rate of meta objective (outer)')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')


def train(model, saver, sess, exp_string, tasks, resume_itr):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    PRINT_INTERVAL = 100

    print('Done model initializing, starting training...')
    prelosses, postlosses = [], []
    if resume_itr != FLAGS.pretrain_iterations + FLAGS.metatrain_iterations - 1:
        if FLAGS.log:
            train_writer = tf.compat.v1.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
        for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
            batch_x, batch_y, cnt_sample = tasksbatch_generator(tasks, FLAGS.meta_batch_size
                                                                , FLAGS.num_samples_each_task,
                                                                FLAGS.dim_input,
                                                                FLAGS.dim_output)  # task_batch[i]: (x, y, features)
            # batch_y = _transform_labels_to_network_format(batch_y, FLAGS.num_classes)
            inputa = batch_x[:, :int(FLAGS.num_samples_each_task / 2), :]  # a used for training
            labela = batch_y[:, :int(FLAGS.num_samples_each_task / 2), :]
            inputb = batch_x[:, int(FLAGS.num_samples_each_task / 2):, :]  # b used for testing
            labelb = batch_y[:, int(FLAGS.num_samples_each_task / 2):, :]
            # # when deal with few-shot problem
            # inputa = batch_x[:, :int(len(batch_x[0]) / 2), :]  # a used for training
            # labela = batch_y[:, :int(len(batch_y[0]) / 2), :]
            # inputb = batch_x[:, int(len(batch_x[0]) / 2):, :]  # b used for testing
            # labelb = batch_y[:, int(len(batch_y[0]) / 2):, :]

            feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela,
                         model.labelb: labelb, model.cnt_sample: cnt_sample}

            if itr < FLAGS.pretrain_iterations:
                input_tensors = [model.pretrain_op]  # for comparison
            else:
                input_tensors = [model.metatrain_op]  # meta_train

            if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
                input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates - 1]])

            result = sess.run(input_tensors, feed_dict)

            if itr % SUMMARY_INTERVAL == 0:
                prelosses.append(result[-2])
                if FLAGS.log:
                    train_writer.add_summary(result[1], itr)  # add summ_op
                postlosses.append(result[-1])

            if (itr != 0) and itr % PRINT_INTERVAL == 0:
                if itr < FLAGS.pretrain_iterations:
                    print_str = 'Pretrain Iteration ' + str(itr)
                else:
                    print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
                print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
                print(print_str)
                prelosses, postlosses = [], []
            #  save model
            if (itr != 0) and itr % SAVE_INTERVAL == 0:
                saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

            # TODO: Once the meta loss arrive at certain threshold, break the iteration
        saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))


def test(model, saver, sess, exp_string, elig_tasks, num_updates=5):
    print('start evaluation...')
    print(exp_string)
    total_Ytest, total_Ypred, total_Ytest1, total_Ypred1, sum_accuracies, sum_accuracies1 = [], [], [], [], [], []

    for i in range(len(elig_tasks)):
        np.random.shuffle(elig_tasks[i])
        train_ = elig_tasks[i][:int(len(elig_tasks[i]) / 2)]
        test_ = elig_tasks[i][int(len(elig_tasks[i]) / 2):]
        """few-steps tuning （不用op跑是因为采用的batch_size（input shape）不一致，且不想更新model.weight）"""
        with tf.compat.v1.variable_scope('model', reuse=True):  # np.normalize()里Variable重用
            inputa, labela = batch_generator(train_, FLAGS.dim_input, FLAGS.dim_output,
                                             FLAGS.test_update_batch_size)
            task_output = model.forward(inputa, model.weights, reuse=True)
            task_loss = model.loss_func(task_output, labela)
            grads = tf.gradients(ys=task_loss, xs=list(model.weights.values()))
            gradients = dict(zip(model.weights.keys(), grads))
            fast_weights = dict(zip(model.weights.keys(), [model.weights[key] -
                                                           model.update_lr * gradients[key] for key in
                                                           model.weights.keys()]))
            for j in range(num_updates - 1):
                inputa, labela = batch_generator(train_, FLAGS.dim_input, FLAGS.dim_output,
                                                 FLAGS.test_update_batch_size)
                loss = model.loss_func(model.forward(inputa, fast_weights, reuse=True),
                                       labela)  # fast_weight和grads（stopped）有关系，但不影响这里的梯度计算
                grads = tf.gradients(ys=loss, xs=list(fast_weights.values()))
                gradients = dict(zip(fast_weights.keys(), grads))
                fast_weights = dict(zip(fast_weights.keys(),
                                        [fast_weights[key] - model.update_lr * gradients[key] for key in
                                         fast_weights.keys()]))
            """Single task test accuracy"""
            inputb, labelb = batch_generator(test_, FLAGS.dim_input, FLAGS.dim_output, len(test_))
            Y_array = sess.run(tf.nn.softmax(model.forward(inputb, fast_weights, reuse=True)))
            total_Ypred1.extend(Y_array)  # prediction
            total_Ytest1.extend(labelb)  # label

            def task_test_acc():
                Y_test = []
                for j in range(len(labelb)):
                    Y_test.append(labelb[j][0])
                    total_Ytest.append(labelb[j][0])
                Y_pred = []
                for j in range(len(labelb)):
                    if Y_array[j][0] > Y_array[j][1]:
                        Y_pred.append(1)
                        total_Ypred.append(1)
                    else:
                        Y_pred.append(0)
                        total_Ypred.append(0)
                accuracy = accuracy_score(Y_test, Y_pred)
                sum_accuracies.append(accuracy)
                # print('Test_Accuracy: %f' % accuracy)

            task_test_acc()  # test accuracy of each task
    """Overall evaluation (test data)"""
    total_Ypred = np.array(total_Ypred).reshape(len(total_Ypred), )
    total_Ytest = np.array(total_Ytest)
    total_acc = accuracy_score(total_Ytest, total_Ypred)
    print('Total_Accuracy: %f' % total_acc)
    cal_measure(total_Ypred, total_Ytest)

    sess.close()


def main():
    """unsupervised pretraining"""
    if not os.path.exists('./DAS_logs/savedmodel.npz'):
        print("start unsupervised pretraining")
        tmp = np.loadtxt('src_data/samples_HK.csv', dtype=np.str, delimiter=",", encoding='UTF-8')
        tmp_feature = tmp[1:, :]
        np.random.shuffle(tmp_feature)  # shuffle
        Unsupervise_pretrain(tmp_feature)
    print('Done unsupervised pretraining')

    """meta task sampling"""

    def tasks_load(taskspath, str_region):
        if os.path.exists(taskspath):
            tasks = read_tasks(taskspath)
            print('     Done reading ' + str_region + ' tasks from previous SLIC result')
        else:
            print('     start meta-task sampling using SLIC algorithm:')

            p = SLICProcessor('./src_data/' + str_region + '/composite.tif', FLAGS.K, FLAGS.M)
            p.iterate_times(loop=FLAGS.loop)
            t = TaskSampling(p.clusters)
            tasks = t.sampling(p.im_geotrans)
            save_tasks(tasks)
            savepts_fortask(p.clusters, './seg_output/' + str_region + 'pts_tasks.xlsx')
        return tasks

    HK_tasks = tasks_load('./seg_output/HK_tasks.xlsx', FLAGS.str_region)

    tasks_train, tasks_test = meta_train_test1(HK_tasks)  # for HK
    print('Done meta-tasks sampling')

    """meta_training"""
    print('model construction...')
    model = MAML(FLAGS.dim_input, FLAGS.dim_output, test_num_updates=5)
    input_tensors_input = (FLAGS.meta_batch_size, int(FLAGS.num_samples_each_task / 2), FLAGS.dim_input)
    input_tensors_label = (FLAGS.meta_batch_size, int(FLAGS.num_samples_each_task / 2), FLAGS.dim_output)
    model.construct_model(input_tensors_input=input_tensors_input, input_tensors_label=input_tensors_label,
                          prefix='metatrain_')
    model.summ_op = tf.compat.v1.summary.merge_all()

    saver = tf.compat.v1.train.Saver(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES),
                                     max_to_keep=10)

    sess = tf.compat.v1.InteractiveSession()
    init = tf.compat.v1.global_variables()  # optimizer里会有额外variable需要初始化
    sess.run(tf.compat.v1.variables_initializer(var_list=init))

    exp_string1 = '.mbs' + str(FLAGS.meta_batch_size) + '.ubs_' + \
                  str(FLAGS.num_samples_each_task) + '.numstep' + str(FLAGS.num_updates) + \
                  '.updatelr' + str(FLAGS.update_lr) + '.meta_lr' + str(FLAGS.meta_lr)

    resume_itr = 0

    # 续点训练
    # if FLAGS.resume or not FLAGS.train:
    if FLAGS.resume:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string1)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1 + 5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)  # 以model_file初始化sess中图

    train(model, saver, sess, exp_string1, tasks_train, resume_itr)

    test(model, saver, sess, exp_string1, tasks_test, num_updates=FLAGS.num_updates)


# TODO: use tf.estimator
if __name__ == "__main__":
    # device=tf.config.list_physical_devices('GPU')
    tf.compat.v1.disable_eager_execution()
    main()
    print('finished!')