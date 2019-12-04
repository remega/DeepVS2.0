import numpy as np
import tensorflow as tf
import time
import yolo_lstmconv20 as Network  # define the CNN
import random
from scipy.optimize import linprog
import os
import cv2
import glob
import imageio
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# global w_img,h_img
# w_img = 640 #
# h_img = 480 #
batch_size = 1
framesnum = 16
inputDim = 224
input_size = (inputDim, inputDim)
outputDim = 56
output_size2 = (64, 64)
output_size = (outputDim, outputDim)
epoch_num = 20
overlapframe = 10 #0~framesnum+frame_skip

dp_in = 1
dp_h = 1
MC_count = 100
CB_FB = 1
numdis = 50
random.seed(a=730)
tf.set_random_seed(730)
frame_skip = 5
dis_type = 'dualKL' # Wassers,dualWassers, KL,dualKL
dislambda = 0.25
flowversion = '2c'
yoloversion = 1



Validfile1 = '../LEDOVTFrecords/test/'
Validfile2 = '../DIEMTFrecords/'
Validfile3 = '../SFUTFrecords/'
Valid_list = [Validfile1] + [Validfile2] + [Validfile3]
# VideoNameFile = 'Traininglist.txt' #'Validationlist.txt'# 'Traininglist.txt'     #choose the data
# Video_dir = 'G:\database\statistics\database'
# CheckpointFile_yolo = './model/pretrain/CNN_YoloFlow_nofinetuned_batch12_premask_lb05_loss05_fea128_1x512_128-185000'
# CheckpointFile_flow = './model/pretrain/flownet-CS.ckpt-0'
SaveFile = './model/'
res_dir = './restest'
# Summary_dir = '/tmp/aremega/deepvs/summary'
# res_dir = '/tmp/aremega/deepvs/res'
# SaveFile = '/tmp/aremega/deepvs/model/'
if not os.path.isdir(SaveFile):
    os.mkdir(SaveFile)
if not os.path.isdir(res_dir):
    os.mkdir(res_dir)

targetname = 'Newlstmconv224_nopre_loss05_dp075_flow2c-19'
CheckpointFile = SaveFile + targetname

ValidData_index = [0, 1, 2]
net = Network.Net()
net.is_training = True
net.dp_in = 1
net.dp_h = 1
net.lambdadis = dislambda
net.disnum = numdis
net.distype = dis_type
net.version_flow = flowversion
net.version_yolo = yoloversion

input = tf.placeholder(tf.float32, (batch_size, framesnum + frame_skip, input_size[0], input_size[1], 3))
GroundTruth = tf.placeholder(tf.float32, (batch_size, framesnum + frame_skip, output_size[0], output_size[1], 1))
RNNmask_in = tf.placeholder(tf.float32, (batch_size, 28, 28, 128, 4 * 2))
RNNmask_h = tf.placeholder(tf.float32, (batch_size, 28, 28, 128, 4 * 2))
#exloss = tf.placeholder(tf.float32)

net.inferenceNew(input, GroundTruth, RNNmask_in, RNNmask_h)
# net._loss(exloss)
# loss_op = net.loss
# extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(extra_update_ops):
#     train_op = net._train()
predicts = net.out

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# saver = tf.train.Saver(net.yolofeatures_colllection)
# saver1 = tf.train.Saver(net.flowfeatures_colllection)

# init = tf.global_variables_initializer()
# sess.run(init)
saver = tf.train.Saver()
saver.restore(sess, CheckpointFile)

subres = os.path.join(res_dir, targetname)
if not os.path.isdir(subres):
    os.mkdir(subres)

def _parse_function(example_proto):
    keys_to_features = {'GTmap': tf.FixedLenFeature([], tf.string),
                        'shape': tf.FixedLenFeature([], tf.string),
                        'image': tf.FixedLenFeature([], tf.string),
                        }
    parsed_features = tf.parse_single_example(example_proto, keys_to_features, name='features')
    image = tf.decode_raw(parsed_features['image'], tf.uint8)
    shape = tf.decode_raw(parsed_features['shape'], tf.int32)
    # the image tensor is flattened out, so we have to reconstruct the shape
    image = tf.reshape(image, [shape[0], shape[1], 3])
    GTmap = tf.decode_raw(parsed_features['GTmap'], tf.uint8)
    GTmap = tf.reshape(GTmap, [shape[0], shape[1], 1])
    GTmap = tf.image.resize_images(GTmap, [output_size[0], output_size[1]])
    image = tf.image.resize_images(image, [input_size[0], input_size[1]])
    image = tf.cast(image, tf.float32)
    GTmap = tf.cast(GTmap, tf.float32)
    image = image / 255.0 * 2 - 1
    GTmap = GTmap / 255.0
    # GTmap = tf.expand_dims(GTmap ,axis = -1)
    return image, shape, GTmap


filenames = tf.placeholder(tf.string)
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)
dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(1))
iterator = dataset.make_initializable_iterator()
image, shape, GTmap = iterator.get_next()

def main():

    iter = 0
    start_time = time.time()
    start_time1 = time.time()
    v_count = 0
    for j in ValidData_index:
        tfdir = Valid_list[j]
        datasetname = tfdir.split('/')[1]
        datasetname = datasetname[:-9]
        sumKL = 0
        sumCC = 0
        count = 0
        sub3res = os.path.join(subres, datasetname)
        if not os.path.isdir(sub3res):
            os.mkdir(sub3res)
        for validfile in glob.glob(tfdir + '*.tfrecords'):

            sess.run(iterator.initializer, feed_dict={filenames: validfile})
            vdir = os.path.split(validfile)[-1]
            vname = vdir.split('.')[0]
            mask_in = np.ones((batch_size, 28, 28, 128, 4 * 2))
            mask_h = np.ones((batch_size, 28, 28, 128, 4 * 2))
            GTall, _, Frameall = sess.run([GTmap, shape, image])
            numframe = 1
            while True:
                try:
                    GTmap1, _, image1 = sess.run([GTmap, shape, image])
                    GTall = np.concatenate((GTall, GTmap1), axis=0)
                    Frameall = np.concatenate((Frameall, image1), axis=0)
                    numframe = numframe + 1
                except tf.errors.OutOfRangeError:
                    break
            #SalOut = np.zeros_like(GTall, np.uint8)
            writer = imageio.get_writer(sub3res + '/' + vname + '.avi', fps=30)
            frameindex = 0
            imageInput = Frameall[frameindex:(frameindex + framesnum + frame_skip), ...]
            GTmapInput = GTall[frameindex:(frameindex + framesnum + frame_skip), ...]
            imageInput = imageInput[np.newaxis, ...]
            GTmapInput = GTmapInput[np.newaxis, ...]
            Input_Batch = imageInput
            GTmap_Batch = GTmapInput
            np_predict = sess.run(predicts,
                                  feed_dict={input: Input_Batch, GroundTruth: GTmap_Batch,
                                             RNNmask_in: mask_in, RNNmask_h: mask_h})

            temp = cv2.resize(np_predict[0, 0, ...], output_size2)
            for i in range(frame_skip):
                writer.append_data(np.uint8(temp * 255))
            for i in range(framesnum):
                temp = cv2.resize(np_predict[0, i, ...], output_size2)
                writer.append_data(np.uint8(temp * 255))

            frameindex = frameindex + 1
            while frameindex + framesnum + frame_skip <= numframe:
                imageInput = Frameall[frameindex:(frameindex + framesnum + frame_skip), ...]
                GTmapInput = GTall[frameindex:(frameindex + framesnum + frame_skip), ...]
                Input_Batch = imageInput[np.newaxis, ...]
                GTmap_Batch = GTmapInput[np.newaxis, ...]
                np_predict = sess.run(predicts,
                                      feed_dict={input: Input_Batch, GroundTruth: GTmap_Batch,
                                                 RNNmask_in: mask_in, RNNmask_h: mask_h})
                temp = cv2.resize(np_predict[0, -1, ...], output_size2)
                writer.append_data(np.uint8(temp * 255))
                frameindex = frameindex + 1

            writer.close()
            usedtime = (time.time() - start_time) / 3600
            v_count = v_count + 1
            print('%d th video: %s; Have used time: %f hrs.' % (v_count, vname, usedtime))

if __name__ == '__main__':
    main()

