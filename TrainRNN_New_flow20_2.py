import numpy as np
import tensorflow as tf
import time
import yolo_lstmconv20 as Network  # define the CNN
import random
from scipy.optimize import linprog
import os
import glob
import imageio
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# global w_img,h_img
# w_img = 640 #
# h_img = 480 #
batch_size = 4
framesnum = 16
inputDim = 224
input_size = (inputDim, inputDim)
outputDim = 56
output_size = (outputDim, outputDim)
epoch_num = 20
overlapframe = 10 #0~framesnum+frame_skip

dp_in = 0.75
dp_h = 0.75
MC_count = 100
CB_FB = 1
numdis = 50
random.seed(a=730)
tf.set_random_seed(730)
frame_skip = 5
dis_type = 'dualKL' # Wassers,dualWassers, KL,dualKL
dislambda = 0.25
flowversion = '2s'
modelname = 'Newlstmconv224_nopre_loss05_dp075_flow2s'

TrainingFile1 = '../LEDOVTFrecords/training/'
TrainingFile2 = '../LEDOVTFrecords/validation/'
TrainingFile3 = '../DHF1KTFrecords/'
Training_list = [TrainingFile1] + [TrainingFile2] + [TrainingFile3]
Validfile1 = '../LEDOVTFrecords/test/'
Validfile2 = '../DIEMTFrecords/'
Validfile3 = '../SFUTFrecords/'
Valid_list = [Validfile1] + [Validfile2] + [Validfile3]
# VideoNameFile = 'Traininglist.txt' #'Validationlist.txt'# 'Traininglist.txt'     #choose the data
# Video_dir = 'G:\database\statistics\database'
# CheckpointFile_yolo = './model/pretrain/CNN_YoloFlow_nofinetuned_batch12_premask_lb05_loss05_fea128_1x512_128-185000'
# CheckpointFile_flow = './model/pretrain/flownet-CS.ckpt-0'
SaveFile = './model/'
Summary_dir = './summary'
res_dir = './res'
# Summary_dir = '/tmp/aremega/deepvs/summary'
# res_dir = '/tmp/aremega/deepvs/res'
# SaveFile = '/tmp/aremega/deepvs/model/'
if not os.path.isdir(Summary_dir):
    os.mkdir(Summary_dir)
if not os.path.isdir(SaveFile):
    os.mkdir(SaveFile)
if not os.path.isdir(res_dir):
    os.mkdir(res_dir)


TrainingData_index = [0, 1]
ValidData_index = [0, 1, 2]


TrainingList = []
for i in range(len(TrainingData_index)):
    TrainingList = TrainingList + glob.glob(Training_list[i] + '*.tfrecords')

net = Network.Net()
net.is_training = True
net.dp_in = dp_in
net.dp_h = dp_h
net.lambdadis = dislambda
net.disnum = numdis
net.distype = dis_type
net.version_flow = flowversion

input = tf.placeholder(tf.float32, (batch_size, framesnum + frame_skip, input_size[0], input_size[1], 3))
GroundTruth = tf.placeholder(tf.float32, (batch_size, framesnum + frame_skip, output_size[0], output_size[1], 1))
RNNmask_in = tf.placeholder(tf.float32, (batch_size, 28, 28, 128, 4 * 2))
RNNmask_h = tf.placeholder(tf.float32, (batch_size, 28, 28, 128, 4 * 2))
exloss = tf.placeholder(tf.float32)

net.inferenceNew(input, GroundTruth, RNNmask_in, RNNmask_h)
net._loss(exloss)
loss_op = net.loss
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_op = net._train()
predicts = net.out

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# saver = tf.train.Saver(net.yolofeatures_colllection)
# saver1 = tf.train.Saver(net.flowfeatures_colllection)

init = tf.global_variables_initializer()
sess.run(init)
# saver.restore(sess, CheckpointFile_yolo)
# saver1.restore(sess, CheckpointFile_flow)

saver2 = tf.train.Saver(max_to_keep=15)
summary_op = tf.summary.merge_all()
subname = os.path.join(Summary_dir, modelname)
subres = os.path.join(res_dir, modelname)
if not os.path.isdir(subres):
    os.mkdir(subres)
if not os.path.isdir(subname):
    os.mkdir(subname)
    summary_writer = tf.summary.FileWriter(subname, sess.graph)
else:
    summary_writer = tf.summary.FileWriter(subname, sess.graph)

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

    # outHisop = net.outHis
    # gtHisop = net.gtHis
    #
    # loss_op1 = net.loss_gt
    # loss_op2 = net.loss_w
    # loss_op3 = net.loss_sparse

    VideoNum = len(TrainingList)
    epochsort = np.arange(0, VideoNum)
    loss = 0
    iter = 0
    start_time = time.time()
    start_time0 = time.time()
    start_time1 = time.time()
    for epoch in range(epoch_num):
        random.shuffle(epochsort)
        print('%d-th epochs' % (epoch))
        v_count = 0
        losslist = np.array([])
        for v in epochsort:
            vfile = TrainingList[v]
            vdir = os.path.split(vfile)[-1]
            vname = vdir.split('.')[0]
            v_count = v_count + 1
            batch_count = 0
            GTmap_Batch = []
            Input_Batch = []
            mask_in = []
            mask_h = []
            sess.run(iterator.initializer, feed_dict={filenames: vfile})
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
            frameindex = 0
            while frameindex+framesnum+frame_skip <= numframe:
                imageInput = Frameall[frameindex:(frameindex+framesnum+frame_skip), ...]
                GTmapInput = GTall[frameindex:(frameindex+framesnum+frame_skip), ...]
                imageInput = imageInput[np.newaxis, ...]
                GTmapInput = GTmapInput[np.newaxis, ...]
                frameindex = frameindex + overlapframe
                distmatrix, meanmask = get_centermask((1, 28, 28, 128, 4 * 2), CB_FB)
                mask_in_s = np.random.binomial(MC_count, dp_in, (1, 28, 28, 128, 4 * 2)) * distmatrix / (
                            MC_count * meanmask)
                mask_h_s = np.random.binomial(MC_count, dp_h, (1, 28, 28, 128, 4 * 2)) * distmatrix / (
                            MC_count * meanmask)
                if batch_count == 0:
                    GTmap_Batch = GTmapInput
                    Input_Batch = imageInput
                    mask_in = mask_in_s
                    mask_h = mask_h_s
                    batch_count = batch_count + 1
                else:
                    GTmap_Batch = np.concatenate((GTmap_Batch, GTmapInput), axis=0)
                    Input_Batch = np.concatenate((Input_Batch, imageInput), axis=0)
                    mask_in = np.concatenate((mask_in, mask_in_s), axis=0)
                    mask_h = np.concatenate((mask_h, mask_h_s), axis=0)
                    batch_count = batch_count + 1
                if batch_count ==  batch_size:
                        batch_count = 0
                        iter += 1
                        _, loss = sess.run([train_op, loss_op],  feed_dict={input: Input_Batch, GroundTruth: GTmap_Batch, RNNmask_in: mask_in, RNNmask_h: mask_h, exloss:0})
                        assert not np.isnan(loss), 'Model diverged with loss = NaN'
                        losslist = np.insert(losslist, 0, values=loss, axis=0)
                        if iter % 100 == 0:
                            # print('%d iterations, %f for each iteration' % (iter, duration))
                            # np_predict, summary, _,l1,l2 = sess.run([predicts, summary_op, train_op,loss_op1,loss_op2], feed_dict={input: Input_Batch, GroundTruth: GTmap_Batch})
                            np_predict, summary, loss = sess.run(
                                [predicts, summary_op, loss_op],
                                feed_dict={input: Input_Batch, GroundTruth: GTmap_Batch, RNNmask_in:mask_in ,RNNmask_h: mask_h, exloss:0})
                            assert not np.isnan(loss), 'Model diverged with loss = NaN'
                            summary_writer.add_summary(summary, iter)

            if epoch == 0:
                usedtime = (time.time() - start_time)/3600
                meanloss = losslist.mean()
                print('%d th video: %s; Have used time: %f hrs, average loss %f' % (v_count, vname, usedtime,meanloss))

        duration = time.time() - start_time1
        start_time1 = time.time()
        meanloss = losslist.mean()
        losslist = np.array([])
        print('Total time for this epoch is %f, average loss %f.' % (
            duration/3600, meanloss))
        hrleft = ((epoch_num - epoch - 1) / (epoch + 1)) * duration
        print('Left hours: %f.' % (hrleft / 3600))
        if epoch % 2 == 0:
            saver2.save(sess, SaveFile + modelname, global_step=epoch+1)
            sub2res = os.path.join(subres, '%03d' % (epoch))
            if not os.path.isdir(sub2res):
                os.mkdir(sub2res)
            for j in ValidData_index:
                tfdir = Valid_list[j]
                datasetname = tfdir.split('/')[1]
                datasetname = datasetname[:-9]
                sumKL = 0
                sumCC = 0
                count = 0
                sub3res = os.path.join(sub2res, datasetname)
                if not os.path.isdir(sub3res):
                    os.mkdir(sub3res)
                for validfile in glob.glob(tfdir + '*.tfrecords'):
                    validCC, validKL = valid(validfile, sub3res)
                    # print(validCC)
                    # print(validKL)
                    sumKL = sumKL + validKL
                    sumCC = sumCC + validCC
                    # sumLoss = sumLoss + validloss
                    count = count + 1

                summary2 = tf.Summary(value=[
                    tf.Summary.Value(tag=datasetname + "_ValKL", simple_value=sumKL / count),
                    tf.Summary.Value(tag=datasetname + "_ValCC", simple_value=sumCC / count),
                    #   tf.Summary.Value(tag=datasetname + "_Valloss", simple_value=sumLoss / count),
                ])
                summary_writer.add_summary(summary2, epoch)


def get_centermask(f_shape, fb):  # shape[batchsize, height, width, channals]
    width = f_shape[2]
    heigh = f_shape[1]
    midw = width // 2
    midh = heigh // 2
    distmatrix = np.zeros([heigh, width])
    for x in range(width):
        for y in range(heigh):
            value = np.sqrt((x - midw) ** 2 + (y - midh) ** 2)
            distmatrix[x, y] = value
    distmatrix = (distmatrix / np.max(distmatrix)) * (1 - fb)
    distmatrix = 1 - distmatrix
    distmatrix = distmatrix[np.newaxis, ..., np.newaxis,np.newaxis]
    meanmask = np.mean(distmatrix)
    # distmatrix = tf.expand_dims(distmatrix, 0)
    # distmatrix = tf.expand_dims(distmatrix, 3)
    # for a in range(f_shape[0]):
    #   for b in range(f_shape[3]):
    #     mask[a, :, :, b] = distmatrix
    return distmatrix,meanmask


def out_loss(disout, disgt, distype):
    #start_time = time.time()
    if distype == 'Wassers' or distype == 'dualWassers':
        shapes = disout.shape
        assert disout.shape == disgt.shape
        assert len(shapes) == 3
        l = numdis
        A_r = np.zeros((l, l, l))
        A_t = np.zeros((l, l, l))
        for i in range(l):
            for j in range(l):
                A_r[i, i, j] = 1
                A_t[i, j, i] = 1
        D = np.ndarray(shape=(l, l))
        for i in range(l):
            for j in range(l):
                D[i, j] = abs(range(l)[i] - range(l)[j])
        A = np.concatenate((A_r.reshape((l, l ** 2)), A_t.reshape((l, l ** 2))), axis=0)
        c = D.reshape((l ** 2))
        loss = 0
        if distype == 'Wassers':
            for i in range(shapes[0]):
                for j in range(shapes[1]):
                    diso = disout[i,j,:]
                    disg = disgt[i,j,:]
                    b = np.concatenate((diso, disg), axis=0)
                    opt_res = linprog(c, A_eq=A, b_eq=b)
                    emd = opt_res.fun
                    loss = loss + emd
        elif distype == 'dualWassers':
            for i in range(shapes[0]):
                for j in range(shapes[1]):
                    diso = disout[i,j,:]
                    disg = disgt[i,j,:]
                    b = np.concatenate((diso, disg), axis=0)
                    opt_res = linprog(-b, A.T, c, bounds=(None, None))
                    emd = -opt_res.fun
                    loss = loss + emd
    else:
        loss = 0
    return loss

def valid(filename, outdir):
    sess.run(iterator.initializer, feed_dict={filenames: filename})
    vdir = os.path.split(filename)[-1]
    vname = vdir.split('.')[0]
    # batch_size = 1
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
    SalOut = np.zeros_like(GTall, np.uint8)
    frameindex = 0

    imageInput = Frameall[frameindex:(frameindex + framesnum + frame_skip), ...]
    GTmapInput = GTall[frameindex:(frameindex + framesnum + frame_skip), ...]
    imageInput = imageInput[np.newaxis, ...]
    GTmapInput = GTmapInput[np.newaxis, ...]
    Input_Batch = imageInput
    GTmap_Batch = GTmapInput

    for j in range(batch_size-1):
        frameindex = frameindex + 1
        imageInput = Frameall[frameindex:(frameindex + framesnum + frame_skip), ...]
        GTmapInput = GTall[frameindex:(frameindex + framesnum + frame_skip), ...]
        imageInput = imageInput[np.newaxis, ...]
        GTmapInput = GTmapInput[np.newaxis, ...]
        GTmap_Batch = np.concatenate((GTmap_Batch, GTmapInput), axis=0)
        Input_Batch = np.concatenate((Input_Batch, imageInput), axis=0)
    np_predict = sess.run(predicts,
                          feed_dict={input: Input_Batch, GroundTruth: GTmap_Batch,
                                     RNNmask_in: mask_in, RNNmask_h: mask_h})
    np_predict = np.uint8(np_predict * 255)

    for i in range(frame_skip):
        SalOut[i, ...] = np_predict[0,0,...]
    SalOut[frame_skip:frame_skip+framesnum, ...] = np_predict[0,...]
    for j in range(batch_size - 1):
        SalOut[j + frame_skip + framesnum, ...] = np_predict[j+1, -1, ...]

    frameindex = frameindex + 1
    batch_count = 0
    while frameindex + framesnum + frame_skip <= numframe:
        imageInput = Frameall[frameindex:(frameindex + framesnum + frame_skip), ...]
        GTmapInput = GTall[frameindex:(frameindex + framesnum + frame_skip), ...]
        imageInput = imageInput[np.newaxis, ...]
        GTmapInput = GTmapInput[np.newaxis, ...]
        if batch_count == 0:
            GTmap_Batch = GTmapInput
            Input_Batch = imageInput
            batch_count = batch_count + 1
        else:
            GTmap_Batch = np.concatenate((GTmap_Batch, GTmapInput), axis=0)
            Input_Batch = np.concatenate((Input_Batch, imageInput), axis=0)
            batch_count = batch_count + 1
        if batch_count == batch_size:
            np_predict = sess.run(predicts,
                                  feed_dict={input: imageInput, GroundTruth: GTmapInput,
                                             RNNmask_in: mask_in, RNNmask_h: mask_h})
            np_predict = np.uint8(np_predict * 255)
            SalOut[(frameindex + frame_skip + framesnum - batch_size):(frameindex + frame_skip + framesnum), ...] = np_predict[:, -1, ...]
        frameindex = frameindex + 1
    writer = imageio.get_writer(outdir + '/' + vname + '.avi', fps=30)
    iter = 0
    sum_CC = 0
    sum_KL = 0
    for indexFrame in range(SalOut.shape[0]):
        assert np.sum(SalOut[indexFrame, ..., 0]) != 0
        tempCC = cacCC(GTall[indexFrame, ..., 0], SalOut[indexFrame, ..., 0])
        tempKL = cacKL(GTall[indexFrame, ..., 0], SalOut[indexFrame, ..., 0])
        if not np.isnan(tempCC) and not np.isnan(tempKL):
            iter += 1
            sum_CC += tempCC
            sum_KL += tempKL
        writer.append_data(SalOut[indexFrame, ..., 0])
    writer.close()
    if iter == 0:
        return 0, 10
    else:
        return sum_CC / iter, sum_KL / iter


def cacCC(gtsAnn, resAnn):
    gtsAnn=gtsAnn.astype(np.float32)
    resAnn = resAnn.astype(np.float32)
    fixationMap = gtsAnn - np.mean(gtsAnn)
    if np.max(fixationMap) > 0:
        fixationMap = fixationMap / np.std(fixationMap)
    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)
    return np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]


def cacKL(gtsAnn, resAnn, eps=1e-7):
    gtsAnn=gtsAnn.astype(np.float32)
    resAnn = resAnn.astype(np.float32)
    if np.sum(gtsAnn) > 0:
        gtsAnn = gtsAnn / np.sum(gtsAnn)
    if np.sum(resAnn) > 0:
        resAnn = resAnn / np.sum(resAnn)
    return np.sum(gtsAnn * np.log(eps + gtsAnn / (resAnn + eps)))

if __name__ == '__main__':
    main()

