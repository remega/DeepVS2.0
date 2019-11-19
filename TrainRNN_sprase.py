import numpy as np
import tensorflow as tf
import cv2
import time
import yolo_lstmconv2 as Network  # define the CNN
import random
from scipy.optimize import linprog


# global w_img,h_img
# w_img = 640 #
# h_img = 480 #
batch_size = 1
framesnum = 16
inputDim = 448
input_size = (inputDim, inputDim)
outputDim = 112
output_size = (outputDim, outputDim)
epoch_num = 20
overlapframe = 10 #0~framesnum+frame_skip
dp_in = 0.75
dp_h = 0.75
MC_count = 100
CB_FB = 0.8
numdis = 10
random.seed(a=730)
tf.set_random_seed(730)
frame_skip = 5
dis_type = 'Wassers' # Wassers,dualWassers, KL,dualKL
dislambda = 0.25

VideoNameFile = 'Traininglist.txt' #'Validationlist.txt'# 'Traininglist.txt'     #choose the data
Video_dir = 'G:\database\statistics\database'
CheckpointFile_yolo = './model/pretrain/CNN_YoloFlow_nofinetuned_batch12_premask_lb05_loss05_fea128_1x512_128-185000'
CheckpointFile_flow = './model/pretrain/CNN_YoloFlow_nofinetuned_batch12_premask_lb05_loss05_fea128_1x512_128-185000'
SaveFile = './model/'
Summary_dir = './summary'

#

def _BatchExtraction(VideoCap, GTCap, batchsize=batch_size, last_input=None, last_GT=None, video_start = True):
    if video_start:
        _, frame = VideoCap.read()
        _, GTframe = GTCap.read()
        GTmap = GTframe[:, :, 2]
        GTmap = GTmap.astype(np.float32)
        GTmap = GTmap / 255.0
        frame = frame.astype(np.float32)
        frame = frame / 255.0 * 2 - 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        GTmap_Batch = GTmap[np.newaxis, ..., np.newaxis]
        Input_Batch = frame[np.newaxis, ...]
        for i in range(batchsize - 1):
            _, frame = VideoCap.read()
            _, GTframe = GTCap.read()
            GTmap = GTframe[:, :, 2]
            GTmap = GTmap.astype(np.float32)
            GTmap = GTmap / 255.0
            frame = frame.astype(np.float32)
            frame = frame / 255.0 * 2 - 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            GTmap = GTmap[np.newaxis, ..., np.newaxis]
            frame = frame[np.newaxis, ...]
            GTmap_Batch = np.concatenate((GTmap_Batch, GTmap), axis=0)
            Input_Batch = np.concatenate((Input_Batch, frame), axis=0)
        Input_Batch = Input_Batch[np.newaxis, ...]
        GTmap_Batch = GTmap_Batch[np.newaxis, ...]
    else:
        Input_Batch = last_input[:,-overlapframe:,...]
        GTmap_Batch = last_GT[:,-overlapframe:,...]
        for i in range(batchsize-overlapframe):
            _, frame = VideoCap.read()
            _, GTframe = GTCap.read()
            GTmap = GTframe[:, :, 2]
            GTmap = GTmap.astype(np.float32)
            GTmap = GTmap / 255.0
            frame = frame.astype(np.float32)
            frame = frame / 255.0 * 2 - 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            GTmap = GTmap[np.newaxis,np.newaxis, ..., np.newaxis]
            frame = frame[np.newaxis,np.newaxis, ...]
            GTmap_Batch = np.concatenate((GTmap_Batch, GTmap), axis=1)
            Input_Batch = np.concatenate((Input_Batch, frame), axis=1)
    return Input_Batch, GTmap_Batch


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


def main():
    net = Network.Net()
    net.is_training = True
    net.dp_in = dp_in
    net.dp_h = dp_h
    net.lambdadis = dislambda
    net.disnum = numdis
    net.distype = dis_type

    input = tf.placeholder(tf.float32, (batch_size, framesnum + frame_skip, input_size[0], input_size[1], 3))
    GroundTruth = tf.placeholder(tf.float32, (batch_size, framesnum + frame_skip, output_size[0], output_size[1], 1))
    RNNmask_in = tf.placeholder(tf.float32, (batch_size, 28, 28, 128, 4 * 2))
    RNNmask_h = tf.placeholder(tf.float32, (batch_size, 28, 28, 128, 4 * 2))
    exloss = tf.placeholder(tf.float32)

    net.inference(input, GroundTruth, RNNmask_in, RNNmask_h)
    net._loss(exloss)
    loss_op = net.loss
    outHisop = net.outHis
    gtHisop = net.gtHis

    loss_op1 = net.loss_gt
    loss_op2 = net.loss_w
    loss_op3 = net.loss_sparse

    train_op = net._train()
    predicts = net.out

    sess = tf.Session()
    saver = tf.train.Saver(net.yolofeatures_colllection)
    saver1 = tf.train.Saver(net.flowfeatures_colllection)

    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, CheckpointFile_yolo)
    saver1.restore(sess, CheckpointFile_flow)

    saver2 = tf.train.Saver()
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(Summary_dir, sess.graph)

    videofile = open(VideoNameFile, 'r')
    allline = videofile.readlines()
    VideoIndex_list = []
    VideoName_list = []
    for line in allline:
        lindex = line.index('\t')
        VideoIndex = int(line[:lindex])
        VideoName = line[lindex + 1:-1]
        VideoIndex_list.append(VideoIndex)
        VideoName_list.append(VideoName)
    VideoNum = len(VideoName_list)
    epochsort = np.arange(0,VideoNum)

    iter = 0
    batch_count = 0

    for epoch in range(epoch_num):
        if epoch % 5 == 0:
            random.shuffle(epochsort)
        print('%d-th epochs' % (epoch))
        for v in epochsort:
            VideoIndex = VideoIndex_list[v]
            VideoName = VideoName_list[v]
            VideoCap = cv2.VideoCapture(Video_dir + '\\' + VideoName + '\\' + VideoName + '_448.avi')
            GTCap = cv2.VideoCapture(Video_dir + '\\' + VideoName + '\\' + 'GT_112.avi')
            VideoSize = (int(VideoCap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(VideoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            VideoFrame = int(VideoCap.get(cv2.CAP_PROP_FRAME_COUNT))
            # assert VideoSize[0] == int(GTCap.get(cv2.CAP_PROP_FRAME_WIDTH)) and VideoSize[1] == int(
            #     GTCap.get(cv2.CAP_PROP_FRAME_HEIGHT)) and VideoFrame == int(GTCap.get(cv2.CAP_PROP_FRAME_COUNT))
            print('New video: %s (%d) with %d frames and size of (%d, %d)' % (VideoName, VideoIndex ,VideoFrame, VideoSize[1], VideoSize[0]))
            start_time = time.time()
            losslist = np.array([])
            videostart = True
            while VideoCap.get(cv2.CAP_PROP_POS_FRAMES) < VideoFrame - framesnum - frame_skip + overlapframe:
                if videostart:
                    Input_slides, GTmap_slides = _BatchExtraction(VideoCap, GTCap, framesnum + frame_skip, video_start = videostart)
                    videostart = False
                    Input_last = Input_slides
                    GT_last = GTmap_slides
                else:
                    Input_slides, GTmap_slides = _BatchExtraction(VideoCap, GTCap, framesnum + frame_skip, last_input = Input_last, last_GT = GT_last,
                                                                  video_start = videostart)
                    Input_last = Input_slides
                    GT_last = GTmap_slides
                distmatrix, meanmask = get_centermask((1, 28, 28, 128, 4 * 2),CB_FB)
                mask_in_s = np.random.binomial(MC_count, dp_in, (1, 28, 28, 128, 4 * 2)) * distmatrix / (MC_count * meanmask)
                mask_h_s = np.random.binomial(MC_count, dp_h, (1, 28, 28, 128, 4 * 2)) * distmatrix / (MC_count * meanmask)
                if batch_count == 0:
                    Input_Batch = Input_slides
                    GTmap_Batch = GTmap_slides
                    batch_count = batch_count + 1
                    mask_in = mask_in_s
                    mask_h = mask_h_s
                else:
                    GTmap_Batch = np.concatenate((GTmap_Batch, GTmap_slides), axis=0)
                    Input_Batch = np.concatenate((Input_Batch, Input_slides), axis=0)
                    mask_in = np.concatenate((mask_in, mask_in_s), axis=0)
                    mask_h = np.concatenate((mask_h, mask_h_s), axis=0)
                    batch_count = batch_count + 1

                if batch_count ==  batch_size:
                        batch_count = 0
                        iter += 1
                        disout, disgt = sess.run([outHisop, gtHisop],
                                           feed_dict={input: Input_Batch, GroundTruth: GTmap_Batch, RNNmask_in: mask_in,
                                                      RNNmask_h: mask_h})
                        metricloss = out_loss(disout, disgt, distype = dis_type)
                        _, loss = sess.run([train_op, loss_op],  feed_dict={input: Input_Batch, GroundTruth: GTmap_Batch, RNNmask_in: mask_in, RNNmask_h: mask_h, exloss:metricloss})
                        assert not np.isnan(loss), 'Model diverged with loss = NaN'
                        losslist = np.insert(losslist, 0, values=loss, axis=0)
                        loss1,loss2,loss3 = sess.run([loss_op1, loss_op2,loss_op3], feed_dict={input: Input_Batch, GroundTruth: GTmap_Batch, RNNmask_in:mask_in ,RNNmask_h: mask_h, exloss:metricloss})
                        print(loss1,loss2,loss3)
                        if iter % 50000 == 0:
                            saver2.save(sess, SaveFile + 'lstmconv_prefinal_loss05_dp075_MC100_centerdrop08', global_step=iter)
                            # summary, loss = sess.run([summary_op, loss_op],
                            #                             feed_dict={input: Input_Batch, GroundTruth: GTmap_Batch, RNNmask_in:mask_in ,RNNmask_h: mask_h})
                            # assert not np.isnan(loss), 'Model diverged with loss = NaN'
                            # summary_writer.add_summary(summary, iter)
                        if iter % 100 == 0:
                            # print('%d iterations, %f for each iteration' % (iter, duration))
                            # np_predict, summary, _,l1,l2 = sess.run([predicts, summary_op, train_op,loss_op1,loss_op2], feed_dict={input: Input_Batch, GroundTruth: GTmap_Batch})
                            np_predict, summary, loss = sess.run(
                                [predicts, summary_op, loss_op],
                                feed_dict={input: Input_Batch, GroundTruth: GTmap_Batch, RNNmask_in:mask_in ,RNNmask_h: mask_h, exloss:metricloss})
                            assert not np.isnan(loss), 'Model diverged with loss = NaN'
                            summary_writer.add_summary(summary, iter)
                            print('The loss is %f' % (loss))
                            # print('The klloss is %f' % (l1))
                            np_predict = np_predict[0,0, :, :, 0]
                            Out_frame = cv2.resize(np_predict, VideoSize)
                            Out_frame = Out_frame * 255
                            Out_frame = np.uint8(Out_frame)
                            cv2.imwrite("./out.jpg", Out_frame)
                            GTmap = GTmap_Batch[0, 0,:, :, 0]
                            GTmap = cv2.resize(GTmap, VideoSize)
                            GTmap = GTmap * 255
                            GTmap = np.uint8(GTmap)
                            cv2.imwrite("./GT.jpg", GTmap)
                            #loss1,loss2,loss3 = sess.run([loss_op1, loss_op2,loss_op3],
                                                       #feed_dict={input: Input_Batch, GroundTruth: GTmap_Batch, RNNmask_in:mask_in ,RNNmask_h: mask_h})

                           # print(loss1,loss2,loss3)

            duration = float(time.time() - start_time)
            meanloss = losslist.mean()
            print('Total time for this video %f, average loss %f ' % (duration, meanloss))
                # print(duration)
            VideoCap.release()
            GTCap.release()
if __name__ == '__main__':
    main()

