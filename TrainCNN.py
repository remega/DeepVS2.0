import numpy as np
import tensorflow as tf
import cv2
import time
import CNN_YoloFlow as Network  # define the CNN
import random

# global w_img,h_img
# w_img = 640 #
# h_img = 480 #
batch_size = 12

inputDim = 448
input_size = (inputDim, inputDim)
outputDim = 112
output_size = (outputDim, outputDim)
epoch_num = 15
overlapframe = 5

random.seed(a=730)
tf.set_random_seed(730)
frame_skip = 5

VideoNameFile = 'Traininglist.txt' #'Validationlist.txt'# 'Traininglist.txt'     #choose the data
Video_dir = 'G:\database\statistics\database'
CheckpointFile_yolo = './model/pretrain/yolo_tiny_rnn'
CheckpointFile_flow = './model/pretrain/FlowNet'
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
    else:
        Input_Batch = last_input[-overlapframe:,...]
        GTmap_Batch = last_GT[-overlapframe:,...]
        for i in range(batchsize-overlapframe):
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
    return Input_Batch, GTmap_Batch



def main():
    net = Network.Net()
    net.is_training = True

    input1 = tf.placeholder(tf.float32, (batch_size, input_size[0], input_size[1], 3))
    input2 = tf.placeholder(tf.float32, (batch_size, input_size[0], input_size[1], 3))
    GroundTruth = tf.placeholder(tf.float32, (batch_size, output_size[0], output_size[1], 1))
    net.inference(input1, input2)
    net._loss(GroundTruth)
    loss_op = net.loss
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
            while VideoCap.get(cv2.CAP_PROP_POS_FRAMES) < VideoFrame - batch_size - frame_skip + overlapframe:
                if videostart:
                    Input_Batch, GTmap_Batch = _BatchExtraction(VideoCap, GTCap, batch_size + frame_skip, video_start = videostart)
                    videostart = False
                    Input_last = Input_Batch
                    GT_last = GTmap_Batch
                else:
                    Input_Batch, GTmap_Batch = _BatchExtraction(VideoCap, GTCap, batch_size + frame_skip, last_input = Input_last, last_GT = GT_last, video_start = videostart)
                    Input_last = Input_Batch
                    GT_last = GTmap_Batch
                Input_ori = Input_Batch[:batch_size,...]
                GT = GTmap_Batch[:batch_size,...]
                Input_post = Input_Batch[-batch_size:,...]
                iter += 1
                # print(iter)
                if iter % 5000 == 0:
                    saver2.save(sess, SaveFile + 'CNN_YoloFlow_nofinetuned_batch12_premask_lb05_loss05_fea128_1x512_128', global_step=iter)
                    summary, loss, _ = sess.run([summary_op, loss_op, train_op],
                                                feed_dict={input1: Input_ori, input2: Input_post, GroundTruth: GT})
                    assert not np.isnan(loss), 'Model diverged with loss = NaN'
                    summary_writer.add_summary(summary, iter)
                elif iter % 100 == 0:
                    # print('%d iterations, %f for each iteration' % (iter, duration))
                    # np_predict, summary, _,l1,l2 = sess.run([predicts, summary_op, train_op,loss_op1,loss_op2], feed_dict={input: Input_Batch, GroundTruth: GTmap_Batch})
                    np_predict, summary, _, loss = sess.run(
                        [predicts, summary_op, train_op, loss_op],
                        feed_dict={input1: Input_ori, input2: Input_post, GroundTruth: GT})
                    assert not np.isnan(loss), 'Model diverged with loss = NaN'
                    summary_writer.add_summary(summary, iter)
                    print('The loss is %f' % (loss))
                    # print('The klloss is %f' % (l1))
                    np_predict = np_predict[0, :, :, 0]
                    Out_frame = cv2.resize(np_predict, VideoSize)
                    Out_frame = Out_frame * 255
                    Out_frame = np.uint8(Out_frame)
                    cv2.imwrite("./out.jpg", Out_frame)
                    GTmap = GTmap_Batch[0,:, :, 0]
                    GTmap = cv2.resize(GTmap, VideoSize)
                    GTmap = GTmap * 255
                    GTmap = np.uint8(GTmap)
                    cv2.imwrite("./GT.jpg", GTmap)
                else:
                    _, loss = sess.run([train_op, loss_op],
                                       feed_dict={input1: Input_ori, input2: Input_post, GroundTruth: GT})
                    losslist = np.insert(losslist, 0, values=loss, axis=0)
                   # print(loss)

            duration = float(time.time() - start_time)
            meanloss = losslist.mean()
            print('Total time for this video %f, average loss %f ' % (duration, meanloss))
                # print(duration)
            VideoCap.release()
            GTCap.release()
if __name__ == '__main__':
    main()

