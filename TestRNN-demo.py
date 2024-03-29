import numpy as np
import tensorflow as tf
import cv2
import time
import h5py
import yolo_lstmconv as Network  # define the CNN
import yolodemo
import random
import glob
import os

# global w_img,h_img
# w_img = 640 #
# h_img = 480 #
batch_size = 1
framesnum = 16
frame_skip = 5
inputDim = 448
input_size = (inputDim, inputDim)
outputDim = 112
output_size = (outputDim, outputDim)
epoch_num = 15
overlapframe = 5# framesnum + frame_skip -1 #5 #0~framesnum+frame_skip

random.seed(a=730)
tf.set_random_seed(730)

dp_in = 1
dp_h = 1
targetname = 'LSTMconv_prefinal_loss05_dp075_075MC100-200000'
#VideoNameFile = 'Testlist.txt'
VideoNameFile = 'Testlist.txt'
Video_dir = 'G:/database/CITIUS/CITIUS/'
savedir = 'G:/database/CITIUS/res/deepvs/'
CheckpointFile = './model/pretrain/'+ targetname

#

def _BatchExtraction(VideoCap, batchsize=batch_size, last_input=None, video_start = True):
    if video_start:
        _, frame = VideoCap.read()
        frame = frame.astype(np.float32)
        frame = frame / 255.0 * 2 - 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, input_size)
        Input_Batch = frame[np.newaxis, ...]
        for i in range(batchsize - 1):
            _, frame = VideoCap.read()
            frame = frame.astype(np.float32)
            frame = frame / 255.0 * 2 - 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, input_size)
            frame = frame[np.newaxis, ...]
            Input_Batch = np.concatenate((Input_Batch, frame), axis=0)
        Input_Batch = Input_Batch[np.newaxis, ...]
    else:
        Input_Batch = last_input[:,-overlapframe:,...]
        for i in range(batchsize-overlapframe):
            _, frame = VideoCap.read()
            frame = frame.astype(np.float32)
            frame = frame / 255.0 * 2 - 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, input_size)
            frame = frame[np.newaxis,np.newaxis, ...]
            Input_Batch = np.concatenate((Input_Batch, frame), axis=1)
    return Input_Batch



def main():
    net = Network.Net()
    net.is_training = False
    input = tf.placeholder(tf.float32, (batch_size, framesnum + frame_skip, input_size[0], input_size[1], 3))
    GroundTruth = tf.placeholder(tf.float32, (batch_size, framesnum + frame_skip, output_size[0], output_size[1], 1))
    RNNmask_in = tf.placeholder(tf.float32, (batch_size, 28, 28, 128, 4 * 2))
    RNNmask_h = tf.placeholder(tf.float32, (batch_size, 28, 28, 128, 4 * 2))
    net.inference(input, GroundTruth, RNNmask_in, RNNmask_h)
    net.dp_in = dp_in
    net.dp_h = dp_h
    # loss_op1 = net.loss_gt
    # loss_op2 = net.loss_gt2
    predicts = net.out

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    # sess = tf.Session(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, CheckpointFile)

    for video_path in glob.glob(Video_dir + '*.avi'):
        full_name = os.path.split(video_path)[-1]
        (VideoName, file_type) = os.path.splitext(full_name)
        VideoCap = cv2.VideoCapture(video_path)
        VideoSize = (int(VideoCap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(VideoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        VideoFrame = int(VideoCap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('New video: %s with %d frames and size of (%d, %d)' % (
        VideoName, VideoFrame, VideoSize[1], VideoSize[0]))
        fps = float(VideoCap.get(cv2.CAP_PROP_FPS))
        videoWriter = cv2.VideoWriter(
            savedir +full_name,
            cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps,
            output_size, isColor=False)
        start_time = time.time()
        videostart = True
        while VideoCap.get(cv2.CAP_PROP_POS_FRAMES) < VideoFrame - framesnum*2 - frame_skip + overlapframe-:
            if videostart:
                Input_Batch = _BatchExtraction(VideoCap, framesnum + frame_skip, video_start=videostart)
                videostart = False
                Input_last = Input_Batch
            else:
                Input_Batch = _BatchExtraction(VideoCap, framesnum + frame_skip, last_input=Input_last,
                                                              video_start=videostart)
                Input_last = Input_Batch
            mask_in = np.ones((1, 28, 28, 128, 4 * 2))
            mask_h = np.ones((1, 28, 28, 128, 4 * 2))
            np_predict = sess.run(predicts,
                                        feed_dict={input: Input_Batch, RNNmask_in: mask_in,
                                                   RNNmask_h: mask_h})

            for index in range(framesnum):
                Out_frame = np_predict[0, index, :, :, 0]
                Out_frame = Out_frame * 255
                Out_frame = np.uint8(Out_frame)
                videoWriter.write(Out_frame)


        duration = float(time.time() - start_time)
        print('Total time for this video %f ' % (duration))
        # print(duration)
        VideoCap.release()

if __name__ == '__main__':
    main()

