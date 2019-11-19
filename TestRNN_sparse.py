
import numpy as np
import tensorflow as tf
import cv2
import time
import h5py
import yolo_lstmconv2 as Network  # define the CNN
import yolodemo
import random



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
targetname = 'lstmconv_prefinal_loss05_dp075_MC100_centerdrop1_dualKL_factor01_numdis50-200000'
#VideoNameFile = 'Testlist.txt'
VideoNameFile = 'Testlist.txt' #,Traininglist,Testlist
Video_dir = 'G:\database\statistics\database'
CheckpointFile = './model/pretrain/'+ targetname

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



def main():
    net = Network.Net()
    net.is_training = False
    input = tf.placeholder(tf.float32, (batch_size, framesnum + frame_skip, input_size[0], input_size[1], 3))
    GroundTruth = tf.placeholder(tf.float32, (batch_size, framesnum + frame_skip, output_size[0], output_size[1], 1))
    RNNmask_in = tf.placeholder(tf.float32, (batch_size, 28, 28, 128, 4 * 2))
    RNNmask_h = tf.placeholder(tf.float32, (batch_size, 28, 28, 128, 4 * 2))
    net.inference(input, GroundTruth, RNNmask_in, RNNmask_h)
    net._loss()
    loss_op = net.loss
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

    videofile = open(VideoNameFile, 'r')
    allline = videofile.readlines()
    for line in allline:
        lindex = line.index('\t')
        VideoIndex = int(line[:lindex])
        VideoName = line[lindex + 1:-1]
        VideoCap = cv2.VideoCapture(Video_dir + '\\' + VideoName + '\\' + VideoName + '_448.avi')
        GTCap = cv2.VideoCapture(Video_dir + '\\' + VideoName + '\\' + 'GT_112.avi')
        VideoSize = (int(VideoCap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(VideoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        VideoFrame = int(VideoCap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('New video: %s (%d) with %d frames and size of (%d, %d)' % (
        VideoName, VideoIndex, VideoFrame, VideoSize[1], VideoSize[0]))
        fps = float(VideoCap.get(cv2.CAP_PROP_FPS))
        videoWriter = cv2.VideoWriter(
        Video_dir + '\\' + VideoName + '\\' + targetname +  '-2.avi', #'_framelen' + str(framesnum) +
            cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps,
            (112, 112), isColor=False)
        start_time = time.time()
        losslist = np.array([])
        videostart = True
        while VideoCap.get(cv2.CAP_PROP_POS_FRAMES) < VideoFrame - framesnum - frame_skip + overlapframe:
            if videostart:
                Input_Batch, GTmap_Batch = _BatchExtraction(VideoCap, GTCap, framesnum + frame_skip, video_start=videostart)
                videostart = False
                Input_last = Input_Batch
                GT_last = GTmap_Batch
            else:
                Input_Batch, GTmap_Batch = _BatchExtraction(VideoCap, GTCap, framesnum + frame_skip, last_input=Input_last,
                                                              last_GT=GT_last,
                                                              video_start=videostart)
                Input_last = Input_Batch
                GT_last = GTmap_Batch
            mask_in = np.ones((1, 28, 28, 128, 4 * 2))
            mask_h = np.ones((1, 28, 28, 128, 4 * 2))
            np_predict, loss = sess.run([predicts,loss_op],
                                        feed_dict={input: Input_Batch, GroundTruth: GTmap_Batch, RNNmask_in: mask_in,
                                                   RNNmask_h: mask_h})
            losslist = np.insert(losslist, 0, values=loss, axis=0)

            for index in range(framesnum):
                Out_frame = np_predict[0, index, :, :, 0]
                Out_frame = Out_frame * 255
                Out_frame = np.uint8(Out_frame)
                videoWriter.write(Out_frame)
            # if videostart:
            #     for index in range(framesnum):
            #        Out_frame = np_predict[0,index, :, :, 0]
            #        Out_frame = Out_frame * 255
            #        Out_frame = np.uint8(Out_frame)
            #        videoWriter.write(Out_frame)
            #     videostart = False
            # else:
            #      Out_frame = np_predict[0, -1, :, :, 0]
            #      Out_frame = Out_frame * 255
            #      Out_frame = np.uint8(Out_frame)
            #      videoWriter.write(Out_frame)


            #    Out_frame = Out_frame * 255
            # for index in range(framesnum):
            #    Out_frame = np_predict[0,index, :, :, 0]
            #    Out_frame = Out_frame * 255
            #    Out_frame = np.uint8(Out_frame)
            #    videoWriter.write(Out_frame)

                    # print(loss)

        duration = float(time.time() - start_time)
        meanloss = losslist.mean()
        print('Total time for this video %f, average loss %f ' % (duration, meanloss))
        # print(duration)
        VideoCap.release()
        GTCap.release()

if __name__ == '__main__':
    main()

