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

targetname = 'CNN_YoloFlow_nofinetuned_batch12_premask_lb05_loss10_fea128_512_128-185000'
VideoNameFile = 'Validationlist.txt'
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
    predicts = net.out
    predicts2 = net.coarse
    predicts3 = net.comb

    loss_op = net.loss
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, CheckpointFile)

    videofile = open(VideoNameFile, 'r')
    allline = videofile.readlines()
    VideoIndex_list = []
    VideoName_list = []
    for line in allline:
        lindex = line.index('\t')
        VideoIndex = int(line[:lindex])
        VideoName = line[lindex + 1:-1]
        VideoCap = cv2.VideoCapture(Video_dir + '\\' + VideoName + '\\' + VideoName + '_448.avi')
        GTCap = cv2.VideoCapture(Video_dir + '\\' + VideoName + '\\' + 'GT_112.avi')
        VideoSize = (int(VideoCap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(VideoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        VideoFrame = int(VideoCap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(VideoCap.get(cv2.CAP_PROP_FPS))
        videoWriter = cv2.VideoWriter(
        Video_dir + '\\' + VideoName + '\\' + targetname + '_final.avi',
            cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps,
            (112, 112), isColor=False)
        # videoWriter2 = cv2.VideoWriter(
        #     Video_dir + '\\' + VideoName + '\\' + targetname + '_pre.avi',
        #     cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps,
        #     (112, 112), isColor=False)
        videoWriter3 = cv2.VideoWriter(
        Video_dir + '\\' + VideoName + '\\' + targetname + '_comb.avi',
            cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps,
            (112, 112), isColor=False)
        print('New video: %s (%d) with %d frames and size of (%d, %d)' % ( VideoName, VideoIndex, VideoFrame, VideoSize[1], VideoSize[0]))
        start_time = time.time()
        losslist = np.array([])

        videostart = True
        while VideoCap.get(cv2.CAP_PROP_POS_FRAMES) < VideoFrame - batch_size - frame_skip + overlapframe:
            if videostart:
                Input_Batch, GTmap_Batch = _BatchExtraction(VideoCap, GTCap, batch_size + frame_skip,
                                                            video_start=videostart)
                videostart = False
                Input_last = Input_Batch
                GT_last = GTmap_Batch
            else:
                Input_Batch, GTmap_Batch = _BatchExtraction(VideoCap, GTCap, batch_size + frame_skip,
                                                            last_input=Input_last, last_GT=GT_last,
                                                            video_start=videostart)
                Input_last = Input_Batch
                GT_last = GTmap_Batch

            Input_ori = Input_Batch[:batch_size,...]
            GT = GTmap_Batch[:batch_size,...]
            Input_post = Input_Batch[-batch_size:,...]
            np_predict, loss,np_predict3 = sess.run([predicts, loss_op,predicts3], feed_dict={input1: Input_ori, input2: Input_post, GroundTruth: GT})
            losslist = np.insert(losslist, 0, values=loss, axis=0)


            for index in range(batch_size):
               Out_frame = np_predict[index, :, :, 0]
               Out_frame = Out_frame * 255
               Out_frame = np.uint8(Out_frame)
               videoWriter.write(Out_frame)
               # Out_frame2 = np_predict2[index, :, :, 0]
               # Out_frame2 = Out_frame2 * 255
               # Out_frame2 = np.uint8(Out_frame2)
               # videoWriter2.write(Out_frame2)
               Out_frame3 = np_predict3[index, :, :, 0]
               Out_frame3 = Out_frame3 * 255
               Out_frame3 = np.uint8(Out_frame3)
               videoWriter3.write(Out_frame3)


        duration = float(time.time() - start_time)
        meanloss = losslist.mean()
        print('Total time for this video %f, average loss %f ' % (duration, meanloss))
                # print(duration)
        VideoCap.release()
        GTCap.release()
if __name__ == '__main__':
    main()

