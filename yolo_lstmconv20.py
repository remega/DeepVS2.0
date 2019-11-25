from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import BasicNet
import BasicConvLSTMCell
from flownetsrc.flownet_cs.flownet_cs import FlowNetCS



class Net(BasicNet.BasicNet):
  # image_size = 448
  batch_size = 16
  framenum =16
  maximgbatch = 4
  init_learning_rate = 10**(-4)
  eps = 1e-7
  gapnum = 5
  salmask_lb = 0.5 #mask cam be salmask_lb~1
  dp_in = 0.25
  dp_h = 0.25
  dropouttype = 'bayes'
  distype  = 'KL'  # Wassers,dualWassers, KL,dualKL
  disnum = 10
  lambdadis = 0
  factordis = 1
  version_flow = 1
  version_yolo = 1
  # num_classes = 20
  cell_size = 7
  # boxes_per_cell = 2
  def __init__(self):
    super(Net, self).__init__() # init the fatther class of YoloTinyNet
    self.global_step = tf.Variable(0, trainable=False)
    self.initial_var_collection.append(self.global_step )
    self.out = []
    self.predict = []
    self.loss = []
    self.loss_gt = []
    self.re = []
    self.loss_gt2 = []
    self.yolofeatures_colllection = []
    self.flowfeatures_colllection = []
    self.startflagcnn = True
    #process params


  def YOLO_tiny_inference(self, images):  # pre128
      cnnpretrain = False
      cnntrainable = True
      self.batch_size = images.get_shape()[0].value
      conv_1 = self.conv_layer('conv1', images, 3, 16, stride=1, pretrain=cnnpretrain, batchnormalization=True,
                               trainable=cnntrainable)
      pool_2 = self.max_pool('pool2', conv_1, 2, stride=2)
      conv_3 = self.conv_layer('conv3', pool_2, 3, 32, stride=1, pretrain=cnnpretrain, batchnormalization=True,
                               trainable=cnntrainable)
      pool_4 = self.max_pool('pool4', conv_3, 2, stride=2)
      conv_5 = self.conv_layer('conv5', pool_4, 3, 64, stride=1, pretrain=cnnpretrain, batchnormalization=True,
                               trainable=cnntrainable)
      pool_6 = self.max_pool('pool6', conv_5, 2, stride=2)
      conv_7 = self.conv_layer('conv7', pool_6, 3, 128, stride=1, pretrain=cnnpretrain, batchnormalization=True,
                               trainable=cnntrainable)
      pool_8 = self.max_pool('pool8', conv_7, 2, stride=2)
      conv_9 = self.conv_layer('conv9', pool_8, 3, 256, stride=1, pretrain=cnnpretrain, batchnormalization=True,
                               trainable=cnntrainable)
      pool_10 = self.max_pool('pool10', conv_9, 2, stride=2)
      conv_11 = self.conv_layer('conv11', pool_10, 3, 512, stride=1, pretrain=cnnpretrain, batchnormalization=True,
                                trainable=cnntrainable)
      pool_12 = self.max_pool('pool12', conv_11, 2, stride=2)
      conv_13 = self.conv_layer('conv13', pool_12, 3, 1024, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
      conv_14 = self.conv_layer('conv14', conv_13, 3, 1024, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
      conv_15 = self.conv_layer('conv15', conv_14, 3, 1024, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
      # temp_conv = tf.transpose(conv_15, (0, 3, 1, 2))
      # fc_16 = self.fc_layer('fc16', temp_conv, 256, flat=True, pretrain=cnnpretrain, trainable=cnntrainable)
      # fc_17 = self.fc_layer('fc17', fc_16, 4096, flat=False, pretrain=cnnpretrain, trainable=cnntrainable)
      # fc_18 = self.fc_layer('fc18', fc_17, 1470, flat=False, linear=True, pretrain=cnnpretrain, trainable=cnntrainable)
      #
      # highFeature = tf.reshape(fc_18, [fc_18.get_shape()[0].value, self.cell_size, self.cell_size, -1])

      conv_15_2 = self.conv_layer('conv_15_2', conv_15, 1, 128, stride=1,pretrain=cnnpretrain, trainable=cnntrainable)
      conv_11_2 = self.conv_layer('conv_11_2', conv_11, 1, 128, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
      conv_9_2 = self.conv_layer('conv_9_2', conv_9, 1, 128, stride=1,pretrain=cnnpretrain, trainable=cnntrainable)
      # conv_7_2 = self.conv_layer('conv_7_2', conv_7, 1, 256, stride=1, pretrain=False)
      tempsize = conv_7.get_shape().as_list()
      newconv_7 = tf.image.resize_images(conv_7, [tempsize[1], tempsize[2]])
      newconv_9 = tf.image.resize_images(conv_9_2, [tempsize[1], tempsize[2]])
      newconv_11_2 = tf.image.resize_images(conv_11_2, [tempsize[1], tempsize[2]])
      newconv_15_2 = tf.image.resize_images(conv_15_2, [tempsize[1], tempsize[2]])
      # highFeature = tf.image.resize_images(highFeature, [tempsize[1], tempsize[2]])
      FeatureMap = tf.concat([newconv_7, newconv_9, newconv_11_2, newconv_15_2], axis=3)
      weight_mask = tf.constant(self.get_centermask(FeatureMap.get_shape().as_list()), dtype=FeatureMap.dtype)
      FeatureMap = FeatureMap * weight_mask
      return FeatureMap

  def Coarse_salmap(self, Yolofeature):  #  tiny pregen256 fea28_128
    cnnpretrain = False
    cnntrainable = True
    conv_19 = self.conv_layer('conv_19', Yolofeature, 3, 512, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
    conv_20 = self.conv_layer('conv_20', conv_19, 1, 256, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
    conv_21 = self.conv_layer('conv_21', conv_20, 3, 128, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
    conv_22 = self.conv_layer('conv_22', conv_21, 1, 128, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
    deconv_23 = self.transpose_conv_layer('deconv_23', conv_22, 4, 16, stride=2, pretrain=cnnpretrain, trainable=cnntrainable)
    deconv_24 = self.transpose_conv_layer('deconv_24', deconv_23, 4, 1, stride=2, linear=True, pretrain=cnnpretrain, trainable=cnntrainable)

    return deconv_24

  def Final_inference(self, cat1, cat2):
      cnnpretrain = False
      cnntrainable = True
      MyFeature = tf.concat([cat1, cat2], axis=3)
      Lastconv_1 = self.conv_layer('Lastconv_1', MyFeature, 3, 512, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
      Lastconv_2 = self.conv_layer('Lastconv_2', Lastconv_1, 1, 512, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
      Lastconv_3 = self.conv_layer('Lastconv_3', Lastconv_2, 3, 256, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
      Lastconv_4 = self.conv_layer('Lastconv_4', Lastconv_3, 1, 128, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)

      return Lastconv_4

  def flownet_with_conv(self, x1, x2, mask):
        cnnpretrain = False
        cnntrainable = True
        input = tf.concat([x1, x2], axis=3, name='FNinput')
        conv_1 = self.leaky_conv(input, 64, 7, 2, 'FNconv1', pretrain=cnnpretrain, trainable=cnntrainable)
        conv_1 = self.conv_mask(conv_1, mask)
        conv_2 = self.leaky_conv(conv_1, 128, 5, 2, 'FNconv2', pretrain=cnnpretrain, trainable=cnntrainable)
        conv_2 = self.conv_mask(conv_2, mask)
        conv_3 = self.leaky_conv(conv_2, 256, 5, 2, 'FNconv3', pretrain=cnnpretrain, trainable=cnntrainable)
        conv_3 = self.conv_mask(conv_3, mask)
        conv_3_1 = self.leaky_conv(conv_3, 256, 3, 1, 'FNconv3_1', pretrain=cnnpretrain, trainable=cnntrainable)
        conv_3_1 = self.conv_mask(conv_3_1, mask)
        conv_4 = self.leaky_conv(conv_3_1, 512, 3, 2, 'FNconv4', pretrain=cnnpretrain, trainable=cnntrainable)
        conv_4 = self.conv_mask(conv_4, mask)
        conv_4_1 = self.leaky_conv(conv_4, 512, 3, 1, 'FNconv4_1', pretrain=cnnpretrain, trainable=cnntrainable)
        # conv_4_1 = self.conv_mask(conv_4_1, mask)
        conv_5 = self.leaky_conv(conv_4_1, 512, 3, 2, 'FNconv5', pretrain=cnnpretrain, trainable=cnntrainable)
        # conv_5 = self.conv_mask(conv_5, mask)
        conv_5_1 = self.leaky_conv(conv_5, 512, 3, 1, 'FNconv5_1', pretrain=cnnpretrain, trainable=cnntrainable)
        # conv_5_1 = self.conv_mask(conv_5_1, mask)
        conv_6 = self.leaky_conv(conv_5_1, 1024, 3, 2, 'FNconv6', pretrain=cnnpretrain, trainable=cnntrainable)
        # conv_6 = self.conv_mask(conv_6, mask)
        conv_6_1 = self.leaky_conv(conv_6, 1024, 3, 1, 'FNconv6_1', pretrain=cnnpretrain, trainable=cnntrainable)
        # conv_6_1 = self.conv_mask(conv_6_1, mask)
        out_cat_size = conv_3.get_shape().as_list()

        Downconv_6_1 = self.conv_layer('FNDownconv_6_1', conv_6_1, 3, 128, stride=1,pretrain=cnnpretrain, trainable=cnntrainable)
        Downconv_5_1 = self.conv_layer('FNDownconv_5_1', conv_5_1, 3, 128, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
        Downconv_4_1 = self.conv_layer('FNDownconv_4_1', conv_4_1, 3, 128, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
        Downconv_3_1 = self.conv_layer('FNDownconv_3_1', conv_3_1, 3, 128, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
        conv_6_1_cat = tf.image.resize_images(Downconv_6_1, [out_cat_size[1],out_cat_size[2]])
        conv_5_1_cat = tf.image.resize_images(Downconv_5_1, [out_cat_size[1],out_cat_size[2]])
        conv_4_1_cat = tf.image.resize_images(Downconv_4_1, [out_cat_size[1],out_cat_size[2]])
        conv_3_1_cat = tf.image.resize_images(Downconv_3_1, [out_cat_size[1],out_cat_size[2]])
        concat_out = tf.concat([conv_6_1_cat, conv_5_1_cat, conv_4_1_cat, conv_3_1_cat], axis=3, name='FNconcat_out')

        return concat_out

  def flownet20(self, x1, x2, mask):
    cnnpretrain = False
    cnntrainable = True
    flownet20 = FlowNetCS()
    inputs = {'input_a': x1,
                'input_b': x2}
    conv_3_1, conv_4_1, conv_5_1, conv_6_1 = flownet20.model(inputs)
    out_cat_size = conv_3_1.get_shape().as_list()
    Downconv_6_1 = self.conv_layer('FNDownconv_6_1', conv_6_1, 3, 128, stride=1, pretrain=cnnpretrain,
                                   trainable=cnntrainable)
    Downconv_5_1 = self.conv_layer('FNDownconv_5_1', conv_5_1, 3, 128, stride=1, pretrain=cnnpretrain,
                                   trainable=cnntrainable)
    Downconv_4_1 = self.conv_layer('FNDownconv_4_1', conv_4_1, 3, 128, stride=1, pretrain=cnnpretrain,
                                   trainable=cnntrainable)
    Downconv_3_1 = self.conv_layer('FNDownconv_3_1', conv_3_1, 3, 128, stride=1, pretrain=cnnpretrain,
                                   trainable=cnntrainable)
    conv_6_1_cat = tf.image.resize_images(Downconv_6_1, [out_cat_size[1], out_cat_size[2]])
    conv_5_1_cat = tf.image.resize_images(Downconv_5_1, [out_cat_size[1], out_cat_size[2]])
    conv_4_1_cat = tf.image.resize_images(Downconv_4_1, [out_cat_size[1], out_cat_size[2]])
    conv_3_1_cat = tf.image.resize_images(Downconv_3_1, [out_cat_size[1], out_cat_size[2]])
    concat_out = tf.concat([conv_6_1_cat, conv_5_1_cat, conv_4_1_cat, conv_3_1_cat], axis=3, name='FNconcat_out')

    return concat_out

  def inference(self, videoslides, GTs, mask_in, mask_h): #videoslides: [batch framenum h w num_features]

    with tf.variable_scope('inference'):
        shapes = videoslides.get_shape().as_list()
        shapes2 = GTs.get_shape().as_list()
        #print(shapes,shapes2)
        assert len(shapes)==5 and len(shapes2)==5
        self.batch_size =  videoslides.get_shape()[0].value
        #self.framenum = videoslides.get_shape()[1].value
        # assert self.framenum % self.maximgbatch == 0 # frmaenum shoube be the multiple of maximgbatch   scope = 'layer_1'
        with tf.variable_scope('conv_lstm', initializer=tf.random_uniform_initializer(-.01, 0.1)):
            # cell_1 = BasicConvLSTMCell.BasicConvLSTMCell([56, 56], [3, 3], 128, state_is_tuple = False)  # input size,fliter size, input channals
            # cell_2 = BasicConvLSTMCell.BasicConvLSTMCell([56, 56], [3, 3], 128, state_is_tuple = False)  # input size,fliter size, input channals
            cell_1 = BasicConvLSTMCell.BasicConvLSTMCell([28, 28], [3, 3], 128,
                                                         state_is_tuple=False)  # input size,fliter size, input channals
            cell_2 = BasicConvLSTMCell.BasicConvLSTMCell([28, 28], [3, 3], 128,
                                                         state_is_tuple=False)  # input size,fliter size, input channals

            new_state_1 = cell_1.zero_state(self.batch_size, 2, tf.float32)
            new_state_2 = cell_2.zero_state(self.batch_size, 2, tf.float32)
           # print(videoslides.get_shape().as_list())
        for indexframe in range(self.framenum):
            frame = videoslides[:, indexframe, ...]
            #print(indexframe+self.gapnum)
            frame_gap = videoslides[:, indexframe+self.gapnum, ...]
            GTframe = GTs[:, indexframe, ...]
            Yolo_features = self.YOLO_tiny_inference(frame)
            Presalmap = self.Coarse_salmap(Yolo_features)
            if self.startflagcnn == True:
                self.yolofeatures_colllection = self.pretrain_var_collection
                self.pretrain_var_collection = []
            salmask = self._normlized_0to1(Presalmap)
            salmask = salmask*(1-self.salmask_lb)+self.salmask_lb
            #salmask = tf.ones_like(salmask)

            Flow_features = self.flownet_with_conv(frame, frame_gap, salmask)
            CNNout = self.Final_inference(Yolo_features, Flow_features)
            if self.startflagcnn == True:
                self.flowfeatures_colllection = self.pretrain_var_collection
            y_1, new_state_1 = cell_1(CNNout, new_state_1,mask_in[...,0:4], mask_h[...,0:4], self.dp_in, self.dp_h, 'lstm_layer1', self.dropouttype)
            y_2, new_state_2 = cell_2(y_1, new_state_2,mask_in[...,4:8], mask_h[...,4:8], self.dp_in, self.dp_h, 'lstm_layer2', self.dropouttype)
            deconv = self.transpose_conv_layer('deconv', y_2, 3, 16, stride=2, pretrain=False, trainable=True)
            deconv2 = self.transpose_conv_layer('deconv2', deconv, 3, 1, stride=1, linear=True, pretrain=False, trainable=True)
            if self.startflagcnn == True:
                tf.get_variable_scope().reuse_variables()
                self.trainable_var_collection.extend(cell_1.trainable_var_collection)
                self.trainable_var_collection.extend(cell_2.trainable_var_collection)
                self.startflagcnn = False
            output = self._normlized_0to1(deconv2)
            norm_GT = self._normlized(GTframe)
            outdis = self._MapHistogram(output, self.disnum)
            gtdis = self._MapHistogram(norm_GT, self.disnum)
            outdis = tf.expand_dims(outdis, 1)
            gtdis = tf.expand_dims(gtdis, 1)
            norm_output = self._normlized(output)
            frame_loss = norm_GT * tf.log(self.eps + norm_GT / (norm_output + self.eps))
            frame_loss = tf.reduce_sum(frame_loss)
            tf.add_to_collection('losses', frame_loss)
            output = tf.expand_dims(output, 1)
            #print(output.get_shape().as_list())
            if indexframe == 0:
                tempout = output
                outdisBatch = outdis
                gtdisBatch = gtdis
            else:
                tempout = tf.concat([tempout, output], axis=1)
                outdisBatch = tf.concat([outdisBatch, outdis], axis=1)
                gtdisBatch = tf.concat([gtdisBatch, gtdis], axis=1)
        self.out = tempout
        self.outHis = outdisBatch
        self.gtHis = gtdisBatch

  def inferenceNew(self, videoslides, GTs, mask_in, mask_h):  # videoslides: [batch framenum h w num_features]


      shapes = videoslides.get_shape().as_list()
      shapes2 = GTs.get_shape().as_list()
      # print(shapes,shapes2)
      assert len(shapes) == 5 and len(shapes2) == 5
      self.batch_size = videoslides.get_shape()[0].value
      # self.framenum = videoslides.get_shape()[1].value
      # assert self.framenum % self.maximgbatch == 0 # frmaenum shoube be the multiple of maximgbatch   scope = 'layer_1'
      with tf.variable_scope('conv_lstm', initializer=tf.random_uniform_initializer(-.01, 0.1)):
          # cell_1 = BasicConvLSTMCell.BasicConvLSTMCell([56, 56], [3, 3], 128, state_is_tuple = False)  # input size,fliter size, input channals
          # cell_2 = BasicConvLSTMCell.BasicConvLSTMCell([56, 56], [3, 3], 128, state_is_tuple = False)  # input size,fliter size, input channals
          cell_1 = BasicConvLSTMCell.BasicConvLSTMCell([28, 28], [3, 3], 128,
                                                       state_is_tuple=False)  # input size,fliter size, input channals
          cell_2 = BasicConvLSTMCell.BasicConvLSTMCell([28, 28], [3, 3], 128,
                                                       state_is_tuple=False)  # input size,fliter size, input channals

          new_state_1 = cell_1.zero_state(self.batch_size, 2, tf.float32)
          new_state_2 = cell_2.zero_state(self.batch_size, 2, tf.float32)
      # print(videoslides.get_shape().as_list())
      for indexframe in range(self.framenum):
          frame = videoslides[:, indexframe, ...]
          # print(indexframe+self.gapnum)
          frame_gap = videoslides[:, indexframe + self.gapnum, ...]
          GTframe = GTs[:, indexframe, ...]
          Yolo_features = self.YOLO_tiny_inference(frame)
          Presalmap = self.Coarse_salmap(Yolo_features)
          if self.startflagcnn == True:
              self.yolofeatures_colllection = self.pretrain_var_collection
              self.pretrain_var_collection = []
          salmask = self._normlized_0to1(Presalmap)
          salmask = salmask * (1 - self.salmask_lb) + self.salmask_lb
          # salmask = tf.ones_like(salmask)
          if self.version_flow == 1:
              Flow_features = self.flownet_with_conv(frame, frame_gap, salmask)
          elif self.version_flow == 2:
              Flow_features = self.flownet20(frame, frame_gap, salmask)
          CNNout = self.Final_inference(Yolo_features, Flow_features)
          if self.startflagcnn == True:
              self.flowfeatures_colllection = self.pretrain_var_collection
          y_1, new_state_1 = cell_1(CNNout, new_state_1, mask_in[..., 0:4], mask_h[..., 0:4], self.dp_in, self.dp_h,
                                    'lstm_layer1', self.dropouttype)
          y_2, new_state_2 = cell_2(y_1, new_state_2, mask_in[..., 4:8], mask_h[..., 4:8], self.dp_in, self.dp_h,
                                    'lstm_layer2', self.dropouttype)
          deconv = self.transpose_conv_layer('deconv', y_2, 3, 16, stride=2, pretrain=False, trainable=True)
          deconv2 = self.transpose_conv_layer('deconv2', deconv, 3, 1, stride=1, linear=True, pretrain=False,
                                              trainable=True)
          if self.startflagcnn == True:
              tf.get_variable_scope().reuse_variables()
              self.trainable_var_collection.extend(cell_1.trainable_var_collection)
              self.trainable_var_collection.extend(cell_2.trainable_var_collection)
              self.startflagcnn = False
          output = self._normlized_0to1(deconv2)
          norm_GT = self._normlized(GTframe)
          outdis = self._MapHistogram(output, self.disnum)
          gtdis = self._MapHistogram(norm_GT, self.disnum)
          outdis = tf.expand_dims(outdis, 1)
          gtdis = tf.expand_dims(gtdis, 1)
          norm_output = self._normlized(output)
          frame_loss = norm_GT * tf.log(self.eps + norm_GT / (norm_output + self.eps))
          frame_loss = tf.reduce_sum(frame_loss)
          tf.add_to_collection('losses', frame_loss)
          output = tf.expand_dims(output, 1)
          # print(output.get_shape().as_list())
          if indexframe == 0:
              tempout = output
              outdisBatch = outdis
              gtdisBatch = gtdis
          else:
              tempout = tf.concat([tempout, output], axis=1)
              outdisBatch = tf.concat([outdisBatch, outdis], axis=1)
              gtdisBatch = tf.concat([gtdisBatch, gtdis], axis=1)
      self.out = tempout
      self.outHis = outdisBatch
      self.gtHis = gtdisBatch

  def _MapHistogram(self,inputmap,binNum = 10):
      shapes = inputmap.get_shape().as_list()
      assert  len(shapes) == 4
      inputmap = tf.cast(inputmap * binNum, tf.int32)
      dismap = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.equal(inputmap,0),dtype = tf.float32),axis = 1), axis = 1)
      for i in range(binNum-1):
          dismap = tf.concat([dismap, tf.reduce_sum(tf.reduce_sum(tf.cast(tf.equal(inputmap,i + 1),dtype = tf.float32),axis = 1), axis = 1)],axis = 1)
      dismap = dismap / tf.reduce_sum(dismap,axis = 1)
      return dismap

  def _normlized(self, mat): # tensor [batch_size, image_height, image_width, channels] normalize each fea map
    mat_shape = mat.get_shape().as_list()
    tempsum = tf.reduce_sum(mat, axis=1)
    tempsum = tf.reduce_sum(tempsum, axis=1) + self.eps
    tempsum = tf.reshape(tempsum, [-1, 1, 1, mat_shape[3]])
    return mat / tempsum

  def _normlized_0to1(self, mat): # tensor [batch_size, image_height, image_width, channels] normalize each fea map
    mat_shape = mat.get_shape().as_list()
    tempmin = tf.reduce_min(mat, axis=1)
    tempmin= tf.reduce_min(tempmin, axis=1)
    tempmin = tf.reshape(tempmin, [-1, 1, 1, mat_shape[3]])
    tempmat = mat - tempmin
    tempmax = tf.reduce_max(tempmat, axis=1)
    tempmax = tf.reduce_max(tempmax, axis=1) + self.eps
    tempmax = tf.reshape(tempmax, [-1, 1, 1, mat_shape[3]])
    return tempmat / tempmax

  def _disloss(self, outdis, gtdis, distype = 'KL', inputloss = None):
      shapes = outdis.get_shape().as_list()
      assert  len(shapes) == 3
      # print(shapes)
      # print(gtdis.get_shape().as_list())
      # assert  shapes == gtdis.get_shape().as_list()
      if  distype == 'KL':
          dis_loss = gtdis * tf.log(self.eps + gtdis / (outdis + self.eps))
          loss = tf.reduce_sum(dis_loss)
      elif distype == 'dualKL':
          adis = (outdis + gtdis) / 2
          dis_loss = (outdis * tf.log(self.eps + outdis / (adis + self.eps)) +  gtdis * tf.log(self.eps + gtdis / (adis + self.eps))) / 2
          loss = tf.reduce_sum(dis_loss)
      else:
          loss = tf.reduce_sum(inputloss)
      return loss




  def _loss(self, exloss):
    weight_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=None)
    loss_weight = tf.add_n(weight_loss)
    loss_kl = tf.get_collection('losses', scope=None)
    loss_kl = tf.add_n(loss_kl)/(self.framenum * self.batch_size)
    loss_dis = self.lambdadis * self._disloss(self.outHis, self.gtHis, distype = self.distype ,inputloss = exloss) / (self.framenum * self.batch_size)
    # self.out = self.predict
    self.loss_gt = loss_kl
    self.loss_w = loss_weight
    self.loss_sparse = loss_dis
    self.loss = loss_kl + loss_weight + loss_dis
    tf.summary.scalar('loss_weight', loss_weight)
    tf.summary.scalar('loss_kl', loss_kl)
    tf.summary.scalar('loss_dis', loss_dis)
    tf.summary.scalar('loss_all', self.loss)


  def _lossmul(self, exloss):
    weight_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=None)
    loss_weight = tf.add_n(weight_loss)
    loss_kl = tf.get_collection('losses', scope=None)
    loss_kl = tf.add_n(loss_kl)/(self.framenum * self.batch_size)
    loss_dis = self._disloss(self.outHis, self.gtHis, distype = self.distype ,inputloss = exloss) / (self.framenum * self.batch_size)
    # self.out = self.predict
    self.loss_gt = loss_kl
    self.loss_w = loss_weight
    self.loss_sparse = loss_dis
    self.loss = (loss_kl + loss_weight) * (1/loss_dis) * self.factordis
    tf.summary.scalar('loss_weight', loss_weight)
    tf.summary.scalar('loss_kl', loss_kl)
    tf.summary.scalar('loss_dis', loss_dis)
    tf.summary.scalar('loss_all', self.loss)

  def _train(self):
    # learning_rate = tf.train.exponential_decay(self.init_learning_rate, self.global_step,
    #                                            100000, 0.95, staircase=True)
    # with tf.variable_scope('trainer'):
        opt = tf.train.AdamOptimizer(self.init_learning_rate,beta1=0.9, beta2=0.999, epsilon=1e-08)
        # for var in self.trainable_var_collection:
        #   print(var.op.name)
        grads = opt.compute_gradients(self.loss,var_list = self.trainable_var_collection)
        apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)
        #apply_gradient_op = tf.train.AdamOptimizer(self.init_learning_rate).minimize(self.loss)
        self.train = apply_gradient_op
        return apply_gradient_op