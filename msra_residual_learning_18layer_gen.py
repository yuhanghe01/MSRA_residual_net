#this file is implemented to msra Residual Learning 18 layers network for Caffe
# author: Yuhang He
# email: heyuhang@dress-plus.com
# date: Jan. 3, 2016

f = file("msra_residualnet_18layers.prototxt", 'w+')
name = "\nname: \"msra_residual_learning_18layers\"\n"
data_layer = "\nlayer{\n\
  name: \"data\"\n\
  type: \"Data\"\n\
  top: \"data\"\n\
  top: \"label\"\n\
  include{\n\
    phase:TRAIN\n\
  }\n\
  transform_param{\n\
    mirror: true\n\
    crop_size: 224\n\
    mean_value: 128\n\
    mean_value: 128\n\
    mean_value: 128\n\
  }\n\
  data_param{\n\
    source: \"/mnt/lvm/data/imagenet2012/train.leveldb\"\n\
    batch_size: 32\n\
    backend: LEVELDB\n\
  }\n\
}\n"

convolution_layer = "\nlayer{\n\
  name: \"%s\"\n\
  type: \"Convolution\"\n\
  bottom: \"%s\"\n\
  top: \"%s\"\n\
  convolution_param{\n\
    num_output: %d\n\
    kernel_size: %d\n\
    stride: %d\n\
    pad: %d\n\
    weight_filler{\n\
      type: \"xavier\"\n\
      std: 0.01\n\
    }\n\
    bias_filler{\n\
      type: \"constant\"\n\
      value: 0.2\n\
    }\n\
  }\n\
}\n"

batch_norm_layer = "\n\
layer{\n\
  name: \"%s\"\n\
  type: \"BatchNorm\"\n\
  bottom: \"%s\"\n\
  top: \"%s\"\n\
}\n"

relu_layer = "\n\
layer{\n\
  name: \"%s\"\n\
  type: \"ReLU\"\n\
  bottom: \"%s\"\n\
  top: \"%s\"\n\
}\n"

pooling_layer = "\n\
layer{\n\
  name: \"%s\"\n\
  type: \"Pooling\"\n\
  bottom: \"%s\"\n\
  top: \"%s\"\n\
  pooling_param{\n\
    pool: MAX\n\
    stride: %d\n\
    kernel_size: %d\n\
  }\n\
}\n"


ave_pooling_layer = "\n\
layer{\n\
  name: \"%s\"\n\
  type: \"Pooling\"\n\
  bottom: \"%s\"\n\
  top: \"%s\"\n\
  pooling_param{\n\
    pool: AVE\n\
    stride: %d\n\
    kernel_size: %d\n\
  }\n\
}\n"


eltwise_layer = "\n\
layer{\n\
  name: \"%s\"\n\
  type: \"Eltwise\"\n\
  bottom: \"%s\"\n\
  bottom: \"%s\"\n\
  top: \"%s\"\n\
  eltwise_param{\n\
    operation: SUM\n\
  }\n\
}\n"

fullconnect_layer = "\n\
layer{\n\
        name: \"%s\"\n\
        type: \"InnerProduct\"\n\
        bottom: \"%s\"\n\
        top: \"%s\"\n\
    param {\n\
      lr_mult: 1\n\
      decay_mult: 1\n\
    }\n\
    param {\n\
      lr_mult: 2\n\
      decay_mult: 0\n\
    }\n\
    inner_product_param {\n\
       num_output: %d\n\
       weight_filler {\n\
       type: \"xavier\"\n\
    }\n\
    bias_filler {\n\
      type: \"constant\"\n\
      value: 0\n\
    }\n\
  }\n\
}\n"

loss_layer = "\n\
layer{\n\
    name: \"%s\"\n\
    type: \"SoftmaxWithLoss\"\n\
    bottom: \"%s\"\n\
    bottom: \"%s\"\n\
    top: \"%s\"\n\
}\n"

accuracy_layer = "\n\
layer{\n\
    name: \"%s\"\n\
    type: \"Accuracy\"\n\
    bottom: \"%s\"\n\
    bottom: \"%s\"\n\
    top: \"%s\"\n\
    include{\n\
      phase: TEST\n\
    }\n\
}\n"

#write author and other relevant info
f.writelines("#prototxt for msra residual learning 18 layers for caffe framework\n\
#author: Yuhang He\n\
#date: Jan. 3, 2016\n\
#email: heyuhang@dress-plus.com\n ")
#write the conv1 layer

f.writelines( name )
f.writelines( data_layer )
f.writelines( data_layer )
f.writelines( convolution_layer%("conv1", "data", "conv1", 64, 7, 2, 3 ) )
f.writelines( batch_norm_layer%("batch_norm1", "conv1", "batch_norm1") )
f.writelines( pooling_layer%("pooling1", "batch_norm1", "pooling1", 2, 3) )

conv_bottom_name = ""
for i in range(1,3):
  print "---writing redisual layer %d --- "%i
  f.writelines("\n#residual_layer%d"%(i))
  conv_layer_name = "residual%d_conv1"%(i)
  if( i == 1 ):
    conv_bottom_name = "pooling1"
  conv_top_name = conv_layer_name
  f.writelines( convolution_layer%( conv_layer_name, conv_bottom_name, conv_top_name, 64, 3, 1, 1 ) )
  bn_layer_name = conv_layer_name + "_bn1"
  f.writelines( batch_norm_layer%( bn_layer_name, conv_top_name, bn_layer_name ) )
  relu_layer_name = bn_layer_name + "_relu1"
  f.writelines( relu_layer%( relu_layer_name, bn_layer_name, relu_layer_name))
  
  conv_layer_name = "residual%d_conv2"%(i)
  f.writelines( convolution_layer%( conv_layer_name, relu_layer_name, conv_layer_name, 64, 3, 1, 1 ) )
  bn_layer_name = conv_layer_name + "_bn2"
  #f.writelines( batch_norm_layer%( bn_layer_name, conv_top_name, bn_layer_name ) )
  f.writelines( batch_norm_layer%( bn_layer_name, conv_layer_name, bn_layer_name ) )
  relu_layer_name = bn_layer_name + "_relu2"
  f.writelines( relu_layer%( relu_layer_name, bn_layer_name, relu_layer_name))
  
  """
  conv_layer_name = "residual%d_conv3"%(i)
  f.writelines( convolution_layer%( conv_layer_name, relu_layer_name, conv_layer_name, 256, 1, 1, 0 ) )
  bn_layer_name = conv_layer_name + "_bn3"
  #f.writelines( batch_norm_layer%( bn_layer_name, conv_top_name, bn_layer_name ) )
  f.writelines( batch_norm_layer%( bn_layer_name, conv_layer_name, bn_layer_name ) )
  relu_layer_name = bn_layer_name + "_relu3"
  f.writelines( relu_layer%( relu_layer_name, bn_layer_name, relu_layer_name))

  #write the dimension matching layer
  if(i == 1):
    dimen_match_name = "residual%d_dimen_match"%(i)
    if( i == 1 ):
      dimen_match_bottom = "pooling1"
    f.writelines( convolution_layer%( dimen_match_name, dimen_match_bottom, dimen_match_name, 256, 1, 1, 0) )
  """
  eletwise_name = "residual%d_eletwise"%(i)
  if( i == 1 ):
    #eletwise_bottom_name1 = dimen_match_bottom
    eletwise_bottom_name1 = "pooling1"
  eletwise_bottom_name2 = relu_layer_name
  f.writelines( eltwise_layer%( eletwise_name, eletwise_bottom_name1, eletwise_bottom_name2, eletwise_name ) )
  
  residual_relu_name = "residual%d_relu"%i
  f.writelines( relu_layer%(residual_relu_name, eletwise_name, residual_relu_name) )
  
  #dimen_match_bottom = redisual_relu_name
  eletwise_bottom_name1 = residual_relu_name
  conv_bottom_name = residual_relu_name
 
eletwise_bottom_name1 = residual_relu_name

for i in range(3,5):
  print "---writing redisual layer %d --- "%i
  f.writelines("\n#residual_layer%d"%(i))
  conv_layer_name = "residual%d_conv1"%(i)
  if( i == 3 ):
    conv_layer_bottom_name = residual_relu_name
  f.writelines( convolution_layer%( conv_layer_name, conv_layer_bottom_name, conv_layer_name, 128, 3, 2, 1) )
  #eletwise_bottom_name1 = conv_layer_name
    
  bn_layer_name = conv_layer_name + "_bn1"
  bn_layer_bottom = conv_layer_name
  f.writelines( batch_norm_layer%( bn_layer_name, bn_layer_bottom, bn_layer_name ) )
    
  relu_layer_name = bn_layer_name + "_relu1"
  f.writelines( relu_layer%( relu_layer_name, bn_layer_name, relu_layer_name ))
    
  conv_layer_name = "residual%d_conv2"%(i)
  f.writelines( convolution_layer%( conv_layer_name, relu_layer_name, conv_layer_name, 128, 3, 1, 1 ) )
  
  #bn_layer_name = conv_layer_name + "_bn2"
  bn_layer_name = conv_layer_name + "_bn2"
  f.writelines( batch_norm_layer%( bn_layer_name, conv_layer_name, bn_layer_name) )
 
  relu_layer_name = bn_layer_name + "_relu2"
  f.writelines( relu_layer%( relu_layer_name, bn_layer_name, relu_layer_name ))

  """
  conv_layer_name = "residual%d_conv3"%(i)
  f.writelines( convolution_layer%( conv_layer_name, relu_layer_name, conv_layer_name, 512, 1, 1, 0 ) )
    
  bn_layer_name = conv_layer_name + "_bn3"
  f.writelines( batch_norm_layer%( bn_layer_name, conv_layer_name, bn_layer_name) )

  relu_layer_name = bn_layer_name + "_relu3"
  f.writelines( relu_layer%( relu_layer_name, bn_layer_name, relu_layer_name ))
  """
  if(i == 3):
    dimen_match_name = "residual%d_match"%i
    f.writelines( convolution_layer%( dimen_match_name, eletwise_bottom_name1, dimen_match_name, 128, 1, 2, 0 ) )
    eletwise_bottom_name1 = dimen_match_name
  
  eletwise_name = "residual%d_eletwise"%i
  eletwise_bottom_name2 = relu_layer_name
  f.writelines( eltwise_layer%( eletwise_name, eletwise_bottom_name1, eletwise_bottom_name2, eletwise_name))
  
  relu_name = "residual%d_relu"%i
  f.writelines( relu_layer%( relu_name, eletwise_name, relu_name ) )
  
  eletwise_bottom_name1 = relu_name
 

eletwise_bottom_name1 = relu_name

for i in range(5, 7):
  print "---writing redisual layer %d --- "%i
  f.writelines("\n#residual_layer%d"%(i)) 
  if( i == 5 ):
    conv_layer_name = "residual%d_conv1"%i
    f.writelines( convolution_layer%( conv_layer_name, relu_name, conv_layer_name, 256, 3, 2, 1 ) )
  #eletwise_bottom_name1 = conv_layer_name
  bn_layer_name = conv_layer_name + "_bn1"
  f.writelines( batch_norm_layer%( bn_layer_name, conv_layer_name, bn_layer_name) )

  relu_layer_name = bn_layer_name + "_relu1"
  f.writelines( relu_layer%( relu_layer_name, bn_layer_name, relu_layer_name ))
  
  conv_layer_name = "residual%d_conv2"%i
  f.writelines( convolution_layer%( conv_layer_name, relu_layer_name, conv_layer_name, 256, 3, 1, 1 ) )

  bn_layer_name = conv_layer_name + "_bn2"
  f.writelines( batch_norm_layer%( bn_layer_name, conv_layer_name, bn_layer_name) )
  
  relu_layer_name = bn_layer_name + "_relu2"
  f.writelines( relu_layer%( relu_layer_name, bn_layer_name, relu_layer_name ))

  """
  conv_layer_name = "residual%d_conv3"%i
  f.writelines( convolution_layer%( conv_layer_name, relu_layer_name, conv_layer_name, 1024, 1, 1, 0 ) )

  bn_layer_name = conv_layer_name + "_bn3"
  f.writelines( batch_norm_layer%( bn_layer_name, conv_layer_name, bn_layer_name) )

  relu_layer_name = bn_layer_name + "_relu3"
  f.writelines( relu_layer%( relu_layer_name, bn_layer_name, relu_layer_name ))
  """
  if( i == 5 ):
    dimen_match_name = "residual%d_match"%i
    f.writelines( convolution_layer%( dimen_match_name, eletwise_bottom_name1, dimen_match_name, 256, 1, 2, 0 ) )
    eletwise_bottom_name1 = dimen_match_name
  
  eletwise_name = "residual%d_eletwise"%i
  eletwise_bottom_name2 = relu_layer_name
  f.writelines( eltwise_layer%( eletwise_name, eletwise_bottom_name1, eletwise_bottom_name2, eletwise_name))
  relu_name = "redisual%d_relu"%i
  f.writelines( relu_layer%( relu_name, eletwise_name, relu_name ) )
  eletwise_bottom_name1 = relu_name


eletwise_bottom_name1 = relu_name

for i in range(7, 9):
  print "---writing redisual layer %d --- "%i
  f.writelines("\n#residual_layer%d"%(i))
  if( i == 7 ):
    conv_layer_name = "residual%d_conv1"%i
    f.writelines( convolution_layer%( conv_layer_name, relu_name, conv_layer_name, 512, 3, 2, 1 ) )
  #eletwise_bottom_name1 = conv_layer_name
  bn_layer_name = conv_layer_name + "_bn1"
  f.writelines( batch_norm_layer%( bn_layer_name, conv_layer_name, bn_layer_name) )

  relu_layer_name = bn_layer_name + "_relu1"
  f.writelines( relu_layer%( relu_layer_name, bn_layer_name, relu_layer_name ))
  
  conv_layer_name = "residual%d_conv2"%i
  f.writelines( convolution_layer%( conv_layer_name, relu_layer_name, conv_layer_name, 512, 3, 1, 1 ) )

  bn_layer_name = conv_layer_name + "_bn2"
  f.writelines( batch_norm_layer%( bn_layer_name, conv_layer_name, bn_layer_name) )
  
  relu_layer_name = bn_layer_name + "_relu2"
  f.writelines( relu_layer%( relu_layer_name, bn_layer_name, relu_layer_name ))
  """
  conv_layer_name = "residual%d_conv3"%i
  f.writelines( convolution_layer%( conv_layer_name, relu_layer_name, conv_layer_name, 2048, 1, 1, 0 ) )

  bn_layer_name = conv_layer_name + "_bn3"
  f.writelines( batch_norm_layer%( bn_layer_name, conv_layer_name, bn_layer_name) )

  relu_layer_name = bn_layer_name + "_relu3"
  f.writelines( relu_layer%( relu_layer_name, bn_layer_name, relu_layer_name ))
  """
  if( i == 7 ):
    dimen_match_name = "residual%d_match"%i
    f.writelines( convolution_layer%( dimen_match_name, eletwise_bottom_name1, dimen_match_name, 512, 1, 2, 0 ) )
    eletwise_bottom_name1 = dimen_match_name
  
  eletwise_name = "residual%d_eletwise"%i
  eletwise_bottom_name2 = relu_layer_name
  f.writelines( eltwise_layer%( eletwise_name, eletwise_bottom_name1, eletwise_bottom_name2, eletwise_name))
  relu_name = conv_layer_name + "_relu4"
  f.writelines( relu_layer%( relu_name, eletwise_name, relu_name ) )
  eletwise_bottom_name1 = relu_name


f.writelines("\n#average pooling layer\n")
ave_pooling_bottom = relu_name
ave_pooling_name = "ave_pooling"
ave_pooling_top = ave_pooling_name
f.writelines( ave_pooling_layer%(ave_pooling_name, ave_pooling_bottom, ave_pooling_top, 1, 7) )

f.writelines("\n#full connection layer\n")
full_connect_name = "full_connect"
full_connect_bottom = ave_pooling_top
full_connect_top = full_connect_name
f.writelines(fullconnect_layer%(full_connect_name, full_connect_bottom, full_connect_top, 6) )



f.writelines("\n#softmax loss layer\n")
loss_layer_name = "softmax_loss_layer"
loss_layer_bottom1 = full_connect_top
loss_layer_bottom2 = "label"
loss_layer_top = loss_layer_name
f.writelines( loss_layer%( loss_layer_name, loss_layer_bottom1, loss_layer_bottom2, loss_layer_top ))

f.writelines("\n#accuracy layer\n")
accuracy_layer_name = "accuracy_layer"
accuracy_layer_bottom1 = full_connect_top
accuracy_layer_bottom2 = "label"
accuracy_layer_top = accuracy_layer_name
f.writelines( accuracy_layer%(accuracy_layer_name, accuracy_layer_bottom1, accuracy_layer_bottom2, accuracy_layer_top ) ) 

f.close()


