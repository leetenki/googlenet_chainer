from chainer.links.caffe import CaffeFunction
from chainer import initializers, optimizers, serializers, Variable
from tuned_googlenet import GoogLeNet
import _pickle as pickle

print('start loading model file...')
caffe_model = CaffeFunction('bvlc_googlenet.caffemodel')
print('done.')

# copy parameters from caffemodel into chainer model
print('start copy params.')
googlenet = GoogLeNet()
googlenet.conv1.W.data = caffe_model['conv1/7x7_s2'].W.data
googlenet.conv1.b.data = caffe_model['conv1/7x7_s2'].b.data

googlenet.conv2_reduce.W.data = caffe_model['conv2/3x3_reduce'].W.data
googlenet.conv2_reduce.b.data = caffe_model['conv2/3x3_reduce'].b.data

googlenet.conv2.W.data = caffe_model['conv2/3x3'].W.data
googlenet.conv2.b.data = caffe_model['conv2/3x3'].b.data

googlenet.inc3a.conv1.W.data = caffe_model['inception_3a/1x1'].W.data
googlenet.inc3a.conv1.b.data = caffe_model['inception_3a/1x1'].b.data
googlenet.inc3a.conv3.W.data = caffe_model['inception_3a/3x3'].W.data
googlenet.inc3a.conv3.b.data = caffe_model['inception_3a/3x3'].b.data
googlenet.inc3a.conv5.W.data = caffe_model['inception_3a/5x5'].W.data
googlenet.inc3a.conv5.b.data = caffe_model['inception_3a/5x5'].b.data
googlenet.inc3a.proj3.W.data = caffe_model['inception_3a/3x3_reduce'].W.data
googlenet.inc3a.proj3.b.data = caffe_model['inception_3a/3x3_reduce'].b.data
googlenet.inc3a.proj5.W.data = caffe_model['inception_3a/5x5_reduce'].W.data
googlenet.inc3a.proj5.b.data = caffe_model['inception_3a/5x5_reduce'].b.data
googlenet.inc3a.projp.W.data = caffe_model['inception_3a/pool_proj'].W.data
googlenet.inc3a.projp.b.data = caffe_model['inception_3a/pool_proj'].b.data

googlenet.inc3b.conv1.W.data = caffe_model['inception_3b/1x1'].W.data
googlenet.inc3b.conv1.b.data = caffe_model['inception_3b/1x1'].b.data
googlenet.inc3b.conv3.W.data = caffe_model['inception_3b/3x3'].W.data
googlenet.inc3b.conv3.b.data = caffe_model['inception_3b/3x3'].b.data
googlenet.inc3b.conv5.W.data = caffe_model['inception_3b/5x5'].W.data
googlenet.inc3b.conv5.b.data = caffe_model['inception_3b/5x5'].b.data
googlenet.inc3b.proj3.W.data = caffe_model['inception_3b/3x3_reduce'].W.data
googlenet.inc3b.proj3.b.data = caffe_model['inception_3b/3x3_reduce'].b.data
googlenet.inc3b.proj5.W.data = caffe_model['inception_3b/5x5_reduce'].W.data
googlenet.inc3b.proj5.b.data = caffe_model['inception_3b/5x5_reduce'].b.data
googlenet.inc3b.projp.W.data = caffe_model['inception_3b/pool_proj'].W.data
googlenet.inc3b.projp.b.data = caffe_model['inception_3b/pool_proj'].b.data

googlenet.inc4a.conv1.W.data = caffe_model['inception_4a/1x1'].W.data
googlenet.inc4a.conv1.b.data = caffe_model['inception_4a/1x1'].b.data
googlenet.inc4a.conv3.W.data = caffe_model['inception_4a/3x3'].W.data
googlenet.inc4a.conv3.b.data = caffe_model['inception_4a/3x3'].b.data
googlenet.inc4a.conv5.W.data = caffe_model['inception_4a/5x5'].W.data
googlenet.inc4a.conv5.b.data = caffe_model['inception_4a/5x5'].b.data
googlenet.inc4a.proj3.W.data = caffe_model['inception_4a/3x3_reduce'].W.data
googlenet.inc4a.proj3.b.data = caffe_model['inception_4a/3x3_reduce'].b.data
googlenet.inc4a.proj5.W.data = caffe_model['inception_4a/5x5_reduce'].W.data
googlenet.inc4a.proj5.b.data = caffe_model['inception_4a/5x5_reduce'].b.data
googlenet.inc4a.projp.W.data = caffe_model['inception_4a/pool_proj'].W.data
googlenet.inc4a.projp.b.data = caffe_model['inception_4a/pool_proj'].b.data

googlenet.inc4b.conv1.W.data = caffe_model['inception_4b/1x1'].W.data
googlenet.inc4b.conv1.b.data = caffe_model['inception_4b/1x1'].b.data
googlenet.inc4b.conv3.W.data = caffe_model['inception_4b/3x3'].W.data
googlenet.inc4b.conv3.b.data = caffe_model['inception_4b/3x3'].b.data
googlenet.inc4b.conv5.W.data = caffe_model['inception_4b/5x5'].W.data
googlenet.inc4b.conv5.b.data = caffe_model['inception_4b/5x5'].b.data
googlenet.inc4b.proj3.W.data = caffe_model['inception_4b/3x3_reduce'].W.data
googlenet.inc4b.proj3.b.data = caffe_model['inception_4b/3x3_reduce'].b.data
googlenet.inc4b.proj5.W.data = caffe_model['inception_4b/5x5_reduce'].W.data
googlenet.inc4b.proj5.b.data = caffe_model['inception_4b/5x5_reduce'].b.data
googlenet.inc4b.projp.W.data = caffe_model['inception_4b/pool_proj'].W.data
googlenet.inc4b.projp.b.data = caffe_model['inception_4b/pool_proj'].b.data

googlenet.inc4c.conv1.W.data = caffe_model['inception_4c/1x1'].W.data
googlenet.inc4c.conv1.b.data = caffe_model['inception_4c/1x1'].b.data
googlenet.inc4c.conv3.W.data = caffe_model['inception_4c/3x3'].W.data
googlenet.inc4c.conv3.b.data = caffe_model['inception_4c/3x3'].b.data
googlenet.inc4c.conv5.W.data = caffe_model['inception_4c/5x5'].W.data
googlenet.inc4c.conv5.b.data = caffe_model['inception_4c/5x5'].b.data
googlenet.inc4c.proj3.W.data = caffe_model['inception_4c/3x3_reduce'].W.data
googlenet.inc4c.proj3.b.data = caffe_model['inception_4c/3x3_reduce'].b.data
googlenet.inc4c.proj5.W.data = caffe_model['inception_4c/5x5_reduce'].W.data
googlenet.inc4c.proj5.b.data = caffe_model['inception_4c/5x5_reduce'].b.data
googlenet.inc4c.projp.W.data = caffe_model['inception_4c/pool_proj'].W.data
googlenet.inc4c.projp.b.data = caffe_model['inception_4c/pool_proj'].b.data

googlenet.inc4d.conv1.W.data = caffe_model['inception_4d/1x1'].W.data
googlenet.inc4d.conv1.b.data = caffe_model['inception_4d/1x1'].b.data
googlenet.inc4d.conv3.W.data = caffe_model['inception_4d/3x3'].W.data
googlenet.inc4d.conv3.b.data = caffe_model['inception_4d/3x3'].b.data
googlenet.inc4d.conv5.W.data = caffe_model['inception_4d/5x5'].W.data
googlenet.inc4d.conv5.b.data = caffe_model['inception_4d/5x5'].b.data
googlenet.inc4d.proj3.W.data = caffe_model['inception_4d/3x3_reduce'].W.data
googlenet.inc4d.proj3.b.data = caffe_model['inception_4d/3x3_reduce'].b.data
googlenet.inc4d.proj5.W.data = caffe_model['inception_4d/5x5_reduce'].W.data
googlenet.inc4d.proj5.b.data = caffe_model['inception_4d/5x5_reduce'].b.data
googlenet.inc4d.projp.W.data = caffe_model['inception_4d/pool_proj'].W.data
googlenet.inc4d.projp.b.data = caffe_model['inception_4d/pool_proj'].b.data

googlenet.inc4e.conv1.W.data = caffe_model['inception_4e/1x1'].W.data
googlenet.inc4e.conv1.b.data = caffe_model['inception_4e/1x1'].b.data
googlenet.inc4e.conv3.W.data = caffe_model['inception_4e/3x3'].W.data
googlenet.inc4e.conv3.b.data = caffe_model['inception_4e/3x3'].b.data
googlenet.inc4e.conv5.W.data = caffe_model['inception_4e/5x5'].W.data
googlenet.inc4e.conv5.b.data = caffe_model['inception_4e/5x5'].b.data
googlenet.inc4e.proj3.W.data = caffe_model['inception_4e/3x3_reduce'].W.data
googlenet.inc4e.proj3.b.data = caffe_model['inception_4e/3x3_reduce'].b.data
googlenet.inc4e.proj5.W.data = caffe_model['inception_4e/5x5_reduce'].W.data
googlenet.inc4e.proj5.b.data = caffe_model['inception_4e/5x5_reduce'].b.data
googlenet.inc4e.projp.W.data = caffe_model['inception_4e/pool_proj'].W.data
googlenet.inc4e.projp.b.data = caffe_model['inception_4e/pool_proj'].b.data

googlenet.inc5a.conv1.W.data = caffe_model['inception_5a/1x1'].W.data
googlenet.inc5a.conv1.b.data = caffe_model['inception_5a/1x1'].b.data
googlenet.inc5a.conv3.W.data = caffe_model['inception_5a/3x3'].W.data
googlenet.inc5a.conv3.b.data = caffe_model['inception_5a/3x3'].b.data
googlenet.inc5a.conv5.W.data = caffe_model['inception_5a/5x5'].W.data
googlenet.inc5a.conv5.b.data = caffe_model['inception_5a/5x5'].b.data
googlenet.inc5a.proj3.W.data = caffe_model['inception_5a/3x3_reduce'].W.data
googlenet.inc5a.proj3.b.data = caffe_model['inception_5a/3x3_reduce'].b.data
googlenet.inc5a.proj5.W.data = caffe_model['inception_5a/5x5_reduce'].W.data
googlenet.inc5a.proj5.b.data = caffe_model['inception_5a/5x5_reduce'].b.data
googlenet.inc5a.projp.W.data = caffe_model['inception_5a/pool_proj'].W.data
googlenet.inc5a.projp.b.data = caffe_model['inception_5a/pool_proj'].b.data

googlenet.inc5b.conv1.W.data = caffe_model['inception_5b/1x1'].W.data
googlenet.inc5b.conv1.b.data = caffe_model['inception_5b/1x1'].b.data
googlenet.inc5b.conv3.W.data = caffe_model['inception_5b/3x3'].W.data
googlenet.inc5b.conv3.b.data = caffe_model['inception_5b/3x3'].b.data
googlenet.inc5b.conv5.W.data = caffe_model['inception_5b/5x5'].W.data
googlenet.inc5b.conv5.b.data = caffe_model['inception_5b/5x5'].b.data
googlenet.inc5b.proj3.W.data = caffe_model['inception_5b/3x3_reduce'].W.data
googlenet.inc5b.proj3.b.data = caffe_model['inception_5b/3x3_reduce'].b.data
googlenet.inc5b.proj5.W.data = caffe_model['inception_5b/5x5_reduce'].W.data
googlenet.inc5b.proj5.b.data = caffe_model['inception_5b/5x5_reduce'].b.data
googlenet.inc5b.projp.W.data = caffe_model['inception_5b/pool_proj'].W.data
googlenet.inc5b.projp.b.data = caffe_model['inception_5b/pool_proj'].b.data

#googlenet.loss3_fc.W.data = caffe_model['loss3/classifier'].W.data
#googlenet.loss3_fc.b.data = caffe_model['loss3/classifier'].b.data

#googlenet.loss1_conv.W.data = caffe_model['loss1/conv'].W.data
#googlenet.loss1_conv.b.data = caffe_model['loss1/conv'].b.data
#googlenet.loss1_fc1.W.data = caffe_model['loss1/fc'].W.data
#googlenet.loss1_fc1.b.data = caffe_model['loss1/fc'].b.data

#googlenet.loss2_conv.W.data = caffe_model['loss2/conv'].W.data
#googlenet.loss2_conv.b.data = caffe_model['loss2/conv'].b.data
#googlenet.loss2_fc1.W.data = caffe_model['loss2/fc'].W.data
#googlenet.loss2_fc1.b.data = caffe_model['loss2/fc'].b.data

serializers.save_npz('tuned_googlenet.model', googlenet)
print('done')
