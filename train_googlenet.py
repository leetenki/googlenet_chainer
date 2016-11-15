import numpy as np
import chainer.functions as F
from chainer import optimizers, serializers, cuda, Variable
import chainer.links as L
from tuned_googlenet import GoogLeNet

# load googlenet
googlenet = GoogLeNet()
serializers.load_npz('tuned_googlenet.model', googlenet)

# optimizer
optimizer = optimizers.Adam()
optimizer.use_cleargrads()
optimizer.setup(googlenet)

# forward backward update...
# y, loss, accuracy = googlenet(x, t)
# optimizer.zero_grads()
# loss.backward()
# optimizer.update()
