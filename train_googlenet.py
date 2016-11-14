import numpy as np
import chainer.functions as F
from chainer import optimizers, serializers, cuda, Variable
import chainer.links as L
from tuned_googlenet import GoogLeNet

googlenet = GoogLeNet()
serializers.load_npz('tuned_googlenet.model', googlenet)

# forward backward update...
