import argparse
import urllib.request
import numpy as np
import chainer.functions as F
from chainer import initializers, optimizers, serializers, cuda, Variable
import chainer.links as L
from googlenet import GoogLeNet
import cv2

def main(url='http://find-travel.cdn-dena.com/picture/articlebody/38214', gpu_num = -1):
    print('Initialized GoogLeNet Model ....')
    googlenet = GoogLeNet()

    print('Load Model Parameter ....')
    serializers.load_npz('googlenet.model', googlenet)

    image_file_path = './sample_images/sample.jpg'
    print('Download Image From {0} ....'.format(url))
    urllib.request.urlretrieve(url, image_file_path)

    # use gpu
    if gpu_num > 0:
         cuda.get_device(gpu_num).use()
         googlenet.to_gpu()

    # load image
    img = cv2.resize(cv2.imread(image_file_path), (224, 224)).astype(np.float32)
    img = img.transpose(2, 0, 1).reshape(1, 3, 224, 224)

    # forward
    x = Variable(img)
    googlenet.train = False
    y = googlenet(x)

    # show prediction
    prediction = F.softmax(y)
    categories = np.loadtxt('synset_words.txt', delimiter="\n", dtype=str)
    result = zip(prediction.data.reshape((prediction.data.size,)), categories)
    result = sorted(result, reverse=True)
    print('--------- predict ---------------------------------------------------------------')
    for i, (score, label) in enumerate(result[:10]):
        print('{:>3d} {:>6.2f}% {}'.format(i + 1, score * 100, label))
    print('---------------------------------------------------------------------------------')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ILSVRC-2014のdatasetで訓練済みのGoogLeNetで、1000クラス分類を試す')
    parser.add_argument('--url', '-u', default='http://find-travel.cdn-dena.com/picture/articlebody/38214', help='ダウンロードするイメージのURLを指定する')
    parser.add_argument('--gpu_num', '-g', type=int, default=-1, help='GPUの番号')
    args = parser.parse_args()

    main(url = args.url, gpu_num = args.gpu_num)
