import cv2
import numpy as np
from tqdm import tqdm
import h5py as h5py
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input

df = pd.read_csv('input/labels.csv')

n = len(df)
breed = set(df['breed'])
n_class = len(breed)
class_to_num = dict(zip(breed, range(n_class)))
num_to_class = dict(zip(range(n_class), breed))

width = 299
X = np.zeros((n, width, width, 3), dtype=np.uint8)
y = np.zeros((n, n_class), dtype=np.uint8)
for i in tqdm(range(n)):
    X[i] = cv2.resize(cv2.imread('input/train/%s.jpg' % df['id'][i]), (width, width))
    y[i][class_to_num[df['breed'][i]]] = 1


def get_features(MODEL, data=X):
    cnn_model = MODEL(include_top=False, input_shape=(width, width, 3), weights='imagenet')

    inputs = Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs, x)

    features = cnn_model.predict(data, batch_size=64, verbose=1)
    return features


xception_features = get_features(Xception, X)
vgg16_features = get_features(VGG16, X)
vgg19_features = get_features(VGG19, X)
resnet50_features = get_features(ResNet50, X)
inception_features = get_features(InceptionV3, X)
inception_resnet_features = get_features(InceptionResNetV2, X)
mobilenet_features = get_features(MobileNet, X)
features = np.concatenate([xception_features,vgg16_features,vgg19_features, resnet50_features,
                           inception_features,inception_resnet_features, mobilenet_features], axis=-1)


