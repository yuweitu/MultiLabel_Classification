import cv2
import numpy as np
from tqdm import tqdm
import h5py as h5py
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
import argparse
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input


def get_features(MODEL, data, args):

    cnn_model = MODEL(include_top=False, input_shape=(args.input_size, args.input_size, 3), weights='imagenet')

    inputs = Input((args.input_size, args.input_size, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs, x)

    features = cnn_model.predict(data, batch_size=args.batch, verbose=1)
    return features


def get_features_combine(model_list, data, args):
    feature_list = []
    for model in model_list:
        feature = get_features(model, data, args)
        feature_list.append(feature)
    features = np.concatenate(feature_list, axis=-1)
    return features

def main(args):

    df = pd.read_csv('input/labels.csv')

    n = len(df)
    breed = set(df['breed'])
    class_to_num = dict(zip(breed, range(args.num_classes)))
    num_to_class = dict(zip(range(args.num_classes), breed))

    X = np.zeros((n, args.input_size, args.input_size, 3), dtype=np.uint8)
    y = np.zeros((n, args.num_classes), dtype=np.uint8)
    for i in tqdm(range(n)):
        X[i] = cv2.resize(cv2.imread('input/train/%s.jpg' % df['id'][i]), (args.input_size, args.input_size))
        y[i][class_to_num[df['breed'][i]]] = 1

    if args.sample:
        X = X[:100]
        y = y[:100]

    # feature engineering
    model_list = [Xception, VGG16, VGG19, ResNet50, InceptionV3, InceptionResNetV2, MobileNet]
    features = get_features_combine(model_list, X, args)
    np.save('feature.npy', features)

    inputs = Input(features.shape[1:])
    x = inputs
    x = Dropout(0.1)(x)
    x = Dense(args.num_classes, activation='softmax')(x)
    model = Model(inputs, x)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    h = model.fit(features, y, batch_size=args.batch, epochs=args.epochs, validation_split=args.split)
    print('Training loss history:\n ', h.history['loss'])
    print('Training accuracy history: \n ', h.history['acc'])
    print('Validation loss history: \n', h.history['val_loss'])
    print('Validation accuracy history: \n', h.history['val_acc'])

    # test
    df2 = pd.read_csv('input/sample_submission.csv')
    n_test = len(df2)
    X_test = np.zeros((n_test, args.input_size, args.input_size, 3), dtype=np.uint8)
    for i in tqdm(range(n_test)):
        X_test[i] = cv2.resize(cv2.imread('input/test/%s.jpg' % df2['id'][i]), (args.input_size,  args.input_size))

    features_test = get_features_combine(model_list, X_test, args)
    y_pred = model.predict(features_test, batch_size=args.batch)
    for b in breed:
        df2[b] = y_pred[:, class_to_num[b]]
    df2.to_csv('pred.csv', index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', type=int, default=128)
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-seed', type=int, default=123)
    parser.add_argument('-split', type=float, default=0.1)
    parser.add_argument('-num_classes', type=float, default=120)
    parser.add_argument('-input_size', type=int, default=224)
    parser.add_argument('-data_dir', type=str, default = 'input/')
    parser.add_argument('-sample', type=bool, default=True)
    args = parser.parse_args()
    args.sample = False
    main(args)