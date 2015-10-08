import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from lasagne import layers, nonlinearities, updates, init, objectives
import numpy as np
from nolearn.lasagne.handlers import EarlyStopping
from itertools import repeat


def sample_from_rotation_x( x ):
    x_extends = []
    y_extends = []
    for i in range(x.shape[0]):
        x_extends.extend([
        np.array([x[i,:,:,0], x[i,:,:,1], x[i,:,:,2]]),
        np.array([np.rot90(x[i,:,:,0]),np.rot90(x[i,:,:,1]), np.rot90(x[i,:,:,2])]),
        np.array([np.rot90(x[i,:,:,0],2),np.rot90(x[i,:,:,1],2), np.rot90(x[i,:,:,2],2)]),
        np.array([np.rot90(x[i,:,:,0],3),np.rot90(x[i,:,:,1],3), np.rot90(x[i,:,:,2],3)])
        ])
    return np.array(x_extends)
 
def sample_from_rotation_y(y):
    y_extends = []
    for i in y:
        y_extends.extend( repeat( i ,4) )
    return np.array(y_extends)


class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1, :]
        return Xb, yb


def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('dropout4', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('dropout5', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64, 64),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(4, 4),
    hidden4_num_units=500, hidden4_W=init.GlorotUniform(gain='relu'),
    dropout4_p=0.5,
    hidden5_num_units=500, hidden5_W=init.GlorotUniform(gain='relu'), 
    hidden5_nonlinearity=nonlinearities.very_leaky_rectify ,
    dropout5_p=0.5,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax, output_W=init.GlorotUniform(),
    update_learning_rate=0.03,
    update_momentum=0.9,
    max_epochs=200,
    on_epoch_finished=[EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)],
    batch_iterator_train=FlipBatchIterator(batch_size=256)
)


class Classifier(BaseEstimator):
    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess_train(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = sample_from_rotation_x( X )
        return X

    def preprocess_test(self, X):
        X = X.transpose((0, 3, 1, 2))
        return X

    def preprocess_y(self, y):
        y = sample_from_rotation_y(y)
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess_train(X)
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess_test(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess_test(X)
        return self.net.predict_proba(X)
