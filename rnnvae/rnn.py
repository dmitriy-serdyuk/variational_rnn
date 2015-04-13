import numpy as np
import theano

import argparse

from theano import tensor

from blocks.algorithms import GradientDescent, Adam
from blocks.bricks import Softmax, Tanh, Linear
from blocks.bricks.recurrent import SimpleRecurrent, Bidirectional, LSTM
from blocks.initialization import Uniform, IsotropicGaussian, Constant
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.main_loop import MainLoop
from blocks.graph import ComputationGraph
from blocks.extensions import Printing
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.extensions.saveload import Checkpoint, Dump
from blocks.bricks.lookup import LookupTable
from blocks.model import Model

from fuel.datasets import OneBillionWord
from fuel.streams import DataStream
from fuel.transformers import Filter, Mapping, Batch, Padding
from fuel.schemes import ConstantScheme

from dataset import BouncingBalls


all_chars = ([chr(ord('a') + i) for i in range(26)] +
             [chr(ord('0') + i) for i in range(10)] +
             [',', '.', '!', '?', '<UNK>'] +
             [' ', '<S>', '</S>'])
code2char = dict(enumerate(all_chars))
char2code = {v: k for k, v in code2char.items()}


def _lower(s):
    return s.lower()


def _transpose(data):
    return tuple(array.T for array in data)


def _filter_long(data):
    return len(data[0]) <= 100


def _make_target(data):
    dt = np.array(data[0], dtype='int64')
    padding = np.zeros_like(dt)
    return np.concatenate([padding, dt], axis=0),


def _make_feature(data):
    padding = np.zeros_like(data[0])
    return np.concatenate([data[0], padding], axis=0), data[1]


def main(model_path, recurrent_type):
    dataset_options = dict(dictionary=char2code, level="character",
                           preprocess=_lower)
    dataset = OneBillionWord("training", [99], **dataset_options)
    data_stream = dataset.get_example_stream()
    data_stream = Filter(data_stream, _filter_long)
    data_stream = Mapping(data_stream, _make_target,
                          add_sources=('target',))
    data_stream = Mapping(data_stream, _make_feature)
    data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(100))
    data_stream = Padding(data_stream)
    data_stream = Mapping(data_stream, _transpose)

    features = tensor.lmatrix('features')
    features_mask = tensor.matrix('features_mask')
    target = tensor.lmatrix('target')
    target_mask = tensor.matrix('target_mask')

    dim = 100
    lookup = LookupTable(len(all_chars), dim,
                         weights_init=IsotropicGaussian(0.01),
                         biases_init=Constant(0.))

    if recurrent_type == 'lstm':
        rnn = LSTM(dim / 4, Tanh(),
                   weights_init=IsotropicGaussian(0.01),
                   biases_init=Constant(0.))
    elif recurrent_type == 'simple':
        rnn = SimpleRecurrent(dim, Tanh())
        rnn = Bidirectional(rnn,
                            weights_init=IsotropicGaussian(0.01),
                            biases_init=Constant(0.))
    else:
        raise ValueError('Not known RNN type')
    rnn.initialize()
    lookup.initialize()
    y_hat = rnn.apply(lookup.apply(features), mask=features_mask)

    print len(all_chars)
    linear = Linear(2 * dim, len(all_chars),
                    weights_init=IsotropicGaussian(0.01),
                    biases_init=Constant(0.))
    linear.initialize()
    y_hat = linear.apply(y_hat)
    seq_lenght = y_hat.shape[0]
    batch_size = y_hat.shape[1]
    y_hat = Softmax().apply(y_hat.reshape((seq_lenght * batch_size, -1))).reshape(y_hat.shape)
    cost = CategoricalCrossEntropy().apply(
        target.flatten(),
        y_hat.reshape((-1, len(all_chars)))) * seq_lenght * batch_size
    cost.name = 'cost'
    cost_per_character = cost / features_mask.sum()
    cost_per_character.name = 'cost_per_character'

    cg = ComputationGraph([cost, cost_per_character])
    model = Model(cost)
    algorithm = GradientDescent(step_rule=Adam(), cost=cost,
                                params=cg.parameters)

    train_monitor = TrainingDataMonitoring(
        [cost, cost_per_character], prefix='train',
        after_batch=True)
    extensions = [train_monitor, Printing(every_n_batches=40),
                  Dump(model_path, every_n_batches=200),
                  #Checkpoint('rnn.pkl', every_n_batches=200)
                  ]
    main_loop = MainLoop(model=model, algorithm=algorithm,
                         data_stream=data_stream, extensions=extensions)
    main_loop.run()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='rnn.bidirectional')
    parser.add_argument('--recurrent-type', default='simple')
    return parser.parse_args()


def _transpose(data):
    return data[0].transpose(1, 0, 2, 3),


if __name__ == '__main__':
    dataset = BouncingBalls(100, 30)
    stream = DataStream(dataset, iteration_scheme=ConstantScheme(1))
    stream = Batch(stream, ConstantScheme(100))
    stream = Batch(stream, ConstantScheme(13))
    stream = Mapping(stream, _transpose)
    from matplotlib import pyplot as plt
    import matplotlib.animation as animation
    for data in stream.get_epoch_iterator():
        print data[0].shape
        images = np.array([data[0][:, 0, :, :],
                           data[0][:, 0, :, :],
                           data[0][:, 0, :, :]]).transpose(1, 2, 3, 0)
        ims = []
        for image in images:
            im = plt.imshow(image)
            ims.append([im])
        fig = plt.figure()
        #plt.imshow(image, interpolation='none')
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
        plt.show()
    #args = parse_args()
    #main(**args.__dict__)