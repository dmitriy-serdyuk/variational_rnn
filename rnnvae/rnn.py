import numpy as np
import theano

from theano import tensor

from blocks.algorithms import GradientDescent, Adam
from blocks.bricks import Softmax, Tanh, Linear
from blocks.bricks.recurrent import SimpleRecurrent
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


def main():
    dataset_options = dict(dictionary=char2code, level="character",
                           preprocess=_lower)
    dataset = OneBillionWord("training", [99], **dataset_options)
    data_stream = dataset.get_example_stream()
    data_stream = Filter(data_stream, _filter_long)
    data_stream = Batch(data_stream,
                        iteration_scheme=ConstantScheme(10))
    data_stream = Padding(data_stream)
    data_stream = Mapping(data_stream, _transpose)

    features = tensor.lmatrix('features')
    features_mask = tensor.matrix('features_mask')
    dim = 100
    lookup = LookupTable(len(all_chars), dim,
                         weights_init=IsotropicGaussian(0.01),
                         biases_init=Constant(0.))

    rnn = SimpleRecurrent(dim, Tanh(),
                          weights_init=IsotropicGaussian(0.01),
                          biases_init=Constant(0.))
    rnn.initialize()
    lookup.initialize()
    y_hat = rnn.apply(lookup.apply(features), mask=features_mask)

    linear = Linear(dim, len(all_chars),
                    weights_init=IsotropicGaussian(0.01),
                    biases_init=Constant(0.))
    linear.initialize()
    y_hat = linear.apply(y_hat)
    seq_lenght = y_hat.shape[0]
    batch_size = y_hat.shape[1]
    y_hat = Softmax().apply(y_hat.reshape((seq_lenght * batch_size, -1))).reshape(y_hat.shape)
    cost = CategoricalCrossEntropy().apply(features[1:, :].reshape((-1,)),
                                           y_hat[:-1, :, :].reshape((-1, len(all_chars))))
    cost.name = 'cost'

    cg = ComputationGraph(cost)
    model = Model(cost)
    algorithm = GradientDescent(step_rule=Adam(), cost=cost,
                                params=cg.parameters)

    train_monitor = TrainingDataMonitoring(
        [cost], prefix='train', after_batch=True)
    extensions = [train_monitor, Printing(every_n_batches=40),
                  Dump('rnn', every_n_batches=200),
                  #Checkpoint('rnn.pkl', every_n_batches=200)
                  ]
    main_loop = MainLoop(model=model, algorithm=algorithm,
                         data_stream=data_stream, extensions=extensions)
    main_loop.run()

if __name__ == '__main__':
    main()