import argparse
import numpy
from theano import tensor, function

from blocks.bricks import (MLP, Sigmoid, application, Random,
                           Initializable, Identity)
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Dump, LoadFromDump
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.dump import load_parameter_values
from blocks.algorithms import Scale, RMSProp, StepClipping, GradientDescent
from blocks.model import Model
from blocks.graph import ComputationGraph

from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, ConstantScheme


class VAEEncoder(Random, Initializable):
    def __init__(self, mlp, hidden_dim):
        super(VAEEncoder, self).__init__()
        self.mlp = mlp
        self.mlp.output_dim = 2 * hidden_dim
        self.hidden_dim = hidden_dim
        self.children = [mlp]

    @application
    def apply(self, x):
        result = self.mlp.apply(x)
        mu, logsigma = tensor.split(result, [self.hidden_dim] * 2, 2, axis=1)
        batch_size = x.shape[0]
        epsilons = self.theano_rng.normal((batch_size, self.hidden_dim),
                                          0., 1.)
        return mu, logsigma, mu + tensor.exp(logsigma) * epsilons


class VAEDecoder(Initializable):
    def __init__(self, mlp, **kwargs):
        super(VAEDecoder, self).__init__(**kwargs)
        self.mlp = mlp
        self.children = [self.mlp]

    @application
    def apply(self, z):
        return self.mlp.apply(z)


class VariationalAutoEncoder(Initializable):
    def __init__(self, input_dim, hidden_dim, **kwargs):
        super(VariationalAutoEncoder, self).__init__(**kwargs)

        encoder_mlp = MLP([Sigmoid(), Identity()],
                          [input_dim, 101, None])
        decoder_mlp = MLP([Sigmoid(), Sigmoid()],
                          [hidden_dim, 101, input_dim])
        self.hidden_dim = hidden_dim
        self.encoder = VAEEncoder(encoder_mlp, hidden_dim)
        self.decoder = VAEDecoder(decoder_mlp)
        self.children = [self.encoder, self.decoder]

    @application
    def apply(self, x):
        mu, logsigma, z = self.encoder.apply(x)
        x_hat = self.decoder.apply(z)
        return mu, logsigma, x_hat

    @application
    def sample(self, batch_size):
        z = self.encoder.theano_rng.normal((batch_size, self.hidden_dim),
                                           0., 1.)
        return self.decoder.apply(z)

    @application
    def cost(self, x):
        mu, logsigma, x_hat = self.apply(x)
        return (self.regularization_cost(mu, logsigma) +
                self.reconstruction_cost(x, x_hat)).mean()

    @application
    def regularization_cost(self, mu, logsigma):
        return - 0.5 * ((1 + 2 * logsigma) - mu ** 2 -
                        tensor.exp(logsigma) ** 2).sum(axis=1)

    @application
    def reconstruction_cost(self, x, x_hat):
        return tensor.nnet.binary_crossentropy(x_hat, x).sum(axis=1)


def main(save, load, sample, path, **kwargs):
    input_dim = 784
    hidden_dim = 10
    batch_size = 100

    features = tensor.matrix('features')

    vae = VariationalAutoEncoder(input_dim, hidden_dim,
                                 weights_init=IsotropicGaussian(0.01),
                                 biases_init=Constant(0.))
    vae.initialize()

    mu, logsigma, x_hat = vae.apply(features)
    cost = vae.cost(features)
    cost.name = 'cost'
    regularization_cost = vae.regularization_cost(mu, logsigma).mean()
    regularization_cost.name = 'regularization_cost'
    reconstruction_cost = vae.reconstruction_cost(features, x_hat).mean()
    reconstruction_cost.name = 'reconstruction_cost'

    cg = ComputationGraph([cost, reconstruction_cost, regularization_cost])
    model = Model(cost)

    algorithm = GradientDescent(step_rule=RMSProp(1e-4), params=cg.parameters,
                                cost=cost)

    extensions = []
    if load:
        extensions.append(LoadFromDump(path))
    if save:
        extensions.append(Dump(path, after_epoch=True))
    extensions.append(FinishAfter(after_n_epochs=6001))

    train_dataset = MNIST('train', binary=True, sources=('features',))
    train_stream = DataStream(train_dataset,
                              iteration_scheme=ShuffledScheme(
                                  examples=train_dataset.num_examples,
                                  batch_size=batch_size))
    train_monitor = TrainingDataMonitoring(
        [cost, regularization_cost, reconstruction_cost],
        prefix='train', after_epoch=True)

    test_dataset = MNIST('test', binary=True, sources=('features',))
    test_stream = DataStream(test_dataset,
                             iteration_scheme=ShuffledScheme(
                                 examples=test_dataset.num_examples,
                                 batch_size=batch_size))
    test_monitor = DataStreamMonitoring([cost], test_stream, prefix='test')
    extensions.extend([train_monitor, test_monitor])
    extensions.extend([Timing(), Printing()])
    main_loop = MainLoop(model=model, algorithm=algorithm,
                         data_stream=train_stream,
                         extensions=extensions)
    if not sample:
        main_loop.run()
    else:
        parameters = load_parameter_values(path + '/params.npz')
        model.set_param_values(parameters)

        num_samples = 10
        samples = vae.sample(num_samples)
        samples = function([], samples)()
        from matplotlib import pyplot as plt
        sample = numpy.zeros((28, 0))
        for i in xrange(num_samples):
            sample = numpy.concatenate([sample, samples[i].reshape((28, 28))],
                                       axis=1)
        plt.imshow(sample)
        plt.show()
        f = function([features], x_hat)
        for data in train_stream.get_epoch_iterator():
            data_hat = f(data[0])
            break
        for image in data_hat:
            plt.imshow(image.reshape((28, 28)))
            plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true', default=False,
                        help='Save model')
    parser.add_argument('--load', action='store_true', default=False,
                        help='Load model')
    parser.add_argument('--sample', action='store_true', default=False,
                        help='Sample images')
    parser.add_argument('--path', type=str, default='vae',
                        help='Experiment path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(**args.__dict__)
