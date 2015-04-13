from theano import tensor

from blocks.bricks import (MLP, Sigmoid, application, Brick, Random,
                           Initializable)
from blocks.bricks.cost import CostMatrix
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Dump, LoadFromDump
from blocks.algorithms import Scale, RMSProp, StepClipping, GradientDescent
from blocks.model import Model
from blocks.graph import ComputationGraph

from fuel.datasets import MNIST
from fuel.streams import DataStream


class VAECost(CostMatrix):
    @application(outputs=["cost"])
    def apply(self, y, y_hat, mu, logsigma):
        return self.cost_matrix(y, y_hat, mu, logsigma).sum(axis=1).mean()



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
        batch_size = result.shape[0]
        epsilons = self.theano_rng.normal(0, 1, (batch_size, self.hidden_dim))
        return mu, logsigma, mu + tensor.exp(logsigma) * epsilons


class VariationalAutoEncoder(Initializable):
    def __init__(self, input_dim, hidden_dim):
        super(VariationalAutoEncoder, self).__init__()

        encoder_mlp = MLP([Sigmoid(), Sigmoid()], [input_dim, 500, None])
        self.encoder = VAEEncoder(encoder_mlp, hidden_dim)
        self.decoder = MLP([Sigmoid(), Sigmoid()], [hidden_dim, 500, input_dim])
        self.children = [self.encoder, self.decoder]

    @application
    def apply(self, x):
        mu, logsigma, z = self.encoder.apply(x)
        x_hat = self.decoder.apply(z)
        return mu, logsigma, x_hat

    @application
    def cost(self, x):
        mu, logsigma, x_hat = self.apply(x)
        return (self.regularization_cost(mu, logsigma) +
                self.reconstruction_cost(x, x_hat))

    @application
    def regularization_cost(self, mu, logsigma):
        return 0.5 * ((1 + 2 * logsigma) - mu ** 2 -
                      tensor.exp(logsigma) ** 2).sum(axis=1)

    @application
    def reconstruction_cost(self, x, x_hat):
        return tensor.nnet.categorical_crossentropy(x_hat, x)


def main():
    input_dim = 784
    hidden_dim = 20

    x = tensor.matrix('x')

    vae = VariationalAutoEncoder(input_dim, hidden_dim)

    mu, logsigma, x_hat = vae.apply(x)

    cost = VAECost().apply(x, x_hat, mu, logsigma)

    cg = ComputationGraph(cost)
    model = Model(cost)

    algorithm = GradientDescent(step_rule=Scale(1e-4), params=cg.parameters,
                                cost=cost)

    extensions = [Dump('VAE'), Timing(), Printing(),
                  FinishAfter(after_n_epochs=10)]

    train_dataset = MNIST('train', binary=True)
    train_stream = DataStream(train_dataset)
    main_loop = MainLoop(model=model, algorithm=algorithm,
                         data_stream=train_stream,
                         extensions=extensions)
    main_loop.run()


if __name__ == '__main__':
    main()
