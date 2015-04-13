from math import ceil, floor

import numpy

import theano

from fuel.datasets import Dataset, IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme

floatX = theano.config.floatX


class Ball(object):
    def __init__(self, ball_coord, ball_v, field_size):
        self.ball_coord = ball_coord
        self.ball_v = ball_v
        self.field_size = field_size

    def move(self):
        self.ball_coord += self.ball_v
        self.ball_coord %= self.field_size


class BouncingBalls(Dataset):
    provides_sources = ('image',)

    def __init__(self, field_size, size, **kwargs):
        super(BouncingBalls, self).__init__(**kwargs)
        self.rng = numpy.random.RandomState(123)
        self.size = size
        self.field_size = field_size
        self.example_iteration_scheme = ConstantScheme(1)

    def open(self):
        ball_coord = self.rng.uniform(0, self.field_size, 2)
        ball_v = self.rng.uniform(-3, 3, 2)
        return Ball(ball_coord, ball_v, self.field_size)

    def get_example_stream(self):
        return DataStream(self, iteration_scheme=self.example_iteration_scheme)

    def get_data(self, state=None, request=None):
        ball_coord, ball_v = state.ball_coord, state.ball_v
        image = numpy.zeros((self.field_size, self.field_size), dtype=floatX)
        for i in xrange(int(floor(ball_coord[0] - self.size / 2)),
                        int(floor(ball_coord[0] + self.size / 2))):
            for j in xrange(int(floor(ball_coord[1] - self.size / 2)),
                            int(floor(ball_coord[1] + self.size / 2))):
                x = min(0, max(i, self.field_size))
                y = min(0, max(j, self.field_size))
                if ((numpy.array([x, y]) - ball_coord) ** 2).sum() < self.size:
                    image[x, y] = 1 - ((numpy.array([x, y]) -
                                        ball_coord) ** 2).sum() / self.size ** 2 / 2

        state.move()
        return (image,)

