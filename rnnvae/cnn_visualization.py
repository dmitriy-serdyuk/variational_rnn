from matplotlib import pyplot as plt
import numpy
import theano
from theano import tensor

from blocks.bricks import Rectifier, Softmax
from blocks.bricks.conv import ConvolutionalLayer, ConvolutionalSequence
from blocks.initialization import IsotropicGaussian, Constant


def main():
    initial = numpy.random.normal(0, 0.1, (1, 1, 200, 200))
    x = theano.shared(initial)

    conv_layer = ConvolutionalLayer(
        Rectifier().apply,
        (16, 16),
        9,
        (4, 4),
        1
    )
    conv_layer2 = ConvolutionalLayer(
        Rectifier().apply,
        (7, 7),
        9,
        (2, 2),
        1
    )
    con_seq = ConvolutionalSequence([conv_layer], 1,
                                    image_size=(200, 200),
                                    weights_init=IsotropicGaussian(0.1),
                                    biases_init=Constant(0.)
                                    )

    con_seq.initialize()
    out = con_seq.apply(x)
    target_out = out[0, 0, 1, 1]

    grad = theano.grad(target_out - .1 * (x ** 2).sum(), x)
    updates = {x: x + 5e-1 * grad}
    #x.set_value(numpy.ones((1, 1, 200, 200)))
    #print theano.function([], out)()

    make_step = theano.function([], target_out, updates=updates)

    for i in xrange(400):
        out_val = make_step()
        print i, out_val

    image = x.get_value()[0][0]
    image = (image - image.mean()) / image.std()
    image = numpy.array([image, image, image]).transpose(1, 2, 0)
    plt.imshow(numpy.cast['uint8'](image * 65. + 128.), interpolation='none')
    plt.show()


if __name__ == '__main__':
    main()