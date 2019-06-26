"""
network.py
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.

一个用于实现前馈神经网络随机梯度下降学习算法的模块。使用反向传播计算梯度。注意，
我的重点是使代码简单、易于阅读和修改。它没有进行优化，并且忽略了许多令人满意的特性。
"""

import random

import numpy as np


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1(Standard Gaussian
        distribution).  Note that the first layer is assumed to be
        an input layer, and by convention we won't set any biases
        for those neurons, since biases are only ever used in
        computing the outputs from later layers.

        “sizes”列表中包含网络各层的神经元数量。例如，如果列表是[2,3,1]，
        那么它将是一个三层网络，第一层包含2个神经元，第二层包含3个神经元，
        第三层包含1个神经元。使用均值为0和方差为1的高斯分布(标准高斯分布)
        随机初始化网络的偏差和权重。注意，第一层假设是一个输入层，按照惯例，
        我们不会为这些神经元设置任何偏差，因为偏差只用于计算后面几层的输出。
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # 首层是输入，无需bias
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # 注意权重初始化时矩阵的顺序，比如前一层a个结点，后一层b个结点，那么连接这两层的权重形状是(b,a),而不是(a,b)
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        # a是⼀个(n, 1)的Numpy ndarray类型，而不是⼀个(n, )的向量
        for b, w in zip(self.biases, self.weights):
            #因为使多分类，其实最后一层的输出应该为softmax的，不过这里用sigmoid，影响也不太大。
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, lr,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.

        利用小批量随机梯度下降训练神经网络。''training_data''是一个元组列表，
        ''(x, y)''表示训练输入和所需输出。其他非可选参数是不言自明的。如果
        提供了''test_data''，则在每个epoch之后根据测试数据对网络进行评估，
        并打印出部分进度。这对于跟踪进度很有用，但是会大大降低速度。
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            # 在每一个epoch的时候，将数据随机打乱，最大限度的保持随机性
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, lr):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate.

        通过将使用反向传播的梯度下降应用于单个小批量处理，更新网络的权重和偏差。
        ''mini_batch''是一个元组列表''(x, y)''， ''lr''是学习率。
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # nabla_w和nabla_b分别代表一个mini_batch训练数据的权重和偏差的总梯度
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # 注意这里是利用mini_batch个训练数据的平均梯度来对权重和偏差进行更新的，所以需要
        # 除以len(mini_batch)
        self.weights = [w - (lr / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (lr / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.

        返回一个元组''(nabla_b, nabla_w)''表示损失函数C_x的梯度。''nabla_b''和
        ''nabla_w''是numpy arrays的逐层列表，类似于''self.bias''和''self.weights''。
        """
        # x.shape:(784,1) y.shape:(10,1)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
        # 最后一层权重和偏差的导数
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
