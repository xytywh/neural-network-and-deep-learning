"""network2.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

network.py的改进版本。实现了一种前馈神经网络的随机梯度下降学习算法。改进
包括增加交叉熵损失函数、正则化和更好的网络权值初始化。注意，我的重点是使
代码简单、易于阅读和修改。它没有进行优化，并且忽略了许多令人满意的特性。

"""

import json
import random
import sys
import numpy as np


# 定义二次损失函数
class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a - y) * sigmoid_prime(z)


# 定义交叉熵损失函数
class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        返回与输出''a''和期望输出''y''关联的损失。注意,使用np.nan_to_num来保证数值的稳定性。
        特别是，如果''a''和''y''在同一个槽中都有一个1.0，那么表达式(1-y)*np.log(1-a)返回nan。
        np.nan_to_num确保将其转换为正确的值(0.0)。

        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        从输出层返回误差delta。注意，该方法没有使用参数''z''。
        它包含在方法的参数中，以便使接口与其他损失类的delta方法一致。

        """
        return (a - y)


class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).


        "sizes"列表包含网络各层的神经元数量。例如，如果列表是[2,3,1]，那么它将是一个三层网络，
        第一层包含2个神经元，第二层包含3个神经元，第三层包含1个神经元。使用''self.default_weight_initializer''
        随机初始化网络的偏差和权重(有关该方法，请参阅docstring)。
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.


        初始化每个权值时使用高斯分布(均值为0，标准差为1)除以连接到同一个神经元的
        权值个数的平方根。使用均值为0、标准差为1的高斯分布初始化偏差。

        注意，第一层假设是一个输入层，按照惯例，我们不会为这些神经元设置任何偏差，
        因为偏差只用于计算后面几层的输出。
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        # randn服从的分布是N(0,1)，除以np.sqrt(x)之后，服从的分布是N(0,1/x)
        # 均值为0,方差为1/np.sqrt(x)，x是输入神经元数量
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.

        使用均值为0、标准差为1的高斯分布初始化权重。使用均值为0、标准差为1的
        高斯分布初始化偏差。注意，第一层假设是一个输入层，按照惯例，我们不会
        为这些神经元设置任何偏差，因为偏差只用于计算后面几层的输出。此权重和
        偏差初始化器使用与第1章相同的方法，并包含在其中以供比较。通常使用默认
        的权重初始化器会更好。

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        # a是⼀个(n, 1)的Numpy ndarray类型，而不是⼀个(n, )的向量
        for b, w in zip(self.biases, self.weights):
            # 因为使多分类，其实最后一层的输出应该为softmax的，不过这里用sigmoid，影响也不太大。
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        利用小批量随机梯度下降训练神经网络。''training_data''是一个元组列表
        ''(x, y)''表示训练输入和所需输出。其他非可选参数是不言自明的，正则化
        参数''lmbda''也是如此。该方法还接受'' evaluation_data''，
        通常是验证数据或测试数据。通过设置适当的标志，我们可以监控评估数据或
        训练数据的成本和准确性。该方法返回一个包含四个列表的元组:评估数据的
        (每epoch)损失、评估数据的准确性、训练数据的损失和训练数据的准确性。
        所有值都在每个训练周期结束时进行评估。例如，如果我们训练30个epoch，
        那么元组的第一个元素将是一个30元素列表，其中包含每个epoch结束时评估
        数据上的损失。注意，如果没有设置相应的标志，列表就是空的。
        """
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, lr, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``lr`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        通过将使用反向传播的梯度下降应用于单个小批量处理，更新网络的权重和偏差。
        ''mini_batch''是一个元组列表''(x, y)''， ''lr''是学习率，''lmbda''
        是正则化参数，''n''是训练数据集的总大小。
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # 正则化参数lmbda除以的是训练集总数n，学习率lr除以的是一个mini_batch的大小
        # 只针对weights，使用的是l2正则化
        self.weights = [(1 - lr * (lmbda / n)) * w - (lr / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]

        # 只针对weights，使用的是l1正则化
        # self.weights = [w - lr * lmbda * sgn(w) / n - (lr / len(mini_batch)) * nw
        #                 for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b - (lr / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
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
        delta = (self.cost).delta(zs[-1], activations[-1], y)
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

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        返回神经网络输出正确结果的“data”中的输入数。假设神经网络的输出为最后一层
        中激活程度最高的神经元的指标。如果数据集是验证数据或测试数据(通常情况下)，
        应该将标志''convert''设置为False，如果数据集是训练数据，则应该将标志
        ''convert''设置为True。之所以需要这个标志，是因为不同的数据集中表示结
        果“y”的方式不同。特别是，它标记了我们是否需要在不同的表示之间进行转换。
        对于不同的数据集使用不同的表示可能看起来很奇怪。为什么不对所有三个数据集
        使用相同的表示呢?这样做是出于效率的考虑——该程序通常根据训练数据和其他数据集
        的准确性来评估损失。这些是不同类型的计算，使用不同的表示可以加快速度。有关
        表示的更多细节可以在mnist_loader.load_data_wrapper中找到。

        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.

        返回数据集''data''的总损失。如果数据集是训练数据(通常情况下)，那么“convert”
        标志应该设置为False，如果数据集是验证或测试数据，那么应该设置为True。请参阅
        对类似(但相反)约定的注释，以了解上面的“准确性”方法。
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(
            np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    返回一个10维单位向量，第j个位置为1.0，其他位置为0。这是用来把一个数字(0…9)
    转换成相应的期望输出从神经网络。

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def sgn(w):
    """Derivative of w."""
    w[w < 0] = -1
    w[w == 0] = 0
    w[w > 0] = 1
