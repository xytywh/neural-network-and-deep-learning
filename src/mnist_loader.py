"""
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.

加载MNIST图像数据的库。有关返回的数据结构的详细信息，请参阅“load_data”和
“load_data_wrapper”的文档字符串。实际上，“load_data_wrapper”
是我们的神经网络代码通常调用的函数。
"""

import pickle
import gzip
import numpy as np


def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.

    将MNIST数据作为一个元组返回，其中包含训练数据、验证数据和测试数据。
    ' training_data ' '作为一个包含两个条目的元组返回。第一个条目
    包含实际的训练图像。这是一个包含50,000个条目的numpy ndarray。
    每个条目依次是一个包含784个值的numpy ndarray，表示单个MNIST
    图像中的28 * 28 = 784个像素。“training_data”元组中的第二个条目
    是一个包含50,000个条目的numpy ndarray。这些条目只是元组第一个条目
    中包含的对应图像的数字值(0…9)。'validation_data'和'test_data'
    是相似的，只是它们都只包含10,000个图像。这是一种很好的数据格式，
    但是对于神经网络来说，稍微修改一下'training_data'的格式是很有帮助的。
    这是在包装器函数'load_data_wrapper() ''中完成的，参见下面。
    """
    f = gzip.open('F:\python\\neural network and deep learning\data\mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    # training_data是含有两个元素的元组
    # training_data[0]是shape为(50000,784)的二维ndarray,每个值是0-1之间，代表每个像素值
    # training_data[1]是shape为(50000,)的一维ndarray，每个值是0-9，代表所属标签
    # validation_data和test_data与training_data的不同在于只含有10000个图像
    return training_data, validation_data, test_data


def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code.

    返回一个包含'' (training_data、validation_data、test_data) ''的元组。
    基于“load_data”，但该格式更便于我们在实现神经网络时使用。特别是，''training_data''
    是一个包含50,000个2元组''(x, y)''的列表。“x”是一个784维的包含输入图像的
    numpy ndarray。y是一个10维的数字。表示单位向量对应于''x''的正确数字。
    'validation_data''和'test_data''是包含10,000个2元组''(x, y)''的列表。
    在每种情况下，“x”都是一个784维的包含输入图像的numpy ndarry，
    ''y''是对应的分类，即，与''x''对应的数字值(整数)。显然，
    这意味着我们对训练数据和验证/测试数据使用的格式略有不同。
    这些格式在我们的神经网络代码中是最方便使用的。
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    # 现在training_data是含有50000个元素的list，每个元素是一个含有两个元素的tuple
    # 每一个tuple中，第一个元素是shape为(784,1)的ndarray，第二个元素是shape为(10,1)的ndarray
    # validation_data和test_data 与 training_data的不同之处在于：1.只含有10000个元素
    # 2.第二个元素是0-9之间的一个数(这里注意：训练数据集是用于训练的，所以最后的输出需要是一个shape为(10,1)的
    # one-hot向量，哪一位最大，代表输入那一类，所以这种one-hot向量形式对于训练数据集市最方便的，而对于
    # 验证数据集和测试数据集，我们不用这两类数据进行训练，我们只想看正确率，而看正确率，将输入数据的输出的(10,1)
    # one-hot向量转换成0-9之间的数，直接和验证或者测试的target进行比较，是否相同，即可，所以，验证和测试的target
    # 没必要是one-hot向量)
    return training_data, validation_data, test_data


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


if __name__ == "__main__":
    training_data, validation_data, test_data = load_data_wrapper()
    print(validation_data[0])
