"""
mnist_average_darkness
~~~~~~~~~~~~~~~~~~~~~~

A naive classifier for recognizing handwritten digits from the MNIST
data set.  The program classifies digits based on how dark they are
--- the idea is that digits like "1" tend to be less dark than digits
like "8", simply because the latter has a more complex shape.  When
shown an image the classifier returns whichever digit in the training
data had the closest average darkness.

The program works in two steps: first it trains the classifier, and
then it applies the classifier to the MNIST test data to see how many
digits are correctly classified.

Needless to say, this isn't a very good way of recognizing handwritten
digits!  Still, it's useful to show what sort of performance we get
from naive ideas.

一种用于从MNIST数据集中识别手写数字的朴素分类器。该程序根据数字的平均暗度对其进行分类，
其思想是，像“1”这样的数字往往比“8”这样的数字暗，因为后者的形状更复杂。当显示图像
时，分类器返回训练数据中平均暗度最近的数字。该程序分为两个步骤:首先训练分类器，
然后将分类器应用于MNIST测试数据，查看正确分类了多少数字。不用说，这不是识别手写
数字的好方法!不过，展示我们从朴素的想法中得到的性能是很有用的。
"""

from collections import defaultdict
import mnist_loader


def main():
    training_data, validation_data, test_data = mnist_loader.load_data()
    # training phase: compute the average darknesses for each digit,
    # based on the training data
    avgs = avg_darknesses(training_data)
    # testing phase: see how many of the test images are classified
    # correctly
    num_correct = sum(int(guess_digit(image, avgs) == digit)
                      for image, digit in zip(test_data[0], test_data[1]))
    print("Baseline classifier using average darkness of image.")
    print("%s of %s values correct." % (num_correct, len(test_data[1])))


def avg_darknesses(training_data):
    """ Return a defaultdict whose keys are the digits 0 through 9.
    For each digit we compute a value which is the average darkness of
    training images containing that digit.  The darkness for any
    particular image is just the sum of the darknesses for each pixel."""
    digit_counts = defaultdict(int)
    darknesses = defaultdict(float)
    for image, digit in zip(training_data[0], training_data[1]):
        digit_counts[digit] += 1
        darknesses[digit] += sum(image)
    avgs = defaultdict(float)
    for digit, n in digit_counts.items():
        avgs[digit] = darknesses[digit] / (n * 784)
    print(avgs)
    return avgs


def guess_digit(image, avgs):
    """Return the digit whose average darkness in the training data is
    closest to the darkness of ``image``.  Note that ``avgs`` is
    assumed to be a defaultdict whose keys are 0...9, and whose values
    are the corresponding average darknesses across the training data.

    返回训练数据中平均暗度最接近“image”暗度的数字。注意，
    ''avgs''被假定为一个defaultdict，其键为0…9，其值为训练数据中相应的平均暗度。
    """
    darkness = sum(image) / 784
    distances = {k: abs(v - darkness) for k, v in avgs.items()}
    # 返回最小的value对应的key
    return min(distances, key=distances.get)


if __name__ == "__main__":
    # 正确率为22.25%
    main()
