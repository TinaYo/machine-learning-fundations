# coding:utf-8

import numpy as np
import re
import random

class Perceptron():
    def __init__(self):
        """
        设定初值w0, b0
        """
        self.w = np.array([0.0 for _ in xrange(4)])
        self.b = 0
        self.learning_rate = 0.5

    def readDataFromFile(self, filename):
        """
        从文件中读取训练数据
        :param filename: 存有训练数据的文件，格式为"+1 3 3\n"
        :return:
        """
        inputs = []
        lineNum = 0
        with open(filename) as fi:
            for line in fi:
                lineNum += 1
                ws = re.split(" |\t|\n", line)
                inputs.append([float(ws[0]), float(ws[1]), float(ws[2]), float(ws[3]), int(ws[4])])
        print lineNum
        return inputs

    def misClassified(self, data):
        """
        判断是否是误分类的点
        :param data: 一条数据
        :return: 如果是误分，就返回True，否则返回False
        """
        x = np.array(data[:-1])
        res = data[-1] * (np.dot(self.w, x) + self.b)
        if res <= 0:
            return True
        else:
            return False

    def train(self, inputs, arrayIndex):
        """
        读取训练数据，更新权值，直到没有误分类点
        :param inputs: 训练数据
        """
        flag = True
        count = 0
        while flag:
            if count == 1000:
                return count
            flag = False
            for i in arrayIndex:
                data = inputs[i]
            # for data in inputs:
                x = np.array(data[:-1])
                y = data[-1]
                if self.misClassified(data):
                    count += 1
                    flag = True
                    self.w += self.learning_rate * y * x
                    self.b += self.learning_rate * y

        return count

    def test(self, testfile, outfile):
        """
        读取测试文件中的测试数据，并对其类型进行预测
        :param testfile: 存有测试数据的文件
        :param outfile:  输出的预测值: {+1, -1}
        """
        with open(testfile) as tf:
            with open(outfile, 'w') as of:
                for line in tf:
                    ws = re.split(" |\n", line)
                    x = np.array([int(ws[0]), int(ws[1])])
                    value = np.dot(self.w, x) + self.b
                    if value >= 0:
                        pre = '+1'
                    else:
                        pre = '-1'
                    of.write(pre + '\n')


if __name__ == '__main__':
    per = Perceptron()
    inputs = per.readDataFromFile("question15.dat")
    times = 0
    updateNum = []
    while times != 2000:
        arrayIndex = [i for i in range(400)]
        random.shuffle(arrayIndex)
        per.__init__()
        updateNum.append(per.train(inputs, arrayIndex))
        times += 1

    print np.average(updateNum)
    # print "weight vector: " + str(per.w)
    # print "bias: " + str(per.b)
    # per.test("..\data\\testSet.txt", "..\data\predict.txt")

