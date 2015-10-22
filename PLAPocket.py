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
        self.learning_rate = 1

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

    def mistakeTimes(self, inputs):
        count = 0
        for data in inputs:
            x = np.array(data[:-1])
            y = data[-1]
            if self.misClassified(data):
                count += 1
        return count

    def trainPCA(self, inputs, arrayIndex):
        """
        读取训练数据，更新权值，直到没有误分类点
        :param inputs: 训练数据
        """
        flag = True
        count = 0
        while flag:
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
                    if count == 50:
                        return

    def train(self, inputs, arrayIndex):
        """
        读取训练数据，更新权值，直到没有误分类点
        :param inputs: 训练数据
        """
        flag = True
        count = 0
        while flag:
            flag = False
            for i in arrayIndex:
                data = inputs[i]
            # for data in inputs:
                x = np.array(data[:-1])
                y = data[-1]
                if self.misClassified(data):
                    oldTimes = self.mistakeTimes(inputs)

                    tempW = self.w
                    tempB = self.b
                    self.w += self.learning_rate * y * x
                    self.b += self.learning_rate * y
                    newTimes = self.mistakeTimes(inputs)

                    if newTimes >= oldTimes:
                        self.w = tempW
                        self.b = tempB
                        continue

                    flag = True
                    count += 1
                    if count == 100:
                        return

    def test(self, testfile):
        """
        读取测试文件中的测试数据，并对其类型进行预测
        :param testfile: 存有测试数据的文件
        :param outfile:  输出的预测值: {+1, -1}
        """
        correct = 0
        wrong = 0
        with open(testfile) as tf:
            # with open(outfile, 'w') as of:
            for line in tf:
                ws = re.split(" |\n|\t", line)
                x = np.array([float(ws[0]), float(ws[1]), float(ws[2]), float(ws[3])])
                label = int(ws[4])
                value = np.dot(self.w, x) + self.b
                if value >= 0:
                    pre = 1
                else:
                    pre = -1

                if label == pre:
                    correct += 1
                else:
                    wrong += 1
                # of.write(pre + '\n')
        return float(wrong) / (float(correct) + float(wrong))

if __name__ == '__main__':
    per = Perceptron()
    inputs = per.readDataFromFile("train.dat")
    trainFileLineNumber = 500
    times = 0
    errorRate = []
    while times != 2000:
        arrayIndex = [i for i in range(trainFileLineNumber)]
        random.shuffle(arrayIndex)
        per.__init__()
        per.train(inputs, arrayIndex)
        # per.trainPCA(inputs, arrayIndex)
        errorRate.append(per.test("test.dat"))
        times += 1

    # print errorRate
    print np.average(errorRate)
    # print "weight vector: " + str(per.w)
    # print "bias: " + str(per.b)
    # per.test("..\data\\testSet.txt", "..\data\predict.txt")

