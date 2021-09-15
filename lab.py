"""
////////////////////////////////////////////////////////////////////////////////////
// Лабораторная работа 3 по дисциплине МРЗвИС
// Выполнена студентом группы 9217023
// БГУИР Павлов Даниил Иванович
// Вариант 2 - модель рециркуляционной сети с адаптивным коэффициентом
// обучения с ненормированными весами
// 14.09.2021
// Использованные материалы:
// https://numpy.org/doc/stable/index.html - методические материалы по numpy
// https://www.learnpython.org/ - методические материалы по python
// https://studfile.net/preview/1557061/page:8/ - описание рециркуляционной сети
"""

import numpy as np
from PIL import Image


class Lab:
    def __init__(self, r=4, m=4, p=3 * 2 * 2, Emax=1000):
        self.r = r + 1
        self.m = m + 1
        self.p = p
        self.Emax = Emax
        self.e = 0.0002

    def open_file(self, path):
        self.initial_matrix = np.asarray(Image.open(path)).astype('float')
        # Image.fromarray(np.uint8(self.initial_matrix)).show()

    def pixel_transformation(self):
        for cell in np.nditer(self.initial_matrix, op_flags=['readwrite']):
            cell[...] = cell * 2 / 255 - 1

    def create_rectangles(self):
        self.rectangles = []
        ok = list(self.initial_matrix)
        for i in range(int(len(ok) / (self.r - 1))):
            temp = []
            for j in range(int(len(ok[i]) / (self.m - 1))):
                temp.append(self.initial_matrix[i * (self.r - 1):(i + 1) * (self.r - 1),
                            j * (self.m - 1):(j + 1) * (self.m - 1)])
            if not (len(ok[i]) % (self.m - 1) == 0):
                temp.append(self.initial_matrix[i * (self.r - 1):(i + 1) * (self.r - 1)
                            , len(self.initial_matrix[i]) - self.m:len(self.initial_matrix[i]) - 1])
            self.rectangles.append(temp)
        if not (len(ok) % (self.r - 1) == 0):
            temp = []
            for j in range(int(len(ok[0]) / (self.m - 1))):
                temp.append(self.initial_matrix[len(self.initial_matrix) - self.r:len(self.initial_matrix) - 1
                            , j * (self.m - 1):(j + 1) * (self.m - 1)])
            if not (len(ok[i]) % (self.m - 1) == 0):
                temp.append(self.initial_matrix[i * (self.r - 1):(i + 1) * (self.r - 1)
                            , len(self.initial_matrix[i]) - self.m:len(self.initial_matrix[i]) - 1])
            self.rectangles.append(temp)
        self.rectangles = np.asanyarray(self.rectangles)

    def create_vectors(self):
        result = []
        for i in range(self.rectangles.shape[0]):
            temp = []
            for j in range(self.rectangles.shape[1]):
                temp.append(np.matrix.flatten(self.rectangles[i][j]))
            result.append(temp)
        self.rectangles = np.array(result)

    def back_to_rectangles(self, transform):
        # for r*m rectangles
        output = []
        for i in transform:
            temp = []
            for j in transform:
                t = []
                for k in j:
                    t.append(np.reshape(k, (self.r - 1, self.m - 1, 3)))
                temp.append(t)
            output.append(temp)
        return np.array(temp)

    def from_rectangles_to_picture(self, rectangles):
        output = []
        for k in rectangles:
            for i in range(self.r - 1):
                temp = []
                for f in k:
                    for j in range(self.m - 1):
                        colours = []
                        for g in range(3):
                            colours.append(f[i][j][g])
                        temp.append(colours)
                output.append(temp)
        return np.array(output)

    def show_image(self, image):
        # Image.fromarray(np.uint8(image)).show()
        Image.fromarray(np.uint8(image)).save("unzip.bmp")

    def init_weight(self):
        self.Wf = (np.random.rand((self.m - 1) * (self.r - 1) * 3, self.p) + 0.01) * 2 - 1
        self.Wb = (np.random.rand(self.p, (self.m - 1) * (self.r - 1) * 3) + 0.01) * 2 - 1

    def train_coeff(self, matr):
        to_return = np.sum(np.square(matr))
        return 0.00007 if 1.0 / to_return == 0 else 1 / to_return

    def train(self):
        ECURR = self.Emax + 1
        epoch = 0
        while ECURR > self.Emax:
            epoch += 1
            for i in self.rectangles:
                for j in i:
                    Y = j @ self.Wf
                    XX = Y @ self.Wb
                    dX = XX - j
                    # 1 / np.sum(np.square(Y))  1 / np.sum(np.square(Y))
                    self.Wb -= (1 / np.sum(np.square(Y))) * (Y.reshape(Y.size, 1) @
                                           dX.reshape(1, dX.size))
                    self.Wf -= (1 / np.sum(np.square(Y))) * (
                            (j.reshape(j.size, 1) @ dX.reshape(1, dX.size)) @ self.Wb.transpose())
            ECURR = 0
            for i in self.rectangles:
                for j in i:
                    Y = np.dot(j, self.Wf)
                    XX = np.dot(Y, self.Wb)
                    dX = XX - j
                    ECURR += np.sum(np.square(dX))
            print(ECURR)
            print(epoch)
        """
        to_return = []
        for i in self.rectangles:
            temp = []
            for j in i:
                Y = np.dot(j, self.Wf)
                XX = np.dot(Y, self.Wb)
                temp.append(XX)
            to_return.append(temp)
        to_return = np.array(to_return)
        for cell in np.nditer(to_return, op_flags=['readwrite']):
            cell[...] = 255 * (min(1, max(-1, cell)) + 1) / 2.0
        return to_return
        """
        return epoch

    def getZ(self):
        return self.rectangles.size / (
                2 + (self.r * self.m + self.rectangles.shape[0] * self.rectangles.shape[1]) * self.p)
