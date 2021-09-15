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

from lab import Lab
import matplotlib.pyplot as mpl
import numpy as np

lab = Lab()
lab.open_file('test.bmp')
lab.pixel_transformation()
lab.create_rectangles()
lab.create_vectors()


def first_qeustion():
    pMass = []
    itMass = []
    for i in range(10):
        print(i)
        lab.e += 0.0001
        pMass.append(lab.e)
        lab.init_weight()
        itMass.append(lab.train())
    print('----------------------------------------------------------------')
    print(pMass)
    print(itMass)
    mpl.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    mpl.xlabel('Коэффициент обучения')
    mpl.ylabel('Количество эпох')
    mpl.plot(pMass, itMass)
    mpl.show()


def sec_question():
    lab.p = 110
    pMass = []
    itMass = []
    for i in range(10):
        print(i)
        lab.p -= 8
        pMass.append(lab.getZ())
        lab.init_weight()
        itMass.append(lab.train())
    print('----------------------------------------------------------------')
    print(pMass)
    print(itMass)
    mpl.xlabel('Коэффициент сжатия')
    mpl.ylabel('Количество эпох')
    mpl.plot(pMass, itMass)
    mpl.show()


def f_question():
    lab.Emax = 10000
    pMass = []
    itMass = []
    for i in range(9):
        print(i)
        lab.Emax -= 1000
        pMass.append(lab.Emax)
        lab.init_weight()
        itMass.append(lab.train())
    print('----------------------------------------------------------------')
    print(pMass)
    print(itMass)
    mpl.xlabel('Заданное значение ошибки')
    mpl.ylabel('Количество эпох')
    mpl.plot(pMass, itMass)
    mpl.show()


f_question()
# lab.show_image(lab.from_rectangles_to_picture(lab.back_to_rectangles(lab.train())))
