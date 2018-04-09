# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 00:16:53 2017

@author: Kartikeya BHardwaj
"""
import matplotlib.pyplot as plt
import random
from sklearn import datasets
from sklearn import svm

m = random.randint(-1000,1000)
digits = datasets.load_digits()
clf = svm.SVC(gamma = 0.001 , C = 100)
x,y = digits.data[:-1] , digits.target[:-1]
clf.fit(x,y)

print("The Prediction is:", clf.predict(digits.data[m]))

plt.imshow(digits.images[m], cmap = plt.cm.gray_r, interpolation = "nearest")
plt.show()

