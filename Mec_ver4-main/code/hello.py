print("hello")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import exp, factorial

# # Generate Distribution:
# randomNums = np.random.normal(6, 2, size=1000)
# randomInts = np.round(randomNums)
# print( randomInts[0])
# for i in range(len(randomInts)):
#     if randomInts[i]<0:
#         randomInts[i] = 0
#     if randomInts[i]>12:
#         randomInts[i] = 12
        
# for index, i in enumerate(randomInts):
#     if (i < -3):
#         print(index, i)

# # Plot:
# axis = np.arange(start=min(randomInts), stop = max(randomInts) + 1)
# plt.hist(randomInts, bins = axis)

# a = np.zeros(200)
# xinc = 25
# yinc = 1150
# for i in range(200):
#     a[i] = (10 - abs(i % 40 - 20)) * xinc + yinc
# b = np.arange(200)
# plt.scatter(b, a)
# plt.axis([0, 40, -20 + yinc - xinc * 10, 20 + yinc + xinc * 10])
# plt.show()

# number_tasks = pd.read_csv("../data_task/summary.csv")
# print(number_tasks)

a = np.ones(200)
landa = 4
for i in range(200):
    j = i % 9
    a[i] = exp(-landa) * (landa**j) / factorial(j)*10000
b = np.arange(200)
plt.scatter(b, a)
plt.axis([0, 20, 0, 3000])
plt.show()















