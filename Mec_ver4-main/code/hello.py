print("hello")

import numpy as np
import matplotlib.pyplot as plt

# Generate Distribution:
randomNums = np.random.normal(6, 2, size=1000)
randomInts = np.round(randomNums)
print( randomInts[0])
for i in range(len(randomInts)):
    if randomInts[i]<0:
        randomInts[i] = 0
    if randomInts[i]>12:
        randomInts[i] = 12
        
for index, i in enumerate(randomInts):
    if (i < -3):
        print(index, i)

# Plot:
axis = np.arange(start=min(randomInts), stop = max(randomInts) + 1)
plt.hist(randomInts, bins = axis)
