import random as rd
import numpy as np
from pathlib import Path
import os
path =os.path.abspath(__file__)
path =Path(path).parent.parent
for i in range(200):
    with open("{}/{}/datatask{}.csv".format(str(path),"data_task",i),"w") as output:
        # indexs=rd.randint(900,1200)
        indexs=rd.randint(1100,1200)
        # m = np.sort(np.random.randint(i*300,(i+1)*300,indexs))
        randomNums = np.random.normal(i*300+150, 50, size=indexs)
        m = np.round(randomNums)
        m = np.sort(m)
        for j in range(len(m)):
            if m[j] < i*300:
                m[j] = i*300
            if m[j] > (i+1)*300 - 1:
                m[j] = (i+1)*300 - 1
        m1 = np.random.randint(1000,2000,indexs)
        m2 = np.random.randint(100,200,indexs)
        m3 = np.random.randint(500,1500,indexs)
        m4 = 1+np.random.rand(indexs)*2
        for j in range(indexs):
            output.write("{},{},{},{},{}\n".format(m[j],m3[j],m1[j],m2[j],m4[j]))
    #import pdb;pdb.set_trace()