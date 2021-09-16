from tensorflow.keras.optimizers import Adam
import copy
import json
import timeit
import warnings
from tempfile import mkdtemp
import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from rl.agents.ddpg import DDPGAgent
#from rl.agents.dqn import DQNAgent
from rl.agents.sarsa import SARSAAgent
from rl.callbacks import Callback, FileLogger, ModelIntervalCheckpoint
from rl.memory import SequentialMemory
#from rl.policy import EpsGreedyQPolicy
from rl.random import OrnsteinUhlenbeckProcess
from tensorflow.keras.backend import cast
from tensorflow.keras.layers import (Activation, Concatenate, Dense, Dropout,
                                     Flatten, Input)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import sys

from fuzzy_controller import *
from enviroment import *
from model import *
from policy import *
from callback import *
from fuzzy_controller import *
import os

from dqnMEC import DQNAgent, BDQNAgent
from SarsaMEC import SARSAAgent

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
def Run_Random():
    files=open("random2.csv","w")
    files.write("kq\n")
    for i in range(15):
        tong=0
        h=0
        a=[]
        c=False
        while c==False:
            a,b,c,d=env.step(np.random.choice([0,0,0,0,0,1,2,3]))
            tong+=b
            if c==True :
                if i!=14:
                    env.reset()
                files.write(str(tong)+"\n")
                print(tong)

def Run_Fuzzy():
    sumreward = 0
    nreward = 0
    fuzzy_logic = Fuzzy_Controller()
    files = open("Fuzzy_5phut.csv","w")
    files1 = open("testFuzzy.csv","w")

    files.write("kq,sl,mean_reward\n")
    env = BusEnv("Fuzzy")
    #env.seed(123)
    start = timeit.default_timer()
    #env.reset()
    for i in range(100):
        tong=0
        h=0
        soluong=0

        a=env.observation
        c=False
        while c==False:
            #Rmi=(10*np.log2(1+46/(np.power(a[(1-1)*3],4)*100)))/8
            #m1=a[11]/a[2+(1-1)*3]+max(a[12]/(Rmi),a[1+(1-1)*3])
            # Rmi=(10*np.log2(1+46/(np.power(a[(2-1)*3],4)*100)))/8
            # m2=a[11]/a[2+(2-1)*3]+max(a[12]/(Rmi),a[1+(2-1)*3])
            # Rmi=(10*np.log2(1+46/(np.power(a[(3-1)*3],4)*100)))/8
            # m3=a[11]/a[2+(3-1)*3]+max(a[12]/(Rmi),a[1+(3-1)*3])
            # m0=a[9]+a[11]/a[10]
            #action=np.argmin([m0,m1,m2,m3])
            
            action=np.random.choice([0,0,0,1,2,3])
            action=0
            action=fuzzy_logic.choose_action(a)
            a,b,c,d=env.step(action)
            tong+=b
            sumreward = sumreward +b
            nreward = nreward + 1
            soluong+=1
            files1.write(str(sumreward / nreward)+"\n")
            if c==True :
                if i!=99:
                    env.reset()
                files.write(str(tong)+","+str(soluong)+","+str(tong/soluong)+"\n")
                print(tong)
    stop = timeit.default_timer()
    print('Time: ', stop - start)  
    files.close()
    
def Run_RGreedy(i, file):
    sumreward = 0
    nreward = 0
    files = open("RGreedy_5phut.csv","w")
    files1 = open("testRGreedy.csv","w")

    files.write("kq,sl,mean_reward\n")
    env = BusEnv("RGreedy")
    env.modifyEnv(i, file)
    for i in range(100):
        tong=0
        h=0
        soluong=0

        a=env.observation
        c=False
        while c==False:
            #Rmi=(10*np.log2(1+46/(np.power(a[(1-1)*3],4)*100)))/8
            #m1=a[11]/a[2+(1-1)*3]+max(a[12]/(Rmi),a[1+(1-1)*3])
            # Rmi=(10*np.log2(1+46/(np.power(a[(2-1)*3],4)*100)))/8
            # m2=a[11]/a[2+(2-1)*3]+max(a[12]/(Rmi),a[1+(2-1)*3])
            # Rmi=(10*np.log2(1+46/(np.power(a[(3-1)*3],4)*100)))/8
            # m3=a[11]/a[2+(3-1)*3]+max(a[12]/(Rmi),a[1+(3-1)*3])
            # m0=a[9]+a[11]/a[10]
            #action=np.argmin([m0,m1,m2,m3])

            predict_reward = env.predict_reward()
            action = int(max(predict_reward, key=predict_reward.get))
            a,b,c,d=env.step(action)
            tong+=b
            sumreward = sumreward +b
            nreward = nreward + 1
            soluong+=1
            files1.write(str(sumreward / nreward)+"\n")
            if c==True :
                if i!=99:
                    env.reset()
                files.write(str(tong)+","+str(soluong)+","+str(tong/soluong)+"\n")
                print(tong)
    stop = timeit.default_timer()
    #print('Time: ', stop - start)  
    files.close()

#using for DQL
def build_model(state_size, num_actions):
    input = Input(shape=(1,state_size))
    x = Flatten()(input)
    x = Dense(16, activation='relu')(x)

    x = Dense(32, activation='relu')(x)

    x = Dense(32, activation='relu')(x)
  
    x = Dense(8, activation='relu')(x)

    output = Dense(num_actions, activation='linear')(x)
    model = Model(inputs=input, outputs=output)
    return model

def Run_DQL(i, file):
    model=build_model(14,4)
    num_actions = 4
    policy = EpsGreedyQPolicy(0.1)
    env = BusEnv("DQL")
    env.modifyEnv(i, file)
    env.seed(123)
    memory = SequentialMemory(limit=25000, window_length=1)
    
    dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,\
              target_model_update=1e-3, policy=policy, gamma=0.8, memory_interval=1)
    files = open("testDQL.csv","w")
    files.write("kq\n")
    #create callback
    callbacks = CustomerTrainEpisodeLogger("./"+ str(file) +"/DQL_5phut_"+ str(i) +".csv")
    callback2 = ModelIntervalCheckpoint("./"+ str(file) +"/weight_DQL_"+ str(i) +".h5f",interval=50000)
    callback3 = TestLogger11(files)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    dqn.fit(env, nb_steps= 100000, visualize=False, verbose=2,callbacks=[callbacks,callback2])
    # dqn.test(env, nb_steps= 50000, visualize=False, verbose=2,callbacks=[callbacks,callback2])
    
def Run_BDQL(i, file):
    model=build_model(14,4)
    num_actions = 4
    # using static by setting policy2 to None
    # for dynamic, epsilon = min(epsilon, epsilon - k(average_reward - baseline))
    # epsilon = max(epsilon, 0.01)
    baseline = 0.5      
    k = 0.3
    epsilon = 0.1
    policy = EpsGreedyQPolicy(epsilon)
    policy2 = None      # None if not used, mean: using dynamic insted
    reward_capacity = 10000      # Queue that save the last "reward_capacity" rewards
    env = BusEnv("BDQL")
    env.modifyEnv(i, file)
    env.seed(123)
    memory = SequentialMemory(limit=25000, window_length=1)
    
    dqn = BDQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,\
              target_model_update=1e-3, policy=policy, gamma=0.8,
              memory_interval=1, i = i, file = file, reward_capacity = reward_capacity,
              k = k, epsilon = epsilon)
    files = open("testDQL.csv","w")
    files.write("kq\n")
    #create callback
    callbacks = CustomerTrainEpisodeLogger("./"+ str(file) +"/BDQL_5phut_"+ str(i) +".csv")
    callback2 = ModelIntervalCheckpoint("./"+ str(file) +"/weight_BDQL_"+ str(i) +".h5f",interval=50000)
    callback3 = TestLogger11(files)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    dqn.fit(env, nb_steps= 100000, visualize=False, verbose=2,callbacks=[callbacks,callback2],
            baseline = baseline)
    # dqn.test(env, nb_steps= 50000, visualize=False, verbose=2,callbacks=[callbacks,callback2])
    
def Run_DDQL(i, file):
    model=build_model(14,4)
    num_actions = 4
    policy = EpsGreedyQPolicy(0.1)
    env = BusEnv("DDQL")
    env.modifyEnv(i, file)
    env.seed(123)
    memory = SequentialMemory(limit=25000, window_length=1)
    
    dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,\
              target_model_update=1e-3, policy=policy,gamma=0.8,memory_interval=1,
              enable_double_dqn=True)
    files = open("testDDQL.csv","w")
    files.write("kq\n")
    #create callback
    callbacks = CustomerTrainEpisodeLogger("./"+ str(file) +"/DDQL_5phut_"+ str(i) +".csv")
    callback2 = ModelIntervalCheckpoint("./"+ str(file) +"/weight_DDQL_"+ str(i) +".h5f",interval=50000)
    callback3 = TestLogger11(files)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    dqn.fit(env, nb_steps= 100000, visualize=False, verbose=2,callbacks=[callbacks,callback2])
    # dqn.test(env, nb_steps= 30000, visualize=False, verbose=2,callbacks=[callbacks,callback2])
    
def Run_Sarsa(i, file):
    model=build_model(14,4)
    num_actions = 4
    policy = EpsGreedyQPolicy(0.1)
    env = BusEnv("Sarsa")
    env.modifyEnv(i, file)
    env.seed(123)
    
    try:
        dqn = SARSAAgent(model=model, nb_actions=num_actions, nb_steps_warmup=10,
                     policy=policy,gamma=0.8)
    except Exception as e:
        print(e)
    files = open("testSarsa.csv","w")
    files.write("kq\n")
    #create callback
    callbacks = CustomerTrainEpisodeLogger("./"+ str(file) +"/Sarsa_5phut_"+ str(i) +".csv")
    callback2 = ModelIntervalCheckpoint("./"+ str(file) +"/weight_Sarsa_"+ str(i) +".h5f",interval=50000)
    callback3 = TestLogger11(files)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    try:
        dqn.fit(env, nb_steps= 100000, visualize=False, verbose=2,callbacks=[callbacks,callback2])
    except Exception as e:
        print(e)

def Run_FDQO(i, file):
    FDQO_method = Model_Deep_Q_Learning(14,4)    #In model  size, action
    baseline = 0  # None if using FDQO, >0 and <1 if using baseline
    threshold = 0.9     # if reward received bigger than threshold, using Fuzzy Logic
    k = 0.3     # Same formula as BDQL
    epsilon = 0.1
    model = FDQO_method.build_model(epsilon = epsilon, name = i, file = file,
                                    k = k, threshold = threshold)
    #Create enviroment FDQO
    env = BusEnv("FDQO")
    env.modifyEnv(i, file)
    env.seed(123)
    #create memory
    memory = SequentialMemory(limit=5000, window_length=1)
    #open files
    files = open("testFDQO.csv","w")
    files.write("kq\n")
    #create callback
    callbacks = CustomerTrainEpisodeLogger("./"+ str(file) +"/FDQO_5phut_"+ str(i) +".csv")
    callback2 = ModelIntervalCheckpoint("./"+ str(file) +"/weight_FDQO_"+ str(i) +".h5f",interval=50000)
    callback3 = TestLogger11(files)
    model.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    model.fit(env, nb_steps= 100000, visualize=False, verbose=2,callbacks=[callbacks,callback2],
              baseline = baseline, eps = 1)
    #model.fit(env, nb_steps= 130000, visualize=False, verbose=2,callbacks=[callbacks,callback2])
    files.close()

if __name__=="__main__":
    file = "csvFilesNorm" # Location to save all the results
    # types = "DQL"
    # if len(sys.argv) > 1:
    #     types = sys.argv[1]
    # if types =="FDQO":
    #     Run_FDQO()
    # elif types == "Random":
    #     Run_Random()
    # elif types == "Fuzzy":
    #     Run_Fuzzy()
    # elif types == "DQL":
    #     Run_DQL()
    # elif types == "DDQL":
    #     Run_DDQL()
    #create model FDQO
    for i in range(0,1):
        try:
            #Run_DQL("M900_1000_mem25_2", file)
            #Run_BDQL("M900_1000_dyn_e0.1_k0.3_que10k_b0.5", file)
            #Run_DDQL("M900_1000_mem25_2", file)
            Run_FDQO("M900_1000_test", file)
            #Run_RGreedy("M900_1000", file)
            #Run_Sarsa("M900_1000", file)
        except:
            continue
