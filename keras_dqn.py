# -*- coding: utf-8 -*-
import math
import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
# from keras.optimizers import Adam
import matplotlib.pyplot as plt
import game_env

EPISODES = 2000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.005
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(300, input_dim=9, activation='relu'))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, invalid_action,apporach=1):
        if apporach ==1 and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        for ia in invalid_action:
            act_values[0][ia] = - math.inf
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # predict the future discounted reward
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            # make the agent to approximately map
            # the current state to future discounted reward
            target_f = self.model.predict(state)
            target_f[0][action] = target
            ##TODO
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
    
    def random_act(self,invalid_action):
        temp = list(np.array(range(0,self.action_size,1)))
        for ia in invalid_action:
            temp.remove(ia)
        action = random.choice(temp)
        return action

def eval(agent,episode = 1):
    win = 0
    for i in range(episode):
        State = game_env.reset()
        for step in range(500):
            state = State.board
            action = agent.act(state,State.get_invalid_action(),0)
            next_state, reward, done = State.step(action)
            if done:
                if reward>0:
                    win+=1
                break
    return win/episode

if __name__ == "__main__":
    game = game_env.new_state()
    state_size = game.n_features
    action_size = game.n_actions
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 16
    win_his = []
    win_p = 0 
    win_ph = []
    print("init success")
    for e in range(EPISODES):
        State = game_env.reset()
        for time in range(500):
            # env.render()
            state = State.board
            action1 = agent.act(state,State.get_invalid_action())
            # This is a random action 
            action2 = agent.random_act(State.get_invalid_action())
            
            while action1==action2:
                action2 = agent.random_act(State.get_invalid_action())
            next_state, reward, done = State.step(action1,action2,singlePlayer=False)
            # next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action1, reward, next_state, done)
            # state = next_state
            # print("time:",time,"reward:",reward)
            if done:
                if reward>0:
                    win_p+=1
                print("episode: {}/{}, score: {},win_p: {} e: {:.4}"
                      .format(e, EPISODES, reward,win_p/(e+1), agent.epsilon))
                win_ph.append(win_p/(e+1))
                break
            if len(agent.memory) > batch_size and time % 5 == 0:
                agent.replay(batch_size)
        
        # if(e % 50 == 0):
        #     State = game_env.reset()
        #     percentage = eval(agent)
        #     win_his.append(percentage)
        #     print("_________eval_______win percentage:",percentage)
    
    agent.save("dqn.h5")
    print('save success')
    # x = np.arange(0,EPISODES,50)
    # plt.plot(x,win_his)
    # plt.xticks(x)
    # plt.xticks(x[::2])
    # plt.ylabel('win_percentage')
    # plt.xlabel('episode')
    # plt.savefig("save1.png")
    # plt.close()
    # print(win_ph)
    plt.plot(win_ph)
    plt.ylabel('win_percentage')
    plt.xlabel('episode')
    plt.savefig("save2.png")