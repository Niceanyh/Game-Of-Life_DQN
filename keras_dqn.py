# -*- coding: utf-8 -*-
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
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
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

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
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
    
    
    # @tf.function
    # def update_gradient(self, target_q, n_step_target_q, states, actions, batch_weights=1):

    #     """
    #     Update main q network by experience replay method.
    #     Args:
    #         target_q (tf.float32): Target Q value for barch.
    #         n_step_target_q (tf.int32): Target Q value after n_step.
    #         states (tf.float32): Batch of states.
    #         actions (tf.int32): Batch of actions.
    #         batch_weights(tf.float32): weights of this batch.
    #     """

    #     self.policy_network.update_lr()
    #     with tf.GradientTape() as tape:
    #         tape.watch(self.policy_network.model.trainable_weights)
    #         main_q = tf.reduce_sum(
    #             self.policy_network.model(states) * tf.one_hot(actions, self.n_actions, 1.0, 0.0),
    #             axis=1)

    #         losses = self.policy_network.loss_function(main_q, target_q) * self.policy_network.one_step_weight
    #         losses += self.policy_network.loss_function(main_q, n_step_target_q) * self.policy_network.n_step_weight

    #         if self.policy_network.l2_weight > 0:
    #             losses += self.policy_network.l2_weight * tf.reduce_sum(
    #                 [tf.reduce_sum(tf.square(layer_weights))
    #                  for layer_weights in self.policy_network.model.trainable_weights])

    #         loss = tf.reduce_mean(losses * batch_weights)

    #     self.policy_network.optimizer.minimize(loss, self.policy_network.model.trainable_variables, tape=tape)

    #     self.loss_metric.update_state(loss)
    #     self.q_metric.update_state(main_q)

    #     return main_q, loss

def eval(agent,episode = 50):
    win = 0
    for i in range(episode):
        State = game_env.reset()
        for step in range(500):
            state = State.board
            action = agent.act(state)
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
    print("init success")
    for e in range(EPISODES):
        State = game_env.reset()
        for time in range(500):
            # env.render()
            state = State.board
            action = agent.act(state)
            next_state, reward, done = State.step(action)
            # next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            # state = next_state
            # print("time:",time,"reward:",reward)
            if done:
                print("episode: {}/{}, score: {}, e: {:.4}"
                      .format(e, EPISODES, reward, agent.epsilon))
                break
            if len(agent.memory) > batch_size and time % 5 == 0:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
        if(e % 50 == 0):
            percentage = eval(agent)
            win_his.append(percentage)
            print("_________eval_______win percentage:",percentage)
    
    agent.save("dqn.h5")
    x = np.arange(0,EPISODES,2)
    plt.plot(x,win_his)
    plt.xticks(x)
    plt.xticks(x[::2])
    plt.ylabel('win_percentage')
    plt.xlabel('episode')
    plt.savefig("save.png")