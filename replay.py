from keras_dqn import DQNAgent
import game_env
import matplotlib.pyplot as plt
import numpy as np
import kmean_sound

EPISODES = 500
if __name__ == "__main__":
    game = game_env.new_state()
    state_size = game.n_features
    action_size = game.n_actions
    agent = DQNAgent(state_size, action_size)
    agent.load("DQN_model/5000+10002p_wr0854.h5")
    agent.epsilon = 0.002
    state_kmeans=[]
    print("init success")
    for e in range(EPISODES):
        print(e)
        State = game_env.reset()
        for time in range(500):
            state = State.board
            action1 = agent.random_act(State.get_invalid_action())
            action2 = agent.random_act(State.get_invalid_action())
            while(action2==action1):
                action2 = agent.random_act(State.get_invalid_action())
            # This is a random action 
            # action = agent.random_act(State,State.get_invalid_action())
            next_state, reward, done = State.step(action1,action2,singlePlayer=False)
            # print("time:",time,"reward:",reward)
            state_kmeans.append(next_state)
            if done:
                break
        

    print(len(state_kmeans))
    state_kmeans = np.array(state_kmeans)
    x = state_kmeans.reshape(len(state_kmeans),81)
    kmean_sound.train(x)
    print("success")
    