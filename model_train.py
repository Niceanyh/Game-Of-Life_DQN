from keras_dqn import DQNAgent
import game_env
import matplotlib.pyplot as plt
import numpy as np
import kmean_sound

EPISODES = 1000

def eval(agent,episode = 50):
    win = 0
    for i in range(episode):
        State = game_env.reset()
        for step in range(500):
            state = State.board
            action1 = agent.act(state,State.get_invalid_action())
            action2 = agent.random_act(State.get_invalid_action())
            while(action2==action1):
                action2 = agent.random_act(State.get_invalid_action())
            # This is a random action 
            # action = agent.random_act(State,State.get_invalid_action())
            next_state, reward, done = State.step(action1,action2,singlePlayer=False)
            
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
    agent.load("DQN_model/5000+10002p_wr0854.h5")
    agent.epsilon = 0.002
    batch_size = 16
    win_his = []
    win_p = 0 
    win_ph = []
    state_kmeans=[]
    print("init success")
    for e in range(EPISODES):
        State = game_env.reset()
        for time in range(500):
            state = State.board
            action1 = agent.act(state,State.get_invalid_action())
            action2 = agent.random_act(State.get_invalid_action())
            while(action2==action1):
                action2 = agent.random_act(State.get_invalid_action())
            # This is a random action 
            # action = agent.random_act(State,State.get_invalid_action())
            next_state, reward, done = State.step(action1,action2,singlePlayer=False)
            agent.memorize(state, action1, reward, next_state, done)
            # print("time:",time,"reward:",reward)
            state_kmeans.append(next_state)
            if done:
                if reward>0:
                    win_p+=1
                print("episode: {}/{}, score: {},win_p: {} e: {:.4}"
                      .format(e, EPISODES, reward,win_p/(e+1), agent.epsilon))
                win_ph.append(win_p/(e+1))
                break
            if len(agent.memory) > batch_size and time % 5 == 0:
                agent.replay(batch_size)
        
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
        if(e % 500 == 0):
            State = game_env.reset()
            percentage = eval(agent)
            win_his.append(percentage)
            print("_________eval_______win percentage:",percentage)
    
    agent.save("dqn.h5")
    # kmean_sound.train(state_kmeans)
    
    print('save success')
    x = np.arange(0,EPISODES,50)
    plt.plot(x,win_his)
    plt.xticks(x)
    plt.xticks(x[::2])
    plt.ylabel('evual_win_percentage')
    plt.xlabel('episode')
    plt.savefig("evual_win_percentage.png")
    plt.close()
    # print(win_ph)
    plt.plot(win_ph)
    plt.ylabel('accumulated_win_percentage')
    plt.xlabel('episode')
    plt.savefig("accumulated_win_percentage.png")