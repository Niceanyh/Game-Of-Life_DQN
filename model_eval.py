from keras_dqn import DQNAgent
import game_env

EPISODES = 1000

game = game_env.new_state()
state_size = game.n_features
action_size = game.n_actions
agent = DQNAgent(state_size, action_size)
# agent.load("5000e,88win.h5")
agent.load("dqn.h5")
agent.epsilon=0
win_acount_p1=0
win_acount_p2=0
e=0
for e in range(EPISODES):
    e+=1
    State = game_env.reset()
    for time in range(500):
        state = State.board
        action1 = agent.act(state,State.get_invalid_action())
        
        # This is a random action 
        # action1 = agent.random_act(State.get_invalid_action())
        action2 = agent.random_act(State.get_invalid_action())
        print(action2)
        while(action2==action1):
            action2 = agent.random_act(State.get_invalid_action())
        next_state, reward, done = State.step(action1,action2,singlePlayer=False)
        # next_state = np.reshape(next_state, [1, state_size])
        # state = next_state
        # print("time:",time,"reward:",reward)
        if done:
            if reward>0:
                win_acount_p1+=1
            break
    print("episode: ",e," and reward: ",reward)
print("fianl wining:",win_acount_p1/EPISODES)