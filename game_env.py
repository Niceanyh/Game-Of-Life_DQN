
import numpy as np
def reset():
    return new_state()
def new_state():
    return State(initial_board())

def initial_board():
    board = np.zeros((9,9),dtype="int")
    board[3][2]=1 
    board[4][1]=1
    board[4][3]=1
    board[5][2]=1
    board[3][6]=2
    board[4][5]=2
    board[4][7]=2
    board[5][6]=2
    return board

def initial_board_max():
    board = np.zeros((9,9),dtype="int")
    for i in range(len(board)):
        for j in range(len(board[0])):
            board[i][j] = 2
    return board

def living_reward(board,player):
        counter=0
        for x in range(len(board[0])):
                for y in range(len(board)):
                    cell_state = board[x][y]
                    if cell_state == player:
                        counter+=1
        return counter

def get_action_id(action):
    id  = action[0]*9+action[1]
    return id 

def get_action(id):
    action = np.array([id//9,id%9])
    return action

class environment():
    def get_generation(self):
        return environment.generation

class State(object):
    def __init__(self,board):
        # self.n_actions=80
        self.board = board
        self.n_actions=81
        self.n_features=81
        self._episode_ended = False
        self.gen=0
    
    def print_board(self):
        print(self.board)
    
    def apply_action(self,action_id,player):
        
        #action : Array[int,int]  player:1 or 2
        action = get_action(action_id)
        
        self.board[action[0]][action[1]]=player
    
    def get_neighbours(self, x, y):# return (a,b) stand for a p1 neibour and b p2 neibour
        player1_neighbour=0
        player2_neighbour=0
        rows=len(self.board[0])
        cols=len(self.board)
        for n in range(-1, 2):
            for m in range(-1, 2):  
                x_adjust= (x+n+rows)% rows
                y_adjust= (y+m+cols)% cols
                if self.board[x_adjust][y_adjust] == 1:
                    player1_neighbour+=1
                elif self.board[x_adjust][y_adjust]==2:
                    player2_neighbour+=1
        if self.board[x][y] == 1:
            player1_neighbour-=1
        elif self.board[x][y] == 2:
            player2_neighbour-=1
        return player1_neighbour,player2_neighbour

    def legal_actions(self):#return a list with the format [int,int]
        legal_actions=[]
        for x in range(len(self.board[0])):
                for y in range(len(self.board)):
                    if self.board[x][y] == 0 :
                        action = 10*x+y
                        legal_actions.append(action)
        return legal_actions

    def evolve(self):
        next = np.zeros(shape=(9,9),dtype="int")
        for x in range(len(self.board[0])):
                for y in range(len(self.board)):
                    cell_state = self.board[x][y]
                    p1_neighbours = self.get_neighbours( x, y)[0] 
                    p2_neighbours = self.get_neighbours( x, y)[1] 
                    neighbours = p1_neighbours + p2_neighbours
                    
                    if cell_state == 0: #empty cell -> born
                        if neighbours == 3:
                            if p1_neighbours > p2_neighbours: 
                                next[x][y] = 1
                            else: next[x][y] = 2
                            
                    elif cell_state == 1: #living cell -> die
                        if p1_neighbours >= 4 or p1_neighbours <= 1:
                            next[x][y] = 0
                        else: #keep the same state
                            next[x][y] = 1
                    elif cell_state == 2: 
                        if p2_neighbours >= 4 or p2_neighbours <= 1:
                            next[x][y] = 0
                        else: #keep the same state
                            next[x][y] = 2
        return next

    def is_not_end(self):
        num_player1 = living_reward(self.board,1)
        num_player2 = living_reward(self.board,2)
        #if (player1 != 0 and player1 < 20 and player2 != 0):
        if (num_player1 != 0 and num_player2 != 0):
            return False
        else: return True
        
    def step(self,action_id,player=1):
        action = get_action(action_id)
        #todo check if an action is valid
        # if(self.board[action[0]][action[1]]!=0):
        self.board[action[0]][action[1]]=player
        
        observation_ = self.evolve() # conway game evolve to next gen
        self.board = observation_
        done = self.is_not_end()
        # if done: reward = living_reward(observation_,player)
        # else: reward=0
        reward = living_reward(observation_,player)
        if done:
            if reward == 0: reward = -100
            else: reward*=10
        return observation_,reward,done
    
    def reset():
    # state at the start of the game
        board = initial_board()
        return State(board).board
    
    def get_invalid_action(self):
        action = []
        board =self.board
        for x in range(len(board[0])):
                for y in range(len(board)):
                    cell_state = board[x][y]
                    if cell_state ==1 or cell_state == 2:
                        temp=[]
                        temp.append(x)
                        temp.append(y)
                        action.append(get_action_id(temp))
        return action



# conway = State(initial_board())
# print(conway.board)
# print(conway.board.shape)
# print(conway.step(2,3))
