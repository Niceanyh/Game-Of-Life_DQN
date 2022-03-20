import pygame
import game_env
import numpy as np
from tkinter import *
import random
from keras_dqn import DQNAgent
import pickle
pygame.init()

def load_music():
    sound = []
    sound.append(pygame.mixer.Sound("sound/A/A1.mp3"))
    sound.append(pygame.mixer.Sound("sound/A/A2.mp3"))
    sound.append(pygame.mixer.Sound("sound/A/A3.mp3"))
    sound.append(pygame.mixer.Sound("sound/A/A4.mp3"))
    sound.append(pygame.mixer.Sound("sound/A/A5.mp3"))
    sound.append(pygame.mixer.Sound("sound/A/A6.mp3"))
    sound.append(pygame.mixer.Sound("sound/A/A7.mp3"))
    
    sound.append(pygame.mixer.Sound("sound/G/G1.mp3"))
    sound.append(pygame.mixer.Sound("sound/G/G2.mp3"))
    sound.append(pygame.mixer.Sound("sound/G/G3.mp3"))
    sound.append(pygame.mixer.Sound("sound/G/G4.mp3"))
    sound.append(pygame.mixer.Sound("sound/G/G5.mp3"))
    sound.append(pygame.mixer.Sound("sound/G/G6.mp3"))
    sound.append(pygame.mixer.Sound("sound/G/G7.mp3"))

    sound.append(pygame.mixer.Sound("sound/D/D1.mp3"))
    sound.append(pygame.mixer.Sound("sound/D/D2.mp3"))
    sound.append(pygame.mixer.Sound("sound/D/D3.mp3"))
    sound.append(pygame.mixer.Sound("sound/D/D4.mp3"))
    sound.append(pygame.mixer.Sound("sound/D/D5.mp3"))
    sound.append(pygame.mixer.Sound("sound/D/D6.mp3"))
    sound.append(pygame.mixer.Sound("sound/D/D7.mp3"))
    return sound
game = game_env.new_state()
state_size = game.n_features
action_size = game.n_actions
agent = DQNAgent(state_size, action_size)
agent.load("DQN_model/5000+10002p_wr0854.h5")
agent.epsilon = 0 #no exploitation
sound = load_music()


myfont = pygame.font.Font(None,20)
game_end_font = pygame.font.Font(None,100)
font_gen = pygame.font.Font(None,50)
WHITE = (200, 200, 200)
BLACK = (0,0,0)
blockSize = 70
menu = 100
WINDOW_WIDTH = 630
WINDOW_HEIGHT = WINDOW_WIDTH+menu
bg = (253,252,223)
p1_color = (225,120,108)
p1_prepare_color = (248,218,187)
p2_prepare_color = (76,172,183)
p2_color = (53,128,163)
screen = pygame.display.set_mode([WINDOW_WIDTH,WINDOW_HEIGHT])


evolve_event = pygame.USEREVENT + 1
ai_pre_event = pygame.USEREVENT + 2


#set timer
pygame.time.set_timer(evolve_event, 2000) #evolve every 3000ms
if pygame.time.get_ticks() == 500:
            pygame.time.set_timer(ai_pre_event, 2000)
            print("set success")


def drawGrid():
     #Set the size of the grid block
    for x in range(0, WINDOW_WIDTH, blockSize):
        for y in range(100, WINDOW_HEIGHT, blockSize):
            rect = pygame.Rect(x, y, blockSize, blockSize)
            pygame.draw.rect(screen, BLACK, rect, 1)

def draw_ai_grid(x_ai_next,y_ai_next):
    x_ai_next = x_ai_next * blockSize
    y_ai_next = menu + y_ai_next * blockSize
    pygame.draw.rect(screen, p1_prepare_color, [x_ai_next, y_ai_next, blockSize, blockSize])
    
def drawP(grid):
    grid == np.flipud(grid)
    for x in range(9):
            for y in range(9):
                x_pos = x * blockSize
                y_pos = menu + y * blockSize
                
                #random_color = (random.randint(10, 255), random.randint(10, 255), random.randint(10, 255))
                if grid[x][y] == 1:
                    pygame.draw.rect(screen, p1_color, [x_pos, y_pos, blockSize, blockSize])
                elif grid[x][y] == 2:
                    pygame.draw.rect(screen, p2_color, [x_pos, y_pos, blockSize, blockSize])
                # else:
    # x_ai_next,y_ai_next = game_env.get_action(agent.act(Board.board))
    # x_ai_next = x_ai_next * blockSize
    # y_ai_next = menu + y_ai_next * blockSize
    # pygame.draw.rect(screen, p1ai_prepare_color, [x_ai_next, y_ai_next, blockSize, blockSize])
    
def get_mouse_action(mouseX,mouseY):
    x,y = mouseX//blockSize,(mouseY-menu)//blockSize
    return y,x

def check_game_over(grid):
    if game_env.living_reward(grid,1) == 0: return 1
    elif game_env.living_reward(grid,2) == 0: return 2
    else: return 0



def main_game():
    kmeans = pickle.load(open("kmeans_sound.pkl", "rb"))
    run = True
    game_over_p2win = False
    game_over_p1win = False
    draw_ai_action = False
    Board = game_env.reset()
    
    while run:
        screen.fill(bg)
        drawP(Board.board.T)
        if draw_ai_action == True:
            print("draw_ai_action")
            x_ai_next,y_ai_next = game_env.get_action(
                    agent.act(Board.board,Board.get_invalid_action()))
            draw_ai_grid(x_ai_next,y_ai_next)
        
        drawGrid()
        GenerationText = font_gen.render("Generation %d "%Board.gen, True,(175, 215, 70),(0,0,120))
        screen.blit(GenerationText, (10,0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                return 0
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    run = False
                if event.key == pygame.K_SPACE:
                    pause = not pause
            
            if event.type == evolve_event:
                sound_index = int(kmeans.predict(Board.board.flatten().reshape(1,81)))
                music = sound[sound_index]
                # before_num = game_env.living_reward(Board,1)
                # music1 = random.choice(list(sound))
                pygame.mixer.Sound.play(music)
                pygame.mixer.music.stop()
                id= agent.act(Board.board,Board.get_invalid_action())
                x,y = game_env.get_action(id)
                observation_,reward,done = Board.step(id)
                # after_num = game_env.living_reward(Board,1)
                Board.gen+=1
                draw_ai_action = False
                # print(x,y)
                # print(Board.gen)
                # print(Board.board)
            if event.type == ai_pre_event:
                draw_ai_action = True
                
                
        if pygame.mouse.get_pressed()[0]:
            mouseX, mouseY = pygame.mouse.get_pos()
            # print(mouseX, mouseY )
            if(mouseY>=100):
                x,y = get_mouse_action(mouseX, mouseY)
                Board.board[x][y] = 2
        
        if (check_game_over(Board.board) == 1):
            # my_custom_menu = InfoBox("Title of the Menu",[Button(title="p2 win!",callback=lambda: None)])
            # menu_manager.open(my_custom_menu)
            run = False
            game_over_p2win = True
            print("p2 win")
        elif (check_game_over(Board.board) == 2):
            # screen.fill(WHITE)
            # GenerationText = myfont.render("p1_win", True,(175, 215, 70),(0,0,120))
            # screen.blit(GenerationText, (200,200))
            run = False
            game_over_p1win = True
            print("p1 win")
        pygame.display.update()
        
        # pygame.draw.circle(screen, (0, 0, 255), (250, 250), 75)
        # pygame.display.flip()
    if game_over_p1win:
        return 1
        
    if game_over_p2win:
        return 2
    
    # pygame.quit()





#TODO time interval visulization
#TODO 