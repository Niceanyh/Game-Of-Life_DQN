import pygame
import game_env
import numpy as np
import pygamepopup
from pygamepopup.menu_manager import MenuManager
from pygamepopup.components import Button, InfoBox
from tkinter import *
from tkinter import messagebox
from keras_dqn import DQNAgent
pygame.init()
pygamepopup.init()

game = game_env.new_state()
state_size = game.n_features
action_size = game.n_actions
agent = DQNAgent(state_size, action_size)
agent.load("dqn.h5")


myfont = pygame.font.Font(None,20)
game_end_font = pygame.font.Font(None,100)
font_gen = pygame.font.Font(None,50)
WHITE = (200, 200, 200)
BLACK = (0,0,0)
blockSize = 70
menu = 100
WINDOW_WIDTH = 630
WINDOW_HEIGHT = WINDOW_WIDTH+menu
p1_color = (94,50,148)
p1ai_prepare_color = (90,44,130)
p2_color = (50,168,78)
screen = pygame.display.set_mode([WINDOW_WIDTH,WINDOW_HEIGHT])
menu_manager = MenuManager(screen)


evolve_event = pygame.USEREVENT + 1
ai_pre_event = pygame.USEREVENT + 2

#set timer
pygame.time.set_timer(evolve_event, 1000) #evolve every 3000ms
pygame.time.set_timer(ai_pre_event, 1400)


def drawGrid():
     #Set the size of the grid block
    for x in range(0, WINDOW_WIDTH, blockSize):
        for y in range(100, WINDOW_HEIGHT, blockSize):
            rect = pygame.Rect(x, y, blockSize, blockSize)
            pygame.draw.rect(screen, BLACK, rect, 1)

def draw_ai_grid(x_ai_next,y_ai_next):
    x_ai_next = x_ai_next * blockSize
    y_ai_next = menu + y_ai_next * blockSize
    pygame.draw.rect(screen, p1ai_prepare_color, [x_ai_next, y_ai_next, blockSize, blockSize])
    
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
    run = True
    game_over_p2win = False
    game_over_p1win = False
    draw_ai_action = False
    Board = game_env.reset()
    while run:
        screen.fill(WHITE)
        drawGrid()
        # if draw_ai_action == True:
        #     draw_ai_grid(x_ai_next,y_ai_next)
        drawP(Board.board.T)
        GenerationText = font_gen.render("Generation %d "%Board.gen, True,(175, 215, 70),(0,0,120))
        screen.blit(GenerationText, (10,0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    run = False
                # if event.key == pygame.K_SPACE:
                #     pause = not pause
            
            if event.type == evolve_event:
                id= agent.act(Board.board)
                x,y = game_env.get_action(id)
                Board.step(id)
                Board.gen+=1
                draw_ai_action = False
                print(x,y)
                print(Board.gen)
                # print(Board.board)
                # Board.step(id)
            if event.type == ai_pre_event:
                draw_ai_action = True
                x_ai_next,y_ai_next = game_env.get_action(agent.act(Board.board))
                
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
        screen.fill(WHITE)
        GenerationText = game_end_font.render("p1_win", True,(175, 215, 70),(0,0,120))
        screen.blit(GenerationText, (100,200))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over_p1win = False
        pygame.display.update()
        
    if game_over_p2win:
        screen.fill(WHITE)
        GenerationText = game_end_font.render("p2_win", True,(175, 215, 70),(0,0,120))
        screen.blit(GenerationText, (100,200))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over_p2win = False
        pygame.display.update()
    
    # pygame.quit()





#TODO time interval visulization
#TODO 