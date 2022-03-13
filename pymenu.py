import pygame
import pygame_menu
import pygame
import time
import random
import numpy as np
import os
import ui

pygame.init()
surface1 = pygame.display.set_mode((630, 730))

def start_the_game_withAI():
    ui.main_game()

def start_the_game():
    
    return 0


menu = pygame_menu.Menu('Game of Life', 630, 730,
                       theme=pygame_menu.themes.THEME_ORANGE)

menu.add.text_input('Name :', default='Player')
menu.add.button('VS AI', start_the_game_withAI)
menu.add.button('Play', start_the_game)
menu.add.button('test', start_the_game)
menu.add.button('Quit', pygame_menu.events.EXIT)

menu.mainloop(surface1)