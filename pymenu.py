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
    win = ui.main_game()
    if win != 0 :
        surface2 = pygame.display.set_mode((630, 730))
        menu = pygame_menu.Menu('Game end', 630, 730,
                        theme=pygame_menu.themes.THEME_DARK)
        if win == 1: 
            # menu.add.button('P1 win')
            menu.add.image("win.jpg")
        else: 
            # menu.add.button('P2 win')
            menu.add.image("win.jpg")
        menu.add.button('Quit', pygame_menu.events.EXIT)
        menu.mainloop(surface2)
def start_the_game():
    
    return 0


menu = pygame_menu.Menu('Game of Life', 630, 730,
                       theme=pygame_menu.themes.THEME_DARK)

# menu.add.text_input('Name :', default='Player')
menu.add.button('VS AI', start_the_game_withAI)
menu.add.button('Play', start_the_game)
menu.add.button('test', start_the_game)
menu.add.button('Quit', pygame_menu.events.EXIT)

menu.mainloop(surface1)