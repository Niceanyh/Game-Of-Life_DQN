import pygame
import pygame_menu
import random
import numpy as np
import ui

sound = ui.load_music()
bg = (253,252,223)
pygame.init()
surface1 = pygame.display.set_mode((630, 730))
music = random.choice(list(sound))
pygame.mixer.Sound.play(music)
def changeTomenu():
    menu.mainloop(surface1)
def start_the_game_withAI():
    win = ui.main_game()
    if win != 0 :
        surface2 = pygame.display.set_mode((630, 730))
        menu = pygame_menu.Menu('Game end', 630, 730,
                        theme=pygame_menu.themes.THEME_DEFAULT)
        if win == 1: 
            # menu.add.button('P1 win')
            menu.add.image("ui_img/aiwin.jpg")
        else: 
            # menu.add.button('P2 win')
            menu.add.image("ui_img/youwin.jpg")
        menu.add.button('Quit', pygame_menu.events.EXIT)
        menu.add.button('Return',changeTomenu)
        menu.mainloop(surface2)
    else:
        pygame_menu.events.EXIT
def get_rule():
    # ismenu = True
        surface3 = pygame.display.set_mode((630, 730))
        menu = pygame_menu.Menu('Survival Rules', 630, 730,
                        theme=pygame_menu.themes.THEME_DEFAULT)
        menu.add.image("ui_img/rule.jpg")
        menu.add.button('Return',changeTomenu)
        menu.mainloop(surface3)

# mytheme = pygame_menu.Theme(background_color=bg, # transparent background
#                 title_background_color=(225,120,108),
#                 title_font_shadow=True,
#                 widget_padding=5,
# )
menu = pygame_menu.Menu('Game of Life', 630, 730,
                       theme=pygame_menu.themes.THEME_DEFAULT)

# menu = pygame_menu.Menu('Game of Life', 630, 730,
#                        theme=mytheme)
# name = menu.add.text_input('Name :', default='Player',textinput_id='name',
#                            enable_selection=True)
menu.add.image("ui_img/game_begin.jpg")
menu.add.button('Play', start_the_game_withAI)
menu.add.button('Rule', get_rule)
menu.add.button('Quit', pygame_menu.events.EXIT)

menu.mainloop(surface1)